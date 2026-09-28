"""Corrected, labeled float64 artifacts. Historical arrays are never imported."""

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from triatomic_batched import SOLVER_VERSION, TriatomicBatchSolver
from triatomic_execution import execution_batches


SCHEMA = "corrected-triatomic-data-v1"
GRID = np.linspace(0.001, 1.0, 500)
POPULATIONS = ("train", "validation_dense", "validation_sparse", "test_dense",
               "test_sparse", "adversarial", "showcase")
SHOWCASE_IDS = ["tri-five-a", "tri-five-b", "tri-five-c", "tri-five-d"]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    """Replace a progress record only after its bytes are durable."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_array(path, array):
    with Path(path).open("xb") as output:
        np.save(output, array, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())


def label_order(interactions):
    if type(interactions) is not int or not 5 <= interactions <= 20:
        raise ValueError("corrected pilot supports 5..20 physical interactions")
    return ["m2", "m3"] + [f"k{i}" for i in range(2, interactions + 1)]


def validate_labels(labels, interactions):
    if labels.ndim != 2 or labels.shape[1] != len(label_order(interactions)) or len(labels) == 0:
        raise ValueError("incorrect physical-label shape")
    if not np.all(np.isfinite(labels)) or np.iscomplexobj(labels):
        raise ValueError("physical labels must be finite and real")
    if np.any((labels[:, :2] < 0.1) | (labels[:, :2] > 10)):
        raise ValueError("mass ratios must lie in [0.1,10]")
    if np.any((labels[:, 2:] < 0) | (labels[:, 2:] > 10)):
        raise ValueError("spring ratios must lie in [0,10]")


def physical_arrays(labels, interactions):
    validate_labels(labels, interactions)
    return (np.column_stack((np.ones(len(labels)), labels[:, :2])),
            np.column_stack((np.ones(len(labels)), labels[:, 2:])))


def sample_labels(count, interactions, seed, population):
    """Separate PCG64 streams per named split; no global RNG or seed defaults."""
    label_order(interactions)
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive integer count and nonnegative integer seed required")
    if population not in POPULATIONS[:5]:
        raise ValueError("unknown random population")
    if population == "train" and count % 2:
        raise ValueError("training count must be even for the exact half/half mixture")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [seed, interactions, POPULATIONS.index(population)])))
    labels = np.column_stack((rng.uniform(0.1, 10, (count, 2)),
                              rng.uniform(0, 10, (count, interactions - 1))))
    if population == "train":
        labels[count // 2:, 2:] *= rng.integers(0, 2, (count // 2, interactions - 1))
    elif population.endswith("sparse"):
        labels[:, 2:] *= rng.integers(0, 2, (count, interactions - 1))
    return labels


def showcase_labels(interactions):
    labels = np.zeros((4, len(label_order(interactions))), dtype=np.float64)
    labels[:, :6] = [[8, 5, 6, 6, 4, 8], [1, 7, 2, 8, 1, 6],
                     [4, 4, 8, 4, 9, 4], [6, 1, 3, 0, 7, 9]]
    return labels


def adversarial_labels(interactions):
    """M5: all 16 masks at four mass corners, equal and near-equal masses.

    Higher counts add each higher neighbor separately and fully dense corners;
    this is bounded coverage, not enumeration of all 2**(K-1) masks.
    """
    width = len(label_order(interactions))
    rows = []
    for masses in ((0.1, 0.1), (0.1, 10), (10, 0.1), (10, 10),
                   (1, 1), (1, 1 + 1e-7)):
        for mask in range(16):
            row = np.zeros(width)
            row[:2] = masses
            row[2:6] = [10 * ((mask >> i) & 1) for i in range(4)]
            rows.append(row)
    if interactions > 5:
        for neighbor in range(6, interactions + 1):
            row = np.zeros(width)
            row[:2] = [0.1, 10]
            row[neighbor] = 10
            rows.append(row)
        for masses in ((0.1, 10), (10, 0.1)):
            row = np.full(width, 10.0)
            row[:2] = masses
            rows.append(row)
    return np.asarray(rows, dtype=np.float64)


def population_labels(config):
    k, seed = config["interactions"], config["seed"]
    counts = {"train": config["train_count"],
              "validation_dense": config["validation_count_per_population"],
              "validation_sparse": config["validation_count_per_population"],
              "test_dense": config["test_count_per_population"],
              "test_sparse": config["test_count_per_population"]}
    result = {name: sample_labels(counts[name], k, seed, name) for name in POPULATIONS[:5]}
    result["adversarial"] = adversarial_labels(k)
    result["showcase"] = showcase_labels(k)
    return result


def flush_mapping(mapping, path):
    mapping.flush()
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _check_split(root, record, grid_size, width):
    labels_path = root / record["labels_file"]
    if sha256_file(labels_path) != record["labels_sha256"]:
        raise ValueError("artifact label checksum mismatch")
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    bands = np.load(root / record["bands_file"], mmap_mode="r", allow_pickle=False)
    if labels.dtype != np.float64 or labels.shape != (record["count"], width):
        raise ValueError("artifact label schema mismatch")
    if bands.dtype != np.float64 or bands.shape != (record["count"], grid_size, 3):
        raise ValueError("artifact band schema mismatch")
    end = 0
    for chunk in record["chunks"]:
        if chunk["start"] != end or not end < chunk["stop"] <= record["count"]:
            raise ValueError("noncontiguous artifact progress")
        if array_hash(bands[end:chunk["stop"]]) != chunk["sha256"]:
            raise ValueError("artifact band checksum mismatch")
        end = chunk["stop"]
    if end != record["completed_rows"]:
        raise ValueError("artifact progress disagrees with chunks")
    return labels, bands


def generate_artifact(root, config, solver, plan, provenance, *, resume=False,
                       after_chunk=None, checkpoint_rows=1, execution_id=None):
    """Durable prefix checkpoints. Explicit resume rewrites only uncommitted rows.

    A crash can leave an unrecorded chunk; its cost is unknown, not reconstructed.
    Completed prefixes are checked before continuing. Labels are small pilot
    arrays; frequency storage and generation workspace stay batch-bounded.
    """
    root = Path(root)
    if type(checkpoint_rows) is not int or checkpoint_rows < 1:
        raise ValueError("positive generation checkpoint row interval required")
    if config["sampling"] != "half-dense-half-independent-p05-zero-mask-v1" or config["mass_bounds"] != [0.1, 10] or config["spring_bounds"] != [0, 10]:
        raise ValueError("unsupported sampling contract")
    if solver.interaction_count != config["interactions"] or not np.array_equal(solver.q_hat_values, GRID):
        raise ValueError("solver does not match the artifact contract")
    labels_by_split = population_labels(config)
    identity = {"schema": SCHEMA, "configuration": config, "provenance": provenance,
                "solver_version": SOLVER_VERSION, "label_order": label_order(config["interactions"]),
                "grid_sha256": array_hash(GRID), "compute_dtype": "float64",
                "storage_dtype": "float64", "references": {"m1": 1, "k1": 1},
                 "band_order": "ascending at each q_hat", "shape_order": ["example", "q_hat", "band"],
                 "checkpoint_rows": checkpoint_rows}
    if resume:
        state = json.loads((root / "manifest.json").read_text())
        if state["identity"] != identity:
            raise ValueError("resume requires identical source, sampling, precision and grid")
        if sha256_file(root / "q_hat.npy") != state["grid_file_sha256"]:
            raise ValueError("grid checksum mismatch")
        if state["complete"]:
            LabeledArtifact(root)
            return state
    else:
        root.mkdir()
        save_array(root / "q_hat.npy", GRID)
        state = {"identity": identity, "grid_file_sha256": sha256_file(root / "q_hat.npy"),
                 "complete": False, "splits": {}, "attempts": []}
        write_json(root / "manifest.json", state)
    for previous in state["attempts"]:
        if previous["status"] == "in_progress":
            previous["status"] = "interrupted_duration_unknown"
    attempt = {"resume": resume, "status": "in_progress", "elapsed_seconds": None,
               "execution_id": execution_id}
    state["attempts"].append(attempt)
    write_json(root / "manifest.json", state)
    tick = time.perf_counter()
    for name, expected_labels in labels_by_split.items():
        labels_path, bands_path = root / f"{name}.labels.npy", root / f"{name}.bands.npy"
        if name not in state["splits"]:
            # Files from an interrupted, uncommitted split initialization belong
            # to this artifact and are regenerated only during explicit resume.
            if resume:
                labels_path.unlink(missing_ok=True)
                bands_path.unlink(missing_ok=True)
            save_array(labels_path, expected_labels)
            bands = np.lib.format.open_memmap(bands_path, mode="w+", dtype=np.float64,
                                              shape=(len(expected_labels), len(GRID), 3))
            flush_mapping(bands, bands_path)
            del bands
            state["splits"][name] = {
                "count": len(expected_labels), "labels_file": labels_path.name,
                "bands_file": bands_path.name, "labels_sha256": sha256_file(labels_path),
                "completed_rows": 0, "chunks": [],
            }
            write_json(root / "manifest.json", state)
        record = state["splits"][name]
        labels, stored = _check_split(root, record, len(GRID), len(identity["label_order"]))
        if not np.array_equal(labels, expected_labels):
            raise ValueError("saved labels disagree with seeded population")
        del stored
        first = record["completed_rows"]
        if first == len(labels):
            continue
        bands = np.load(bands_path, mmap_mode="r+", allow_pickle=False)
        masses, springs = physical_arrays(labels[first:], config["interactions"])
        with_context = execution_batches(solver, masses, springs, plan)
        try:
            chunk_tick = time.perf_counter()
            committed = first
            clipped = 0
            for relative, result in with_context:
                start, stop = first + relative, first + relative + len(result.frequencies)
                bands[start:stop] = result.frequencies
                clipped += result.negative_eigenvalues_clipped
                # Coalesce small compute batches into bounded durable blocks.
                # Otherwise millions of rows cause quadratic JSON rewrites and
                # excessive network-volume flushes. Resume redoes only this block.
                if stop - committed < checkpoint_rows and stop != len(labels):
                    continue
                flush_mapping(bands, bands_path)
                record["chunks"].append({
                    "start": committed, "stop": stop, "sha256": array_hash(bands[committed:stop]),
                    "execution_id": execution_id,
                    "negative_eigenvalues_clipped": clipped,
                    "generation_write_hash_seconds": time.perf_counter() - chunk_tick,
                })
                record["completed_rows"] = stop
                write_json(root / "manifest.json", state)
                if after_chunk is not None:
                    after_chunk(name, stop)
                committed = stop
                clipped = 0
                chunk_tick = time.perf_counter()
        finally:
            with_context.close()
            del bands
    attempt["status"] = "completed"
    attempt["elapsed_seconds"] = time.perf_counter() - tick
    state["complete"] = True
    write_json(root / "manifest.json", state)
    return state


class LabeledArtifact:
    """Checksum-verified mmap reader. Only requested rows become dense batches."""

    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        identity = self.manifest["identity"]
        if identity["schema"] != SCHEMA or not self.manifest["complete"]:
            raise ValueError("incomplete or unsupported corrected artifact")
        self.interactions = identity["configuration"]["interactions"]
        if identity["label_order"] != label_order(self.interactions):
            raise ValueError("artifact label order mismatch")
        if identity["compute_dtype"] != "float64" or identity["storage_dtype"] != "float64":
            raise ValueError("unsupported artifact precision")
        if identity["references"] != {"m1": 1, "k1": 1} or identity["band_order"] != "ascending at each q_hat":
            raise ValueError("unsupported artifact physical convention")
        if sha256_file(self.root / "q_hat.npy") != self.manifest["grid_file_sha256"]:
            raise ValueError("artifact grid checksum mismatch")
        self.grid = np.load(self.root / "q_hat.npy", allow_pickle=False)
        if not np.array_equal(self.grid, GRID) or array_hash(self.grid) != identity["grid_sha256"]:
            raise ValueError("artifact grid does not match the corrected contract")
        if set(self.manifest["splits"]) != set(POPULATIONS):
            raise ValueError("artifact population set mismatch")
        self.arrays = {}
        for name, record in self.manifest["splits"].items():
            if record["completed_rows"] != record["count"]:
                raise ValueError("artifact has incomplete split")
            labels, bands = _check_split(self.root, record, len(self.grid), len(label_order(self.interactions)))
            validate_labels(labels, self.interactions)
            self.arrays[name] = (labels, bands)

    def batches(self, population, batch_size, *, permutation=None):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("positive batch size required")
        labels, bands = self.arrays[population]
        indices = np.arange(len(labels)) if permutation is None else np.asarray(permutation)
        if indices.shape != (len(labels),) or not np.array_equal(np.sort(indices), np.arange(len(labels))):
            raise ValueError("batch ordering must include each row exactly once")
        for start in range(0, len(labels), batch_size):
            rows = indices[start:start + batch_size]
            yield rows, np.array(bands[rows], copy=True), np.array(labels[rows], copy=True)
