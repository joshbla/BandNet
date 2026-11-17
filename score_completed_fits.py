"""Score completed production fits on a CPU, without the all-fits training barrier.

The production runner evaluates only after all 17 fits have trained. When a run
stops with some fits untrained, this applies the runner's final-evaluation step
(checkpoint qualification, then test dense/sparse, adversarial, showcase and,
for study fits, the shared M5 cross-count population) to completed fits only.

Input is a run root restored by `scoring_restore.ts`: training records, best
checkpoints and data, with training bands reduced to a verified prefix. Training
records are never modified. Scoring state, its CPU execution records and source
snapshots live under `<run>/scoring/`; evaluation records use the runner's
`<run>/<fit>/<label>/` layout. Rerunning with more fits appends a new execution.
`verify_corrected_pilot.py --production --completed-fits` audits the result.
"""

import argparse
import gc
import json
import platform
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from corrected_pilot import (ROOT, SOURCE_FILES, check_execution_source, production_dataset, production_protocol,
                             source_identity, validate_execution_history)
from generation_policy import load_generation_policy
from generation_resources import detect_resources
from triatomic_data import (GRID, LabeledArtifact, _check_split, array_hash, label_order, sha256_file,
                            validate_labels, write_json, SCHEMA)
from triatomic_learning import compact_evaluation, load_model, qualify_checkpoint_model

EXECUTION_SCHEMA = "corrected-cpu-scoring-execution-v1"
STATE_SCHEMA = "corrected-completed-fit-scoring-v1"
SCORER = Path(__file__).resolve()


class PrefixTrainingArtifact(LabeledArtifact):
    """A `LabeledArtifact` whose training bands hold only the first dataset chunk.

    Every other split, the training labels and the first training band chunk are
    checksum-verified against the dataset manifest. Scoring reads training bands
    only for the fixed reload-probe rows, which lie in that first chunk.
    """

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
        width = len(label_order(self.interactions))
        self.arrays = {}
        for name, record in self.manifest["splits"].items():
            if record["completed_rows"] != record["count"]:
                raise ValueError("artifact has incomplete split")
            if name == "train":
                labels, bands = self._training_prefix(record, width)
            else:
                labels, bands = _check_split(self.root, record, len(self.grid), width)
            validate_labels(labels, self.interactions)
            self.arrays[name] = (labels, bands)
        if set(self.arrays) != set(self.manifest["splits"]) or "train" not in self.arrays:
            raise ValueError("artifact population set mismatch")

    def _training_prefix(self, record, width):
        labels_path = self.root / record["labels_file"]
        if sha256_file(labels_path) != record["labels_sha256"]:
            raise ValueError("artifact label checksum mismatch")
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if labels.dtype != np.float64 or labels.shape != (record["count"], width):
            raise ValueError("artifact label schema mismatch")
        chunk = record["chunks"][0]
        if chunk["start"] != 0:
            raise ValueError("noncontiguous artifact progress")
        prefix = self.root / f"{record['bands_file']}.prefix"
        with prefix.open("rb") as source:
            version = np.lib.format.read_magic(source)
            read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, fortran, dtype = read(source)
            offset = source.tell()
        if dtype != np.float64 or fortran or shape != (record["count"], len(self.grid), 3):
            raise ValueError("artifact band schema mismatch")
        row_bytes = len(self.grid) * 3 * 8
        if (prefix.stat().st_size - offset) // row_bytes < chunk["stop"]:
            raise ValueError("training band prefix is shorter than its first dataset chunk")
        bands = np.memmap(prefix, dtype=np.float64, mode="r", offset=offset, shape=(chunk["stop"], len(self.grid), 3))
        if array_hash(bands) != chunk["sha256"]:
            raise ValueError("artifact band checksum mismatch")
        return labels, bands

    def batches(self, population, batch_size, *, permutation=None):
        if population == "train":
            raise ValueError("only the verified training prefix is present")
        return super().batches(population, batch_size, permutation=permutation)


def frozen_protocol(root):
    """Training records and the fixed protocol, checked as the auditor does."""
    protocol = json.loads((root / "protocol.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    batch_sizes = {fit["training"]["batch_size"] for fit in protocol["fits"]}
    if len(batch_sizes) != 1:
        raise ValueError("production fits disagree on batch size")
    batch_size = batch_sizes.pop()
    expected = production_protocol({"batch_size": batch_size}, schema=protocol["schema"])
    for key, value in expected.items():
        if protocol[key] != value:
            raise ValueError(f"production protocol differs from the fixed experiment: {key}")
    for name, digest in protocol["provenance"]["source_sha256"].items():
        if sha256_file(root / "sources" / name) != digest:
            raise ValueError(f"production source snapshot mismatch: {name}")
    validate_execution_history(root, protocol, progress["attempts"])
    return protocol, progress, batch_size


def score(root, names):
    root = Path(root).resolve()
    protocol, progress, batch_size = frozen_protocol(root)
    order = [fit["name"] for fit in protocol["fits"]]
    if not names or len(set(names)) != len(names) or any(name not in progress["fits"] for name in names):
        raise ValueError("requested fits must be distinct and have completed training")
    source = source_identity()
    check_execution_source(protocol, source)
    resources = detect_resources()
    threads = resources.cpu_budget
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    policy = load_generation_policy(ROOT / ".env.local")

    directory = root / "scoring"
    state_path = directory / "progress.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["schema"] != STATE_SCHEMA:
            raise ValueError("unsupported scoring state")
    else:
        (directory / "executions").mkdir(parents=True)
        state = {"schema": STATE_SCHEMA, "status": "running", "fits": {}, "evaluations": {}, "executions": []}
    execution_id = uuid.uuid4().hex
    snapshot = directory / "sources" / execution_id
    snapshot.mkdir(parents=True)
    scorer_sha256 = sha256_file(SCORER)
    for name, digest in dict(source["source_sha256"], **{SCORER.name: scorer_sha256}).items():
        shutil.copyfile(ROOT / name, snapshot / name)
        if sha256_file(snapshot / name) != digest:
            raise ValueError("scoring source changed while taking snapshot")
    execution = {"schema": EXECUTION_SCHEMA, "execution_id": execution_id, "device": "cpu",
                 "batch_size": batch_size, "torch_threads": threads, "source_sha256": dict(source["source_sha256"]),
                 "scorer_sha256": scorer_sha256, "code_base_commit": source["code_base_commit"],
                 "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform(),
                 "resources": resources.as_dict(), "generation_policy": dict(policy.__dict__),
                 "requested_fits": list(names), "checkpoint_qualifications": {}, "status": "running"}
    execution_path = directory / "executions" / f"{execution_id}.json"
    write_json(execution_path, execution)
    state["executions"].append(execution_id)
    state["status"] = "running"
    write_json(state_path, state)
    started = time.monotonic()

    def budget_check():
        if shutil.disk_usage(root).free < 20 * 10**9:
            raise OSError("less than 20 GB disk reserve remains")

    try:
        shared = None
        for fit in protocol["fits"]:
            name = fit["name"]
            if name not in names:
                continue
            record = progress["fits"][name]
            checkpoint = root / name / "best.pt"
            if sha256_file(checkpoint) != record["checkpoint_sha256"]:
                raise ValueError("completed checkpoint checksum mismatch")
            artifact = PrefixTrainingArtifact(root / f"{name}-data")
            if (artifact.manifest["identity"]["configuration"] != fit["data"]
                    or artifact.manifest["identity"]["provenance"] != protocol["provenance"]
                    or sha256_file(artifact.root / "manifest.json") != record["dataset_manifest_sha256"]):
                raise ValueError("fit data configuration mismatch")
            model, payload = load_model(checkpoint)
            if payload["dataset_manifest_sha256"] != record["dataset_manifest_sha256"]:
                raise ValueError("checkpoint dataset mismatch")
            execution["checkpoint_qualifications"][name] = qualify_checkpoint_model(
                model, artifact, payload["reload_probe"], "cpu", gradients=False)
            write_json(execution_path, execution)
            evaluations = [(population, artifact, population) for population in ("test_dense", "test_sparse", "adversarial", "showcase")]
            if name.startswith("study-"):
                if shared is None:
                    # The runner's shared M5 targets, generated here because the
                    # runner never reached its evaluation phase.
                    shared = production_dataset(root, "shared-m5", protocol["shared"], protocol["provenance"],
                                                policy, execution_id, budget_check)
                evaluations.extend((f"shared_m5_{population}", shared, population) for population in ("test_dense", "test_sparse"))
            for label, targets, population in evaluations:
                budget_check()
                path = root / name / label
                summary = compact_evaluation(path, model, targets, population, batch_size, record["checkpoint_sha256"],
                                             resume=path.exists(), execution_id=execution_id)
                state["evaluations"][f"{name}/{label}"] = summary
                write_json(state_path, state)
                print(json.dumps({"fit": name, "population": label, "primary": summary["primary"],
                                  "invalid": summary["invalid_prediction_count"]}), flush=True)
                if summary["invalid_prediction_count"]:
                    raise ArithmeticError("invalid final designs; complete records retained for review")
            state["fits"][name] = dict(record)
            write_json(state_path, state)
            del model, artifact
            gc.collect()
        state["status"] = "completed_pending_independent_audit"
        state["scored_fits"] = [name for name in order if name in state["fits"]]
        state["untrained_fits"] = [name for name in order if name not in progress["fits"]]
        execution["status"] = "completed"
    except BaseException as error:
        state["status"] = execution["status"] = "failed"
        execution["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        execution["elapsed_seconds"] = time.monotonic() - started
        write_json(execution_path, execution)
        write_json(state_path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, type=Path, help="run root restored by scoring_restore.ts")
    parser.add_argument("--fits", required=True, nargs="+", help="completed fits to score, e.g. main-m5")
    args = parser.parse_args()
    score(args.run, args.fits)


if __name__ == "__main__":
    main()
