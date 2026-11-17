"""Reload a corrected checkpoint and evaluate frequency-only target NPY arrays.

Batch size and explicit CPU threads are read from local .env.local. A stored
grid is mandatory. Target curves may originate from a different interaction
count, supporting the retained shared-M5-target M5-M20 study interface.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from corrected_pilot import ROOT, settings
from triatomic_data import GRID, flush_mapping, label_order, sha256_file, write_json
from triatomic_learning import evaluate_designs, load_model, predict, summarize_errors


def infer_files(checkpoint, target_file, grid_file, output, *, batch_size, torch_threads):
    checkpoint, target_file, grid_file, output = map(Path, (checkpoint, target_file, grid_file, output))
    if output.exists() or not output.parent.is_dir():
        raise ValueError("inference output must be new with an existing parent")
    if type(batch_size) is not int or batch_size < 1 or type(torch_threads) is not int or torch_threads < 1:
        raise ValueError("explicit positive batch size and CPU thread count required")
    grid = np.load(grid_file, allow_pickle=False)
    if not np.array_equal(grid, GRID):
        raise ValueError("target grid differs from the trained grid")
    targets = np.load(target_file, mmap_mode="r", allow_pickle=False)
    if targets.dtype != np.float64 or targets.ndim != 3 or targets.shape[1:] != (500, 3) or not len(targets):
        raise ValueError("targets must be nonempty float64 (N,500,3) arrays")
    torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)
    model, payload = load_model(checkpoint)
    output.mkdir()
    shapes = {"predictions": (len(targets), model.interactions + 1),
              "reconstructed_bands": targets.shape, "per_band_errors": (len(targets), 3)}
    arrays = {name: np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=np.float64, shape=shape)
              for name, shape in shapes.items()}
    failures = []
    started = time.perf_counter()
    for start in range(0, len(targets), batch_size):
        stop = min(start + batch_size, len(targets))
        target = np.array(targets[start:stop])
        predictions = predict(model, target, batch_size)
        curves, errors, failed = evaluate_designs(target, predictions, model.interactions)
        arrays["predictions"][start:stop] = predictions
        arrays["reconstructed_bands"][start:stop] = curves
        arrays["per_band_errors"][start:stop] = errors
        failures.extend({"row": start + row["row"], "reason": row["reason"]} for row in failed)
    for name, array in arrays.items():
        flush_mapping(array, output / f"{name}.npy")
    report = summarize_errors(arrays["per_band_errors"], failures)
    report.update({"schema": "corrected-inference-v1", "interactions": model.interactions,
                   "label_order": label_order(model.interactions), "grid_sha256": sha256_file(grid_file),
                   "targets_sha256": sha256_file(target_file), "checkpoint_sha256": sha256_file(checkpoint),
                   "training_dataset_manifest_sha256": payload["dataset_manifest_sha256"],
                   "inference_reconstruction_write_seconds": time.perf_counter() - started,
                   "source_sha256": {name: sha256_file(ROOT / name) for name in
                                     ("corrected_inference.py", "triatomic_learning.py", "triatomic_data.py", "triatomic_batched.py")},
                   "files": {name: sha256_file(output / f"{name}.npy") for name in arrays}})
    write_json(output / "scores.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--grid", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _, config = settings(ROOT / ".env.local")
    report = infer_files(args.checkpoint, args.targets, args.grid, args.output,
                         batch_size=config["batch_size"], torch_threads=config["torch_threads"])
    print(f"Targets: {report['count']}; invalid: {report['invalid_prediction_count']}; primary: {report['primary']}")
    if report["invalid_prediction_count"]:
        raise RuntimeError("invalid predictions recorded; primary score is undefined")


if __name__ == "__main__":
    main()
