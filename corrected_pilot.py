"""Small local corrected run. All run controls come from this repo's .env.local."""

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from generation_policy import load_generation_policy
from generation_resources import detect_resources
from triatomic_batched import TriatomicBatchSolver
from triatomic_data import (GRID, POPULATIONS, SHOWCASE_IDS, LabeledArtifact, adversarial_labels,
                            array_hash, generate_artifact, label_order, physical_arrays,
                            sample_labels, save_array, sha256_file, showcase_labels, write_json)
from triatomic_execution import tune_execution
from triatomic_genuine_formula import triatomic_frequencies
from triatomic_learning import (BandInverse, DifferentiableTriatomic, band_loss,
                                evaluate_population, train_model)


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("triatomic_data.py", "triatomic_learning.py", "corrected_pilot.py", "corrected_inference.py",
                "triatomic_batched.py", "triatomic_genuine_formula.py", "triatomic_execution.py",
                "generation_policy.py", "generation_resources.py", "test_triatomic_pipeline.py",
                "pyproject.toml", "uv.lock")


def settings(path):
    names = {"INTERACTIONS": int, "TRAIN_COUNT": int, "VALIDATION_COUNT": int,
             "TEST_COUNT": int, "EPOCHS": int, "BATCH_SIZE": int, "WIDTH": int,
             "LEARNING_RATE": float, "SEED": int, "TORCH_THREADS": int}
    values = {}
    for line in Path(path).read_text().splitlines():
        key, separator, raw = line.partition("=")
        key = key.strip()
        if not key.startswith("BANDNET_PILOT_"):
            continue
        name = key.removeprefix("BANDNET_PILOT_")
        if name not in names or not separator or name in values:
            raise ValueError(f"unknown, duplicate or malformed pilot setting: {key}")
        values[name] = names[name](raw.strip())
    if set(values) != set(names):
        raise ValueError("all BANDNET_PILOT settings must be explicit in local .env.local")
    label_order(values["INTERACTIONS"])
    # This command is deliberately a local pilot, not the full M5-M20 study.
    bounds = {"TRAIN_COUNT": (32, 10000), "VALIDATION_COUNT": (2, 1024),
              "TEST_COUNT": (2, 1024), "EPOCHS": (1, 20), "BATCH_SIZE": (1, 128),
              "WIDTH": (4, 256), "TORCH_THREADS": (1, detect_resources().cpu_budget),
              "LEARNING_RATE": (1e-6, 0.01), "SEED": (0, 2**32 - 1)}
    for name, (lower, upper) in bounds.items():
        if not np.isfinite(values[name]) or not lower <= values[name] <= upper:
            raise ValueError(f"pilot {name} must be in [{lower},{upper}]")
    if values["TRAIN_COUNT"] % 2:
        raise ValueError("training count must be even")
    data = {"interactions": values["INTERACTIONS"], "seed": values["SEED"],
            "train_count": values["TRAIN_COUNT"],
            "validation_count_per_population": values["VALIDATION_COUNT"],
            "test_count_per_population": values["TEST_COUNT"],
            "sampling": "half-dense-half-independent-p05-zero-mask-v1",
            "mass_bounds": [0.1, 10], "spring_bounds": [0, 10]}
    train = {name.lower(): values[name] for name in
             ("EPOCHS", "BATCH_SIZE", "WIDTH", "LEARNING_RATE", "SEED", "TORCH_THREADS")}
    return data, train


def preflight(interactions, seed, width):
    """New-domain/gradient/precision checks, not a repeat solver benchmark."""
    labels = np.concatenate((adversarial_labels(interactions), showcase_labels(interactions),
                             sample_labels(16, interactions, seed, "train")))
    masses, springs = physical_arrays(labels, interactions)
    solver = TriatomicBatchSolver(GRID, interactions)
    forward = DifferentiableTriatomic(GRID, interactions)
    reference = np.stack([triatomic_frequencies(m, k, GRID) for m, k in zip(masses, springs)])
    cpu = solver.evaluate(masses, springs).frequencies
    tensor = torch.tensor(labels, dtype=torch.float64, requires_grad=True)
    differentiable = forward(tensor)
    actual = differentiable.detach().numpy()
    scale = np.maximum(1, reference.max(axis=(1, 2)))[:, None, None]
    limits = 128 * np.finfo(np.float64).eps
    measures = {}
    for name, candidate in (("cpu", cpu), ("differentiable", actual)):
        frequency_error = float(np.max(np.abs(candidate - reference) / scale))
        squared_error = float(np.max(np.abs(candidate**2 - reference**2) / scale**2))
        if frequency_error > 1e-10 or squared_error > limits:
            raise AssertionError(f"new-domain {name} parity failed")
        measures[name] = {"maximum_scaled_frequency_error": frequency_error,
                          "maximum_scaled_squared_frequency_error": squared_error}
    # Deliberate residual exercises gradients, including the boundary and
    # repeated-band diagnostic rows. Finiteness is not differentiability proof.
    loss = band_loss(torch.tensor(reference * 1.01), differentiable)
    loss.backward()
    if not torch.isfinite(tensor.grad).all():
        raise AssertionError("nonfinite boundary/degeneracy diagnostic gradient")
    interior = sample_labels(1, interactions, seed + 11, "validation_dense")
    interior[:, 2:] = 1 + 0.8 * interior[:, 2:]
    probe = torch.tensor(interior, requires_grad=True)
    gradient_check = torch.autograd.gradcheck(
        DifferentiableTriatomic([0.03, 0.21, 0.69, 0.97], interactions),
        (probe,), eps=1e-5, atol=2e-5, rtol=2e-4, raise_exception=True)
    torch.manual_seed(seed)
    model = BandInverse(interactions, width, np.sqrt(np.mean(reference**2, axis=(0, 1))))
    with torch.no_grad():
        target_cpu, target_reference = torch.tensor(cpu), torch.tensor(reference)
        cpu_loss = band_loss(target_cpu, forward(model(target_cpu))).item()
        reference_loss = band_loss(target_reference, forward(model(target_reference))).item()
    if abs(cpu_loss - reference_loss) > 1e-10 * max(1, abs(reference_loss)):
        raise AssertionError("reference/production training-loss parity failed")
    return {"examples": len(labels), "label_sha256": array_hash(labels), "comparison": measures,
            "finite_boundary_and_repeated_band_gradients": True, "interior_gradcheck": gradient_check,
            "reference_input_loss": reference_loss, "production_input_loss": cpu_loss,
            "loss_absolute_difference": abs(cpu_loss - reference_loss),
            "precision": "float64 compute, storage, network, differentiable physics and metrics",
            "precision_selection": "no quantization; exact stored payload checks; affordable measured pilot storage",
            "gradient_scope": "finite diagnostics at repeated eigenvalues; finite-difference agreement at a nondegenerate interior input only"}


def export_figure(path, artifact, records, training_report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target = artifact.arrays["showcase"][1]
    design = records["reconstructed_bands"]
    errors = records["per_band_errors"]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for row, axis in enumerate(axes.flat):
        for band, color in enumerate(("#0072B2", "#D55E00", "#009E73")):
            axis.plot(GRID, target[row, :, band], color=color, label="Target" if band == 0 else None)
            axis.plot(GRID, design[row, :, band], color=color, linestyle="--", label="Predicted" if band == 0 else None)
        axis.set_title(f"{SHOWCASE_IDS[row]} | normalized error {errors[row].mean():.4f}")
        axis.set_xlabel("Normalized wave number qL / pi")
        axis.set_ylabel("Dimensionless frequency")
        axis.legend()
    figure.suptitle("Corrected pilot: held-out showcase targets")
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    history = training_report["history"]
    axis.plot([row["epoch"] for row in history], [row["validation_primary"] for row in history], marker="o")
    axis.set_xlabel("Epoch (0 = untrained)")
    axis.set_ylabel("Held-out mean target-normalized band error")
    figure.savefig(path.with_name("validation.png"), dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        raise ValueError("output must be a new directory inside an existing parent")
    data_config, training_config = settings(ROOT / ".env.local")
    policy = load_generation_policy(ROOT / ".env.local")
    required_bytes = (data_config["train_count"] + 2 * data_config["validation_count_per_population"]
                      + 2 * data_config["test_count_per_population"] + 200) * len(GRID) * 3 * 8
    if shutil.disk_usage(args.output.parent).free < required_bytes + 128 * 1024**2:
        raise OSError("insufficient disk headroom for the pilot artifact")
    torch.set_num_threads(training_config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    provenance = {"code_base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "source_sha256": sources,
        "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform()}
    args.output.mkdir()
    (args.output / "sources").mkdir()
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, args.output / "sources" / name)
    write_json(args.output / "protocol.json", {
        "data": data_config, "training": training_config, "provenance": provenance,
        "success_criteria": ["numerical and gradient preflight passes", "complete checksum-verified labeled artifacts",
                             "at least one finite epoch", "held-out band error improves from initialization",
                             "checkpoint reload preserves score", "all final predictions admissible",
                             "consistent per-band/primary scores and figure export"],
        "selection": "lowest pooled dense/sparse validation primary; final tests and showcases never select checkpoints",
        "scope": "local pilot; no publication accuracy threshold or full-study budget established",
    })
    started = time.perf_counter()
    validation = preflight(data_config["interactions"], data_config["seed"], training_config["width"])
    validation["elapsed_seconds"] = time.perf_counter() - started
    write_json(args.output / "preflight.json", validation)
    solver = TriatomicBatchSolver(GRID, data_config["interactions"])
    probe_labels = sample_labels(min(data_config["train_count"], 1024), data_config["interactions"], data_config["seed"], "train")
    plan, tuning = tune_execution(solver, *physical_arrays(probe_labels, data_config["interactions"]), policy)
    write_json(args.output / "calibration.json", tuning)
    tick = time.perf_counter()
    generate_artifact(args.output / "data", data_config, solver, plan, provenance)
    generation_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    artifact = LabeledArtifact(args.output / "data")
    loading_seconds = time.perf_counter() - tick
    model, training_report = train_model(args.output / "training", artifact, training_config, provenance)
    summaries, showcase_records = {}, None
    (args.output / "evaluation").mkdir()
    for population in ("test_dense", "test_sparse", "adversarial", "showcase"):
        summary, records = evaluate_population(model, artifact, population, training_config["batch_size"])
        summaries[population] = summary
        for name, array in records.items():
            save_array(args.output / "evaluation" / f"{population}.{name}.npy", array)
        if population == "showcase":
            showcase_records = records
    write_json(args.output / "evaluation" / "scores.json", summaries)
    if any(summary["invalid_prediction_count"] for summary in summaries.values()):
        raise ArithmeticError("invalid final predictions; retained failure records")
    export_figure(args.output / "showcase.png", artifact, showcase_records, training_report)
    showcase_rows = []
    for i, case_id in enumerate(SHOWCASE_IDS):
        showcase_rows.append({"id": case_id, "target_labels": artifact.arrays["showcase"][0][i].tolist(),
                              "predicted_labels": showcase_records["predictions"][i].tolist(),
                              "per_band_error": showcase_records["per_band_errors"][i].tolist(),
                              "primary": float(showcase_records["per_band_errors"][i].mean())})
    write_json(args.output / "evaluation" / "showcase.json", {"label_order": label_order(artifact.interactions), "rows": showcase_rows})
    files = {str(path.relative_to(args.output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in sorted(args.output.rglob("*")) if path.is_file()}
    improved = training_report["best_validation_primary"] < training_report["initial_validation_primary"]
    report = {"protocol": "corrected-local-pilot-v1", "provenance": provenance,
              "artifact_directory": str(args.output.resolve()), "data_configuration": data_config,
              "training_configuration": training_config, "preflight": validation,
              "calibration": tuning, "generation_write_seconds": generation_seconds,
              "artifact_checksum_load_seconds": loading_seconds,
              "training": training_report, "evaluation": summaries,
              "small_end_to_end_passed": improved,
              "process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
              "memory_scope": "process lifetime high-water RSS, including preflight, calibration and training",
              "data_bytes": sum(item["bytes"] for name, item in files.items() if name.startswith("data/")),
              "files": files, "population_counts": {name: len(artifact.arrays[name][0]) for name in POPULATIONS},
              "limitations": ["one local pilot seed; no production-size or accuracy conclusion",
                              "float64 CPU neural/differentiable path; GPU untested",
                              "M5-M20 remains required; only the configured interaction count was trained",
                              "no real-pod resource preflight or approved pod reserves",
                              "source snapshots identify this working-tree implementation, not the base commit alone"]}
    write_json(args.output / "report.json", report)
    print(json.dumps({"output": str(args.output), "passed": improved,
                      "initial_validation": training_report["initial_validation_primary"],
                      "best_validation": training_report["best_validation_primary"],
                      "training_seconds": training_report["training_seconds_excluding_validation"],
                      "generation_write_seconds": generation_seconds}, indent=2))
    if not improved:
        raise RuntimeError("pilot ran but held-out learning criterion failed; report preserved")


if __name__ == "__main__":
    main()
