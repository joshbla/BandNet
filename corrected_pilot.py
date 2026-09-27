"""Corrected pilot and fixed production execution. Controls come from .env.local."""

import argparse
import gc
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
from triatomic_learning import (BandInverse, ProductionBandInverse, DifferentiableTriatomic, band_loss,
                                checked_device, compact_evaluation, load_model, production_payload,
                                synchronize, training_step, train_production, _save_checkpoint,
                                evaluate_designs, evaluate_population, predict,
                                summarize_errors, train_model)


ROOT = Path(__file__).resolve().parent
SOURCE_FILES = ("triatomic_data.py", "triatomic_learning.py", "corrected_pilot.py", "corrected_inference.py",
                "triatomic_batched.py", "triatomic_genuine_formula.py", "triatomic_execution.py",
                "generation_policy.py", "generation_resources.py", "test_triatomic_pipeline.py", "verify_corrected_pilot.py",
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


def preflight(interactions, seed, width, *, device="cpu"):
    """New-domain/gradient/precision checks, not a repeat solver benchmark."""
    labels = np.concatenate((adversarial_labels(interactions), showcase_labels(interactions),
                             sample_labels(16, interactions, seed, "train")))
    masses, springs = physical_arrays(labels, interactions)
    solver = TriatomicBatchSolver(GRID, interactions)
    selected_device = checked_device(device)
    forward = DifferentiableTriatomic(GRID, interactions).to(selected_device)
    reference = np.stack([triatomic_frequencies(m, k, GRID) for m, k in zip(masses, springs)])
    cpu = solver.evaluate(masses, springs).frequencies
    tensor = torch.tensor(labels, dtype=torch.float64, requires_grad=True, device=selected_device)
    differentiable = forward(tensor)
    actual = differentiable.detach().cpu().numpy()
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
    loss = band_loss(torch.tensor(reference * 1.01, device=selected_device), differentiable)
    loss.backward()
    if not torch.isfinite(tensor.grad).all():
        raise AssertionError("nonfinite boundary/degeneracy diagnostic gradient")
    interior = sample_labels(1, interactions, seed + 11, "validation_dense")
    interior[:, 2:] = 1 + 0.8 * interior[:, 2:]
    probe = torch.tensor(interior, requires_grad=True, device=selected_device)
    gradient_check = torch.autograd.gradcheck(
        DifferentiableTriatomic([0.03, 0.21, 0.69, 0.97], interactions).to(selected_device),
        (probe,), eps=1e-5, atol=2e-5, rtol=2e-4, raise_exception=True)
    torch.manual_seed(seed)
    model = BandInverse(interactions, width, np.sqrt(np.mean(reference**2, axis=(0, 1)))).to(selected_device)
    with torch.no_grad():
        target_cpu, target_reference = torch.tensor(cpu, device=selected_device), torch.tensor(reference, device=selected_device)
        cpu_loss = band_loss(target_cpu, forward(model(target_cpu))).item()
        reference_loss = band_loss(target_reference, forward(model(target_reference))).item()
    if abs(cpu_loss - reference_loss) > 1e-10 * max(1, abs(reference_loss)):
        raise AssertionError("reference/production training-loss parity failed")
    return {"device": str(selected_device), "examples": len(labels), "label_sha256": array_hash(labels), "comparison": measures,
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


def sizing_exercise(output):
    """Fixed five-fit sizing protocol; validation only, no final test selection."""
    data_base, training_base = settings(ROOT / ".env.local")
    policy = load_generation_policy(ROOT / ".env.local")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("sizing output must be new with an existing parent")
    if shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise OSError("sizing exercise requires two GiB disk headroom")
    output.mkdir()
    (output / "sources").mkdir()
    provenance = {"code_base_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform()}
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, output / "sources" / name)
    training_base.update(epochs=60)
    protocol = {"name": "bounded-five-fit-sizing-v1", "provenance": provenance,
                "training_base": training_base, "data_seed": data_base["seed"],
                "generation_policy": policy.__dict__,
                "comparisons": [[5, 2048, 128], [5, 8192, 128], [5, 8192, 256]],
                "selection": "lowest pooled validation primary; prefer fewer parameters/examples within 5% relative of minimum",
                "confirmation": "selected M5 with seed+1, then M20 selected configuration; 60 epochs each",
                "evaluation": "512 dense/512 sparse validation, development boundary diagnostics; no final tests/showcases"}
    write_json(output / "protocol.json", protocol)
    torch.set_num_threads(training_base["torch_threads"])
    torch.use_deterministic_algorithms(True)
    datasets, generation, runs = {}, {}, []
    exercise_start = time.perf_counter()

    def dataset(interactions, count):
        key = f"m{interactions}-n{count}"
        if key in datasets:
            return datasets[key]
        config = dict(data_base, interactions=interactions, train_count=count,
                      validation_count_per_population=512, test_count_per_population=2)
        started = time.perf_counter()
        check = preflight(interactions, config["seed"], 128)
        check["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / f"{key}-preflight.json", check)
        solver = TriatomicBatchSolver(GRID, interactions)
        probe = sample_labels(1024, interactions, config["seed"], "train")
        plan, tuning = tune_execution(solver, *physical_arrays(probe, interactions), policy)
        tick = time.perf_counter()
        generate_artifact(output / f"{key}-data", config, solver, plan, provenance)
        generated = time.perf_counter() - tick
        tick = time.perf_counter()
        artifact = LabeledArtifact(output / f"{key}-data")
        generation[key] = {"configuration": config, "preflight": check, "calibration": tuning,
                           "generation_write_seconds": generated,
                           "checksum_load_seconds": time.perf_counter() - tick,
                           "manifest_sha256": sha256_file(artifact.root / "manifest.json")}
        datasets[key] = artifact
        write_json(output / "generation.json", generation)
        return artifact

    def fit(interactions, count, width, seed, name):
        artifact = dataset(interactions, count)
        config = dict(training_base, width=width, seed=seed)
        tick = time.perf_counter()
        model, training = train_model(output / name, artifact, config, provenance)
        fit_seconds = time.perf_counter() - tick
        diagnostic = {}
        for population in ("validation_dense", "validation_sparse", "adversarial"):
            summary, records = evaluate_population(model, artifact, population, config["batch_size"])
            diagnostic[population] = summary
            for kind, array in records.items():
                save_array(output / name / f"{population}.{kind}.npy", array)
        if interactions == 20:
            shared = datasets[f"m5-n{count}"]
            for population in ("validation_dense", "validation_sparse", "adversarial"):
                all_errors, all_predictions, failures = [], [], []
                for rows, target, _ in shared.batches(population, config["batch_size"]):
                    prediction = predict(model, target, config["batch_size"])
                    _, errors, failed = evaluate_designs(target, prediction, interactions)
                    failures.extend({"row": int(rows[item["row"]]), "reason": item["reason"]} for item in failed)
                    all_errors.append(errors)
                    all_predictions.append(prediction)
                errors = np.concatenate(all_errors)
                diagnostic[f"shared_m5_{population}"] = summarize_errors(errors, failures)
                save_array(output / name / f"shared_m5_{population}.per_band_errors.npy", errors)
                save_array(output / name / f"shared_m5_{population}.predictions.npy", np.concatenate(all_predictions))
        row = {"name": name, "interactions": interactions, "train_count": count,
               "configuration": config, "best_epoch": training["best_epoch"],
               "best_validation_primary": training["best_validation_primary"],
               "training_seconds": training["training_seconds_excluding_validation"],
               "fit_wall_seconds": fit_seconds, "checkpoint_sha256": training["checkpoint_sha256"],
               "diagnostics": diagnostic, "history": training["history"]}
        runs.append(row)
        write_json(output / "runs.json", runs)
        print(json.dumps({key: row[key] for key in ("name", "best_epoch", "best_validation_primary", "training_seconds", "fit_wall_seconds")}), flush=True)
        return row

    seed = training_base["seed"]
    fit(5, 2048, 128, seed, "m5-small")
    fit(5, 8192, 128, seed, "m5-large")
    fit(5, 8192, 256, seed, "m5-wide")
    minimum = min(row["best_validation_primary"] for row in runs)
    eligible = [row for row in runs if row["best_validation_primary"] <= minimum * 1.05]
    selected = min(eligible, key=lambda row: (row["configuration"]["width"], row["train_count"]))
    write_json(output / "selection.json", {"selected": selected["name"], "rule": protocol["selection"],
                                           "minimum": minimum, "eligible": [row["name"] for row in eligible]})
    fit(5, selected["train_count"], selected["configuration"]["width"], seed + 1, "m5-repeat")
    fit(20, selected["train_count"], selected["configuration"]["width"], seed, "m20-endpoint")
    files = {str(path.relative_to(output)): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
             for path in sorted(output.rglob("*")) if path.is_file()}
    report = {"protocol": protocol, "generation": generation, "selected": selected["name"], "runs": runs,
              "total_wall_seconds": time.perf_counter() - exercise_start,
              "process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys.platform == "darwin" else 1024),
              "artifact_directory": str(output.resolve()), "files": files,
              "limitations": ["development validation, not publication final tests", "CPU float64 only; no CUDA validation",
                              "one fixed-data optimizer-seed repeat; no population replication", "M6-M19 not trained"]}
    write_json(output / "report.json", report)


def production_settings(path):
    types = {"DEVICE": str, "BATCH_SIZE": int, "TORCH_THREADS": int,
             "CHECKPOINT_STEPS": int, "MAX_SECONDS": int}
    values = {}
    for line in Path(path).read_text().splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if not key.startswith("BANDNET_PRODUCTION_"):
            continue
        name = key.removeprefix("BANDNET_PRODUCTION_")
        if not separator or name not in types or name in values:
            raise ValueError(f"unknown, duplicate or malformed production setting: {key}")
        values[name] = types[name](value.strip())
    if set(values) != set(types):
        raise ValueError("all BANDNET_PRODUCTION settings must be explicit in .env.local")
    if values["DEVICE"] != "cuda:0":
        raise ValueError("production execution requires explicit cuda:0")
    for name in types.keys() - {"DEVICE"}:
        if values[name] < 1:
            raise ValueError(f"positive {name} required")
    if values["BATCH_SIZE"] != 1024:
        raise ValueError("baseline batch size is 1024; changing it requires a protocol revision")
    if values["TORCH_THREADS"] > detect_resources().cpu_budget:
        raise ValueError("requested threads exceed detected CPU allocation")
    return {name.lower(): value for name, value in values.items()}


def production_protocol(controls):
    """A fixed 17-fit experiment, with fresh named streams and no search loop."""
    train = {"architecture": "original-five-relu-corrected-io-v1", "device": controls["device"],
             "batch_size": controls["batch_size"], "torch_threads": controls["torch_threads"],
             "checkpoint_steps": controls["checkpoint_steps"], "learning_rate": 0.001, "seed": 271829}
    base = {"sampling": "half-dense-half-independent-p05-zero-mask-v1",
            "mass_bounds": [0.1, 10], "spring_bounds": [0, 10],
            "validation_count_per_population": 2048}
    fits = [{"name": "main-m5", "data": dict(base, interactions=5, seed=271828,
             train_count=2500000, test_count_per_population=125000),
             "training": dict(train, epochs=5)}]
    fits.extend({"name": f"study-m{k}", "data": dict(base, interactions=k, seed=314159,
                 train_count=100000, test_count_per_population=5000),
                 "training": dict(train, epochs=100, seed=314160)} for k in range(5, 21))
    return {"schema": "corrected-full-three-band-v1", "fits": fits,
            "shared": dict(base, interactions=5, seed=161803, train_count=32,
                           validation_count_per_population=2, test_count_per_population=10000),
            "shared_use": "only test_dense/test_sparse form the common 20000-target M5 population; auxiliary schema rows unused",
            "selection": "minimum pooled dense/sparse validation primary, including initialization; final tests after all fits",
            "precision": "float64 network/data/metrics and complex128 physics",
            "optimizer": "adopted corrected Adam 0.001, no scheduler, no label penalty",
            "input": "1500 frequency features with training-only band RMS scales; no repeated q column",
            "output": "adopted bounded sigmoid ratios m2,m3,k2..kM; fixed m1=k1=1",
            "seed_policy": "single fixed optimizer seed per family; independent main/study/shared data namespaces; no search"}


def source_identity():
    return {"code_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda}


def gpu_identity():
    device = checked_device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    return {"name": properties.name, "total_memory_bytes": properties.total_memory,
            "capability": [properties.major, properties.minor],
            "nvidia_smi": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip()}


def retain_sources(output, provenance):
    (output / "sources").mkdir()
    for name, digest in provenance["source_sha256"].items():
        shutil.copyfile(ROOT / name, output / "sources" / name)
        if sha256_file(output / "sources" / name) != digest:
            raise ValueError("source changed while taking snapshot")


def gpu_preflight(output):
    """Bounded numerical and full-architecture throughput checks, not a fit search."""
    controls = production_settings(ROOT / ".env.local")
    policy = load_generation_policy(ROOT / ".env.local")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("preflight output must be new with an existing parent")
    device = checked_device(controls["device"])
    torch.set_num_threads(controls["torch_threads"])
    torch.use_deterministic_algorithms(True)
    hardware = gpu_identity()
    provenance = source_identity()
    output.mkdir()
    retain_sources(output, provenance)
    report = {"passed": False, "provenance": provenance, "hardware": hardware,
              "controls": controls, "generation_policy": policy.__dict__,
              "resources": detect_resources().as_dict(), "counts": {},
              "scope": "M5/M20 numerical checks and full-size batch updates; no accuracy/sizing campaign"}
    write_json(output / "report.json", report)
    started = time.monotonic()
    # Use a separate stream from every development/production population.
    for interactions in (5, 20):
        if time.monotonic() - started >= controls["max_seconds"]:
            raise TimeoutError("preflight time allowance exhausted")
        numerical = preflight(interactions, 424242, 8, device=controls["device"])
        solver = TriatomicBatchSolver(GRID, interactions)
        labels = sample_labels(controls["batch_size"], interactions, 424243, "train")
        plan, tuning = tune_execution(solver, *physical_arrays(labels, interactions), policy)
        data_config = dict(production_protocol(controls)["shared"], interactions=interactions,
                           seed=424244, train_count=controls["batch_size"],
                           validation_count_per_population=2048, test_count_per_population=2)
        tick = time.perf_counter()
        generate_artifact(output / f"m{interactions}-data", data_config, solver, plan, provenance, checkpoint_rows=4096)
        generation_seconds = time.perf_counter() - tick
        artifact = LabeledArtifact(output / f"m{interactions}-data")
        target = np.array(artifact.arrays["train"][1], copy=True)
        torch.manual_seed(424245)
        model = ProductionBandInverse(interactions, np.sqrt(np.mean(target**2, axis=(0, 1)))).to(device)
        forward = DifferentiableTriatomic(GRID, interactions).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, foreach=False)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        for _ in range(2):
            training_step(model, forward, optimizer, target, device)
        durations, losses = [], []
        for _ in range(5):
            if time.monotonic() - started >= controls["max_seconds"]:
                raise TimeoutError("preflight time allowance exhausted")
            synchronize(device)
            tick = time.perf_counter()
            # Include the same shuffled mmap read and transfer used by production.
            order = np.random.default_rng(424246).permutation(len(target))
            batch = np.array(artifact.arrays["train"][1][order], copy=True)
            losses.append(training_step(model, forward, optimizer, batch, device))
            synchronize(device)
            durations.append(time.perf_counter() - tick)
        training_config = dict(production_protocol(controls)["fits"][0]["training"])
        checkpoint = output / f"m{interactions}.pt"
        tick = time.perf_counter()
        payload = production_payload(model, artifact, training_config, provenance, 0, None)
        _save_checkpoint(checkpoint, payload)
        save_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        _save_checkpoint(output / f"m{interactions}-resume-probe.pt",
                         {"current": model.state_dict(), "optimizer": optimizer.state_dict(), "best": payload})
        resume_save_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        for population in ("validation_dense", "validation_sparse"):
            validation, _ = evaluate_population(model, artifact, population, controls["batch_size"], retain_curves=False)
            if validation["invalid_prediction_count"]:
                raise ArithmeticError("full-validation probe produced invalid designs")
        validation_seconds = time.perf_counter() - tick
        probe = np.array(artifact.arrays["showcase"][1], copy=True)
        before = predict(model, probe, controls["batch_size"])
        restored, _ = load_model(checkpoint, device=controls["device"])
        np.testing.assert_array_equal(before, predict(restored, probe, controls["batch_size"]))
        cpu_model, _ = load_model(checkpoint)
        cpu_predictions = predict(cpu_model, probe, controls["batch_size"])
        np.testing.assert_allclose(before, cpu_predictions, rtol=1e-10, atol=1e-10)
        _, scores, failed = evaluate_designs(probe, before, interactions)
        if failed or not np.all(np.isfinite(scores)):
            raise ArithmeticError("GPU checkpoint predictions failed common CPU scoring")
        row = {"numerical": numerical, "calibration": tuning,
               "generation_write_seconds": generation_seconds,
               "generated_rows": sum(len(pair[0]) for pair in artifact.arrays.values()),
               "warmup_updates": 2, "measured_updates": 5, "batch_size": controls["batch_size"],
               "transfer_and_mmap_inclusive_step_seconds": durations, "losses": losses,
               "median_step_seconds": float(np.median(durations)),
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "checkpoint_write_seconds": save_seconds, "checkpoint_sha256": sha256_file(checkpoint),
               "resume_checkpoint_write_seconds": resume_save_seconds,
               "full_validation_4096_seconds": validation_seconds,
               "cuda_reload_exact": True, "cpu_reload_prediction_tolerance": {"rtol": 1e-10, "atol": 1e-10},
               "common_cpu_scoring_passed": True}
        report["counts"][str(interactions)] = row
        write_json(output / "report.json", report)
        del model, restored, cpu_model, optimizer, forward, artifact, payload
        gc.collect()
        torch.cuda.empty_cache()
    median = max(row["median_step_seconds"] for row in report["counts"].values())
    steps = 5 * ((2500000 + 1023) // 1024) + 16 * 100 * ((100000 + 1023) // 1024)
    report.update(passed=True, elapsed_seconds=time.monotonic() - started,
                  optimizer_steps_in_full_matrix=steps,
                  training_only_projection_seconds=steps * median,
                  projection_scope="endpoint median extrapolation only; excludes validation, checkpoints, generation, evaluation, auditing and interruptions",
                  process_lifetime_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    report["files"] = {str(path.relative_to(output)): sha256_file(path)
                       for path in sorted(output.rglob("*")) if path.is_file() and path.name != "report.json"}
    write_json(output / "report.json", report)
    print(json.dumps({"passed": True, "report": str(output / "report.json"),
                      "training_only_projection_seconds": report["training_only_projection_seconds"]}), flush=True)


def production_run(output, preflight_root, *, resume=False):
    controls = production_settings(ROOT / ".env.local")
    policy = load_generation_policy(ROOT / ".env.local")
    device = checked_device(controls["device"])
    torch.set_num_threads(controls["torch_threads"])
    torch.use_deterministic_algorithms(True)
    provenance = source_identity()
    precheck = json.loads((preflight_root / "report.json").read_text())
    expected_controls = dict(precheck["controls"], max_seconds=controls["max_seconds"])
    if (not precheck["passed"] or precheck["provenance"] != provenance
            or precheck["hardware"] != gpu_identity() or expected_controls != controls
            or precheck["generation_policy"] != policy.__dict__):
        raise ValueError("a successful preflight on this exact source/configuration/GPU stack is required")
    for name, digest in precheck["files"].items():
        if sha256_file(preflight_root / name) != digest:
            raise ValueError("preflight evidence checksum mismatch")
    protocol = production_protocol(controls)
    protocol.update(provenance=provenance, generation_policy=policy.__dict__,
                    preflight_report_sha256=sha256_file(preflight_root / "report.json"), hardware=precheck["hardware"])
    if resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("resume requires the identical frozen production protocol")
        state = json.loads((output / "progress.json").read_text())
    else:
        if output.exists() or not output.parent.is_dir():
            raise ValueError("production output must be new with an existing parent")
        if shutil.disk_usage(output.parent).free < 150 * 10**9:
            raise OSError("production requires at least 150 GB free on the persistent volume")
        output.mkdir()
        retain_sources(output, provenance)
        write_json(output / "protocol.json", protocol)
        state = {"status": "running", "fits": {}, "evaluations": {}, "attempts": []}
    for attempt in state["attempts"]:
        if attempt["status"] == "running":
            attempt["status"] = "interrupted_duration_unknown"
    attempt = {"resume": resume, "status": "running", "elapsed_seconds": None,
               "max_seconds": controls["max_seconds"]}
    state["attempts"].append(attempt)
    state["status"] = "running"
    write_json(output / "progress.json", state)
    started = time.monotonic()
    deadline = started + controls["max_seconds"]

    def budget_check():
        if time.monotonic() >= deadline:
            raise TimeoutError("execution allowance exhausted; resume only under an approved further budget")
        if shutil.disk_usage(output).free < 20 * 10**9:
            raise OSError("less than 20 GB disk reserve remains")

    def dataset(name, config):
        budget_check()
        path = output / f"{name}-data"
        if (path / "manifest.json").exists():
            saved = json.loads((path / "manifest.json").read_text())
            if saved["identity"]["configuration"] != config or saved["identity"]["provenance"] != provenance:
                raise ValueError("production dataset identity mismatch")
            if saved["complete"]:
                return LabeledArtifact(path)
        solver = TriatomicBatchSolver(GRID, config["interactions"])
        probe = sample_labels(1024, config["interactions"], config["seed"], "train")
        plan, tuning = tune_execution(solver, *physical_arrays(probe, config["interactions"]), policy)
        calibration_path = output / f"{name}-calibrations.json"
        calibrations = json.loads(calibration_path.read_text()) if calibration_path.exists() else []
        calibrations.append(tuning)
        write_json(calibration_path, calibrations)
        generate_artifact(path, config, solver, plan, provenance, resume=path.exists(),
                          after_chunk=lambda name, stop: budget_check(), checkpoint_rows=4096)
        return LabeledArtifact(path)

    try:
        # Finish all fixed training before examining fresh final populations.
        for fit in protocol["fits"]:
            budget_check()
            name, config = fit["name"], fit["data"]
            if name in state["fits"]:
                continue
            numerical_path = output / f"{name}-preflight.json"
            if not numerical_path.exists():
                write_json(numerical_path, preflight(config["interactions"], 424242, 8, device=controls["device"]))
            artifact = dataset(name, config)
            training_path = output / name
            model, training = train_production(training_path, artifact, fit["training"], provenance,
                                                resume=training_path.exists(), deadline=deadline)
            state["fits"][name] = {"checkpoint_sha256": training["checkpoint_sha256"],
                                    "dataset_manifest_sha256": sha256_file(artifact.root / "manifest.json")}
            write_json(output / "progress.json", state)
            del model, artifact
            gc.collect()
            torch.cuda.empty_cache()
        shared = dataset("shared-m5", protocol["shared"])
        for fit in protocol["fits"]:
            name = fit["name"]
            budget_check()
            checkpoint = output / name / "best.pt"
            if sha256_file(checkpoint) != state["fits"][name]["checkpoint_sha256"]:
                raise ValueError("completed checkpoint checksum mismatch")
            model, payload = load_model(checkpoint, device=controls["device"])
            artifact = dataset(name, fit["data"])
            if payload["dataset_manifest_sha256"] != sha256_file(artifact.root / "manifest.json"):
                raise ValueError("checkpoint dataset mismatch")
            evaluations = [(population, artifact, population) for population in ("test_dense", "test_sparse", "adversarial", "showcase")]
            if name.startswith("study-"):
                evaluations.extend((f"shared_m5_{population}", shared, population) for population in ("test_dense", "test_sparse"))
            for label, targets, population in evaluations:
                budget_check()
                path = output / name / label
                summary = compact_evaluation(path, model, targets, population, controls["batch_size"],
                                             state["fits"][name]["checkpoint_sha256"], resume=path.exists(),
                                             after_chunk=lambda stop: budget_check())
                state["evaluations"][f"{name}/{label}"] = summary
                write_json(output / "progress.json", state)
                if summary["invalid_prediction_count"]:
                    raise ArithmeticError("invalid final designs; complete records retained for review")
            del model, artifact
            gc.collect()
            torch.cuda.empty_cache()
        state["status"] = "completed_pending_independent_audit"
        attempt["status"] = "completed"
    except BaseException as error:
        state["status"] = "paused" if isinstance(error, TimeoutError) else "failed"
        attempt["status"] = state["status"]
        attempt["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        attempt["elapsed_seconds"] = time.monotonic() - started
        write_json(output / "progress.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sizing", action="store_true", help="historical bounded sizing exercise")
    modes.add_argument("--gpu-preflight", action="store_true", help="required actual-GPU checks and full-batch throughput")
    modes.add_argument("--production", action="store_true", help="fixed full-size CUDA matrix; paid launch requires owner approval")
    parser.add_argument("--preflight", type=Path, help="successful GPU preflight artifact required by production")
    parser.add_argument("--resume", action="store_true", help="explicit production resume with identical source/configuration")
    args = parser.parse_args()
    if (args.resume or args.preflight is not None) and not args.production:
        parser.error("--resume and --preflight are production-only")
    if args.production:
        if args.preflight is None:
            parser.error("--production requires --preflight")
        production_run(args.output, args.preflight, resume=args.resume)
        return
    if args.gpu_preflight:
        gpu_preflight(args.output)
        return
    if args.sizing:
        sizing_exercise(args.output)
        return
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
