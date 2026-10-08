"""Corrected pilot and fixed production execution. Controls come from .env.local."""

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch

from generation_policy import load_generation_policy
from generation_resources import detect_resources
from triatomic_batched import TriatomicBatchSolver
from triatomic_data import (GRID, POPULATIONS, SHOWCASE_IDS, LabeledArtifact, adversarial_labels,
                            array_hash, generate_artifact, label_order, physical_arrays,
                            sample_labels, save_array, sha256_file, showcase_labels, write_json)
from triatomic_execution import tune_execution, working_bytes
from triatomic_genuine_formula import triatomic_frequencies
from triatomic_learning import (BandInverse, ProductionBandInverse, DifferentiableTriatomic, band_loss,
                                checked_device, compact_evaluation, load_model, production_payload,
                                synchronize, training_step, train_production, _save_checkpoint,
                                save_resume_checkpoint, read_resume_checkpoint, qualify_checkpoint_model,
                                evaluate_designs, evaluate_population, predict,
                                summarize_errors, train_model, checkpoint_probe)


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


def update_local_settings(path, updates):
    """Change only named local settings, without exposing any other local values."""
    lines, found = [], set()
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if key in updates:
            if not separator or key in found:
                raise ValueError(f"duplicate or malformed local setting: {key}")
            found.add(key)
            line = f"{key}={updates[key]}"
        lines.append(line)
    if found != set(updates):
        raise ValueError("automatic configuration requires the documented local setting names")
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".settings-", delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write("\n".join(lines) + "\n")
            output.flush()
            os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def configure_machine_resources(path):
    resources = detect_resources()
    # Dedicated benchmark allocation: scale headroom with the actual machine.
    cpu_reserve = min(resources.cpu_budget - 1, max(1, int(np.ceil(resources.cpu_budget * .1))))
    ram_mib = int(np.ceil(resources.available_memory_bytes * .1 / 1024**2))
    available_threads = resources.cpu_budget - cpu_reserve
    update_local_settings(path, {"BANDNET_GENERATION_CPU_RESERVE": cpu_reserve,
                                "BANDNET_GENERATION_RAM_RESERVE_MIB": ram_mib,
                                "BANDNET_PRODUCTION_TORCH_THREADS": available_threads})
    return {"resources": resources.as_dict(), "cpu_reserve": cpu_reserve,
            "ram_reserve_mib": ram_mib, "available_threads": available_threads,
            "policy": "10 percent CPU/RAM headroom; retain at least one compute CPU"}


def tune_training_threads(model, forward, optimizer, target, device, available, seconds):
    candidates = sorted({1, available, *(2**power for power in range(available.bit_length()) if 2**power <= available)})
    started = time.perf_counter()
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    measured = []
    for threads in candidates:
        if measured and time.perf_counter() - started >= seconds:
            break
        torch.set_num_threads(threads)
        model.load_state_dict(initial)
        optimizer.state.clear()
        training_step(model, forward, optimizer, target, device)
        durations = []
        for _ in range(2):
            synchronize(device)
            tick = time.perf_counter()
            training_step(model, forward, optimizer, target, device)
            synchronize(device)
            durations.append(time.perf_counter() - tick)
        measured.append({"threads": threads, "seconds": durations, "median_seconds": float(np.median(durations))})
    winner = min(measured, key=lambda row: row["median_seconds"])
    torch.set_num_threads(winner["threads"])
    model.load_state_dict(initial)
    model.zero_grad(set_to_none=True)
    optimizer.state.clear()
    return {"selected_threads": winner["threads"], "candidates": candidates, "measurements": measured,
            "elapsed_seconds": time.perf_counter() - started, "search_complete": len(measured) == len(candidates),
             "selection": "fastest measured transfer-inclusive full-batch update; bounded search, not a global optimum"}


def learning_health(states, *, minimum_relative_improvement=0.001, validation_required=True):
    """A bounded learning witness, separate from numerical/timing qualification.

    Disposable timing records the small validation witness without gating on it:
    a few updates make it noisy, while collapse shows in the other checks."""
    if not states:
        raise ValueError("learning health requires observed states")
    first, last = states[0], states[-1]
    reasons = []
    if not all(row["finite"] and np.isfinite(row["training_loss"]) and np.isfinite(row["validation_loss"])
               and np.isfinite(row["predictions"]).all() for row in states):
        reasons.append("nonfinite learning witness")
    if any(count == 0 for count in last["sigmoid_derivative_nonzero_by_output"]):
        reasons.append("entire output column has zero sigmoid derivative")
    if last["unique_prediction_rows"] < 2 or len(np.unique(last["predictions"], axis=0)) < 2:
        reasons.append("varied inputs have identical predictions")
    if (not last["useful_current_gradients"] or not last["gradients"]
            or not all(g["present"] and g["finite"] and g["norm"] is not None
                       and np.isfinite(g["norm"]) and g["norm"] > 0 and g["nonzero"] > 0
                       for g in last["gradients"])):
        reasons.append("current gradients are missing, nonfinite or entirely zero in a parameter tensor")
    progress = {}
    for key in ("training_loss", "validation_loss"):
        initial, final = first[key], last[key]
        improved = np.isfinite(initial) and np.isfinite(final) and initial > 0 and final <= initial * (1 - minimum_relative_improvement)
        progress[key] = {"initial": initial, "final": final, "improved": bool(improved)}
        if not improved and (key == "training_loss" or validation_required):
            reasons.append(f"no meaningful bounded-window {key} improvement")
    return {"passed": not reasons, "reasons": reasons, "progress": progress,
            "minimum_relative_improvement": minimum_relative_improvement,
            "validation_required": validation_required,
            "scope": "fixed training/validation witness only; not generalization or GPU qualification"}


def check_learning_evidence(row, *, validation_required=True):
    """Recompute the health decision from observations, never trust a pass flag."""
    try:
        observations = row["learning_observations"]
        states = [observations["initial"], observations["final"]]
        for state in states:
            live = np.asarray(state["sigmoid_derivative_nonzero_by_output"])
            if live.ndim != 1 or not len(live) or np.any(live < 0) or np.any(live > row["batch_size"]):
                raise ValueError("invalid derivative counts")
        actual = learning_health(states, validation_required=validation_required)
        if not actual["passed"] or row["learning_health"] != actual:
            raise ValueError("health decision differs from observations")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid learning health evidence: {error}") from error
    return actual


def learning_observation(model, forward, target, validation, device, *, gradients):
    """Observe the full logical training batch and fixed validation witnesses."""
    tensor = torch.as_tensor(target, dtype=torch.float64, device=device)
    layers, captured = [], {}

    def capture(name, final):
        def hook(module, inputs, output):
            value = output.detach()
            layers.append({"layer": name, "min": value.min().item(), "max": value.max().item(),
                           "rms": value.square().mean().sqrt().item(),
                           "positive_fraction": (value > 0).double().mean().item()})
            if final:
                captured["raw"] = value
        return hook

    linear = [(name, module) for name, module in model.network.named_modules() if isinstance(module, torch.nn.Linear)]
    handles = [module.register_forward_hook(capture(name, index == len(linear) - 1))
               for index, (name, module) in enumerate(linear)]
    model.zero_grad(set_to_none=True)
    try:
        with torch.set_grad_enabled(gradients):
            predictions = model(tensor)
            loss = band_loss(tensor, forward(predictions))
            if gradients:
                loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    raw = captured["raw"]
    unit = raw.sigmoid()
    derivative = unit * (1 - unit)
    gradient_rows = []
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        gradient_rows.append({"parameter": name, "present": grad is not None,
                              "finite": bool(grad is not None and torch.isfinite(grad).all()),
                              "norm": None if grad is None else grad.norm().item(),
                              "nonzero": 0 if grad is None else int(torch.count_nonzero(grad))})
    with torch.no_grad():
        witness = torch.as_tensor(validation, dtype=torch.float64, device=device)
        validation_predictions = model(witness)
        validation_loss = band_loss(witness, forward(validation_predictions)).item()
    chosen = torch.linspace(0, len(predictions) - 1, min(16, len(predictions)), device=device).long()
    finite = bool(torch.isfinite(raw).all() and torch.isfinite(predictions).all()
                  and torch.isfinite(loss) and np.isfinite(validation_loss))
    return {"finite": finite, "training_loss": loss.item(), "validation_loss": validation_loss,
            "layers": layers, "logit_min_by_output": raw.amin(0).tolist(),
            "logit_max_by_output": raw.amax(0).tolist(),
            "sigmoid_zero_by_output": (unit == 0).sum(0).tolist(),
            "sigmoid_one_by_output": (unit == 1).sum(0).tolist(),
            "sigmoid_derivative_nonzero_by_output": (derivative != 0).sum(0).tolist(),
            "sigmoid_derivative_below_1e12_by_output": (derivative < 1e-12).sum(0).tolist(),
            "sigmoid_derivative_min_by_output": derivative.amin(0).tolist(),
            "sigmoid_derivative_max_by_output": derivative.amax(0).tolist(),
            "unique_prediction_rows": len(torch.unique(predictions.detach(), dim=0)),
            "prediction_rows": chosen.tolist(), "predictions": predictions.detach()[chosen].tolist(),
            "validation_predictions": validation_predictions.tolist(),
            "gradients": gradient_rows,
            "useful_current_gradients": all(row["finite"] and row["nonzero"] > 0 for row in gradient_rows)}


def bounded_learning_diagnostic(output):
    """Owner-approved local cap: five fixed trials, at most seven updates each."""
    if output.exists() or not output.parent.is_dir():
        raise ValueError("diagnostic output must be new with an existing parent")
    _, controls = settings(ROOT / ".env.local")
    torch.set_num_threads(controls["torch_threads"])
    torch.use_deterministic_algorithms(True)
    retained = ROOT / "artifacts/runpod-timing-retry-us/remote/disposable-timing"
    output.mkdir()
    (output / "sources").mkdir()
    sources = {}
    for name in SOURCE_FILES:
        shutil.copyfile(ROOT / name, output / "sources" / name)
        sources[name] = sha256_file(output / "sources" / name)
    report = {"schema": "bounded-learning-diagnostic-v1", "device": "cpu", "torch": str(torch.__version__),
              "numpy": np.__version__, "torch_threads": controls["torch_threads"], "source_sha256": sources,
              "maximum_updates": 35, "actual_updates": 0, "trials": [], "completed": False,
              "scope": "fixed retained fixtures; no production-rate adoption, GPU or generalization claim",
              "retained_report_sha256": sha256_file(retained / "report.json")}

    def fixture(interactions):
        artifact = LabeledArtifact(retained / f"m{interactions}-data")
        target = np.array(artifact.arrays["train"][1], copy=True)
        if len(target) != 1024 or not np.array_equal(artifact.grid, GRID):
            raise ValueError("diagnostic requires retained full logical batch and native grid")
        rows = np.linspace(0, 2047, 16, dtype=int)
        validation = np.concatenate([artifact.arrays[population][1][rows]
                                     for population in ("validation_dense", "validation_sparse")])
        return artifact, target, validation

    def trial(interactions, seed, rate, name, *, baseline=False):
        artifact, target, validation = fixture(interactions)
        torch.manual_seed(seed)
        model = ProductionBandInverse(interactions, np.sqrt(np.mean(target**2, axis=(0, 1))))
        forward = DifferentiableTriatomic(GRID, interactions)
        optimizer = torch.optim.Adam(model.parameters(), lr=rate, foreach=False)
        row = {"name": name, "interactions": interactions, "seed": seed, "learning_rate": rate,
               "batch_size": 1024, "input_scale": model.input_scale.tolist(),
               "dataset_manifest_sha256": sha256_file(artifact.root / "manifest.json"),
               "validation_rows_per_population": np.linspace(0, 2047, 16, dtype=int).tolist(),
               "states": [], "parameter_movement": []}
        report["trials"].append(row)
        samples = []
        for step in range(8):
            observation = learning_observation(model, forward, target, validation, "cpu", gradients=True)
            observation["updates_completed"] = step
            row["states"].append(observation)
            write_json(output / "report.json", report)
            print(json.dumps({"trial": name, "updates": step, "loss": observation["training_loss"],
                              "validation_loss": observation["validation_loss"],
                              "live_derivatives": observation["sigmoid_derivative_nonzero_by_output"]}), flush=True)
            collapsed = any(count == 0 for count in observation["sigmoid_derivative_nonzero_by_output"])
            if (step == 7 or not observation["finite"]
                    or any(not gradient["finite"] for gradient in observation["gradients"])
                    or (collapsed and not baseline)):
                break
            # Sample actual movement, not Adam moments; avoid a full weight copy per step.
            samples = []
            for key, parameter in model.named_parameters():
                indices = torch.linspace(0, parameter.numel() - 1, min(256, parameter.numel())).long()
                samples.append((key, parameter, indices, parameter.detach().flatten()[indices].clone()))
            optimizer.step()
            report["actual_updates"] += 1
            row["parameter_movement"].append({"update": step + 1,
                "scope": "up to 256 uniformly spaced entries per tensor; actual sampled deltas, not full norms",
                "parameters": [{"parameter": key, "samples": len(indices),
                                "delta_norm": (parameter.detach().flatten()[indices] - before).norm().item(),
                                "max_abs_delta": (parameter.detach().flatten()[indices] - before).abs().max().item()}
                               for key, parameter, indices, before in samples]})
        row["health"] = learning_health(row["states"])
        write_json(output / "report.json", report)
        del model, optimizer, forward, samples
        gc.collect()
        return row

    artifact, target, validation = fixture(5)
    negative, _ = load_model(retained / "m5.pt")
    observation = learning_observation(negative, DifferentiableTriatomic(GRID, 5), target, validation, "cpu", gradients=True)
    report["negative_control"] = {"checkpoint_sha256": sha256_file(retained / "m5.pt"),
                                  "observation": observation, "health": learning_health([observation])}
    del negative, artifact, target, validation, _
    gc.collect()
    original = trial(5, 424245, 0.001, "m5-original", baseline=True)
    overshoot = any(any(count == 0 for count in state["sigmoid_derivative_nonzero_by_output"])
                    for state in original["states"][1:])
    report["baseline_overshoot_confirmed"] = overshoot
    if overshoot:
        candidate = trial(5, 424245, 0.0001, "m5-candidate")
        if candidate["health"]["passed"]:
            for interactions, seed, name in ((5, 271829, "m5-main-seed"), (5, 314160, "m5-study-seed"),
                                              (20, 424245, "m20-control")):
                result = trial(interactions, seed, 0.0001, name)
                if not result["health"]["passed"]:
                    report["stopped_after_failed_trial"] = name
                    break
        else:
            report["stopped_after_failed_trial"] = "m5-candidate"
    report["completed"] = True
    write_json(output / "report.json", report)
    print(json.dumps({"output": str(output), "actual_updates": report["actual_updates"],
                      "trials": [{"name": row["name"], "health": row["health"]} for row in report["trials"]]}), flush=True)


def production_protocol(controls):
    """A fixed 17-fit experiment, with fresh named streams and no search loop."""
    train = {"architecture": "original-five-relu-corrected-io-v1",
             "batch_size": controls["batch_size"], "learning_rate": 0.0001, "seed": 271829}
    base = {"sampling": "half-dense-half-independent-p05-zero-mask-v1",
            "mass_bounds": [0.1, 10], "spring_bounds": [0, 10],
            "validation_count_per_population": 2048}
    fits = [{"name": "main-m5", "data": dict(base, interactions=5, seed=271828,
             train_count=2500000, test_count_per_population=125000),
             "training": dict(train, epochs=5)}]
    fits.extend({"name": f"study-m{k}", "data": dict(base, interactions=k, seed=314159,
                 train_count=100000, test_count_per_population=5000),
                 "training": dict(train, epochs=100, seed=314160)} for k in range(5, 21))
    return {"schema": "corrected-full-three-band-v3", "fits": fits,
            "shared": dict(base, interactions=5, seed=161803, train_count=32,
                           validation_count_per_population=2, test_count_per_population=10000),
            "shared_use": "only test_dense/test_sparse form the common 20000-target M5 population; auxiliary schema rows unused",
            "selection": "minimum pooled dense/sparse validation primary, including initialization; final tests after all fits",
            "precision": "float64 network/data/metrics and complex128 physics",
            "optimizer": "adopted corrected Adam 0.0001, no scheduler, no label penalty",
            "input": "1500 frequency features with training-only band RMS scales; no repeated q column",
            "output": "adopted bounded sigmoid ratios m2,m3,k2..kM; fixed m1=k1=1",
            "seed_policy": "single fixed optimizer seed per family; independent main/study/shared data namespaces; no search"}


def source_identity():
    return {"code_base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
            "numpy": np.__version__, "torch": str(torch.__version__), "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda}


def production_provenance(source):
    """Exact source/lockfile contents define the experiment; host/runtime do not."""
    return {"source_sha256": dict(source["source_sha256"])}


def check_preflight_evidence(report, *, historical=False):
    allowed = ("corrected-gpu-preflight-v2", "corrected-gpu-preflight-v3", "corrected-gpu-preflight-v4") if historical else ("corrected-gpu-preflight-v4",)
    if "schema" not in report or report["schema"] not in allowed or report["passed"] is not True:
        raise ValueError("successful portable GPU preflight required")
    if report["schema"] in ("corrected-gpu-preflight-v3", "corrected-gpu-preflight-v4"):
        if report["numerical_timing_passed"] is not True or report["learning_health_passed"] is not True:
            raise ValueError("successful numerical/timing and learning qualification required")
    if report["schema"] == "corrected-gpu-preflight-v4" and report["probe_training"] != {
            "learning_rate": .0001, "seed": 424245, "updates": 7, "production_rate": True}:
        raise ValueError("preflight must qualify the adopted production rate")
    if set(report["counts"]) != {"5", "20"}:
        raise ValueError("preflight must qualify both M5 and M20")
    for row in report["counts"].values():
        if (row["cuda_reload_exact"] is not True or row["adam_reload_next_update_exact"] is not True
                or row["common_cpu_scoring_passed"] is not True
                or row["numerical"]["interior_gradcheck"] is not True
                or row["numerical"]["finite_boundary_and_repeated_band_gradients"] is not True
                or row["cpu_reload_prediction_tolerance"] != {"rtol": 1e-10, "atol": 1e-10}
                or row["batch_size"] != report["controls"]["batch_size"]
                or row["warmup_updates"] != 2 or row["measured_updates"] != 5
                or not np.isfinite(row["median_step_seconds"]) or row["median_step_seconds"] <= 0):
            raise ValueError("incomplete numerical or timing preflight evidence")
        if report["schema"] in ("corrected-gpu-preflight-v3", "corrected-gpu-preflight-v4"):
            check_learning_evidence(row)


def validate_execution_history(root, protocol, attempts):
    """Portable run archives keep the exact admitted report for every machine."""
    executions = {}
    if protocol["schema"] not in ("corrected-full-three-band-v2", "corrected-full-three-band-v3"):
        raise ValueError("portable execution requires the v2 or v3 production protocol")
    for attempt in attempts:
        identity = attempt["execution_id"]
        if (not isinstance(identity, str) or len(identity) != 32
                or any(char not in "0123456789abcdef" for char in identity) or identity in executions):
            raise ValueError("invalid or duplicate execution identity")
        path = Path(root) / "executions" / f"{identity}.json"
        if sha256_file(path) != attempt["preflight_report_sha256"]:
            raise ValueError("archived preflight checksum mismatch")
        report = json.loads(path.read_text())
        check_preflight_evidence(report, historical=True)
        if production_provenance(report["provenance"]) != protocol["provenance"]:
            raise ValueError("execution source differs from the experiment")
        expected = production_protocol(report["controls"])
        if protocol["schema"] == "corrected-full-three-band-v2":
            expected["schema"] = protocol["schema"]
            expected["optimizer"] = "adopted corrected Adam 0.001, no scheduler, no label penalty"
            for fit in expected["fits"]:
                fit["training"]["learning_rate"] = .001
        elif report["schema"] != "corrected-gpu-preflight-v4":
            raise ValueError("v3 production requires adopted-rate preflight evidence")
        for key, value in expected.items():
            if protocol[key] != value:
                raise ValueError("execution changed the scientific protocol")
        if (dict(report["controls"], max_seconds=attempt["controls"]["max_seconds"]) != attempt["controls"]
                or report["generation_policy"] != attempt["generation_policy"]):
            raise ValueError("execution operational settings differ from its qualification")
        executions[identity] = attempt
    return executions


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


def measure_preflight_checkpoints(output, model, optimizer, artifact, config, provenance, device):
    """Disjoint selected-save preparation/write and complete recovery-wrapper timing."""
    tick = time.perf_counter()
    payload = production_payload(model, artifact, config, provenance, 0, None)
    preparation = time.perf_counter() - tick
    tick = time.perf_counter()
    _save_checkpoint(output / f"m{model.interactions}.pt", payload)
    selected = time.perf_counter() - tick
    tick = time.perf_counter()
    synchronize(device)
    save_resume_checkpoint(output / f"m{model.interactions}-resume-probe.pt",
                           {"current": model.state_dict(), "optimizer": optimizer.state_dict(), "best": payload,
                            "reload_probe": checkpoint_probe(model, artifact)})
    recovery = time.perf_counter() - tick
    return payload, preparation, selected, recovery


def timing_io_rows(path):
    values = []
    for line in Path(path).read_text().splitlines():
        key, separator, raw = line.partition("=")
        if key.strip() == "BANDNET_TIMING_IO_ROWS":
            if not separator:
                raise ValueError("malformed timing I/O setting")
            values.append(int(raw.strip()))
    if len(values) != 1 or values[0] < 2 or values[0] % 2:
        raise ValueError("one explicit positive even BANDNET_TIMING_IO_ROWS required")
    return values[0]


def measure_training_reads(path, batch_size, device, budget_check):
    """One shuffled epoch, matching production's mmap indexing/copy and transfer."""
    cache = {"method": "posix_fadvise DONTNEED", "advice_succeeded": False,
             "may_be_page_cached": True, "reason": "advice unavailable; no root cache drop attempted"}
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED"):
        try:
            with Path(path).open("rb") as stream:
                os.fsync(stream.fileno())
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            cache.update(advice_succeeded=True, reason="advisory eviction requested, not proof of cold storage")
        except OSError as error:
            cache["reason"] = f"advisory eviction failed: {error}"
    bands = np.load(path, mmap_mode="r")
    if bands.dtype != np.float64 or bands.shape[1:] != (500, 3):
        raise ValueError("timing reads require production float64 three-band shape")
    order = np.random.default_rng(np.random.SeedSequence([271829, 100, 1])).permutation(len(bands))
    synchronize(device)
    tick = time.perf_counter()
    for start in range(0, len(order), batch_size):
        budget_check()
        target = np.array(bands[order[start:start + batch_size]], copy=True)
        transferred = torch.tensor(target, dtype=torch.float64, device=device)
        synchronize(device)
        del transferred
    seconds = time.perf_counter() - tick
    return {"rows": len(bands), "bytes": bands.nbytes, "seconds": seconds,
            "rows_per_second": len(bands) / seconds, "cache": cache,
            "scope": "one complete shuffled epoch: mmap read, host copy, device transfer; no optimizer updates"}


def timing_slice_allocation(requested, remaining_seconds, measured_rows_per_second, batch_size):
    if (requested < 2 or requested % 2 or not np.isfinite(measured_rows_per_second)
            or measured_rows_per_second <= 0):
        raise ValueError("explicit even slice target and positive measured generation rate required")
    allowance = min(120, remaining_seconds - 20)
    rows = min(requested, int(allowance * measured_rows_per_second / 1.5)) // 2 * 2
    if allowance <= 0 or rows < batch_size:
        raise TimeoutError("insufficient allowance for a real dataset I/O slice")
    return rows, allowance


def gpu_preflight(output, *, timing_only=False):
    """Bounded numerical and full-architecture throughput checks, not a fit search."""
    started = time.monotonic()
    automatic = None
    if timing_only:
        checked_device("cuda:0")
        automatic = configure_machine_resources(ROOT / ".env.local")
    controls = production_settings(ROOT / ".env.local")

    def budget_check():
        if time.monotonic() - started >= controls["max_seconds"]:
            raise TimeoutError("preflight time allowance exhausted")

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
    bulk = output / "bulk" if timing_only else output
    if timing_only:
        bulk.mkdir()
    report = {"schema": "corrected-timing-check-v4" if timing_only else "corrected-gpu-preflight-v4",
               "passed": False, "provenance": provenance, "hardware": hardware,
               "checkpoint_schedule": "initial-periodic-terminal-v1",
               "numerical_timing_passed": False, "learning_health_passed": False,
              "automatic_resources": automatic, "disposable_timing_test": timing_only,
              "controls": controls, "generation_policy": policy.__dict__,
              "resources": detect_resources().as_dict(), "counts": {},
                "scope": "M5/M20 numerical checks; all M5..M20 throughput in timing mode; no accuracy/sizing campaign"}
    probe_rate = production_protocol(controls)["fits"][0]["training"]["learning_rate"]
    report["probe_training"] = {"learning_rate": probe_rate, "seed": 424245,
                                 "updates": 7, "production_rate": True}
    if timing_only:
        report["timing_policy"] = {"recovery_saves": "measured every count, no interpolation",
                                  "intermediate_updates": 5, "endpoint_updates": 7,
                                  "generation_tuning": "M5/M20 tuned; M6-M19 reuse the M5 plan and are charged the slower endpoint tuning time",
                                  "io_target_rows": timing_io_rows(ROOT / ".env.local"),
                                  "io_max_seconds": 120, "io_generation_safety_factor": 1.5,
                                  "export_scope": "report.json, sources/ and small arrays; bulk/ excluded"}
    write_json(output / "report.json", report)
    # Use a separate stream from every development/production population.
    for interactions in (range(5, 21) if timing_only else (5, 20)):
        budget_check()
        numerical = preflight(interactions, 424242, 8, device=controls["device"]) if interactions in (5, 20) else None
        measured_updates = 5 if interactions in (5, 20) else 3
        budget_check()
        solver = TriatomicBatchSolver(GRID, interactions)
        labels = sample_labels(controls["batch_size"], interactions, 424243, "train")
        if timing_only and interactions not in (5, 20):
            # Retuning every count would exhaust the allocation; the verifier charges
            # endpoint tuning time for these counts instead of omitting it.
            plan = dataclasses.replace(m5_plan, estimated_working_bytes=working_bytes(
                len(solver.q_hat_values), interactions, m5_plan.chunk_size, m5_plan.workers))
            tuning = {"reused_plan_from_interactions": 5, "selected_plan": dataclasses.asdict(plan),
                      "elapsed_seconds": None}
        else:
            plan, tuning = tune_execution(solver, *physical_arrays(labels, interactions), policy)
        if interactions == 5:
            m5_plan = plan
        budget_check()
        data_config = dict(production_protocol(controls)["shared"], interactions=interactions,
                           seed=424244, train_count=controls["batch_size"],
                           validation_count_per_population=2048, test_count_per_population=2)
        tick = time.perf_counter()
        generate_artifact(bulk / f"m{interactions}-data", data_config, solver, plan, provenance,
                          checkpoint_rows=4096, after_chunk=lambda *args: budget_check())
        generation_seconds = time.perf_counter() - tick
        budget_check()
        artifact = LabeledArtifact(bulk / f"m{interactions}-data")
        target = np.array(artifact.arrays["train"][1], copy=True)
        torch.manual_seed(424245)
        model = ProductionBandInverse(interactions, np.sqrt(np.mean(target**2, axis=(0, 1)))).to(device)
        forward = DifferentiableTriatomic(GRID, interactions).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=probe_rate, foreach=False)
        validation_witness = np.concatenate([
            artifact.arrays[population][1][np.linspace(0, len(artifact.arrays[population][1]) - 1,
                                                     min(16, len(artifact.arrays[population][1])), dtype=int)]
            for population in ("validation_dense", "validation_sparse")])
        initial_learning = learning_observation(model, forward, target, validation_witness, device, gradients=True)

        def preserve_learning_failure(stage, error, final=None):
            failure = {"stage": stage, "error": f"{type(error).__name__}: {error}",
                       "numerical": numerical, "batch_size": controls["batch_size"],
                       "learning_observations": {"initial": initial_learning},
                       "checkpoint_export_started": False}
            if final is None:
                try:
                    final = learning_observation(model, forward, target, validation_witness, device, gradients=True)
                except ArithmeticError as observation_error:
                    failure["final_observation_error"] = str(observation_error)
            if final is not None:
                failure["learning_observations"]["final"] = final
                failure["learning_health"] = learning_health([initial_learning, final],
                                                             validation_required=not timing_only)
                failure["learning_health"]["passed"] = False
                failure["learning_health"]["reasons"].append(f"training execution failed at {stage}")
            report["counts"][str(interactions)] = failure
            report["failure"] = {"interactions": interactions, "stage": stage, "error": str(error)}

            def failure_json(value):
                # Preserve nonfinite evidence as tags, never repaired numeric results.
                # This encoding is failure-only and cannot qualify as passing health.
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    if np.isnan(value):
                        return {"nonfinite": "nan"}
                    return {"nonfinite": "positive_infinity" if value > 0 else "negative_infinity"}
                if isinstance(value, dict):
                    return {key: failure_json(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [failure_json(item) for item in value]
                return value

            write_json(output / "report.json", failure_json(report))

        if timing_only and interactions == 5:
            budget_check()
            tuning_limit = min(policy.tuning_seconds, controls["max_seconds"] - (time.monotonic() - started))
            try:
                automatic["training_threads"] = tune_training_threads(
                    model, forward, optimizer, target, device, automatic["available_threads"], tuning_limit)
            except ArithmeticError as error:
                preserve_learning_failure("thread_tuning", error)
                raise
            controls["torch_threads"] = automatic["training_threads"]["selected_threads"]
            update_local_settings(ROOT / ".env.local", {"BANDNET_PRODUCTION_TORCH_THREADS": controls["torch_threads"]})
        def observed_update(batch, stage):
            try:
                return training_step(model, forward, optimizer, batch, device)
            except ArithmeticError as error:
                preserve_learning_failure(stage, error)
                raise
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        for _ in range(2):
            budget_check()
            observed_update(target, "warmup")
        durations, losses = [], []
        for _ in range(measured_updates):
            budget_check()
            synchronize(device)
            tick = time.perf_counter()
            # Include the same shuffled mmap read and transfer used by production.
            order = np.random.default_rng(424246).permutation(len(target))
            batch = np.array(artifact.arrays["train"][1][order], copy=True)
            losses.append(observed_update(batch, "timed_update"))
            synchronize(device)
            durations.append(time.perf_counter() - tick)
        final_learning = learning_observation(model, forward, target, validation_witness, device, gradients=True)
        health = learning_health([initial_learning, final_learning], validation_required=not timing_only)
        if not health["passed"]:
            error = ArithmeticError("bounded learning health failed; observations retained before checkpoint/export")
            preserve_learning_failure("learning_health", error, final_learning)
            raise error
        training_config = {"architecture": "original-five-relu-corrected-io-v1",
                           "batch_size": controls["batch_size"], "learning_rate": probe_rate,
                            "seed": 424245, "updates": 2 + measured_updates, "scope": "disposable-timing-or-preflight"}
        checkpoint = bulk / f"m{interactions}.pt"
        budget_check()
        payload, payload_prepare_seconds, save_seconds, resume_save_seconds = measure_preflight_checkpoints(
            bulk, model, optimizer, artifact, training_config, provenance, device)
        tick = time.perf_counter()
        for population in ("validation_dense", "validation_sparse"):
            budget_check()
            validation, _ = evaluate_population(model, artifact, population, controls["batch_size"], retain_curves=False)
            if validation["invalid_prediction_count"]:
                raise ArithmeticError("full-validation probe produced invalid designs")
        validation_seconds = time.perf_counter() - tick
        probe = np.array(artifact.arrays["showcase"][1], copy=True)
        budget_check()
        before = predict(model, probe, controls["batch_size"])
        tick = time.perf_counter()
        restored, cpu_model = None, None
        if interactions in (5, 20):
            restored, _ = load_model(checkpoint, device=controls["device"])
            np.testing.assert_array_equal(before, predict(restored, probe, controls["batch_size"]))
            cpu_model, _ = load_model(checkpoint)
            cpu_predictions = predict(cpu_model, probe, controls["batch_size"])
            np.testing.assert_allclose(before, cpu_predictions, rtol=1e-10, atol=1e-10)
        _, scores, failed = evaluate_designs(probe, before, interactions)
        if failed or not np.all(np.isfinite(scores)):
            raise ArithmeticError("GPU checkpoint predictions failed common CPU scoring")
        reload_check_seconds = time.perf_counter() - tick
        # Exercise actual serialized Adam continuation on this CUDA stack. These
        # two verification updates are separate from the five throughput samples.
        budget_check()
        if not timing_only:
            tick = time.perf_counter()
            resume_state = read_resume_checkpoint(output / f"m{interactions}-resume-probe.pt")
            resumed = ProductionBandInverse(interactions, payload["input_scale"]).to(device)
            resumed.load_state_dict(resume_state["current"])
            resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=probe_rate, foreach=False)
            resumed_optimizer.load_state_dict(resume_state["optimizer"])
            original_loss = training_step(model, forward, optimizer, target, device)
            budget_check()
            resumed_loss = training_step(resumed, forward, resumed_optimizer, target, device)
            if original_loss != resumed_loss:
                raise AssertionError("GPU Adam reload changed the next loss")
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
            synchronize(device)
            resume_verification_seconds = time.perf_counter() - tick
            del resumed, resumed_optimizer, resume_state
        row = {"numerical": numerical, "calibration": tuning, "training_configuration": training_config,
               "generation_write_seconds": generation_seconds,
               "generated_rows": sum(len(pair[0]) for pair in artifact.arrays.values()),
                "warmup_updates": 2, "measured_updates": measured_updates, "batch_size": controls["batch_size"],
               "transfer_and_mmap_inclusive_step_seconds": durations, "losses": losses,
               "median_step_seconds": float(np.median(durations)),
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
               "checkpoint_write_seconds": save_seconds, "checkpoint_sha256": sha256_file(checkpoint),
               "learning_health": health,
               "learning_observations": {"initial": initial_learning, "final": final_learning},
               "checkpoint_timing_scope": "selected serialization only; payload preparation separate; recovery includes fresh probe preparation and integrity hash",
               "best_payload_prepare_seconds": payload_prepare_seconds,
               "reload_and_scoring_check_seconds": reload_check_seconds,
               "resume_checkpoint_write_seconds": resume_save_seconds,
               "adam_reload_next_update_exact": None if timing_only else True,
               "resume_verification_updates": 0 if timing_only else 2,
               "resume_verification_seconds": None if timing_only else resume_verification_seconds,
               "full_validation_4096_seconds": validation_seconds,
                "cuda_reload_exact": True if interactions in (5, 20) else None, "cpu_reload_prediction_tolerance": {"rtol": 1e-10, "atol": 1e-10},
               "common_cpu_scoring_passed": True}
        if timing_only:
            budget_check()
            tick = time.perf_counter()
            compact_evaluation(output / f"m{interactions}-timing-records", model, artifact, "validation_dense",
                               controls["batch_size"], row["checkpoint_sha256"])
            row["compact_evaluation_seconds"] = time.perf_counter() - tick
            row["compact_evaluation_rows"] = len(artifact.arrays["validation_dense"][0])
            indices = np.linspace(0, row["compact_evaluation_rows"] - 1, 8, dtype=int)
            save_array(output / f"m{interactions}-timing-records" / "sample_targets.npy",
                       np.array(artifact.arrays["validation_dense"][1][indices], copy=True))
        report["counts"][str(interactions)] = row
        write_json(output / "report.json", report)
        del model, restored, cpu_model, optimizer, forward, artifact, payload
        gc.collect()
        torch.cuda.empty_cache()
    if timing_only:
        budget_check()
        remaining = controls["max_seconds"] - (time.monotonic() - started)
        rate = report["counts"]["5"]["generated_rows"] / report["counts"]["5"]["generation_write_seconds"]
        requested = report["timing_policy"]["io_target_rows"]
        rows, allowance = timing_slice_allocation(requested, remaining, rate, controls["batch_size"])
        config = dict(production_protocol(controls)["fits"][0]["data"], seed=424247,
                      train_count=rows, validation_count_per_population=2, test_count_per_population=2)
        tick = time.perf_counter()
        generate_artifact(bulk / "main-m5-io", config, TriatomicBatchSolver(GRID, 5), m5_plan, provenance,
                          checkpoint_rows=4096, after_chunk=lambda *args: budget_check())
        generation = time.perf_counter() - tick
        io_artifact = LabeledArtifact(bulk / "main-m5-io")
        bands_path = Path(io_artifact.arrays["train"][1].filename)
        del io_artifact
        report["disk_read"] = measure_training_reads(bands_path, controls["batch_size"], device, budget_check)
        report["disk_read"].update(requested_rows=requested, generation_seconds=generation,
                                   reduced_for_budget=rows != requested, generation_allowance_seconds=allowance,
                                   reduction_rule="min(target, floor(min(120s, remaining minus 20s reserve) times M5 generation rate / 1.5)); even rows")
    median = max(row["median_step_seconds"] for row in report["counts"].values())
    budget_check()
    steps = 5 * ((2500000 + 1023) // 1024) + 16 * 100 * ((100000 + 1023) // 1024)
    healthy = all(row["learning_health"]["passed"] for row in report["counts"].values())
    report.update(passed=healthy, numerical_timing_passed=True, learning_health_passed=healthy,
                  elapsed_seconds=time.monotonic() - started,
                  optimizer_steps_in_full_matrix=steps,
                  training_only_projection_seconds=steps * median,
                   projection_scope="conservative max-count training median only; use independent stage projection for the full estimate",
                  process_lifetime_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)
    report["files"] = {str(path.relative_to(output)): sha256_file(path)
                       for path in sorted(output.rglob("*")) if path.is_file() and path.name != "report.json"
                       and (not timing_only or not path.is_relative_to(bulk))}
    write_json(output / "report.json", report)
    if timing_only:
        (output / "report.sha256").write_text(sha256_file(output / "report.json") + "\n")
    print(json.dumps({"passed": healthy, "report": str(output / "report.json"),
                       "training_only_projection_seconds": report["training_only_projection_seconds"]}), flush=True)
    if not healthy:
        raise RuntimeError("numerical/timing checks completed but bounded learning health failed; evidence retained")


def production_run(output, preflight_root, *, resume=False):
    invocation_start = time.monotonic()
    controls = production_settings(ROOT / ".env.local")
    policy = load_generation_policy(ROOT / ".env.local")
    device = checked_device(controls["device"])
    torch.set_num_threads(controls["torch_threads"])
    torch.use_deterministic_algorithms(True)
    source = source_identity()
    provenance = production_provenance(source)
    precheck_bytes = (preflight_root / "report.json").read_bytes()
    precheck = json.loads(precheck_bytes)
    check_preflight_evidence(precheck)
    expected_controls = dict(precheck["controls"], max_seconds=controls["max_seconds"])
    if (precheck["provenance"] != source
            or precheck["hardware"] != gpu_identity() or expected_controls != controls
            or precheck["generation_policy"] != policy.__dict__):
        raise ValueError("a successful preflight on this exact source/configuration/GPU stack is required")
    for name, digest in precheck["files"].items():
        path = (preflight_root / name).resolve()
        if not path.is_relative_to(preflight_root.resolve()) or sha256_file(path) != digest:
            raise ValueError("preflight evidence checksum mismatch")
    protocol = production_protocol(controls)
    protocol.update(provenance=provenance)
    if resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("resume requires the identical frozen production protocol")
        state = json.loads((output / "progress.json").read_text())
        validate_execution_history(output, protocol, state["attempts"])
        for name, digest in provenance["source_sha256"].items():
            if sha256_file(output / "sources" / name) != digest:
                raise ValueError("retained production source checksum mismatch")
    else:
        if output.exists() or not output.parent.is_dir():
            raise ValueError("production output must be new with an existing parent")
        if shutil.disk_usage(output.parent).free < 150 * 10**9:
            raise OSError("production requires at least 150 GB free on the persistent volume")
        output.mkdir()
        (output / "executions").mkdir()
        retain_sources(output, provenance)
        write_json(output / "protocol.json", protocol)
        state = {"status": "running", "fits": {}, "evaluations": {}, "attempts": []}
    for attempt in state["attempts"]:
        if attempt["status"] == "running":
            attempt["status"] = "interrupted_duration_unknown"
    execution_id = uuid.uuid4().hex
    archive_path = output / "executions" / f"{execution_id}.json"
    with archive_path.open("xb") as archive:
        archive.write(precheck_bytes)
        archive.flush()
        os.fsync(archive.fileno())
    descriptor = os.open(archive_path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    attempt = {"resume": resume, "status": "running", "elapsed_seconds": None,
               "max_seconds": controls["max_seconds"], "execution_id": execution_id,
               "preflight_report_sha256": hashlib.sha256(precheck_bytes).hexdigest(),
               "controls": dict(controls), "generation_policy": dict(policy.__dict__),
               "resources_at_start": detect_resources().as_dict(),
               "checkpoint_qualifications": {}}
    state["attempts"].append(attempt)
    state["status"] = "running"
    write_json(output / "progress.json", state)
    started = invocation_start
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
        calibrations.append({"execution_id": execution_id, "tuning": tuning})
        write_json(calibration_path, calibrations)
        generate_artifact(path, config, solver, plan, provenance, resume=path.exists(),
                          after_chunk=lambda name, stop: budget_check(), checkpoint_rows=4096,
                          execution_id=execution_id)
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
                                               execution=controls, execution_id=execution_id,
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
            model, payload = load_model(checkpoint)
            artifact = dataset(name, fit["data"])
            if payload["dataset_manifest_sha256"] != sha256_file(artifact.root / "manifest.json"):
                raise ValueError("checkpoint dataset mismatch")
            attempt["checkpoint_qualifications"][name] = qualify_checkpoint_model(
                model, artifact, payload["reload_probe"], device, gradients=False)
            write_json(output / "progress.json", state)
            budget_check()
            evaluations = [(population, artifact, population) for population in ("test_dense", "test_sparse", "adversarial", "showcase")]
            if name.startswith("study-"):
                evaluations.extend((f"shared_m5_{population}", shared, population) for population in ("test_dense", "test_sparse"))
            for label, targets, population in evaluations:
                budget_check()
                path = output / name / label
                summary = compact_evaluation(path, model, targets, population, controls["batch_size"],
                                             state["fits"][name]["checkpoint_sha256"], resume=path.exists(),
                                             after_chunk=lambda stop: budget_check(), execution_id=execution_id)
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
    modes.add_argument("--timing-check", action="store_true", help="disposable automatic-resource timing test; never a production checkpoint")
    modes.add_argument("--production", action="store_true", help="fixed full-size CUDA matrix; paid launch requires owner approval")
    modes.add_argument("--bounded-learning-diagnostic", action="store_true", help="approved local retained-fixture diagnostic, at most 35 updates")
    parser.add_argument("--preflight", type=Path, help="successful GPU preflight artifact required by production")
    parser.add_argument("--resume", action="store_true", help="resume the same experiment on a newly qualified compatible machine")
    args = parser.parse_args()
    if (args.resume or args.preflight is not None) and not args.production:
        parser.error("--resume and --preflight are production-only")
    if args.bounded_learning_diagnostic:
        bounded_learning_diagnostic(args.output)
        return
    if args.production:
        if args.preflight is None:
            parser.error("--production requires --preflight")
        production_run(args.output, args.preflight, resume=args.resume)
        return
    if args.gpu_preflight or args.timing_check:
        gpu_preflight(args.output, timing_only=args.timing_check)
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
