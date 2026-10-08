"""Verify retained pilot files and export its original report plus audit evidence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from corrected_pilot import ROOT, settings, production_settings, production_protocol, validate_execution_history, check_learning_evidence
from triatomic_data import (GRID, LabeledArtifact, array_hash, sha256_file, write_json,
                            adversarial_labels, showcase_labels, physical_arrays)
from triatomic_genuine_formula import triatomic_frequencies
from triatomic_learning import band_errors, evaluate_designs, evaluate_population, load_model, predict, summarize_errors


def verify_run(root, batch_size):
    root = Path(root)
    original = json.loads((root / "report.json").read_text())
    for name, expected in original["files"].items():
        path = root / name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"retained artifact checksum mismatch: {name}")
    for name, expected in original["provenance"]["source_sha256"].items():
        if sha256_file(root / "sources" / name) != expected:
            raise ValueError(f"source snapshot differs from measured implementation: {name}")
    artifact = LabeledArtifact(root / "data")
    model, checkpoint = load_model(root / "training" / "best.pt")
    if checkpoint["dataset_manifest_sha256"] != sha256_file(root / "data" / "manifest.json"):
        raise ValueError("checkpoint belongs to another dataset manifest")
    checked = 0
    for population, summary in original["evaluation"].items():
        if summary["invalid_prediction_count"]:
            raise ValueError("pilot contains invalid predictions")
        predictions = np.load(root / "evaluation" / f"{population}.predictions.npy", mmap_mode="r")
        curves = np.load(root / "evaluation" / f"{population}.reconstructed_bands.npy", mmap_mode="r")
        errors = np.load(root / "evaluation" / f"{population}.per_band_errors.npy", mmap_mode="r")
        for rows, targets, _ in artifact.batches(population, batch_size):
            # Small floating-point differences from a changed GEMM batch shape
            # have a stated tight tolerance; stored metric recomputation is exact.
            np.testing.assert_allclose(predict(model, targets, batch_size), predictions[rows], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(band_errors(targets, curves[rows]), errors[rows])
            checked += len(rows)
        if float(np.mean(np.mean(errors, axis=1))) != summary["primary"]:
            raise ValueError("stored population metric mismatch")
    standalone = json.loads((root / "standalone-showcase" / "scores.json").read_text())
    if standalone["checkpoint_sha256"] != sha256_file(root / "training" / "best.pt"):
        raise ValueError("standalone checkpoint mismatch")
    for name, digest in standalone["files"].items():
        path = root / "standalone-showcase" / f"{name}.npy"
        if sha256_file(path) != digest:
            raise ValueError("standalone output checksum mismatch")
        np.testing.assert_array_equal(np.load(path), np.load(root / "evaluation" / f"showcase.{name}.npy"))
    if standalone["primary"] != original["evaluation"]["showcase"]["primary"]:
        raise ValueError("standalone score differs from integrated evaluation")
    return {"original_report": original,
            "audit": {"original_report_sha256": sha256_file(root / "report.json"),
                      "auditor_sha256": sha256_file(Path(__file__)),
                      "retained_file_count_checked": len(original["files"]),
                      "source_snapshots_verified": True, "labeled_artifact_verified": True,
                      "checkpoint_dataset_identity_verified": True,
                      "predictions_and_metrics_checked": checked,
                      "standalone_arrays_equal_integrated_arrays": True,
                      "standalone_report": standalone}}


def verify_sizing(root, batch_size):
    """Audit all retained development predictions without evaluating final tests."""
    root = Path(root)
    original = json.loads((root / "report.json").read_text())
    for name, expected in original["files"].items():
        path = root / name
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError(f"retained artifact checksum mismatch: {name}")
    provenance = original["protocol"]["provenance"]
    for name, digest in provenance["source_sha256"].items():
        if sha256_file(root / "sources" / name) != digest:
            raise ValueError(f"measured source mismatch: {name}")
    artifacts = {key: LabeledArtifact(root / f"{key}-data") for key in original["generation"]}
    first, second = artifacts["m5-n2048"], artifacts["m5-n8192"]
    for population in ("validation_dense", "validation_sparse"):
        for left, right in zip(first.arrays[population], second.arrays[population]):
            np.testing.assert_array_equal(left, right)
    checked, checkpoints, learning_summary = 0, [], []
    for run in original["runs"]:
        model, payload = load_model(root / run["name"] / "best.pt")
        key = f"m{run['interactions']}-n{run['train_count']}"
        artifact = artifacts[key]
        if payload["dataset_manifest_sha256"] != sha256_file(artifact.root / "manifest.json"):
            raise ValueError("sizing checkpoint dataset mismatch")
        if payload["provenance"] != provenance or payload["training_configuration"] != run["configuration"]:
            raise ValueError("sizing checkpoint source/configuration mismatch")
        best = min(run["history"], key=lambda row: row["validation_primary"])
        if payload["epoch"] != best["epoch"] or payload["validation_primary"] != best["validation_primary"]:
            raise ValueError("checkpoint was not selected by minimum validation")
        for name, summary in run["diagnostics"].items():
            shared = name.startswith("shared_m5_")
            population = name.removeprefix("shared_m5_") if shared else name
            targets = artifacts[f"m5-n{run['train_count']}"] if shared else artifact
            predictions = np.load(root / run["name"] / f"{name}.predictions.npy", mmap_mode="r")
            errors = np.load(root / run["name"] / f"{name}.per_band_errors.npy", mmap_mode="r")
            for rows, bands, _ in targets.batches(population, batch_size):
                np.testing.assert_allclose(predict(model, bands, batch_size), predictions[rows], rtol=1e-12, atol=1e-12)
                curves, scores, failures = evaluate_designs(bands, predictions[rows], run["interactions"])
                if failures:
                    raise ValueError("invalid retained development design")
                np.testing.assert_array_equal(scores, errors[rows])
                if not shared:
                    stored = np.load(root / run["name"] / f"{name}.reconstructed_bands.npy", mmap_mode="r")
                    np.testing.assert_array_equal(curves, stored[rows])
                checked += len(rows)
            recomputed = summarize_errors(errors, [])
            for metric in recomputed:
                if recomputed[metric] != summary[metric]:
                    raise ValueError(f"stored sizing summary differs: {name}/{metric}")
        checkpoints.append({"name": run["name"], "best_epoch": best["epoch"],
                            "checkpoint_sha256": sha256_file(root / run["name"] / "best.pt")})
        milestones = {}
        for epoch in (20, 40, 60):
            prefix = run["history"][:epoch + 1]
            selected_epoch = min(prefix, key=lambda row: row["validation_primary"])
            milestones[str(epoch)] = {"best_epoch": selected_epoch["epoch"],
                                     "best_validation_primary": selected_epoch["validation_primary"],
                                     "last_validation_primary": prefix[-1]["validation_primary"]}
        learning_summary.append({"name": run["name"], "milestones": milestones,
                                 "train_seconds_per_example_epoch": run["training_seconds"] / (run["train_count"] * 60),
                                 "validation_seconds_total": sum(row["validation_dense"]["inference_reconstruction_scoring_seconds"]
                                                                 + row["validation_sparse"]["inference_reconstruction_scoring_seconds"]
                                                                 for row in run["history"]),
                                 "diagnostics": {name: {key: summary[key] for key in ("count", "primary", "p95_per_sample", "maximum_per_sample", "invalid_prediction_count")}
                                                 for name, summary in run["diagnostics"].items()}})
    minimum = min(run["best_validation_primary"] for run in original["runs"][:3])
    eligible = [run for run in original["runs"][:3] if run["best_validation_primary"] <= minimum * 1.05]
    selected = min(eligible, key=lambda run: (run["configuration"]["width"], run["train_count"]))
    if selected["name"] != original["selected"]:
        raise ValueError("sizing selection rule mismatch")
    return {"original_report": original, "learning_summary": learning_summary,
            "audit": {"original_report_sha256": sha256_file(root / "report.json"),
                      "auditor_sha256": sha256_file(Path(__file__)),
                      "retained_file_count_checked": len(original["files"]),
                      "source_snapshots_verified": True, "labeled_artifacts_verified": True,
                      "shared_m5_validation_identity_verified": True,
                      "selection_rule_verified": True, "checkpoints": checkpoints,
                      "predictions_and_metrics_checked": checked,
                      "final_tests_and_showcases_evaluated": False}}


def timing_workload(checkpoint_steps, checkpoint_schedule):
    """Independent arithmetic for the agreed fixed experiment, not fitted timing."""
    if type(checkpoint_steps) is not int or checkpoint_steps < 1:
        raise ValueError("positive checkpoint interval required")
    if checkpoint_schedule not in ("initial-periodic-epoch-v1", "initial-periodic-terminal-v1"):
        raise ValueError("unsupported checkpoint schedule")
    fits = [(5, 2500000, 5, 125000), *[(k, 100000, 100, 5000) for k in range(5, 21)]]
    work = {"fits": 17, "optimizer_updates": 0, "generated_rows": 0,
            "generated_payload_bytes": 0, "validation_passes": 0,
            "resumable_checkpoint_writes": 0, "final_records": 0, "generation_calibrations": 18}
    for index, (k, train, epochs, test) in enumerate(fits):
        extras = len(adversarial_labels(k)) + len(showcase_labels(k))
        rows = train + 4096 + 2 * test + extras
        steps = epochs * ((train + 1023) // 1024)
        work["optimizer_updates"] += steps
        work["generated_rows"] += rows
        work["generated_payload_bytes"] += rows * (1500 + k + 1) * 8
        work["validation_passes"] += epochs + 1
        boundary_saves = epochs + 2 if checkpoint_schedule == "initial-periodic-epoch-v1" else 2
        work["resumable_checkpoint_writes"] += boundary_saves + steps // checkpoint_steps
        work["final_records"] += 2 * test + extras + (20000 if index else 0)
    shared_rows = 32 + 4 + 20000 + len(adversarial_labels(5)) + len(showcase_labels(5))
    work["generated_rows"] += shared_rows
    work["generated_payload_bytes"] += shared_rows * 1506 * 8
    return work


def timing_projection_all_counts(report, hash_bytes_per_second):
    if set(report["counts"]) != {str(k) for k in range(5, 21)}:
        raise ValueError("timing requires measured M5..M20 rates without interpolation")
    if report["checkpoint_schedule"] != "initial-periodic-terminal-v1":
        raise ValueError("new timing schema requires the lean checkpoint schedule")
    io = report["disk_read"]
    if (io["rows"] < 1 or io["bytes"] != io["rows"] * 1500 * 8
            or not np.isfinite(io["seconds"]) or io["seconds"] <= 0
            or io["rows_per_second"] != io["rows"] / io["seconds"]):
        raise ValueError("invalid measured disk-read rate")
    if not np.isfinite(hash_bytes_per_second) or hash_bytes_per_second <= 0:
        raise ValueError("positive measured checksum rate required")
    stages = {}
    fits = []

    def add(name, seconds, upper=None):
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(f"missing positive measurements for {name}")
        if upper is None:
            upper = seconds
        if not np.isfinite(upper) or upper < seconds:
            raise ValueError(f"invalid upper measurement for {name}")
        if name not in stages:
            stages[name] = {"lower_seconds": 0., "upper_seconds": 0.}
        stages[name]["lower_seconds"] += seconds
        stages[name]["upper_seconds"] += upper

    work = timing_workload(report["controls"]["checkpoint_steps"], report["checkpoint_schedule"])
    for index, (k, train, epochs, test) in enumerate([(5, 2500000, 5, 125000),
                                                   *[(k, 100000, 100, 5000) for k in range(5, 21)]]):
        row = report["counts"][str(k)]
        updates = epochs * ((train + 1023) // 1024)
        extras = len(adversarial_labels(k)) + len(showcase_labels(k))
        generated = train + 4096 + 2 * test + extras
        records = 2 * test + extras + (20000 if index else 0)
        samples = np.asarray(row["transfer_and_mmap_inclusive_step_seconds"], dtype=float)
        if not len(samples) or not np.all(np.isfinite(samples)) or np.any(samples <= 0):
            raise ValueError("invalid measured per-count update rate")
        training = float(np.median(samples)) * updates
        reads = train * epochs / io["rows_per_second"]
        fits.append({"name": "main-m5" if index == 0 else f"study-m{k}",
                     "interactions": k, "optimizer_updates": updates,
                     "training_seconds": training, "disk_read_seconds": reads})
        add("training_updates", training)
        add("full_dataset_shuffled_reads", reads)
        add("data_generation_and_writing", row["generation_write_seconds"] / row["generated_rows"] * generated)
        if k in (5, 20):
            add("generation_calibration", row["calibration"]["elapsed_seconds"])
        else:
            if row["calibration"]["reused_plan_from_interactions"] != 5:
                raise ValueError("intermediate counts must record the reused M5 generation plan")
            add("generation_calibration", max(report["counts"][str(e)]["calibration"]["elapsed_seconds"] for e in (5, 20)))
        add("training_validation", row["full_validation_4096_seconds"] * (epochs + 1))
        add("periodic_checkpoint_writes", row["resume_checkpoint_write_seconds"] *
            (2 + updates // report["controls"]["checkpoint_steps"]))
        add("selected_model_writes", row["checkpoint_write_seconds"])
        add("best_weight_copies", row["best_payload_prepare_seconds"], row["best_payload_prepare_seconds"] * (epochs + 1))
        record_rate = row["compact_evaluation_seconds"] / row["compact_evaluation_rows"]
        add("final_inference_scoring_writing", record_rate * records)
        add("independent_audit_inference_proxy", record_rate * records)
        add("independent_audit_validation", row["full_validation_4096_seconds"])
    # Shared targets and model reload checks are measured at their actual endpoint.
    m5 = report["counts"]["5"]
    shared = 32 + 4 + 20000 + len(adversarial_labels(5)) + len(showcase_labels(5))
    add("data_generation_and_writing", m5["generation_write_seconds"] / m5["generated_rows"] * shared)
    add("generation_calibration", m5["calibration"]["elapsed_seconds"])
    reloads = [report["counts"][str(k)]["reload_and_scoring_check_seconds"] for k in (5, 20)]
    add("model_reload_checks", min(reloads) * 34, max(reloads) * 34)
    add("dataset_integrity_scans", 3 * work["generated_payload_bytes"] / hash_bytes_per_second)
    lower = sum(row["lower_seconds"] for row in stages.values())
    upper = sum(row["upper_seconds"] for row in stages.values())
    return {"checkpoint_schedule": report["checkpoint_schedule"], "workload": work, "fits": fits,
            "stages": stages, "lower_seconds": lower, "upper_seconds": upper,
            "planning_seconds_with_25_percent_margin": upper * 1.25,
            "range_meaning": "measured per-count rates; best-selection and endpoint reload brackets, not a confidence interval",
            "limitations": ["cache eviction is advisory; disk reads may have been page cached",
                            "M5 shuffled-read bandwidth applies to identical 1500-feature study rows",
                            "full shuffled reads added conservatively to cached-read-inclusive updates; some I/O double counting",
                            "checksum bandwidth measured on small evidence, not cold full datasets",
                            "independent audit inference uses write-inclusive evaluation as a proxy",
                            "provisioning, installation and export are additional",
                            "25 percent planning margin is an explicit assumption"]}


def timing_projection(report, hash_bytes_per_second, hourly_price=None):
    result = timing_projection_seconds(report, hash_bytes_per_second)
    result["lower_hours"] = result["lower_seconds"] / 3600
    result["upper_hours"] = result["upper_seconds"] / 3600
    result["planning_hours_with_25_percent_margin"] = result["planning_seconds_with_25_percent_margin"] / 3600
    if hourly_price is not None:
        if not np.isfinite(hourly_price) or hourly_price <= 0:
            raise ValueError("positive input hourly price required")
        result["hourly_price"] = hourly_price
        result["lower_cost"] = result["lower_hours"] * hourly_price
        result["upper_cost"] = result["upper_hours"] * hourly_price
        result["planning_cost_with_25_percent_margin"] = result["planning_hours_with_25_percent_margin"] * hourly_price
    return result


def timing_projection_seconds(report, hash_bytes_per_second):
    if report["schema"] == "corrected-timing-check-v4":
        return timing_projection_all_counts(report, hash_bytes_per_second)
    if report["schema"] == "corrected-timing-check-v1":
        checkpoint_schedule = "initial-periodic-epoch-v1"
        if "checkpoint_schedule" in report:
            raise ValueError("historical timing schema must retain its original checkpoint schedule")
    elif report["schema"] == "corrected-timing-check-v2":
        checkpoint_schedule = report["checkpoint_schedule"]
        if checkpoint_schedule != "initial-periodic-terminal-v1":
            raise ValueError("new timing schema requires the lean checkpoint schedule")
    else:
        raise ValueError("unsupported timing report schema")
    work = timing_workload(report["controls"]["checkpoint_steps"], checkpoint_schedule)
    rows = list(report["counts"].values())
    stages = {}

    def stage(name, samples, count, upper_count=None):
        values = np.asarray(samples, dtype=float)
        if not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"missing positive measurements for {name}")
        high_count = count if upper_count is None else upper_count
        stages[name] = {"lower_seconds": float(values.min() * count),
                        "upper_seconds": float(values.max() * high_count)}

    stage("training_updates", [np.median(row["transfer_and_mmap_inclusive_step_seconds"]) for row in rows], work["optimizer_updates"])
    stage("data_generation_and_writing", [row["generation_write_seconds"] / row["generated_rows"] for row in rows], work["generated_rows"])
    stage("generation_calibration", [row["calibration"]["elapsed_seconds"] for row in rows], work["generation_calibrations"])
    stage("training_validation", [row["full_validation_4096_seconds"] for row in rows], work["validation_passes"])
    stage("periodic_checkpoint_writes", [row["resume_checkpoint_write_seconds"] for row in rows], work["resumable_checkpoint_writes"])
    stage("selected_model_writes", [row["checkpoint_write_seconds"] for row in rows], work["fits"])
    stage("best_weight_copies", [row["best_payload_prepare_seconds"] for row in rows], work["fits"], work["validation_passes"])
    record_rates = [row["compact_evaluation_seconds"] / row["compact_evaluation_rows"] for row in rows]
    stage("final_inference_scoring_writing", record_rates, work["final_records"])
    stage("independent_audit_inference_proxy", record_rates, work["final_records"])
    stage("independent_audit_validation", [row["full_validation_4096_seconds"] for row in rows], work["fits"])
    stage("model_reload_checks", [row["reload_and_scoring_check_seconds"] for row in rows], 2 * work["fits"])
    stage("dataset_integrity_scans", [1 / hash_bytes_per_second], 3 * work["generated_payload_bytes"])
    lower = sum(row["lower_seconds"] for row in stages.values())
    upper = sum(row["upper_seconds"] for row in stages.values())
    return {"checkpoint_schedule": checkpoint_schedule,
            "workload": work, "stages": stages, "lower_seconds": lower, "upper_seconds": upper,
            "planning_seconds_with_25_percent_margin": upper * 1.25,
            "range_meaning": "M5/M20 measured-rate bracket plus best-selection-count bracket; not a confidence interval or guarantee",
            "limitations": ["unmeasured intermediate interaction counts", "small cached sample versus full dataset",
                            "cached checksum bandwidth may exceed cold-storage bandwidth",
                            "audit inference uses the measured write-inclusive evaluation rate as a proxy",
                            "future provisioning, software installation and final artifact export are additional",
                            "25 percent planning margin is an explicit assumption, not measured uncertainty"]}


def verify_timing_small(root, report, hash_rate, hourly_price):
    if (root / "report.sha256").read_text().strip() != sha256_file(root / "report.json"):
        raise ValueError("timing report checksum mismatch")
    required = {f"sources/{name}" for name in report["provenance"]["source_sha256"]}
    checked = 0
    for k in range(5, 21):
        row = report["counts"][str(k)]
        measured = 5 if k in (5, 20) else 3
        seconds = np.asarray(row["transfer_and_mmap_inclusive_step_seconds"])
        if (seconds.shape != (measured,) or not np.all(np.isfinite(seconds)) or np.any(seconds <= 0)
                or row["median_step_seconds"] != float(np.median(seconds))
                or row["warmup_updates"] != 2 or row["measured_updates"] != measured
                or row["batch_size"] != 1024 or row["resume_verification_updates"] != 0
                or row["common_cpu_scoring_passed"] is not True):
            raise ValueError("invalid per-count timing evidence")
        expected_rows = 1024 + 4096 + 4 + len(adversarial_labels(k)) + len(showcase_labels(k))
        if row["generated_rows"] != expected_rows or row["compact_evaluation_rows"] != 2048:
            raise ValueError("timing row counts differ from the measured workload")
        config = row["training_configuration"]
        if (config["learning_rate"] != .0001 or config["seed"] != 424245
                or config["updates"] != measured + 2 or config["batch_size"] != 1024 or "epochs" in config):
            raise ValueError("timing training configuration mismatch")
        if k in (5, 20):
            if (row["numerical"]["interior_gradcheck"] is not True
                    or row["numerical"]["finite_boundary_and_repeated_band_gradients"] is not True
                    or row["cuda_reload_exact"] is not True
                    or row["cpu_reload_prediction_tolerance"] != {"rtol": 1e-10, "atol": 1e-10}):
                raise ValueError("missing endpoint numerical checks")
        directory = root / f"m{k}-timing-records"
        for name in ("manifest.json", "scores.json", "predictions.npy", "per_band_errors.npy", "sample_targets.npy"):
            required.add(f"m{k}-timing-records/{name}")
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest["complete"] is not True or manifest["completed_rows"] != 2048
                or manifest["identity"]["checkpoint_sha256"] != row["checkpoint_sha256"]
                or manifest["failures"]):
            raise ValueError("incomplete small timing records")
        predictions = np.load(directory / "predictions.npy", mmap_mode="r")
        scores = np.load(directory / "per_band_errors.npy", mmap_mode="r")
        targets = np.load(directory / "sample_targets.npy")
        if (predictions.shape != (2048, k + 1) or scores.shape != (2048, 3)
                or targets.shape != (8, 500, 3) or not np.all(np.isfinite(predictions))
                or not np.all(np.isfinite(scores)) or not np.all(np.isfinite(targets))):
            raise ValueError("invalid small timing arrays")
        indices = np.linspace(0, 2047, 8, dtype=int)
        masses, springs = physical_arrays(np.asarray(predictions[indices]), k)
        independent = np.stack([triatomic_frequencies(m, s, GRID) for m, s in zip(masses, springs)])
        np.testing.assert_allclose(band_errors(targets, independent), scores[indices], rtol=1e-10, atol=1e-10)
        checked += len(indices)
    if not required.issubset(report["files"]):
        raise ValueError("timing manifest omits required evidence checksums")
    io = report["disk_read"]
    if (io["requested_rows"] != report["timing_policy"]["io_target_rows"] or io["rows"] > io["requested_rows"]
            or io["reduced_for_budget"] != (io["rows"] != io["requested_rows"])
            or io["cache"]["may_be_page_cached"] is not True
            or report["timing_policy"]["recovery_saves"] != "measured every count, no interpolation"):
        raise ValueError("inconsistent I/O or recovery measurement scope")
    projection = timing_projection(report, hash_rate, hourly_price)
    if report["optimizer_steps_in_full_matrix"] != projection["workload"]["optimizer_updates"]:
        raise ValueError("full-study update count mismatch")
    return {"passed": True, "report_sha256": sha256_file(root / "report.json"),
            "auditor_sha256": sha256_file(Path(__file__)), "checked_reference_scores_on_cpu": checked,
            "raw_timings_and_resource_selection_verified": True, "artifact_checksums_verified": True,
            "observed_checksum_bytes_per_second": hash_rate, "estimate": projection,
            "verification_scope": "small evidence: sampled independent reference scores, not checkpoint prediction reexecution or bulk data audit",
            "production_start": "fresh initialization and fresh production data after owner approval; never resume this test"}


def verify_timing(root, hourly_price=None):
    """Check raw evidence and independently derive a full-run estimate on CPU."""
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    if report["schema"] not in ("corrected-timing-check-v1", "corrected-timing-check-v2", "corrected-timing-check-v4") or report["passed"] is not True:
        raise ValueError("a completed disposable timing check is required")
    new = report["schema"] == "corrected-timing-check-v4"
    expected_counts = {str(k) for k in range(5, 21)} if new else {"5", "20"}
    if report["controls"]["batch_size"] != 1024 or set(report["counts"]) != expected_counts:
        raise ValueError("timing must measure the fixed batch size and both endpoints")
    if report["schema"] in ("corrected-timing-check-v2", "corrected-timing-check-v4"):
        expected_probe = {"learning_rate": .0001, "seed": 424245, "updates": 7}
        expected_probe.update({"production_rate": True} if new else {"candidate_only": True})
        if report["probe_training"] != expected_probe:
            raise ValueError("timing must identify the approved single-rate candidate")
        if (report["checkpoint_schedule"] != "initial-periodic-terminal-v1"
                or report["numerical_timing_passed"] is not True or report["learning_health_passed"] is not True):
            raise ValueError("new timing requires explicit schedule and learning qualification")
        for row in report["counts"].values():
            check_learning_evidence(row, validation_required=not new)
    checked_bytes = 0
    tick = time.perf_counter()
    for name, digest in report["files"].items():
        path = (root / name).resolve()
        if new and Path(name).parts[0] == "bulk":
            raise ValueError("exported timing evidence must exclude bulk data")
        if not path.is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError("timing artifact checksum mismatch")
        checked_bytes += path.stat().st_size
    hash_rate = checked_bytes / (time.perf_counter() - tick)
    for name, digest in report["provenance"]["source_sha256"].items():
        if sha256_file(root / "sources" / name) != digest:
            raise ValueError("timing source snapshot mismatch")
    tuning = report["automatic_resources"]["training_threads"]
    for candidate in tuning["measurements"]:
        if len(candidate["seconds"]) != 2 or candidate["median_seconds"] != float(np.median(candidate["seconds"])):
            raise ValueError("thread calibration arithmetic mismatch")
    winner = min(tuning["measurements"], key=lambda row: row["median_seconds"])
    if winner["threads"] != tuning["selected_threads"] or report["controls"]["torch_threads"] != winner["threads"]:
        raise ValueError("timing did not use the fastest measured thread setting")
    torch.set_num_threads(report["controls"]["torch_threads"])
    if new:
        return verify_timing_small(root, report, hash_rate, hourly_price)
    checked = 0
    for k in (5, 20):
        row = report["counts"][str(k)]
        seconds = np.asarray(row["transfer_and_mmap_inclusive_step_seconds"])
        if (seconds.shape != (5,) or not np.all(np.isfinite(seconds)) or np.any(seconds <= 0)
                or row["median_step_seconds"] != float(np.median(seconds))
                or row["warmup_updates"] != 2 or row["measured_updates"] != 5
                or row["batch_size"] != 1024
                or row["numerical"]["interior_gradcheck"] is not True
                or row["numerical"]["finite_boundary_and_repeated_band_gradients"] is not True
                or row["resume_verification_updates"] != 0 or row["cuda_reload_exact"] is not True
                or row["common_cpu_scoring_passed"] is not True):
            raise ValueError("invalid raw timing or correctness evidence")
        model, payload = load_model(root / f"m{k}.pt")
        if report["schema"] == "corrected-timing-check-v2":
            config = payload["training_configuration"]
            if (config["seed"] != 424245 or config["updates"] != 7 or "epochs" in config
                    or config["batch_size"] != 1024 or config["learning_rate"] != .0001):
                raise ValueError("timing checkpoint provenance differs from actual updates")
        artifact = LabeledArtifact(root / f"m{k}-data")
        if (len(artifact.arrays["train"][0]) != 1024
                or any(len(artifact.arrays[p][0]) != 2048 for p in ("validation_dense", "validation_sparse"))
                or row["generated_rows"] != sum(len(pair[0]) for pair in artifact.arrays.values())):
            raise ValueError("timing artifact row counts differ from the measured workload")
        records = root / f"m{k}-timing-records"
        predictions = np.load(records / "predictions.npy", mmap_mode="r")
        scores = np.load(records / "per_band_errors.npy", mmap_mode="r")
        indices = np.linspace(0, len(predictions) - 1, min(8, len(predictions)), dtype=int)
        target = np.array(artifact.arrays["validation_dense"][1][indices], copy=True)
        actual = predict(model, target, len(indices))
        np.testing.assert_allclose(actual, predictions[indices], rtol=1e-10, atol=1e-10)
        masses, springs = physical_arrays(np.asarray(predictions[indices]), k)
        independent = np.stack([triatomic_frequencies(m, s, artifact.grid) for m, s in zip(masses, springs)])
        np.testing.assert_allclose(band_errors(target, independent), scores[indices], rtol=1e-10, atol=1e-10)
        checked += len(indices)
        del model, artifact
    projection = timing_projection(report, hash_rate, hourly_price)
    if report["optimizer_steps_in_full_matrix"] != projection["workload"]["optimizer_updates"]:
        raise ValueError("full-study update count mismatch")
    return {"passed": True, "report_sha256": sha256_file(root / "report.json"),
            "auditor_sha256": sha256_file(Path(__file__)), "checked_predictions_on_cpu": checked,
            "raw_timings_and_resource_selection_verified": True, "artifact_checksums_verified": True,
            "observed_checksum_bytes_per_second": hash_rate, "estimate": projection,
            "production_start": "fresh initialization and fresh production data after owner approval; never resume this test"}


def verify_production(root, controls):
    """Independent-process checkpoint/target/compact-record audit before publication."""
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text())
    progress = json.loads((root / "progress.json").read_text())
    if progress["status"] != "completed_pending_independent_audit":
        raise ValueError("production matrix has not completed")
    expected = production_protocol(controls)
    for key, value in expected.items():
        if protocol[key] != value:
            raise ValueError(f"production protocol differs from the fixed experiment: {key}")
    for name, digest in protocol["provenance"]["source_sha256"].items():
        if sha256_file(root / "sources" / name) != digest:
            raise ValueError(f"production source snapshot mismatch: {name}")
    executions = validate_execution_history(root, protocol, progress["attempts"])
    if not executions:
        raise ValueError("production is missing qualified execution history")

    def check_data_executions(artifact):
        for attempt in artifact.manifest["attempts"]:
            if attempt["execution_id"] not in executions:
                raise ValueError("unqualified dataset generation execution")
        for split in artifact.manifest["splits"].values():
            for chunk in split["chunks"]:
                if chunk["execution_id"] not in executions:
                    raise ValueError("unqualified dataset chunk execution")

    def check_qualification(record, *, gradients):
        if (record["passed"] is not True or record["population"] != "train"
                or record["tolerance"] != {"rtol": 1e-10, "atol": 1e-10}
                or record["finite_gradients_checked"] is not gradients
                or any(not np.isfinite(record[key]) or record[key] < 0 for key in
                       ("maximum_cpu_prediction_difference", "maximum_device_prediction_difference",
                        "maximum_common_score_difference", "elapsed_seconds"))):
            raise ValueError("invalid checkpoint transfer qualification")

    shared = LabeledArtifact(root / "shared-m5-data")
    check_data_executions(shared)
    if (shared.manifest["identity"]["configuration"] != protocol["shared"]
            or shared.manifest["identity"]["provenance"] != protocol["provenance"]):
        raise ValueError("shared target configuration mismatch")
    checked, summaries, checkpoints = 0, {}, []
    for fit in protocol["fits"]:
        name = fit["name"]
        artifact = LabeledArtifact(root / f"{name}-data")
        check_data_executions(artifact)
        if (artifact.manifest["identity"]["configuration"] != fit["data"]
                or artifact.manifest["identity"]["provenance"] != protocol["provenance"]):
            raise ValueError("fit data configuration mismatch")
        checkpoint = root / name / "best.pt"
        digest = sha256_file(checkpoint)
        if digest != progress["fits"][name]["checkpoint_sha256"]:
            raise ValueError("production checkpoint checksum mismatch")
        model, payload = load_model(checkpoint, device=controls["device"])
        if (payload["dataset_manifest_sha256"] != sha256_file(artifact.root / "manifest.json")
                or payload["training_configuration"] != fit["training"]
                or payload["provenance"] != protocol["provenance"]):
            raise ValueError("checkpoint data/configuration/source mismatch")
        training = json.loads((root / name / "training.json").read_text())
        training_attempts = json.loads((root / name / "attempts.json").read_text())
        for attempt in training_attempts:
            if attempt["execution_id"] not in executions or attempt["execution"] != executions[attempt["execution_id"]]["controls"]:
                raise ValueError("unqualified training execution")
            if attempt["resume"]:
                check_qualification(attempt["checkpoint_qualification"], gradients=True)
        steps = training["steps_by_execution"]
        expected_steps = fit["training"]["epochs"] * ((len(artifact.arrays["train"][0]) + fit["training"]["batch_size"] - 1) // fit["training"]["batch_size"])
        if (any(identity not in executions or type(count) is not int or count < 1 for identity, count in steps.items())
                or sum(steps.values()) != expected_steps):
            raise ValueError("training execution step accounting mismatch")
        if any(row["execution_id"] not in executions for row in training["history"]):
            raise ValueError("unqualified validation execution")
        if [row["epoch"] for row in training["history"]] != list(range(fit["training"]["epochs"] + 1)):
            raise ValueError("incomplete or duplicated training epochs")
        best = min(training["history"], key=lambda row: row["validation_primary"])
        if (payload["epoch"] != best["epoch"] or payload["validation_primary"] != best["validation_primary"]
                or payload["execution_id"] != best["execution_id"]
                or training["checkpoint_sha256"] != digest):
            raise ValueError("checkpoint selection was not the recorded validation minimum")
        dense, _ = evaluate_population(model, artifact, "validation_dense", controls["batch_size"], retain_curves=False)
        sparse, _ = evaluate_population(model, artifact, "validation_sparse", controls["batch_size"], retain_curves=False)
        if dense["invalid_prediction_count"] or sparse["invalid_prediction_count"]:
            raise ValueError("invalid saved-model validation prediction")
        score = (dense["primary"] * dense["count"] + sparse["primary"] * sparse["count"]) / (dense["count"] + sparse["count"])
        np.testing.assert_allclose(score, payload["validation_primary"], rtol=1e-10, atol=1e-10)
        evaluations = [(population, artifact, population) for population in ("test_dense", "test_sparse", "adversarial", "showcase")]
        if name.startswith("study-"):
            evaluations.extend((f"shared_m5_{population}", shared, population) for population in ("test_dense", "test_sparse"))
        for label, targets, population in evaluations:
            directory = root / name / label
            manifest = json.loads((directory / "manifest.json").read_text())
            identity = manifest["identity"]
            count = len(targets.arrays[population][0])
            if (not manifest["complete"] or manifest["completed_rows"] != count or manifest["failures"]
                    or identity["target_manifest_sha256"] != sha256_file(targets.root / "manifest.json")
                    or identity["checkpoint_sha256"] != digest or identity["population"] != population
                    or identity["model_interactions"] != model.interactions):
                raise ValueError("compact evaluation identity/completion mismatch")
            arrays = {kind: np.load(directory / f"{kind}.npy", mmap_mode="r", allow_pickle=False)
                      for kind in ("predictions", "per_band_errors")}
            if (arrays["predictions"].shape != (count, model.interactions + 1)
                    or arrays["per_band_errors"].shape != (count, 3)
                    or any(array.dtype != np.float64 for array in arrays.values())):
                raise ValueError("compact evaluation schema mismatch")
            end = 0
            for chunk in manifest["chunks"]:
                if chunk["execution_id"] not in executions:
                    raise ValueError("unqualified evaluation execution")
                execution = executions[chunk["execution_id"]]
                if chunk["inference_device"] != execution["controls"]["device"]:
                    raise ValueError("evaluation device differs from its qualification")
                check_qualification(execution["checkpoint_qualifications"][name], gradients=False)
                if chunk["start"] != end or not end < chunk["stop"] <= count:
                    raise ValueError("compact evaluation chunk order mismatch")
                for kind, array in arrays.items():
                    if array_hash(array[end:chunk["stop"]]) != chunk["sha256"][kind]:
                        raise ValueError("compact evaluation checksum mismatch")
                end = chunk["stop"]
            if end != count:
                raise ValueError("compact evaluation is truncated")
            for rows, bands, _ in targets.batches(population, controls["batch_size"]):
                predicted = predict(model, bands, controls["batch_size"])
                np.testing.assert_allclose(predicted, arrays["predictions"][rows], rtol=1e-10, atol=1e-10)
                _, errors, failed = evaluate_designs(bands, arrays["predictions"][rows], model.interactions)
                if failed:
                    raise ValueError("invalid final design in production audit")
                np.testing.assert_allclose(errors, arrays["per_band_errors"][rows], rtol=1e-12, atol=1e-12)
                checked += len(rows)
            summary = json.loads((directory / "scores.json").read_text())
            for metric, value in summarize_errors(arrays["per_band_errors"], []).items():
                if metric in ("primary", "mean_per_band", "median_per_sample", "p95_per_sample", "maximum_per_sample"):
                    np.testing.assert_allclose(summary[metric], value, rtol=1e-12, atol=1e-12)
                elif summary[metric] != value:
                    raise ValueError(f"production summary mismatch: {name}/{label}/{metric}")
            if targets.interactions == model.interactions:
                mae = np.abs(arrays["predictions"] - targets.arrays[population][0]).mean(axis=0)
                np.testing.assert_allclose(mae, summary["parameter_mae_by_label"], rtol=1e-12, atol=1e-12)
            if summary != progress["evaluations"][f"{name}/{label}"]:
                raise ValueError("progress/evaluation summary mismatch")
            summaries[f"{name}/{label}"] = summary
        checkpoints.append({"name": name, "sha256": digest, "best_epoch": best["epoch"]})
        del model, artifact
    return {"protocol_sha256": sha256_file(root / "protocol.json"),
            "auditor_sha256": sha256_file(Path(__file__)), "checkpoints": checkpoints,
            "source_snapshots_verified": True, "dataset_checksums_verified": True,
            "qualified_executions_verified": list(executions),
            "portability_scope": "qualified numerical agreement, not bit-identical cross-hardware training",
            "predictions_and_metrics_checked": checked, "evaluation": summaries,
            "prediction_tolerance": {"rtol": 1e-10, "atol": 1e-10},
            "score_tolerance": {"rtol": 1e-12, "atol": 1e-12},
            "publication_status": "audited numerical records; manuscript replacement still required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--sizing", action="store_true")
    modes.add_argument("--production", action="store_true")
    modes.add_argument("--timing", action="store_true")
    parser.add_argument("--hourly-price", type=float, help="explicit allocation price for timing cost projection")
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("audit report must be new in an existing directory")
    if args.timing:
        write_json(args.report, verify_timing(args.run, args.hourly_price))
        return
    if args.production:
        config = production_settings(ROOT / ".env.local")
        torch.set_num_threads(config["torch_threads"])
        write_json(args.report, verify_production(args.run, config))
        return
    _, config = settings(ROOT / ".env.local")
    torch.set_num_threads(config["torch_threads"])
    if args.sizing:
        report = verify_sizing(args.run, config["batch_size"])
    else:
        report = verify_run(args.run, config["batch_size"])
    write_json(args.report, report)
    print(f"Verified {report['audit']['predictions_and_metrics_checked']} predictions; report: {args.report}")
    if args.sizing:
        print(json.dumps(report["learning_summary"], indent=2))


if __name__ == "__main__":
    main()
