"""Verify retained pilot files and export its original report plus audit evidence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from corrected_pilot import ROOT, settings, production_settings, production_protocol, validate_execution_history
from triatomic_data import (LabeledArtifact, array_hash, sha256_file, write_json,
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


def timing_workload(checkpoint_steps):
    """Independent arithmetic for the agreed fixed experiment, not fitted timing."""
    if type(checkpoint_steps) is not int or checkpoint_steps < 1:
        raise ValueError("positive checkpoint interval required")
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
        work["resumable_checkpoint_writes"] += epochs + 2 + steps // checkpoint_steps
        work["final_records"] += 2 * test + extras + (20000 if index else 0)
    shared_rows = 32 + 4 + 20000 + len(adversarial_labels(5)) + len(showcase_labels(5))
    work["generated_rows"] += shared_rows
    work["generated_payload_bytes"] += shared_rows * 1506 * 8
    return work


def timing_projection(report, hash_bytes_per_second):
    work = timing_workload(report["controls"]["checkpoint_steps"])
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
    return {"workload": work, "stages": stages, "lower_seconds": lower, "upper_seconds": upper,
            "planning_seconds_with_25_percent_margin": upper * 1.25,
            "range_meaning": "M5/M20 measured-rate bracket plus best-selection-count bracket; not a confidence interval or guarantee",
            "limitations": ["unmeasured intermediate interaction counts", "small cached sample versus full dataset",
                            "cached checksum bandwidth may exceed cold-storage bandwidth",
                            "audit inference uses the measured write-inclusive evaluation rate as a proxy",
                            "future provisioning, software installation and final artifact export are additional",
                            "25 percent planning margin is an explicit assumption, not measured uncertainty"]}


def verify_timing(root):
    """Check raw evidence and independently derive a full-run estimate on CPU."""
    root = Path(root)
    report = json.loads((root / "report.json").read_text())
    if report["schema"] != "corrected-timing-check-v1" or report["passed"] is not True:
        raise ValueError("a completed disposable timing check is required")
    if report["controls"]["batch_size"] != 1024 or set(report["counts"]) != {"5", "20"}:
        raise ValueError("timing must measure the fixed batch size and both endpoints")
    checked_bytes = 0
    tick = time.perf_counter()
    for name, digest in report["files"].items():
        path = (root / name).resolve()
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
        model, _ = load_model(root / f"m{k}.pt")
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
    projection = timing_projection(report, hash_rate)
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
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("audit report must be new in an existing directory")
    if args.timing:
        write_json(args.report, verify_timing(args.run))
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
