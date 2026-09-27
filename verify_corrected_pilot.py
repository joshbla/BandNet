"""Verify retained pilot files and export its original report plus audit evidence."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from corrected_pilot import ROOT, settings
from triatomic_data import LabeledArtifact, sha256_file, write_json
from triatomic_learning import band_errors, evaluate_designs, load_model, predict, summarize_errors


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--sizing", action="store_true")
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("audit report must be new in an existing directory")
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
