"""Verify retained pilot files and export its original report plus audit evidence."""

import argparse
import json
from pathlib import Path

import numpy as np

from corrected_pilot import ROOT, settings
from triatomic_data import LabeledArtifact, sha256_file, write_json
from triatomic_learning import band_errors, load_model, predict


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("audit report must be new in an existing directory")
    _, config = settings(ROOT / ".env.local")
    report = verify_run(args.run, config["batch_size"])
    write_json(args.report, report)
    print(f"Verified {report['audit']['predictions_and_metrics_checked']} predictions; report: {args.report}")


if __name__ == "__main__":
    main()
