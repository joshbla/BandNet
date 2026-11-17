"""Rebuild corrected paper figures and traced values from local audited records.

Run from this checkout with the locked numerical environment:
    uv run --frozen --extra training python three_band_paper.py RECORDS_DIRECTORY MANUSCRIPT_DIRECTORY

No training, environment configuration, or network services are needed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from triatomic_genuine_formula import triatomic_frequencies

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("records", type=Path, help="Local audited production records")
parser.add_argument("manuscript", type=Path, help="Manuscript checkout for generated figures")
args = parser.parse_args()
records, manuscript = args.records, args.manuscript
sources = {}

def source(relative):
    path = records / relative
    sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path

def read_json(relative):
    return json.loads(source(relative).read_text())

def load(relative):
    return np.load(source(relative), allow_pickle=False)

def summary(fit, population):
    prefix = f"{fit}/{population}"
    score = read_json(prefix + "/scores.json")
    errors = load(prefix + "/per_band_errors.npy")
    assert errors.shape == (score["count"], 3)
    assert score["invalid_prediction_count"] == 0 and np.isfinite(errors).all()
    samples = errors.mean(axis=1)
    actual = {
        "primary": float(samples.mean()),
        "median_per_sample": float(np.median(samples)),
        "p95_per_sample": float(np.quantile(samples, .95, method="linear")),
        "maximum_per_sample": float(samples.max()),
    }
    for key, value in actual.items():
        np.testing.assert_allclose(value, score[key], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(errors.mean(axis=0), score["mean_per_band"], rtol=1e-12, atol=1e-14)
    return {"source": prefix + "/scores.json", **score}

manifest = read_json("main-m5-data/manifest.json")
training = read_json("main-m5/training.json")
q = load("main-m5-data/q_hat.npy")
targets = load("main-m5-data/showcase.bands.npy")
labels = load("main-m5-data/showcase.labels.npy")
predictions = load("main-m5/showcase/predictions.npy")
stored = load("main-m5/showcase/per_band_errors.npy")
assert labels.shape == predictions.shape == (4, 6)
np.testing.assert_array_equal(labels, [[8,5,6,6,4,8],[1,7,2,8,1,6],[4,4,8,4,9,4],[6,1,3,0,7,9]])
assert manifest["identity"]["label_order"] == ["m2", "m3", "k2", "k3", "k4", "k5"]
assert hashlib.sha256(Path("triatomic_genuine_formula.py").read_bytes()).hexdigest() == manifest["identity"]["provenance"]["source_sha256"]["triatomic_genuine_formula.py"]

def reconstruct(vectors):
    return np.stack([triatomic_frequencies(np.r_[1, x[:2]], np.r_[1, x[2:]], q) for x in vectors])

achieved = reconstruct(predictions)
target_check = reconstruct(labels)
np.testing.assert_allclose(target_check, targets, atol=1e-10, rtol=1e-10)
residual = achieved - targets
rmse = np.sqrt(np.mean(residual**2, axis=1))
l1 = np.mean(np.abs(residual), axis=1)
ranges = np.ptp(targets, axis=1)
rms = np.sqrt(np.mean(targets**2, axis=1))
assert np.all(ranges > 0) and np.all(rms > 0)
normalized = rmse / rms
np.testing.assert_allclose(normalized, stored, rtol=1e-9, atol=1e-11)
cases = []
for i, panel in enumerate("abcd"):
    cases.append({
        "id": "tri-five-" + panel,
        "source": "main-m5/showcase/{predictions,per_band_errors}.npy and main-m5-data/showcase.{bands,labels}.npy",
        "target_masses": np.r_[1, labels[i,:2]].tolist(),
        "target_stiffnesses": np.r_[1, labels[i,2:]].tolist(),
        "achieved_masses": np.r_[1, predictions[i,:2]].tolist(),
        "achieved_stiffnesses": np.r_[1, predictions[i,2:]].tolist(),
        "table_rmse_sum": float(rmse[i].sum()),
        "table_l1_sum": float(l1[i].sum()),
        "table_nrmse_range_sum": float((rmse[i]/ranges[i]).sum()),
        "adopted_primary": float(normalized[i].mean()),
        "per_band_rmse": rmse[i].tolist(),
        "per_band_l1": l1[i].tolist(),
        "per_band_target_range": ranges[i].tolist(),
        "per_band_target_rms": rms[i].tolist(),
        "per_band_adopted_error": normalized[i].tolist(),
    })

populations = {p: summary("main-m5", p) for p in ["test_sparse", "test_dense", "adversarial", "showcase"]}
study = []
for m in range(5, 21):
    fit = f"study-m{m}"
    t = read_json(fit + "/training.json")
    data = read_json(fit + "-data/manifest.json")
    assert data["identity"]["configuration"]["train_count"] == 100000
    assert t["configuration"]["epochs"] == 100 and len(t["history"]) == 101
    study.append({
        "interactions": m,
        "training_source": fit + "/training.json",
        "data_source": fit + "-data/manifest.json",
        "train_count": 100000,
        "epochs": 100,
        "best_epoch": t["best_epoch"],
        "training_seconds_excluding_validation": t["training_seconds_excluding_validation"],
        "populations": {p: summary(fit, p) for p in ["test_dense", "test_sparse", "shared_m5_test_dense", "shared_m5_test_sparse", "adversarial", "showcase"]},
    })

plt.rcParams.update({"font.size": 15, "axes.labelsize": 19, "legend.fontsize": 13})
fig, axes = plt.subplots(2, 2, figsize=(11, 10), layout="constrained")
colors = ["dodgerblue", "limegreen", "red"]
for i, ax in enumerate(axes.flat):
    for band, color in enumerate(colors):
        ax.plot(q, targets[i,:,band], color=color, lw=3.0)
        ax.plot(q, achieved[i,:,band], color="black", ls="--", lw=2.2)
    ax.set(xlim=(0, 1), ylim=(0, 1.08*max(targets[i].max(), achieved[i].max())), xlabel=r"$\hat q$", ylabel=r"$\omega$")
    ax.set_xticks(np.linspace(0,1,6))
    ax.grid(True, color="0.7", lw=.7)
    ax.text(-.15, 1.01, "(" + "abcd"[i] + ")", transform=ax.transAxes, fontsize=25, fontfamily="serif")
    ax.legend(handles=[Line2D([0],[0],color="black",lw=3,label=r"$\omega_i$ Target"),Line2D([0],[0],color="black",lw=2.2,ls="--",label=r"$\omega_i$ Design")],loc="lower right")
fig.savefig(manuscript / "new_figs/3Band Samples.png", dpi=300)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 8), layout="constrained")
counts = [row["interactions"] for row in study]
for population, label, marker in [("shared_m5_test_dense", "Uniform", "o"),("shared_m5_test_sparse", "Stratified", "x")]:
    assert all(row["populations"][population]["count"] == 10000 for row in study)
    ax.plot(counts, [row["populations"][population]["primary"] for row in study], marker=marker, lw=2.3, label=label)
ax.set(xlabel="Model complexity", ylabel="Mean target-RMS-normalized RMSE", title="Model error on shared M5 testing bands", ylim=(0,.20))
ax.set_xticks(counts, [f"M{m}" for m in counts], rotation=45)
ax.grid(True, color="0.7", lw=.7)
ax.legend(loc="upper left")
fig.savefig(manuscript / "media/3Band_Generalization.png", dpi=300)
plt.close(fig)

generation = {name: sum(chunk["generation_write_hash_seconds"] for chunk in split["chunks"]) for name, split in manifest["splits"].items()}
execution_ids = list(training["steps_by_execution"])
assert len(execution_ids) == 1
execution_source = "executions/" + execution_ids[0] + ".json"
execution = read_json(execution_source)
assert "H100" in execution["hardware"]["name"]
values = {
    "protocol": "corrected three-band v4",
    "definitions": {
        "showcase_table": "For each ascending band, RMSE=sqrt(mean_q((prediction-target)^2)), L1=mean_q(abs(prediction-target)), NRMSE=RMSE/(max_q(target)-min_q(target)); sum each quantity over the three bands. Same aggregation as historical one-/two-band showcase rows, using corrected stored q_hat grid and targets.",
        "adopted_metric": "For each sample, mean over three bands of per-band RMSE / sqrt(mean_q(target^2)); population primary is arithmetic mean over samples. Wave-number columns excluded.",
        "quantile": "Median and 95th percentile of per-sample adopted scores; NumPy quantile method=linear, checked against scores.json.",
        "population_labels": {"stratified": "test_sparse", "uniform": "test_dense", "boundary": "adversarial"},
        "ambiguity": "NRMSE alone does not identify range versus RMS normalization or band sum versus mean. The showcase table uses historical range-normalized band sums; population and generalization use adopted RMS-normalized band means. No zero-range fallback: this computation fails if any showcased band has zero range.",
    },
    "grid": {"source": "main-m5-data/q_hat.npy", "count": len(q), "minimum": float(q.min()), "maximum": float(q.max())},
    "showcase": cases,
    "main_populations": populations,
    "study": study,
    "main_resources": {
        "training_source": "main-m5/training.json",
        "data_source": "main-m5-data/manifest.json",
        "hardware_source": execution_source,
        "hardware_for_paper": "H100",
        "train_count": manifest["identity"]["configuration"]["train_count"],
        "epochs": training["configuration"]["epochs"],
        "best_epoch": training["best_epoch"],
        "best_validation_primary": training["best_validation_primary"],
        "training_configuration": training["configuration"],
        "training_seconds_excluding_validation": training["training_seconds_excluding_validation"],
        "training_minutes_excluding_validation": training["training_seconds_excluding_validation"]/60,
        "training_timing_scope": training["timing_scope"],
        "generation_write_hash_seconds_by_split": generation,
        "generation_write_hash_seconds_all_splits": sum(generation.values()),
        "generation_train_minutes": generation["train"]/60,
        "generation_timing_scope": "Sum of manifest chunk generation_write_hash_seconds; includes curve generation, writing and hashing. Training-only and all-split totals are separate. Does not measure total pipeline wall time or all setup/label generation.",
        "peak_memory": None,
    },
    "presentation_choices": {
        "showcase": "2x2 panels, ascending bands colored blue/green/red, solid targets and black dashed designs, independent panel y limits, same style as the one- and two-band showcase figures; q_hat is the normalized wave number.",
        "generalization": "Two population means on shared M5 targets, 10000 examples each; linear zero-based axis; no own-count series or uncertainty band. One fit per interaction count.",
        "population_table": "Rows mean, median, 95th percentile, count and invalid designs; columns stratified, uniform and boundary.",
    },
    "checks": {
        "independent_reference_source_matches_recorded_sha256": True,
        "target_reconstruction_max_absolute_error": float(np.max(np.abs(target_check-targets))),
        "adopted_per_band_max_absolute_error_vs_audited": float(np.max(np.abs(normalized-stored))),
        "all_population_summaries_verified_from_per_band_arrays": True,
        "all_populations_have_zero_invalid_predictions": True,
    },
    "source_sha256": sources,
}
Path("three_band_paper_values.json").write_text(json.dumps(values, indent=2, allow_nan=False)+"\n")
print(json.dumps({"showcase": cases, "main_resources": values["main_resources"], "checks": values["checks"]},indent=2))
