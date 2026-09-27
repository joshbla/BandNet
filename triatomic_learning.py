"""Band-objective learning and common corrected inference/evaluation.

The pilot is explicitly CPU/float64. No historical model, label loss, device
fallback, post-prediction clipping, or failed-design score substitution is used.
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from triatomic_batched import TriatomicBatchSolver
from triatomic_data import GRID, label_order, physical_arrays, sha256_file, write_json


MODEL_SCHEMA = "corrected-triatomic-band-model-v1"
LOSS_NAME = "mean_example_mean_band_target_normalized_squared_frequency_residual"
METRIC_NAME = "mean_example_mean_band_target_rms_normalized_rmse"


def validate_bands(bands):
    if bands.ndim != 3 or bands.shape[1:] != (len(GRID), 3) or len(bands) == 0:
        raise ValueError("bands must have shape (N,500,3)")
    if np.iscomplexobj(bands) or not np.all(np.isfinite(bands)) or np.any(bands < 0):
        raise ValueError("bands must be real, finite and nonnegative")
    if np.any(np.diff(bands, axis=-1) < 0):
        raise ValueError("bands must be ascending at each wave number")
    if np.any(np.mean(bands**2, axis=1) <= 0):
        raise ValueError("each target band must have positive RMS")


def band_errors(target, design):
    """Return per-example/per-band scores; never include q or predicted scales."""
    validate_bands(target)
    validate_bands(design)
    if target.shape != design.shape:
        raise ValueError("target and design band shapes differ")
    return np.sqrt(np.mean((design - target)**2, axis=1) / np.mean(target**2, axis=1))


def band_loss(target, design):
    if target.shape != design.shape or target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError("loss expects matching frequency-only tensors")
    if not torch.isfinite(target).all() or not torch.isfinite(design).all():
        raise ArithmeticError("nonfinite bands in loss")
    denominator = target.square().mean(dim=1)
    if torch.any(denominator <= 0):
        raise ValueError("target band RMS must be positive")
    return ((design - target).square().mean(dim=1) / denominator).mean()


class DifferentiableTriatomic(nn.Module):
    """Float64 Hermitian bond construction with eigenvalue-only gradients.

    Sorted bands are piecewise differentiable. At repeated eigenvalues a unique
    parameter gradient is not asserted. Domain and finite-gradient checks remain
    required; no jitter or altered band ordering is introduced.
    """

    def __init__(self, grid, interactions):
        super().__init__()
        label_order(interactions)
        grid = np.asarray(grid)
        if grid.ndim != 1 or not len(grid) or not np.all(np.isfinite(grid)):
            raise ValueError("finite one-dimensional grid required")
        if np.any((grid < 0.001) | (grid > 1)) or np.any(np.diff(grid) <= 0):
            raise ValueError("differentiable path is limited to increasing q_hat in [0.001,1]")
        self.interactions = interactions
        q = torch.tensor(grid, dtype=torch.float64)
        basis = torch.zeros((interactions, len(grid), 3, 3), dtype=torch.complex128)
        for distance in range(1, interactions + 1):
            for site in range(3):
                shift, neighbor = divmod(site + distance, 3)
                angle = torch.pi * q * shift
                if neighbor == site:
                    basis[distance - 1, :, site, site] += 4 * torch.sin(angle / 2).square()
                else:
                    phase = torch.exp(1j * angle)
                    basis[distance - 1, :, site, site] += 1
                    basis[distance - 1, :, neighbor, neighbor] += 1
                    basis[distance - 1, :, site, neighbor] -= phase
                    basis[distance - 1, :, neighbor, site] -= phase.conj()
        self.register_buffer("basis", basis)

    def forward(self, labels):
        if labels.dtype != torch.float64 or labels.device.type != "cpu":
            raise ValueError("pilot physics requires CPU float64 labels")
        if labels.ndim != 2 or labels.shape[1] != self.interactions + 1 or not len(labels):
            raise ValueError("incorrect differentiable label shape")
        if not torch.isfinite(labels).all():
            raise ArithmeticError("nonfinite differentiable physical labels")
        if torch.any((labels[:, :2] < 0.1) | (labels[:, :2] > 10)):
            raise ValueError("differentiable mass ratios outside [0.1,10]")
        if torch.any((labels[:, 2:] < 0) | (labels[:, 2:] > 10)):
            raise ValueError("differentiable spring ratios outside [0,10]")
        one = torch.ones((len(labels), 1), dtype=labels.dtype, device=labels.device)
        masses = torch.cat((one, labels[:, :2]), dim=1)
        springs = torch.cat((one, labels[:, 2:]), dim=1)
        stiffness = torch.einsum("nk,kqij->nqij", springs.to(torch.complex128), self.basis)
        inverse_mass = torch.rsqrt(masses)
        dynamic = stiffness * inverse_mass[:, None, :, None] * inverse_mass[:, None, None, :]
        scale = dynamic.abs().sum(dim=-1).amax(dim=-1).clamp_min(1)
        tolerance = 64 * torch.finfo(torch.float64).eps * scale
        residual = (dynamic - dynamic.mH).abs().amax(dim=(-2, -1))
        if not torch.isfinite(dynamic).all() or torch.any(residual > tolerance):
            raise ArithmeticError("nonfinite or non-Hermitian differentiable matrix")
        eigenvalues = torch.linalg.eigvalsh(dynamic)
        # The adopted positive-k1 domain and q>=.001 have strictly positive
        # eigenvalues. A zero would make sqrt's derivative singular: fail it.
        if not torch.isfinite(eigenvalues).all() or torch.any(eigenvalues <= 0):
            raise ArithmeticError("nonpositive or nonfinite eigenvalue in differentiable domain")
        return torch.sqrt(eigenvalues)


class BandInverse(nn.Module):
    def __init__(self, interactions, width, input_scale):
        super().__init__()
        outputs = len(label_order(interactions))
        if type(width) is not int or width < 1:
            raise ValueError("positive hidden width required")
        scale = torch.as_tensor(input_scale, dtype=torch.float64)
        if scale.shape != (3,) or not torch.isfinite(scale).all() or torch.any(scale <= 0):
            raise ValueError("three positive training-only input scales required")
        self.interactions, self.width = interactions, width
        self.register_buffer("input_scale", scale.clone())
        self.network = nn.Sequential(nn.Linear(len(GRID) * 3, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh(), nn.Linear(width, outputs)).double()

    def forward(self, bands):
        raw = self.network((bands / self.input_scale).flatten(start_dim=1))
        unit = torch.sigmoid(raw)
        # Bounds are part of the model, not a post-hoc repair. Finite logits
        # usually produce interior springs; exact zero targets can be approximated
        # without a thresholding rule or a claim of exact support recovery.
        return torch.cat((0.1 + 9.9 * unit[:, :2], 10 * unit[:, 2:]), dim=1)


def predict(model, target, batch_size):
    validate_bands(target)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive inference batch size required")
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(target), batch_size):
            outputs.append(model(torch.tensor(target[start:start + batch_size], dtype=torch.float64)).numpy())
    return np.concatenate(outputs)


def evaluate_designs(target, predictions, interactions):
    """All rows stay accounted for. Any invalid design makes the primary null."""
    validate_bands(target)
    predictions = np.asarray(predictions)
    if predictions.shape != (len(target), len(label_order(interactions))):
        raise ValueError("prediction count/order does not match targets")
    valid = (np.all(np.isfinite(predictions), axis=1)
             & np.all((predictions[:, :2] >= 0.1) & (predictions[:, :2] <= 10), axis=1)
             & np.all((predictions[:, 2:] >= 0) & (predictions[:, 2:] <= 10), axis=1))
    failures = [{"row": int(i), "reason": "nonfinite or out-of-domain prediction"}
                for i in np.flatnonzero(~valid)]
    curves = np.full(target.shape, np.nan)
    errors = np.full((len(target), 3), np.nan)
    solver = TriatomicBatchSolver(GRID, interactions)
    # This function processes one application batch; generation remains batched.
    if np.any(valid):
        masses, springs = physical_arrays(predictions[valid], interactions)
        try:
            curves[valid] = solver.evaluate(masses, springs).frequencies
            errors[valid] = band_errors(target[valid], curves[valid])
        except ArithmeticError as error:
            # No numerical substitution or valid-only success average. Record
            # the whole attempted batch as failed if its forward solve fails.
            for row in np.flatnonzero(valid):
                failures.append({"row": int(row), "reason": str(error)})
            valid[:] = False
            curves[:] = np.nan
            errors[:] = np.nan
    return curves, errors, failures


def summarize_errors(errors, failures):
    summary = {"metric": METRIC_NAME, "count": len(errors), "invalid_prediction_count": len(failures),
               "primary": None, "mean_per_band": None, "median_per_sample": None,
               "p95_per_sample": None, "maximum_per_sample": None, "failures": failures}
    if not failures:
        if not np.all(np.isfinite(errors)):
            raise ArithmeticError("unaccounted-for nonfinite evaluation score")
        sample_errors = errors.mean(axis=1)
        summary.update(primary=float(sample_errors.mean()), mean_per_band=errors.mean(axis=0).tolist(),
                       median_per_sample=float(np.median(sample_errors)),
                       p95_per_sample=float(np.quantile(sample_errors, 0.95)),
                       maximum_per_sample=float(sample_errors.max()))
    return summary


def evaluate_population(model, artifact, population, batch_size):
    if model.interactions != artifact.interactions:
        raise ValueError("labeled population evaluation requires matching label dimensions; use frequency-only inference for cross-count targets")
    started = time.perf_counter()
    predictions, curves, errors, failures, parameter_errors = [], [], [], [], []
    for rows, target, labels in artifact.batches(population, batch_size):
        predicted = predict(model, target, batch_size)
        design, scores, failed = evaluate_designs(target, predicted, artifact.interactions)
        failures.extend({"row": int(rows[item["row"]]), "reason": item["reason"]} for item in failed)
        predictions.append(predicted)
        curves.append(design)
        errors.append(scores)
        parameter_errors.append(np.abs(predicted - labels))
    errors = np.concatenate(errors)
    summary = summarize_errors(errors, failures)
    summary["parameter_mae_by_label"] = None
    summary["inference_reconstruction_scoring_seconds"] = time.perf_counter() - started
    if not failures:
        summary["parameter_mae_by_label"] = np.concatenate(parameter_errors).mean(axis=0).tolist()
    return summary, {"predictions": np.concatenate(predictions),
                     "reconstructed_bands": np.concatenate(curves), "per_band_errors": errors}


def training_scales(artifact, batch_size):
    sums, count = np.zeros(3), 0
    for _, target, _ in artifact.batches("train", batch_size):
        sums += np.sum(target**2, axis=(0, 1))
        count += target.shape[0] * target.shape[1]
    return np.sqrt(sums / count)


def _save_checkpoint(path, payload):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as output:
        torch.save(payload, output)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def load_model(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["schema"] != MODEL_SCHEMA or payload["loss"] != LOSS_NAME or payload["metric"] != METRIC_NAME:
        raise ValueError("unsupported checkpoint contract")
    if payload["dtype"] != "float64" or any(value.dtype != torch.float64 for value in payload["state_dict"].values()):
        raise ValueError("checkpoint precision differs from the checked float64 model")
    if payload["grid"] != GRID.tolist() or payload["label_order"] != label_order(payload["interactions"]):
        raise ValueError("checkpoint grid or labels mismatch")
    model = BandInverse(payload["interactions"], payload["width"], payload["input_scale"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("checkpoint has nonfinite weights")
    if not torch.equal(model.input_scale, torch.tensor(payload["input_scale"], dtype=torch.float64)):
        raise ValueError("checkpoint input scales disagree")
    model.eval()
    return model, payload


def train_model(root, artifact, config, provenance):
    root = Path(root)
    root.mkdir()
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["torch_threads"])
    torch.use_deterministic_algorithms(True)
    batch_size = config["batch_size"]
    model = BandInverse(artifact.interactions, config["width"], training_scales(artifact, batch_size))
    forward = DifferentiableTriatomic(artifact.grid, artifact.interactions)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    history = []
    best = None
    best_epoch = None
    training_seconds = 0.0
    for epoch in range(config["epochs"] + 1):
        loss_sum, count = 0.0, 0
        if epoch:
            model.train()
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], 100, epoch]))
            order = rng.permutation(len(artifact.arrays["train"][0]))
            started = time.perf_counter()
            for _, target, _ in artifact.batches("train", batch_size, permutation=order):
                target_tensor = torch.tensor(target, dtype=torch.float64)
                optimizer.zero_grad(set_to_none=True)
                design = forward(model(target_tensor))
                loss = band_loss(target_tensor, design)
                if not torch.isfinite(loss):
                    raise ArithmeticError("nonfinite training loss")
                loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise ArithmeticError("missing or nonfinite model gradient")
                optimizer.step()
                loss_sum += loss.item() * len(target)
                count += len(target)
            elapsed = time.perf_counter() - started
            training_seconds += elapsed
        else:
            elapsed = 0.0
        dense, _ = evaluate_population(model, artifact, "validation_dense", batch_size)
        sparse, _ = evaluate_population(model, artifact, "validation_sparse", batch_size)
        if dense["invalid_prediction_count"] or sparse["invalid_prediction_count"]:
            write_json(root / "validation_failure.json", {"epoch": epoch, "dense": dense, "sparse": sparse})
            raise ArithmeticError("invalid validation predictions")
        validation = (dense["primary"] * dense["count"] + sparse["primary"] * sparse["count"]) / (dense["count"] + sparse["count"])
        history.append({"epoch": epoch, "training_loss": loss_sum / count if epoch else None,
                        "training_seconds": elapsed, "validation_primary": validation,
                        "validation_dense": dense, "validation_sparse": sparse})
        if best is None or validation < best:
            best, best_epoch = validation, epoch
            _save_checkpoint(root / "best.pt", {
                "schema": MODEL_SCHEMA, "state_dict": model.state_dict(), "interactions": artifact.interactions,
                "width": config["width"], "input_scale": model.input_scale.tolist(),
                "grid": artifact.grid.tolist(), "label_order": label_order(artifact.interactions),
                "loss": LOSS_NAME, "metric": METRIC_NAME, "epoch": epoch,
                "dtype": "float64",
                "validation_primary": validation, "training_configuration": config,
                "dataset_manifest_sha256": sha256_file(artifact.root / "manifest.json"),
                "provenance": provenance,
            })
        write_json(root / "history.json", history)
    restored, payload = load_model(root / "best.pt")
    dense, _ = evaluate_population(restored, artifact, "validation_dense", batch_size)
    sparse, _ = evaluate_population(restored, artifact, "validation_sparse", batch_size)
    reloaded_score = (dense["primary"] * dense["count"] + sparse["primary"] * sparse["count"]) / (dense["count"] + sparse["count"])
    if reloaded_score != best or payload["epoch"] != best_epoch:
        raise AssertionError("checkpoint roundtrip changed validation predictions")
    report = {"configuration": config, "loss": LOSS_NAME, "metric": METRIC_NAME,
              "initial_validation_primary": history[0]["validation_primary"],
              "best_validation_primary": best, "best_epoch": best_epoch,
              "training_seconds_excluding_validation": training_seconds,
              "checkpoint_roundtrip_passed": True, "checkpoint_sha256": sha256_file(root / "best.pt"),
              "history": history}
    write_json(root / "training.json", report)
    return restored, report
