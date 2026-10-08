import importlib.util
import hashlib
import io
import json
import tempfile
import unittest
import copy
from dataclasses import replace
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np

from generation_policy import GenerationPolicy
from generation_resources import detect_resources
from triatomic_batched import TriatomicBatchSolver
from triatomic_data import (GRID, LabeledArtifact, adversarial_labels, generate_artifact,
                            physical_arrays, sample_labels, showcase_labels)
from triatomic_execution import ExecutionPlan, working_bytes
from triatomic_genuine_formula import triatomic_frequencies


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from corrected_pilot import (settings, production_protocol, production_settings, production_run,
                                 production_provenance, check_preflight_evidence, gpu_preflight, configure_machine_resources,
                                 learning_health, learning_observation, check_learning_evidence, tune_training_threads,
                                 measure_preflight_checkpoints, timing_io_rows, measure_training_reads, timing_slice_allocation)
    from corrected_inference import infer_files
    from verify_corrected_pilot import verify_production, timing_workload, timing_projection, verify_timing
    from triatomic_learning import (BandInverse, DifferentiableTriatomic, band_errors, band_loss,
                                    evaluate_designs, evaluate_population, load_model, predict, train_model,
                                    ProductionBandInverse, checked_device, compact_evaluation, train_production,
                                    read_resume_checkpoint, qualify_checkpoint_model, bounded_eigvalsh, cuda_eigen_call_limit,
                                    save_resume_checkpoint, training_step)


def fixture_plan(chunk=7):
    resources = detect_resources()
    native = ("backend-managed; one outer worker" if
              np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"] == "accelerate"
              else "one BLAS thread per worker")
    return ExecutionPlan(chunk, 1, working_bytes(500, 5, chunk, 1),
                         resources.available_memory_bytes, native, GenerationPolicy(0, 0, 1))


def fixture_config():
    return {"interactions": 5, "seed": 317, "train_count": 8,
            "validation_count_per_population": 3, "test_count_per_population": 3,
            "sampling": "half-dense-half-independent-p05-zero-mask-v1",
            "mass_bounds": [.1, 10], "spring_bounds": [0, 10]}


def fixture_artifact(root, **kwargs):
    return generate_artifact(root, fixture_config(), TriatomicBatchSolver(GRID, 5),
                             fixture_plan(), {"test_fixture": True}, **kwargs)


def fixture_preflight(root, source, hardware, controls, policy):
    """Synthetic machine evidence for CPU orchestration tests, never CUDA evidence."""
    root.mkdir(exist_ok=True)
    probe = root / "fixture-probe.txt"
    probe.write_text("CPU orchestration fixture, not a GPU measurement")
    row = {"cuda_reload_exact": True, "adam_reload_next_update_exact": True, "common_cpu_scoring_passed": True,
           "numerical": {"interior_gradcheck": True, "finite_boundary_and_repeated_band_gradients": True},
           "cpu_reload_prediction_tolerance": {"rtol": 1e-10, "atol": 1e-10},
           "batch_size": controls["batch_size"], "warmup_updates": 2,
           "measured_updates": 5, "median_step_seconds": 1}
    row.update(fixture_learning_evidence())
    report = {"schema": "corrected-gpu-preflight-v4", "passed": True, "provenance": source,
              "probe_training": {"learning_rate": .0001, "seed": 424245, "updates": 7, "production_rate": True},
               "numerical_timing_passed": True, "learning_health_passed": True,
              "hardware": hardware, "controls": controls, "generation_policy": policy.__dict__,
              "counts": {"5": row, "20": row},
              "files": {probe.name: hashlib.sha256(probe.read_bytes()).hexdigest()}}
    (root / "report.json").write_text(json.dumps(report))


def fixture_learning_evidence(*, validation_required=True):
    initial = {"finite": True, "training_loss": 1., "validation_loss": 1.,
               "sigmoid_derivative_nonzero_by_output": [2, 2], "unique_prediction_rows": 2,
               "predictions": [[1., 2.], [2., 3.]], "useful_current_gradients": True,
               "gradients": [{"present": True, "finite": True, "norm": .2, "nonzero": 2}]}
    final = dict(initial, training_loss=.8, validation_loss=.9)
    return {"learning_observations": {"initial": initial, "final": final},
            "learning_health": learning_health([initial, final], validation_required=validation_required)}


class CorrectedDataTests(unittest.TestCase):
    def test_generation_resume_preserves_prefix_across_execution_plans(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            def interrupt(name, stop):
                raise RuntimeError("generation machine departed")
            with self.assertRaisesRegex(RuntimeError, "generation machine departed"):
                fixture_artifact(root, after_chunk=interrupt, execution_id="machine-a")
            prefix = np.load(root / "train.bands.npy")[:7].copy()
            generate_artifact(root, fixture_config(), TriatomicBatchSolver(GRID, 5),
                              fixture_plan(chunk=3), {"test_fixture": True}, resume=True, execution_id="machine-b")
            artifact = LabeledArtifact(root)
            np.testing.assert_array_equal(prefix, artifact.arrays["train"][1][:7])
            attempts = artifact.manifest["attempts"]
            self.assertEqual([row["execution_id"] for row in attempts], ["machine-a", "machine-b"])
            self.assertEqual(attempts[0]["status"], "interrupted_duration_unknown")
            self.assertEqual({chunk["execution_id"] for chunk in artifact.manifest["splits"]["train"]["chunks"]},
                             {"machine-a", "machine-b"})

    def test_sampling_contract_and_independent_populations(self):
        for k in (5, 20):
            train = sample_labels(128, k, 7, "train")
            np.testing.assert_array_equal(train, sample_labels(128, k, 7, "train"))
            masses, springs = physical_arrays(train, k)
            self.assertTrue(np.all((masses[:, 1:] >= .1) & (masses[:, 1:] < 10)))
            np.testing.assert_array_equal(masses[:, 0], 1)
            np.testing.assert_array_equal(springs[:, 0], 1)
            self.assertTrue(np.all(springs[:64] > 0))
            self.assertGreater(np.count_nonzero(springs[64:, 1:] == 0), 0)
            seen = set(map(tuple, train))
            for name in ("validation_dense", "validation_sparse", "test_dense", "test_sparse"):
                labels = sample_labels(128, k, 7, name)
                self.assertFalse(seen.intersection(map(tuple, labels)))
                seen.update(map(tuple, labels))
            physical_arrays(showcase_labels(k), k)
        with self.assertRaises(ValueError):
            sample_labels(3, 5, 7, "train")

    def test_boundary_masks_and_showcase_identity(self):
        rows = adversarial_labels(5)
        self.assertEqual(rows.shape, (96, 6))
        self.assertEqual(len(set(map(tuple, rows[:16, 2:] > 0))), 16)
        np.testing.assert_array_equal(showcase_labels(5)[3], [6, 1, 3, 0, 7, 9])
        self.assertTrue(np.all(showcase_labels(20)[:, 6:] == 0))

    def test_durable_roundtrip_partial_batch_and_no_dense_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            fixture_artifact(root)
            artifact = LabeledArtifact(root)
            self.assertIsInstance(artifact.arrays["train"][1], np.memmap)
            batches = list(artifact.batches("train", 3))
            self.assertEqual([len(item[0]) for item in batches], [3, 3, 2])
            labels, curves = artifact.arrays["train"]
            masses, springs = physical_arrays(labels[[0, 7]], 5)
            reference = np.stack([triatomic_frequencies(m, k, GRID) for m, k in zip(masses, springs)])
            np.testing.assert_allclose(curves[[0, 7]], reference, rtol=1e-10, atol=1e-10)
            with self.assertRaises(FileExistsError):
                fixture_artifact(root)
            with self.assertRaises(ValueError):
                list(artifact.batches("train", 3, permutation=np.zeros(8, dtype=int)))

    def test_explicit_resume_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            def interrupt(name, stop):
                if name == "train":
                    raise RuntimeError("simulated interruption")
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                fixture_artifact(root, after_chunk=interrupt)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                LabeledArtifact(root)
            before = np.load(root / "train.bands.npy").copy()
            fixture_artifact(root, resume=True)
            artifact = LabeledArtifact(root)
            np.testing.assert_array_equal(artifact.arrays["train"][1][:7], before[:7])
            self.assertEqual(len(artifact.manifest["attempts"]), 2)
            # An interrupted attempt has unknown duration, never a zero substitute.
            self.assertIsNone(artifact.manifest["attempts"][0]["elapsed_seconds"])
            changed = np.load(root / "train.bands.npy", mmap_mode="r+")
            changed[0, 0, 0] += 1
            changed.flush()
            del changed
            with self.assertRaisesRegex(ValueError, "checksum"):
                LabeledArtifact(root)
            with self.assertRaisesRegex(ValueError, "checksum"):
                fixture_artifact(root, resume=True)

    def test_resume_rejects_changed_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            fixture_artifact(root)
            config = fixture_config()
            config["seed"] += 1
            with self.assertRaisesRegex(ValueError, "identical"):
                generate_artifact(root, config, TriatomicBatchSolver(GRID, 5), fixture_plan(),
                                   {"test_fixture": True}, resume=True)

    def test_generation_coalesces_compute_batches_into_resumable_disk_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            config = dict(fixture_config(), train_count=32)
            def interrupt(name, stop):
                if name == "train":
                    self.assertEqual(stop, 14)
                    raise RuntimeError("simulated grouped-write interruption")
            with self.assertRaisesRegex(RuntimeError, "grouped-write"):
                generate_artifact(root, config, TriatomicBatchSolver(GRID, 5), fixture_plan(),
                                  {"test_fixture": True}, checkpoint_rows=10, after_chunk=interrupt)
            generated = generate_artifact(root, config, TriatomicBatchSolver(GRID, 5), fixture_plan(),
                                          {"test_fixture": True}, checkpoint_rows=10, resume=True)
            self.assertEqual([chunk["stop"] for chunk in generated["splits"]["train"]["chunks"]], [14, 28, 32])
            artifact = LabeledArtifact(root)
            expected = TriatomicBatchSolver(GRID, 5).evaluate(*physical_arrays(artifact.arrays["train"][0], 5)).frequencies
            np.testing.assert_array_equal(artifact.arrays["train"][1], expected)


@unittest.skipUnless(HAS_TORCH, "corrected training checks require uv --extra training")
class CorrectedLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_forward_parity_for_retained_interaction_counts(self):
        for k in range(5, 21):
            with self.subTest(k=k):
                labels = sample_labels(4, k, 61, "train")
                masses, springs = physical_arrays(labels, k)
                grid = np.array([.001, .13, .37, .81, 1.0])
                actual = DifferentiableTriatomic(grid, k)(torch.tensor(labels)).detach().numpy()
                reference = np.stack([triatomic_frequencies(m, s, grid) for m, s in zip(masses, springs)])
                scale = np.maximum(1, reference.max(axis=(1, 2)))[:, None, None]
                self.assertLess(float(np.max(np.abs(actual - reference) / scale)), 1e-10)
                self.assertLess(float(np.max(np.abs(actual**2 - reference**2) / scale**2)), 128 * np.finfo(float).eps)

    def test_native_boundary_parity_and_finite_degenerate_gradients(self):
        labels = adversarial_labels(5)
        masses, springs = physical_arrays(labels, 5)
        reference = np.stack([triatomic_frequencies(m, s, GRID) for m, s in zip(masses, springs)])
        tensor = torch.tensor(labels, requires_grad=True)
        actual = DifferentiableTriatomic(GRID, 5)(tensor)
        scale = np.maximum(1, reference.max(axis=(1, 2)))[:, None, None]
        self.assertLess(float(np.max(np.abs(actual.detach().numpy() - reference) / scale)), 1e-10)
        band_loss(torch.tensor(reference * 1.01), actual).backward()
        self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_interior_gradient_matches_finite_differences(self):
        for k in (5, 20):
            probe = sample_labels(1, k, 74, "validation_dense")
            probe[:, 2:] = 1 + .8 * probe[:, 2:]
            self.assertTrue(torch.autograd.gradcheck(
                DifferentiableTriatomic([.03, .21, .69, .97], k),
                (torch.tensor(probe, requires_grad=True),), eps=1e-5, atol=2e-5, rtol=2e-4))

    def test_target_normalization_equal_band_weights_and_loss_distinction(self):
        target = np.tile([1., 10., 100.], (2, 500, 1))
        design = target * [1.1, 1.2, 1.3]
        errors = band_errors(target, design)
        np.testing.assert_allclose(errors, [[.1, .2, .3], [.1, .2, .3]], atol=1e-14)
        self.assertAlmostEqual(errors.mean(), .2)
        self.assertAlmostEqual(band_loss(torch.tensor(target), torch.tensor(design)).item(), (.01 + .04 + .09) / 3)
        with self.assertRaises(ValueError):
            band_errors(np.zeros_like(target), design)

    def test_nonunique_mass_swap_is_not_a_label_penalty(self):
        labels = np.array([[2., 7., 3., 4., 5., 6.]])
        alternative = labels.copy()
        alternative[:, :2] = alternative[:, :2][:, ::-1]
        solver = TriatomicBatchSolver(GRID, 5)
        target = solver.evaluate(*physical_arrays(labels, 5)).frequencies
        design = solver.evaluate(*physical_arrays(alternative, 5)).frequencies
        self.assertLess(float(band_errors(target, design).max()), 1e-10)
        self.assertGreater(float(np.abs(labels - alternative).mean()), 1)

    def test_decoder_admissibility_and_invalid_design_accounting(self):
        model = BandInverse(5, 4, [1, 2, 3])
        labels = predict(model, np.tile([1., 2., 3.], (3, 500, 1)), 2)
        physical_arrays(labels, 5)
        target = TriatomicBatchSolver(GRID, 5).evaluate(*physical_arrays(labels, 5)).frequencies
        labels[0, 0] = 0
        labels[1, 2] = np.nan
        _, errors, failures = evaluate_designs(target, labels, 5)
        self.assertEqual([item["row"] for item in failures], [0, 1])
        self.assertTrue(np.isnan(errors[:2]).all())
        np.testing.assert_allclose(errors[2], 0, atol=1e-14)
        with self.assertRaises(ValueError):
            DifferentiableTriatomic([0, 1], 5)

    def test_small_training_checkpoint_inference_and_batch_weighting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            config = {"epochs": 2, "batch_size": 3, "width": 8, "seed": 317,
                      "torch_threads": 1, "learning_rate": .001}
            model, report = train_model(root / "model", artifact, config, {"test_fixture": True})
            self.assertTrue(report["checkpoint_roundtrip_passed"])
            self.assertLess(report["best_validation_primary"], report["initial_validation_primary"])
            restored, _ = load_model(root / "model" / "best.pt")
            target = artifact.arrays["showcase"][1]
            np.testing.assert_array_equal(predict(model, target, 3), predict(restored, target, 3))
            first, records = evaluate_population(model, artifact, "showcase", 3)
            second, _ = evaluate_population(model, artifact, "showcase", 2)
            self.assertAlmostEqual(first["primary"], records["per_band_errors"].mean(), places=14)
            self.assertAlmostEqual(first["primary"], second["primary"], places=12)
            self.assertEqual(first["invalid_prediction_count"], 0)
            external = infer_files(root / "model" / "best.pt", root / "data" / "showcase.bands.npy",
                                   root / "data" / "q_hat.npy", root / "inference",
                                   batch_size=3, torch_threads=1)
            self.assertEqual(external["primary"], first["primary"])
            np.testing.assert_array_equal(np.load(root / "inference" / "predictions.npy"), records["predictions"])

    def test_local_settings_are_required_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env.local"
            path.write_text("BANDNET_PILOT_INTERACTIONS=5\n")
            with self.assertRaisesRegex(ValueError, "explicit"):
                settings(path)
            example = (Path(__file__).parent / ".env.local.example").read_text()
            path.write_text(example)
            data, train = settings(path)
            self.assertEqual(data["interactions"], 5)
            self.assertGreater(train["epochs"], 0)
            path.write_text(example + "\nBANDNET_PILOT_INTERACTIONS=20\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                settings(path)

    def test_production_matrix_and_fresh_stream_isolation(self):
        controls = {"device": "cuda:0", "batch_size": 1024, "torch_threads": 1, "checkpoint_steps": 500}
        protocol = production_protocol(controls)
        self.assertEqual(len(protocol["fits"]), 17)
        self.assertEqual(protocol["fits"][0]["data"]["train_count"], 2500000)
        self.assertEqual(protocol["fits"][0]["training"]["epochs"], 5)
        self.assertEqual([fit["data"]["interactions"] for fit in protocol["fits"]], [5, *range(5, 21)])
        for fit in protocol["fits"][1:]:
            self.assertEqual((fit["data"]["train_count"], fit["training"]["epochs"]), (100000, 100))
        seen = set()
        for config in [fit["data"] for fit in protocol["fits"] if fit["data"]["interactions"] == 5] + [protocol["shared"]]:
            for population in ("train", "validation_dense", "validation_sparse", "test_dense", "test_sparse"):
                rows = set(map(tuple, sample_labels(32, 5, config["seed"], population)))
                self.assertFalse(seen.intersection(rows))
                seen.update(rows)
        self.assertEqual(protocol["shared"]["test_count_per_population"] * 2, 20000)

    def test_cuda_request_never_silently_runs_on_cpu(self):
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                checked_device("cuda:0")
        with self.assertRaises(ValueError):
            checked_device("auto")

    def test_production_settings_require_cuda_and_baseline_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env.local"
            example = (Path(__file__).parent / ".env.local.example").read_text().replace("BANDNET_PRODUCTION_TORCH_THREADS=8", "BANDNET_PRODUCTION_TORCH_THREADS=1")
            path.write_text(example)
            self.assertEqual(production_settings(path)["batch_size"], 1024)
            path.write_text(example.replace("BANDNET_PRODUCTION_DEVICE=cuda:0", "BANDNET_PRODUCTION_DEVICE=cpu"))
            with self.assertRaisesRegex(ValueError, "cuda"):
                production_settings(path)

    def test_recorded_production_network_shape_without_training_it(self):
        linear = torch.nn.Linear
        with patch("torch.nn.Linear", side_effect=lambda left, right, **kwargs: linear(left, right, device="meta", **kwargs)):
            model = ProductionBandInverse(5, [1, 2, 3])
        layers = [layer for layer in model.network if isinstance(layer, linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in layers],
                         [(1500, 5000), (5000, 2500), (2500, 5000), (5000, 2500), (2500, 5000), (5000, 6)])
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 57550006)

    def test_training_resume_preserves_adam_cursor_and_final_selection(self):
        # Reduce widths only inside this unit fixture. CLI production has no such override.
        with tempfile.TemporaryDirectory() as temporary, patch("triatomic_learning.PRODUCTION_ARCHITECTURE", [8, 4, 8, 4, 8]):
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            config = {"architecture": "original-five-relu-corrected-io-v1",
                      "epochs": 2, "batch_size": 3, "seed": 74, "learning_rate": .001}
            execution = {"device": "cpu", "batch_size": 3, "torch_threads": 1, "checkpoint_steps": 1}
            provenance = {"fixture": True}
            direct, direct_report = train_production(root / "direct", artifact, config, provenance,
                                                     execution=execution, execution_id="direct")
            def interrupt(epoch, row):
                if epoch == 1 and row == 3:
                    raise RuntimeError("simulated training interruption")
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                train_production(root / "resumed", artifact, config, provenance, execution=execution,
                                 execution_id="original", after_checkpoint=interrupt)
            with self.assertRaisesRegex(ValueError, "identical"):
                train_production(root / "resumed", artifact, dict(config, seed=75), provenance, resume=True,
                                 execution=execution, execution_id="replacement")
            resumed, resumed_report = train_production(root / "resumed", artifact, config, provenance, resume=True,
                                                       execution=dict(execution, checkpoint_steps=2), execution_id="replacement")
            for name, value in direct.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(direct_report["best_epoch"], resumed_report["best_epoch"])
            self.assertEqual([row["validation_primary"] for row in direct_report["history"]],
                              [row["validation_primary"] for row in resumed_report["history"]])
            self.assertEqual(resumed_report["steps_by_execution"], {"original": 1, "replacement": 5})
            restored, _ = load_model(root / "resumed" / "best.pt")
            np.testing.assert_array_equal(predict(direct, artifact.arrays["showcase"][1], 3),
                                          predict(restored, artifact.arrays["showcase"][1], 3))
            # Reopening a finished fit neither adds epochs nor touches final tests.
            _, again = train_production(root / "resumed", artifact, config, provenance, resume=True,
                                       execution=execution, execution_id="reopened")
            self.assertEqual(len(again["history"]), 3)

    def test_lean_recovery_replays_unsaved_validation_and_preserves_current_adam(self):
        with tempfile.TemporaryDirectory() as temporary, patch("triatomic_learning.PRODUCTION_ARCHITECTURE", [8, 4, 8, 4, 8]):
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            config = {"architecture": "original-five-relu-corrected-io-v1",
                      "epochs": 3, "batch_size": 3, "seed": 74, "learning_rate": .001}
            execution = {"device": "cpu", "batch_size": 3, "torch_threads": 1, "checkpoint_steps": 4}
            provenance = {"fixture": True}
            validation_calls = 0

            def earlier_best(*args, **kwargs):
                nonlocal validation_calls
                summary, records = evaluate_population(*args, **kwargs)
                # Deliberately select epoch zero while real optimizer updates
                # continue, so best/current confusion cannot pass this fixture.
                summary["primary"] = 1.0 if validation_calls < 2 else 2.0
                validation_calls += 1
                return summary, records

            with patch("triatomic_learning.evaluate_population", side_effect=earlier_best), \
                    patch("triatomic_learning.save_resume_checkpoint", wraps=save_resume_checkpoint) as saves:
                _, direct = train_production(root / "direct", artifact, config, provenance,
                                             execution=execution, execution_id="fixture")
            self.assertEqual(saves.call_count, 4)  # initial, updates 4/8, terminal
            self.assertEqual(direct["best_epoch"], 0)
            terminal = read_resume_checkpoint(root / "direct" / "latest.pt")
            self.assertEqual((terminal["epoch"], terminal["next_row"], terminal["steps"]), (4, 0, 9))
            self.assertTrue(any(not torch.equal(value, terminal["best"]["state_dict"][name])
                                for name, value in terminal["current"].items()))
            self.assertTrue(all(values["step"].item() == 9 for values in terminal["optimizer"]["state"].values()))
            selected, _ = load_model(root / "direct" / "best.pt")
            for name, value in selected.state_dict().items():
                torch.testing.assert_close(value, terminal["best"]["state_dict"][name], rtol=0, atol=0)

            validation_calls = 0
            update_calls = 0

            def interrupt_unsaved(*args, **kwargs):
                nonlocal update_calls
                update_calls += 1
                if update_calls == 8:
                    raise RuntimeError("unsaved interruption")
                return training_step(*args, **kwargs)

            with patch("triatomic_learning.evaluate_population", side_effect=earlier_best), \
                    patch("triatomic_learning.training_step", side_effect=interrupt_unsaved):
                with self.assertRaisesRegex(RuntimeError, "unsaved interruption"):
                    train_production(root / "replay", artifact, config, provenance,
                                     execution=execution, execution_id="fixture")
            saved = read_resume_checkpoint(root / "replay" / "latest.pt")
            self.assertEqual(saved["steps"], 4)
            self.assertEqual([row["epoch"] for row in saved["history"]], [0, 1])
            ahead = json.loads((root / "replay" / "history.json").read_text())
            self.assertEqual([row["epoch"] for row in ahead], [0, 1, 2])

            def verify_history_before_update(*args, **kwargs):
                history = json.loads((root / "replay" / "history.json").read_text())
                self.assertEqual([row["epoch"] for row in history], [0, 1])
                return training_step(*args, **kwargs)

            # Observe the first replayed update before later validations rewrite
            # history. Then the actual training function handles subsequent calls.
            replay_calls = 0

            def replay_update(*args, **kwargs):
                nonlocal replay_calls
                replay_calls += 1
                if replay_calls == 1:
                    return verify_history_before_update(*args, **kwargs)
                return training_step(*args, **kwargs)

            with patch("triatomic_learning.evaluate_population", side_effect=earlier_best), \
                    patch("triatomic_learning.training_step", side_effect=replay_update):
                _, replayed = train_production(root / "replay", artifact, config, provenance,
                                               execution=execution, execution_id="fixture", resume=True)
            self.assertEqual(replay_calls, 5)
            self.assertEqual([row["validation_primary"] for row in replayed["history"]],
                             [row["validation_primary"] for row in direct["history"]])
            recovered = read_resume_checkpoint(root / "replay" / "latest.pt")
            for name, value in terminal["current"].items():
                torch.testing.assert_close(value, recovered["current"][name], rtol=0, atol=0)
            for parameter, values in terminal["optimizer"]["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, recovered["optimizer"]["state"][parameter][name], rtol=0, atol=0)
            self.assertEqual(terminal["optimizer"]["param_groups"], recovered["optimizer"]["param_groups"])
            # Even an ahead/stale reporting file on a completed fit must be
            # reconciled without another update, validation, or recovery save.
            (root / "replay" / "history.json").write_text("[]")
            with patch("triatomic_learning.training_step", side_effect=AssertionError("completed update")), \
                    patch("triatomic_learning.evaluate_population", side_effect=AssertionError("completed validation")), \
                    patch("triatomic_learning.save_resume_checkpoint", side_effect=AssertionError("completed save")):
                _, reopened = train_production(root / "replay", artifact, config, provenance,
                                              execution=execution, execution_id="fixture", resume=True)
            self.assertEqual(reopened["history"], recovered["history"])
            self.assertEqual(json.loads((root / "replay" / "history.json").read_text()), recovered["history"])

    def test_lean_recovery_deadline_forces_validation_cursor_save(self):
        with tempfile.TemporaryDirectory() as temporary, patch("triatomic_learning.PRODUCTION_ARCHITECTURE", [8, 4, 8, 4, 8]):
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            config = {"architecture": "original-five-relu-corrected-io-v1",
                      "epochs": 1, "batch_size": 3, "seed": 74, "learning_rate": .001}
            execution = {"device": "cpu", "batch_size": 3, "torch_threads": 1, "checkpoint_steps": 5000}
            with patch("triatomic_learning.save_resume_checkpoint", wraps=save_resume_checkpoint) as saves:
                with self.assertRaisesRegex(TimeoutError, "durable training cursor"):
                    train_production(root / "fit", artifact, config, {"fixture": True},
                                     execution=execution, execution_id="fixture", deadline=0)
            self.assertEqual(saves.call_count, 2)
            saved = read_resume_checkpoint(root / "fit" / "latest.pt")
            self.assertEqual((saved["epoch"], saved["next_row"], saved["steps"]), (1, 0, 0))
            self.assertEqual(saved["best"]["epoch"], 0)
            with patch("triatomic_learning.save_resume_checkpoint", wraps=save_resume_checkpoint) as saves:
                _, report = train_production(root / "fit", artifact, config, {"fixture": True},
                                             execution=execution, execution_id="fixture", resume=True)
            self.assertEqual(saves.call_count, 1)
            self.assertEqual([row["epoch"] for row in report["history"]], [0, 1])

    def test_atomic_recovery_write_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "latest.pt"
            state = {"current": {"weight": torch.tensor([1.0], dtype=torch.float64)}, "steps": 1}
            save_resume_checkpoint(path, state)
            before = path.read_bytes()
            replacement = {"current": {"weight": torch.tensor([2.0], dtype=torch.float64)}, "steps": 2}
            with patch("triatomic_learning.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    save_resume_checkpoint(path, replacement)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(read_resume_checkpoint(path)["steps"], 1)
            # A subsequent save replaces the leftover partial file normally.
            save_resume_checkpoint(path, replacement)
            self.assertEqual(read_resume_checkpoint(path)["steps"], 2)

    def test_runtime_changes_do_not_change_the_scientific_contract(self):
        controls = {"device": "cuda:0", "batch_size": 1024, "torch_threads": 8,
                    "checkpoint_steps": 500, "max_seconds": 600}
        self.assertEqual(production_protocol(controls), production_protocol(
            dict(controls, torch_threads=2, checkpoint_steps=20, max_seconds=200)))
        first = {"source_sha256": {"uv.lock": "locked"}, "platform": "host-a", "torch": "runtime-a"}
        self.assertEqual(production_provenance(first), production_provenance(
            dict(first, platform="host-b", torch="runtime-b")))
        self.assertNotEqual(production_provenance(first), production_provenance(
            dict(first, source_sha256={"uv.lock": "changed"})))

    def test_checkpoint_transfer_checks_actual_weights_and_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as temporary, patch("triatomic_learning.PRODUCTION_ARCHITECTURE", [8, 4, 8, 4, 8]):
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            config = {"architecture": "original-five-relu-corrected-io-v1", "epochs": 1,
                      "batch_size": 3, "seed": 74, "learning_rate": .001}
            execution = {"device": "cpu", "batch_size": 3, "torch_threads": 1, "checkpoint_steps": 1}
            def interrupt(epoch, row):
                if row:
                    raise RuntimeError("interrupt")
            with self.assertRaisesRegex(RuntimeError, "interrupt"):
                train_production(root / "fit", artifact, config, {}, execution=execution,
                                 execution_id="source", after_checkpoint=interrupt)
            state = read_resume_checkpoint(root / "fit" / "latest.pt")
            model = ProductionBandInverse(5, state["input_scale"])
            model.load_state_dict(state["current"])
            qualified = qualify_checkpoint_model(model, artifact, state["reload_probe"], torch.device("cpu"), gradients=True)
            self.assertTrue(qualified["passed"])
            self.assertTrue(qualified["finite_gradients_checked"])
            with torch.no_grad():
                model.network[-1].bias.add_(1)
            with self.assertRaises(AssertionError):
                qualify_checkpoint_model(model, artifact, state["reload_probe"], torch.device("cpu"), gradients=True)
            model.load_state_dict(state["current"])
            bad_probe = dict(state["reload_probe"], per_band_errors=state["reload_probe"]["per_band_errors"] + .01)
            with self.assertRaises(AssertionError):
                qualify_checkpoint_model(model, artifact, bad_probe, torch.device("cpu"), gradients=False)
            for field in ("next_row", "optimizer", "current"):
                tampered = torch.load(root / "fit" / "latest.pt", weights_only=True)
                if field == "next_row":
                    tampered[field] += 1
                elif field == "optimizer":
                    next(iter(tampered[field]["state"].values()))["exp_avg"].add_(.01)
                else:
                    next(iter(tampered[field].values())).add_(.01)
                torch.save(tampered, root / "tampered.pt")
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "integrity"):
                    read_resume_checkpoint(root / "tampered.pt")
            torch.save({"current": state["current"]}, root / "legacy.pt")
            with self.assertRaisesRegex(ValueError, "portable v2"):
                read_resume_checkpoint(root / "legacy.pt")

    def test_learning_health_recomputes_observations_and_rejects_forged_pass(self):
        row = dict(fixture_learning_evidence(), batch_size=4)
        self.assertTrue(check_learning_evidence(row)["passed"])
        # The observed M8 timing stop: training improved, small validation witness rose.
        noisy = copy.deepcopy(fixture_learning_evidence())["learning_observations"]
        noisy["final"]["validation_loss"] = 1.07
        self.assertFalse(learning_health([noisy["initial"], noisy["final"]])["passed"])
        timing = learning_health([noisy["initial"], noisy["final"]], validation_required=False)
        self.assertTrue(timing["passed"])
        self.assertFalse(timing["progress"]["validation_loss"]["improved"])
        stalled = dict(noisy["final"], training_loss=1.)
        self.assertFalse(learning_health([noisy["initial"], stalled], validation_required=False)["passed"])
        for field in ("sigmoid_derivative_nonzero_by_output", "predictions", "gradients", "validation_loss"):
            broken = copy.deepcopy(row)
            final = broken["learning_observations"]["final"]
            if field == "sigmoid_derivative_nonzero_by_output":
                final[field] = [0, 0]
            elif field == "predictions":
                final[field] = [[1., 1.], [1., 1.]]
            elif field == "gradients":
                final[field][0]["norm"] = 0.
                final[field][0]["nonzero"] = 0
            else:
                final[field] = 1.1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "learning health"):
                check_learning_evidence(broken)  # Stored pass and useful-gradient flags are still true.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_preflight(root, {}, {}, {"batch_size": 1024}, GenerationPolicy(0, 0, 1))
            report = json.loads((root / "report.json").read_text())
            report["counts"]["5"]["learning_observations"]["final"]["validation_loss"] = 2.
            with self.assertRaisesRegex(ValueError, "learning health"):
                check_preflight_evidence(report)
            report.update(schema="corrected-timing-check-v2", checkpoint_schedule="initial-periodic-terminal-v1")
            report["probe_training"] = {"learning_rate": .0001, "seed": 424245,
                                        "updates": 7, "candidate_only": True}
            (root / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "learning health"):
                verify_timing(root)  # Reject before expensive files/checkpoint inspection.
            report["schema"] = "corrected-gpu-preflight-v2"
            check_preflight_evidence(report, historical=True)
            with self.assertRaisesRegex(ValueError, "portable GPU preflight"):
                check_preflight_evidence(report)  # Historical qualification cannot admit a new production run.

    def test_saturated_decoder_blocks_momentum_only_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            target = np.array(artifact.arrays["train"][1], copy=True)
            model = BandInverse(5, 8, [1., 1., 1.])
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, foreach=False)
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            optimizer.step()  # Prime nonzero Adam momentum in this tiny negative-control fixture.
            with torch.no_grad():
                model.network[-1].weight.zero_()
                model.network[-1].bias.copy_(torch.tensor([1000., 1000., -1000., -1000., -1000., -1000.]))
            forward = DifferentiableTriatomic(GRID, 5)
            observed = learning_observation(model, forward, target, target[:2], "cpu", gradients=True)
            self.assertEqual(observed["sigmoid_derivative_nonzero_by_output"], [0] * 6)
            self.assertEqual(observed["unique_prediction_rows"], 1)
            self.assertFalse(learning_health([observed, observed])["passed"])
            self.assertFalse(learning_health([observed, observed], validation_required=False)["passed"])
            before = copy.deepcopy(model.state_dict())
            state = copy.deepcopy(optimizer.state_dict())
            with self.assertRaisesRegex(ArithmeticError, "momentum-only"):
                training_step(model, forward, optimizer, target, "cpu")
            for name, value in before.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            for parameter, values in state["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(optimizer.state_dict()["state"][parameter][name], value, rtol=0, atol=0)

    def test_training_guard_allows_individually_zero_parameter_gradients(self):
        class Partial(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.active = torch.nn.Parameter(torch.tensor(2., dtype=torch.float64))
                self.inactive = torch.nn.Parameter(torch.tensor(3., dtype=torch.float64))
            def forward(self, target):
                return target * (self.active + 0 * self.inactive)
        model = Partial()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        training_step(model, torch.nn.Identity(), optimizer, np.ones((2, 3, 3)), "cpu")
        self.assertLess(model.active.item(), 2.)
        self.assertEqual(model.inactive.grad.item(), 0.)

    def test_thread_tuner_restores_weights_and_pristine_adam_next_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            target = np.array(artifact.arrays["train"][1], copy=True)
            torch.manual_seed(74)
            model = BandInverse(5, 8, np.sqrt(np.mean(target**2, axis=(0, 1))))
            untouched = copy.deepcopy(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, foreach=False)
            reference = torch.optim.Adam(untouched.parameters(), lr=.001, foreach=False)
            forward = DifferentiableTriatomic(GRID, 5)
            tune_training_threads(model, forward, optimizer, target, torch.device("cpu"), 1, 1)
            self.assertEqual(len(optimizer.state), 0)
            for name, value in untouched.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            self.assertEqual(training_step(model, forward, optimizer, target, "cpu"),
                             training_step(untouched, forward, reference, target, "cpu"))
            for name, value in untouched.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            for parameter, values in reference.state_dict()["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(optimizer.state_dict()["state"][parameter][name], value, rtol=0, atol=0)

    def test_checkpoint_timer_scopes_separate_preparation_and_full_recovery(self):
        clock = [0.]
        def elapsed(seconds, result):
            def operation(*args, **kwargs):
                clock[0] += seconds
                return result
            return operation
        model = SimpleNamespace(interactions=5, state_dict=lambda: {"weight": "fixture"})
        optimizer = SimpleNamespace(state_dict=lambda: {"optimizer": "fixture"})
        with patch("corrected_pilot.time.perf_counter", side_effect=lambda: clock[0]), \
                patch("corrected_pilot.production_payload", side_effect=elapsed(13, {"payload": True})), \
                patch("corrected_pilot._save_checkpoint", side_effect=elapsed(5, None)), \
                patch("corrected_pilot.synchronize"), \
                patch("corrected_pilot.checkpoint_probe", side_effect=elapsed(7, {"fresh_probe": True})), \
                patch("corrected_pilot.save_resume_checkpoint", side_effect=elapsed(11, None)) as recovery:
            _, preparation, selected, resumed = measure_preflight_checkpoints(
                Path("unused-fixture"), model, optimizer, None, {}, {}, "cpu")
            self.assertEqual((preparation, selected, resumed), (13., 5., 18.))
            self.assertEqual(recovery.call_args.args[1]["reload_probe"], {"fresh_probe": True})

    def test_preflight_requires_numerics_adam_continuation_and_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "preflight"
            fixture_preflight(root, {"source_sha256": {}}, {"fixture": True}, {"batch_size": 1024}, GenerationPolicy(0, 0, 1))
            original = (root / "report.json").read_text()
            check_preflight_evidence(json.loads(original))
            report = json.loads(original)
            report["probe_training"]["learning_rate"] = .001
            with self.assertRaisesRegex(ValueError, "production rate"):
                check_preflight_evidence(report)
            report["schema"] = "corrected-gpu-preflight-v3"
            check_preflight_evidence(report, historical=True)
            with self.assertRaisesRegex(ValueError, "portable GPU preflight"):
                check_preflight_evidence(report)
            for key in ("cuda_reload_exact", "adam_reload_next_update_exact", "common_cpu_scoring_passed"):
                report = json.loads(original)
                report["counts"]["20"][key] = False
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, "incomplete numerical"):
                    check_preflight_evidence(report)
            report = json.loads(original)
            report["counts"]["5"]["median_step_seconds"] = 0
            with self.assertRaisesRegex(ValueError, "timing"):
                check_preflight_evidence(report)
            report = json.loads(original)
            del report["counts"]["20"]
            with self.assertRaisesRegex(ValueError, "both M5 and M20"):
                check_preflight_evidence(report)

    def test_preflight_executes_serialized_adam_check_in_cpu_fixture(self):
        # Execute the complete preflight control flow with tiny CPU fixtures.
        # CUDA availability/identity/memory queries are mocked, not claimed tested.
        controls = {"device": "cpu", "batch_size": 4, "torch_threads": 1,
                    "checkpoint_steps": 2, "max_seconds": 30}
        policy = GenerationPolicy(0, 0, 1)
        def tune(solver, *args):
            return replace(fixture_plan(), estimated_working_bytes=working_bytes(500, solver.interaction_count, 7, 1)), {"fixture": True}
        def small_data(path, config, *args, **kwargs):
            return generate_artifact(path, dict(config, validation_count_per_population=2), *args, **kwargs)
        observation_calls = 0
        def synthetic_observation(*args, **kwargs):
            # This fixture qualifies control flow, not tiny-network learning accuracy.
            nonlocal observation_calls
            key = "initial" if observation_calls % 2 == 0 else "final"
            observation_calls += 1
            return fixture_learning_evidence()["learning_observations"][key]
        with (tempfile.TemporaryDirectory() as temporary,
              patch("triatomic_learning.PRODUCTION_ARCHITECTURE", [8, 4, 8, 4, 8]),
              patch("corrected_pilot.production_settings", return_value=controls),
              patch("corrected_pilot.load_generation_policy", return_value=policy),
              patch("corrected_pilot.gpu_identity", return_value={"cpu_fixture": True}),
              patch("corrected_pilot.source_identity", return_value={"source_sha256": {}}),
              patch("corrected_pilot.checked_device", return_value=torch.device("cpu")),
              patch("corrected_pilot.configure_machine_resources", return_value={"available_threads": 2}),
              patch.multiple("corrected_pilot", update_local_settings=lambda *args: None,
                             timing_io_rows=lambda path: 8),
              patch("corrected_pilot.tune_execution", side_effect=tune),
              patch("corrected_pilot.generate_artifact", side_effect=small_data),
              patch("corrected_pilot.learning_observation", side_effect=synthetic_observation),
              patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.empty_cache"),
              patch("torch.cuda.max_memory_allocated", return_value=0),
              patch("torch.cuda.max_memory_reserved", return_value=0), redirect_stdout(io.StringIO())):
            root = Path(temporary) / "preflight"
            gpu_preflight(root)
            report = json.loads((root / "report.json").read_text())
            check_preflight_evidence(report)
            self.assertEqual(set(report["counts"]), {"5", "20"})
            for count in (5, 20):
                self.assertTrue(report["counts"][str(count)]["adam_reload_next_update_exact"])
                self.assertEqual(report["counts"][str(count)]["resume_verification_updates"], 2)
                read_resume_checkpoint(root / f"m{count}-resume-probe.pt")
                _, payload = load_model(root / f"m{count}.pt")
                self.assertEqual(payload["training_configuration"]["seed"], 424245)
                self.assertEqual(payload["training_configuration"]["updates"], 7)
                self.assertEqual(payload["training_configuration"]["learning_rate"], .0001)
                self.assertNotIn("epochs", payload["training_configuration"])
            disposable = Path(temporary) / "timing"
            gpu_preflight(disposable, timing_only=True)
            timing = json.loads((disposable / "report.json").read_text())
            self.assertEqual(timing["schema"], "corrected-timing-check-v4")
            self.assertEqual(timing["checkpoint_schedule"], "initial-periodic-terminal-v1")
            self.assertTrue(timing["disposable_timing_test"])
            self.assertEqual(timing["probe_training"], {"learning_rate": .0001, "seed": 424245,
                                                        "updates": 7, "production_rate": True})
            self.assertEqual(set(timing["counts"]), {str(k) for k in range(5, 21)})
            self.assertEqual(timing["disk_read"]["rows"], 8)
            self.assertEqual(timing["disk_read"]["bytes"], 8 * 1500 * 8)
            self.assertTrue(timing["disk_read"]["cache"]["may_be_page_cached"])
            self.assertFalse(any(name.startswith("bulk/") for name in timing["files"]))
            self.assertEqual((disposable / "report.sha256").read_text().strip(),
                             hashlib.sha256((disposable / "report.json").read_bytes()).hexdigest())
            for count in range(6, 20):
                self.assertEqual(timing["counts"][str(count)]["measured_updates"], 3)
                self.assertIsNone(timing["counts"][str(count)]["numerical"])
            for count in (5, 20):
                _, payload = load_model(disposable / "bulk" / f"m{count}.pt")
                self.assertEqual(payload["training_configuration"]["learning_rate"], .0001)
                state = read_resume_checkpoint(disposable / "bulk" / f"m{count}-resume-probe.pt")
                self.assertEqual(state["optimizer"]["param_groups"][0]["lr"], .0001)
            self.assertEqual(production_protocol(controls)["fits"][0]["training"]["learning_rate"], .0001)
            self.assertEqual(production_protocol(controls)["schema"], "corrected-full-three-band-v3")
            self.assertIn(timing["controls"]["torch_threads"], (1, 2))
            self.assertTrue(all(row["resume_verification_updates"] == 0 for row in timing["counts"].values()))
            with self.assertRaisesRegex(ValueError, "fixed batch size"):
                verify_timing(disposable)  # Tiny fixture rates must never price production.
            with patch("corrected_pilot.training_step", side_effect=ArithmeticError("all current gradients are zero")), \
                    patch("corrected_pilot.measure_preflight_checkpoints") as writes:
                failed = Path(temporary) / "zero-gradients"
                with self.assertRaisesRegex(ArithmeticError, "all current gradients"):
                    gpu_preflight(failed, timing_only=True)
                writes.assert_not_called()
                failure = json.loads((failed / "report.json").read_text())
                self.assertEqual(failure["failure"]["stage"], "thread_tuning")
                self.assertEqual(set(failure["counts"]), {"5"})
                self.assertFalse(failure["passed"])
                self.assertIn("final", failure["counts"]["5"]["learning_observations"])
                warmup_failed = Path(temporary) / "zero-warmup-gradients"
                with self.assertRaisesRegex(ArithmeticError, "all current gradients"):
                    gpu_preflight(warmup_failed)
                failure = json.loads((warmup_failed / "report.json").read_text())
                self.assertEqual(failure["failure"]["stage"], "warmup")
                writes.assert_not_called()
            with patch("corrected_pilot.learning_observation", return_value=fixture_learning_evidence()["learning_observations"]["initial"]), \
                    patch("corrected_pilot.measure_preflight_checkpoints") as writes:
                failed = Path(temporary) / "unhealthy"
                with self.assertRaisesRegex(ArithmeticError, "learning health"):
                    gpu_preflight(failed)
                writes.assert_not_called()
                failure = json.loads((failed / "report.json").read_text())
                self.assertEqual(failure["failure"]["stage"], "learning_health")
                self.assertEqual(set(failure["counts"]), {"5"})
                self.assertFalse(failure["counts"]["5"]["learning_health"]["passed"])
            nonfinite = copy.deepcopy(fixture_learning_evidence()["learning_observations"]["final"])
            nonfinite["gradients"][0]["norm"] = float("nan")
            nonfinite["logit_min_by_output"] = [float("-inf"), 0.]
            nonfinite["logit_max_by_output"] = [float("inf"), 1.]
            original_error = ArithmeticError("missing or nonfinite gradient")
            with patch("corrected_pilot.learning_observation", side_effect=[
                    fixture_learning_evidence()["learning_observations"]["initial"], nonfinite]), \
                    patch("corrected_pilot.training_step", side_effect=original_error), \
                    patch("corrected_pilot.measure_preflight_checkpoints") as writes:
                failed = Path(temporary) / "nonfinite-gradient"
                with self.assertRaises(ArithmeticError) as raised:
                    gpu_preflight(failed)
                self.assertIs(raised.exception, original_error)
                writes.assert_not_called()
                def reject_nonstandard_constant(value):
                    self.fail(f"nonstandard JSON numeric constant: {value}")
                failure = json.loads((failed / "report.json").read_text(), parse_constant=reject_nonstandard_constant)
                self.assertEqual(failure["failure"]["stage"], "warmup")
                self.assertEqual(failure["failure"]["error"], str(original_error))
                self.assertEqual(set(failure["counts"]), {"5"})
                self.assertFalse(failure["passed"])
                row = failure["counts"]["5"]
                self.assertFalse(row["learning_health"]["passed"])
                observed = row["learning_observations"]["final"]
                self.assertEqual(observed["gradients"][0]["norm"], {"nonfinite": "nan"})
                self.assertEqual(observed["logit_min_by_output"][0], {"nonfinite": "negative_infinity"})
                self.assertEqual(observed["logit_max_by_output"][0], {"nonfinite": "positive_infinity"})
                with self.assertRaisesRegex(ValueError, "learning health"):
                    check_learning_evidence(row)
            controls["max_seconds"] = 0
            with self.assertRaisesRegex(TimeoutError, "preflight time allowance"):
                gpu_preflight(Path(temporary) / "expired")

    def test_automatic_resources_scale_to_the_detected_machine(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env.local"
            path.write_text((Path(__file__).parent / ".env.local.example").read_text() + "\nUNRELATED=preserve-me\n")
            for cpus, gib in ((1, 4), (8, 16), (64, 128)):
                resources = SimpleNamespace(cpu_budget=cpus, available_memory_bytes=gib * 1024**3,
                                            as_dict=lambda: {"cpu_budget": cpus})
                with patch("corrected_pilot.detect_resources", return_value=resources):
                    selected = configure_machine_resources(path)
                self.assertEqual(selected["available_threads"], cpus - selected["cpu_reserve"])
                self.assertGreaterEqual(selected["available_threads"], 1)
                self.assertLessEqual(selected["available_threads"], cpus)
                self.assertLess(selected["ram_reserve_mib"] * 1024**2, resources.available_memory_bytes)
                self.assertIn("UNRELATED=preserve-me", path.read_text())
                self.assertEqual(path.stat().st_mode & 0o077, 0)

    def test_bounded_eigensolver_preserves_values_gradients_and_tail_rows(self):
        torch.manual_seed(72)
        source = torch.randn(3, 5, 3, 3, dtype=torch.complex128, requires_grad=True)
        matrices = source @ source.mH + torch.diag(torch.tensor([1., 2., 3.], dtype=torch.complex128))
        expected = torch.linalg.eigvalsh(matrices)
        eigvalsh = torch.linalg.eigvalsh
        with patch("torch.linalg.eigvalsh", wraps=eigvalsh) as calls:
            actual = bounded_eigvalsh(matrices, 4)
        self.assertEqual([tuple(call.args[0].shape) for call in calls.call_args_list], [(4, 3, 3)] * 4)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        weights = torch.tensor([.3, .7, 1.1])
        first = torch.autograd.grad((actual.square() * weights).sum(), source, retain_graph=True)[0]
        second = torch.autograd.grad((expected.square() * weights).sum(), source)[0]
        torch.testing.assert_close(first, second, rtol=1e-11, atol=1e-11)
        for gib in (16, 24, 80):
            with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=gib * 1024**3)):
                limit = cuda_eigen_call_limit(torch.device("cuda:0"))
            self.assertLessEqual(limit * 2 * 1024**2, gib * 1024**3 // 8)

    def test_independent_timing_accounting_includes_all_large_stages(self):
        work = timing_workload(500, "initial-periodic-epoch-v1")
        self.assertEqual(work["optimizer_updates"], 169010)
        self.assertEqual(work["generated_rows"], 4601618)
        self.assertEqual(work["validation_passes"], 1622)
        self.assertEqual(work["resumable_checkpoint_writes"], 1967)
        self.assertEqual(work["final_records"], 731850)
        row = {"transfer_and_mmap_inclusive_step_seconds": [1, 2, 3, 4, 5],
               "generation_write_seconds": 10, "generated_rows": 5500,
               "calibration": {"elapsed_seconds": 1}, "full_validation_4096_seconds": 2,
               "resume_checkpoint_write_seconds": .1, "checkpoint_write_seconds": .05,
               "best_payload_prepare_seconds": .01, "compact_evaluation_seconds": 1,
               "compact_evaluation_rows": 2048, "reload_and_scoring_check_seconds": .2}
        historical = {"schema": "corrected-timing-check-v1", "controls": {"checkpoint_steps": 500},
                      "counts": {"5": row, "20": row}}
        result = timing_projection(historical, 1e9)
        self.assertEqual(result["stages"]["training_updates"]["upper_seconds"], 3 * 169010)
        self.assertGreater(result["upper_seconds"], 3 * 169010)
        self.assertEqual(result["planning_seconds_with_25_percent_margin"], result["upper_seconds"] * 1.25)
        lean = timing_workload(5000, "initial-periodic-terminal-v1")
        self.assertEqual(lean["resumable_checkpoint_writes"], 52)
        for name in work:
            if name != "resumable_checkpoint_writes":
                self.assertEqual(lean[name], work[name])
        current = dict(historical, schema="corrected-timing-check-v2",
                       checkpoint_schedule="initial-periodic-terminal-v1", controls={"checkpoint_steps": 5000})
        projection = timing_projection(current, 1e9)
        self.assertEqual(projection["stages"]["periodic_checkpoint_writes"]["upper_seconds"], 5.2)
        self.assertEqual(result["workload"]["resumable_checkpoint_writes"], 1967)
        with self.assertRaisesRegex(ValueError, "historical"):
            timing_projection(dict(historical, checkpoint_schedule="initial-periodic-terminal-v1"), 1e9)
        with self.assertRaisesRegex(ValueError, "lean"):
            timing_projection(dict(current, checkpoint_schedule="initial-periodic-epoch-v1"), 1e9)

    def test_timing_io_setting_and_complete_shuffled_read(self):
        self.assertEqual(timing_slice_allocation(250000, 160, 4000, 1024), (250000, 120))
        self.assertEqual(timing_slice_allocation(250000, 80, 1000, 1024), (40000, 60))
        with self.assertRaises(TimeoutError):
            timing_slice_allocation(250000, 20, 1000, 1024)
        with self.assertRaises(ValueError):
            timing_slice_allocation(250000, 80, 0, 1024)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            setting = root / ".env.local"
            setting.write_text("BANDNET_TIMING_IO_ROWS=250000\n")
            self.assertEqual(timing_io_rows(setting), 250000)
            for text in ("", "BANDNET_TIMING_IO_ROWS=3\n", "BANDNET_TIMING_IO_ROWS=2\nBANDNET_TIMING_IO_ROWS=4\n"):
                setting.write_text(text)
                with self.assertRaises(ValueError):
                    timing_io_rows(setting)
            bands = np.arange(7 * 1500, dtype=np.float64).reshape(7, 500, 3)
            np.save(root / "bands.npy", bands)
            calls = []
            transfer = torch.tensor
            def record_transfer(target, **kwargs):
                calls.append(target.copy())
                return transfer(target, **kwargs)
            with patch("corrected_pilot.torch.tensor", side_effect=record_transfer):
                result = measure_training_reads(root / "bands.npy", 3, torch.device("cpu"), lambda: None)
            self.assertEqual([len(batch) for batch in calls], [3, 3, 1])
            np.testing.assert_array_equal(np.sort(np.concatenate(calls)[:, 0, 0]), bands[:, 0, 0])
            self.assertEqual(result["bytes"], bands.nbytes)
            self.assertEqual(result["rows_per_second"], 7 / result["seconds"])
            self.assertTrue(result["cache"]["may_be_page_cached"])
            def expired():
                raise TimeoutError("expired fixture")
            with self.assertRaises(TimeoutError):
                measure_training_reads(root / "bands.npy", 3, torch.device("cpu"), expired)

    def test_all_count_timing_projection_and_small_export_verification(self):
        row = {"transfer_and_mmap_inclusive_step_seconds": [1., 2., 3., 4., 5.],
               "median_step_seconds": 3., "warmup_updates": 2, "measured_updates": 5, "batch_size": 1024,
               "generation_write_seconds": 10., "generated_rows": 5500,
               "calibration": {"elapsed_seconds": 1.}, "full_validation_4096_seconds": 2.,
               "resume_checkpoint_write_seconds": .1, "checkpoint_write_seconds": .05,
               "best_payload_prepare_seconds": .01, "compact_evaluation_seconds": 1.,
               "compact_evaluation_rows": 2048, "reload_and_scoring_check_seconds": .2,
               "checkpoint_sha256": "fixture-checkpoint-not-exported", "resume_verification_updates": 0,
               "cuda_reload_exact": True, "common_cpu_scoring_passed": True,
               "cpu_reload_prediction_tolerance": {"rtol": 1e-10, "atol": 1e-10},
               "numerical": {"interior_gradcheck": True, "finite_boundary_and_repeated_band_gradients": True}}
        row.update(fixture_learning_evidence(validation_required=False))
        report = {"schema": "corrected-timing-check-v4", "passed": True,
                  "numerical_timing_passed": True, "learning_health_passed": True,
                  "probe_training": {"learning_rate": .0001, "seed": 424245, "updates": 7, "production_rate": True},
                  "checkpoint_schedule": "initial-periodic-terminal-v1",
                  "controls": {"checkpoint_steps": 5000, "batch_size": 1024, "torch_threads": 1},
                  "counts": {}, "provenance": {"source_sha256": {}},
                  "automatic_resources": {"training_threads": {"selected_threads": 1,
                       "measurements": [{"threads": 1, "seconds": [1., 1.], "median_seconds": 1.}]}},
                  "disk_read": {"rows": 250000, "bytes": 250000 * 1500 * 8, "seconds": 10.,
                                "rows_per_second": 25000., "requested_rows": 250000,
                                "reduced_for_budget": False, "cache": {"may_be_page_cached": True}},
                  "timing_policy": {"io_target_rows": 250000, "recovery_saves": "measured every count, no interpolation"},
                  "optimizer_steps_in_full_matrix": 169010}
        for k in range(5, 21):
            measured = 5 if k in (5, 20) else 3
            current = copy.deepcopy(row)
            current.update(measured_updates=measured,
                           generated_rows=1024 + 4096 + 4 + len(adversarial_labels(k)) + len(showcase_labels(k)),
                           transfer_and_mmap_inclusive_step_seconds=[float(k)] * measured,
                           median_step_seconds=float(k),
                           training_configuration={"learning_rate": .0001, "seed": 424245,
                                                   "updates": measured + 2, "batch_size": 1024})
            if k not in (5, 20):
                current["calibration"] = {"reused_plan_from_interactions": 5, "elapsed_seconds": None}
            report["counts"][str(k)] = current
        report["counts"]["20"]["calibration"] = {"elapsed_seconds": 4.}
        projection = timing_projection(report, 1e9, 2.)
        # M5 main/study and the shared targets use M5 tuning; the rest are charged the slower endpoint.
        self.assertEqual(projection["stages"]["generation_calibration"]["lower_seconds"], 3 * 1. + 15 * 4.)
        expected_training = 5 * 12210 + sum(k * 9800 for k in range(5, 21))
        self.assertEqual(projection["stages"]["training_updates"]["lower_seconds"], expected_training)
        self.assertEqual(projection["stages"]["full_dataset_shuffled_reads"]["upper_seconds"],
                         (2500000 * 5 + 16 * 100000 * 100) / 25000)
        self.assertEqual(len(projection["fits"]), 17)
        self.assertEqual(projection["planning_cost_with_25_percent_margin"], projection["upper_seconds"] * 1.25 / 3600 * 2)
        incomplete = copy.deepcopy(report)
        del incomplete["counts"]["12"]
        with self.assertRaisesRegex(ValueError, "without interpolation"):
            timing_projection(incomplete, 1e9)
        incomplete = copy.deepcopy(report)
        del incomplete["disk_read"]
        with self.assertRaises(KeyError):
            timing_projection(incomplete, 1e9)
        with self.assertRaises(ValueError):
            timing_projection(report, 1e9, 0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for k in range(5, 21):
                directory = root / f"m{k}-timing-records"
                directory.mkdir()
                labels = sample_labels(8, k, 100 + k, "train")
                masses, springs = physical_arrays(labels, k)
                targets = np.stack([triatomic_frequencies(m, s, GRID) for m, s in zip(masses, springs)])
                predictions = np.tile(labels[0], (2048, 1))
                indices = np.linspace(0, 2047, 8, dtype=int)
                predictions[indices] = labels
                np.save(directory / "predictions.npy", predictions)
                np.save(directory / "per_band_errors.npy", np.zeros((2048, 3)))
                np.save(directory / "sample_targets.npy", targets)
                (directory / "manifest.json").write_text(json.dumps({"complete": True, "completed_rows": 2048,
                      "failures": [], "identity": {"checkpoint_sha256": row["checkpoint_sha256"]}}))
                (directory / "scores.json").write_text("{}")
            report["files"] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in root.rglob("*") if path.is_file()}
            (root / "report.json").write_text(json.dumps(report))
            (root / "report.sha256").write_text(hashlib.sha256((root / "report.json").read_bytes()).hexdigest())
            verified = verify_timing(root, 2.)
            self.assertTrue(verified["passed"])
            self.assertEqual(verified["checked_reference_scores_on_cpu"], 128)
            (root / "m6-timing-records" / "sample_targets.npy").write_bytes(b"partial copy")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_timing(root)

    def test_compact_cross_count_resume_checksums_and_common_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_artifact(root / "data")
            artifact = LabeledArtifact(root / "data")
            model = BandInverse(20, 4, [1, 2, 3])
            def interrupt(stop):
                raise RuntimeError("simulated record interruption")
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                compact_evaluation(root / "records", model, artifact, "showcase", 3, "fixture-checkpoint", after_chunk=interrupt)
            summary = compact_evaluation(root / "records", model, artifact, "showcase", 3, "fixture-checkpoint", resume=True)
            original_summary = (root / "records" / "scores.json").read_bytes()
            with patch("triatomic_learning.predict", side_effect=AssertionError("completed predictions must not be repeated")):
                reopened = compact_evaluation(root / "records", model, artifact, "showcase", 3,
                                              "fixture-checkpoint", resume=True, execution_id="new-machine")
            self.assertEqual(reopened, summary)
            self.assertEqual((root / "records" / "scores.json").read_bytes(), original_summary)
            predictions = np.load(root / "records" / "predictions.npy")
            np.testing.assert_array_equal(predictions, predict(model, artifact.arrays["showcase"][1], 3))
            _, scores, failures = evaluate_designs(artifact.arrays["showcase"][1], predictions, 20)
            self.assertFalse(failures)
            self.assertAlmostEqual(summary["primary"], scores.mean())
            self.assertFalse((root / "records" / "reconstructed_bands.npy").exists())
            with self.assertRaisesRegex(ValueError, "identity"):
                compact_evaluation(root / "records", model, artifact, "showcase", 3, "different", resume=True)
            corrupted = np.load(root / "records" / "predictions.npy", mmap_mode="r+")
            corrupted[0, 0] += 1
            corrupted.flush()
            del corrupted
            with self.assertRaisesRegex(ValueError, "checksum"):
                compact_evaluation(root / "records", model, artifact, "showcase", 3, "fixture-checkpoint", resume=True)

    def test_matrix_runner_finishes_training_before_final_scoring_and_resumes(self):
        # Three synthetic machines, real tiny training/Adam/data, no CUDA claim.
        controls = {"device": "cpu", "batch_size": 3, "torch_threads": 1,
                    "checkpoint_steps": 2, "max_seconds": 60}
        protocol = production_protocol(controls)
        protocol["fits"] = [protocol["fits"][0], protocol["fits"][-1]]
        for fit in protocol["fits"]:
            fit["data"].update(train_count=32, validation_count_per_population=3, test_count_per_population=3)
            fit["training"]["epochs"] = 1
        protocol["shared"].update(train_count=32, validation_count_per_population=3, test_count_per_population=3)
        policy = GenerationPolicy(0, 0, 1)
        provenance = {"source_sha256": {}, "platform": "fixture-host-a", "torch": "fixture-runtime-a"}
        hardware = {"name": "fixture-gpu-a", "uuid": "fixture-uuid-a"}
        def tune(solver, *args):
            plan = fixture_plan()
            return replace(plan, estimated_working_bytes=working_bytes(500, solver.interaction_count, 7, 1)), {"fixture": True}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            precheck = root / "precheck"
            fixture_preflight(precheck, provenance, hardware, controls, policy)
            output = root / "production"
            scored = []
            interrupted = {"training": False, "evaluation": False}
            def train(*args, **kwargs):
                def checkpoint(epoch, row):
                    if not interrupted["training"] and epoch == 1 and row == 6:
                        interrupted["training"] = True
                        raise RuntimeError("training machine departed")
                return train_production(*args, **kwargs, after_checkpoint=checkpoint)
            def evaluate(*args, **kwargs):
                progress = json.loads((output / "progress.json").read_text())
                self.assertEqual(len(progress["fits"]), 2)
                scored.append(args[3])
                budget_check = kwargs["after_chunk"]
                def checkpoint(stop):
                    budget_check(stop)
                    if not interrupted["evaluation"]:
                        interrupted["evaluation"] = True
                        raise RuntimeError("evaluation machine departed")
                kwargs["after_chunk"] = checkpoint
                return compact_evaluation(*args, **kwargs)
            with (patch("triatomic_learning.PRODUCTION_ARCHITECTURE", [8, 4, 8, 4, 8]),
                  patch("corrected_pilot.production_settings", return_value=controls),
                  patch("corrected_pilot.production_protocol", return_value=protocol),
                  patch("corrected_pilot.load_generation_policy", return_value=policy) as policy_reader,
                  patch("corrected_pilot.checked_device", return_value=torch.device("cpu")),
                  patch("triatomic_learning.checked_device", return_value=torch.device("cpu")),
                  patch("corrected_pilot.source_identity", return_value=provenance),
                  patch("corrected_pilot.gpu_identity", return_value=hardware),
                  patch("corrected_pilot.preflight", return_value={"fixture": True}),
                  patch("corrected_pilot.tune_execution", side_effect=tune),
                  patch("corrected_pilot.shutil.disk_usage", return_value=SimpleNamespace(free=200 * 10**9)),
                  patch("torch.cuda.empty_cache"),
                  patch("corrected_pilot.train_production", side_effect=train),
                  patch("corrected_pilot.compact_evaluation", side_effect=evaluate)):
                with self.assertRaisesRegex(RuntimeError, "training machine departed"):
                    production_run(output, precheck)
                self.assertFalse(scored)
                frozen_protocol = (output / "protocol.json").read_bytes()
                # A different machine cannot borrow the first machine's preflight.
                hardware.update(name="fixture-gpu-b", uuid="fixture-uuid-b")
                provenance.update(platform="fixture-host-b", torch="fixture-runtime-b")
                controls.update(torch_threads=2, checkpoint_steps=3)
                policy = GenerationPolicy(0, 0, 2)
                policy_reader.return_value = policy
                with self.assertRaisesRegex(ValueError, "exact source/configuration/GPU"):
                    production_run(output, precheck, resume=True)
                new_precheck = root / "precheck-b"
                fixture_preflight(new_precheck, provenance, hardware, controls, policy)
                # Relocating the entire artifact tree must not change its identity.
                moved = root / "moved-production"
                output.rename(moved)
                output = moved
                with self.assertRaisesRegex(RuntimeError, "evaluation machine departed"):
                    production_run(output, new_precheck, resume=True)
                self.assertEqual((output / "protocol.json").read_bytes(), frozen_protocol)
                hardware.update(name="fixture-gpu-c", uuid="fixture-uuid-c")
                provenance.update(platform="fixture-host-c", torch="fixture-runtime-c")
                newest_precheck = root / "precheck-c"
                fixture_preflight(newest_precheck, provenance, hardware, controls, policy)
                production_run(output, newest_precheck, resume=True)
                first = json.loads((output / "progress.json").read_text())
                self.assertEqual(first["status"], "completed_pending_independent_audit")
                self.assertEqual(len(first["evaluations"]), 10)
                self.assertEqual(len(first["attempts"]), 3)
                ids = [attempt["execution_id"] for attempt in first["attempts"]]
                training = json.loads((output / "main-m5" / "training.json").read_text())
                self.assertEqual(training["steps_by_execution"], {ids[0]: 2, ids[1]: 9})
                self.assertEqual((output / "protocol.json").read_bytes(), frozen_protocol)
                # Completed fits/evaluation chunks survive an additional reopen unchanged.
                production_run(output, newest_precheck, resume=True)
                second = json.loads((output / "progress.json").read_text())
                self.assertEqual(first["fits"], second["fits"])
                self.assertEqual(first["evaluations"], second["evaluations"])
                self.assertEqual(len(second["attempts"]), 4)
                with patch("verify_corrected_pilot.production_protocol", return_value=production_protocol(controls)):
                    with self.assertRaisesRegex(ValueError, "protocol differs"):
                        verify_production(output, controls)
                with patch("verify_corrected_pilot.production_protocol", return_value=protocol):
                    audited = verify_production(output, controls)
                    archived = output / "executions" / f"{ids[0]}.json"
                    original = archived.read_bytes()
                    archived.write_text("{}")
                    with self.assertRaisesRegex(ValueError, "archived preflight checksum"):
                        verify_production(output, controls)
                    archived.write_bytes(original)
                self.assertGreater(audited["predictions_and_metrics_checked"], 200)
                self.assertEqual(len(audited["qualified_executions_verified"]), 4)
                # Neither corrupted qualification files nor changed science can resume.
                (newest_precheck / "fixture-probe.txt").write_text("corrupted")
                with self.assertRaisesRegex(ValueError, "preflight evidence checksum"):
                    production_run(output, newest_precheck, resume=True)
                fixture_preflight(newest_precheck, provenance, hardware, controls, policy)
                changed = json.loads(frozen_protocol)
                changed["fits"][0]["training"]["learning_rate"] = .002
                (output / "protocol.json").write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, "identical frozen production protocol"):
                    production_run(output, newest_precheck, resume=True)


if __name__ == "__main__":
    unittest.main()
