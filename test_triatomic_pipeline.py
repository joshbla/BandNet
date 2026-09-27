import importlib.util
import tempfile
import unittest
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
    from corrected_pilot import settings
    from corrected_inference import infer_files
    from triatomic_learning import (BandInverse, DifferentiableTriatomic, band_errors, band_loss,
                                    evaluate_designs, evaluate_population, load_model, predict, train_model)


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


class CorrectedDataTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
