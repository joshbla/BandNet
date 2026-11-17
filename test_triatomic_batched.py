import tempfile
import unittest
from pathlib import Path

import numpy as np

from triatomic_batched import TriatomicBatchSolver, _solve_dynamic_matrices
from triatomic_genuine_formula import triatomic_frequencies


class TriatomicBatchTests(unittest.TestCase):
    def assert_reference_agreement(self, masses, stiffnesses, grid):
        actual = TriatomicBatchSolver(grid, stiffnesses.shape[1]).evaluate(
            masses, stiffnesses
        ).frequencies
        expected = np.stack([
            triatomic_frequencies(mass, springs, grid)
            for mass, springs in zip(masses, stiffnesses)
        ])
        scales = np.maximum(1.0, np.max(expected, axis=(1, 2)))
        # Squared frequencies remain well-conditioned at exact acoustic zeros.
        for row, scale in enumerate(scales):
            squared_tolerance = 128 * np.finfo(float).eps * scale**2
            np.testing.assert_allclose(
                actual[row] ** 2, expected[row] ** 2,
                rtol=0, atol=squared_tolerance,
            )
            unresolved_zero = np.maximum(actual[row], expected[row]) ** 2 <= squared_tolerance
            self.assertTrue(np.all(
                (np.abs(actual[row] - expected[row]) <= 1e-10 * scale) | unresolved_zero
            ))
        return actual, expected, scales

    def test_seeded_dense_and_sparse_m5_native_grid(self):
        rng = np.random.default_rng(1847)
        masses = np.column_stack((np.ones(24), rng.uniform(0.1, 10.0, (24, 2))))
        springs = np.column_stack((np.ones(24), rng.uniform(0.0, 10.0, (24, 4))))
        springs[12:, 1:] *= rng.integers(0, 2, (12, 4))
        actual, expected, scales = self.assert_reference_agreement(
            masses, springs, np.linspace(0.001, 1.0, 500)
        )
        self.assertTrue(np.all(np.max(np.abs(actual - expected), axis=(1, 2)) <= 1e-10 * scales))
        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all(np.diff(actual, axis=-1) >= 0))

    def test_zone_center_near_zero_extreme_masses_and_degeneracy(self):
        masses = np.array([[1, 1, 1], [1, 1e-3, 1e3], [1, 1e3, 1e-3], [1, 2, 3]], dtype=float)
        springs = np.array([[1, 0, 0, 0, 0], [1, 10, 0, 10, 0], [1, 0, 10, 0, 10], [0, 0, 0, 0, 0]], dtype=float)
        actual, expected, scales = self.assert_reference_agreement(
            masses, springs, np.array([0.0, 1e-12, 1e-9, 1e-6, 0.001, 0.37, 1.0])
        )
        self.assertTrue(np.all(actual[:, 0, 0] <= np.sqrt(128 * np.finfo(float).eps) * scales))
        np.testing.assert_array_equal(actual[-1], np.zeros((7, 3)))

    def test_seeded_zone_center_roundoff_is_bounded(self):
        rng = np.random.default_rng(1847)
        masses = np.column_stack((np.ones(64), rng.uniform(0.1, 10, (64, 2))))
        springs = np.column_stack((np.ones(64), rng.uniform(0, 10, (64, 4))))
        actual, expected, scales = self.assert_reference_agreement(
            masses, springs, np.array([0, 1e-12, 1e-9, 1e-6, 0.001, 0.37, 1])
        )
        # Every k1=1 case has a rigid mode; do not assert equality of square
        # roots of independent positive roundoff residues at that mode.
        limit = np.sqrt(128 * np.finfo(float).eps) * scales
        self.assertTrue(np.all(actual[:, 0, 0] <= limit))
        self.assertTrue(np.all(expected[:, 0, 0] <= limit))

    def test_matches_reference_for_each_neighbor_and_padding(self):
        grid = np.array([0.0, 0.17, 0.51, 1.0])
        for count in (1, 2, 3, 4, 5, 6, 7):
            with self.subTest(interactions=count):
                springs = np.eye(count)
                masses = np.tile([1.0, 1.7, 2.4], (count, 1))
                self.assert_reference_agreement(masses, springs, grid)
        springs = np.array([[1, 2, 3, 4, 5]], dtype=float)
        five = TriatomicBatchSolver(grid, 5).evaluate([[1, 2, 3]], springs)
        six = TriatomicBatchSolver(grid, 6).evaluate([[1, 2, 3]], np.pad(springs, ((0, 0), (0, 1))))
        np.testing.assert_array_equal(five.frequencies, six.frequencies)

    def test_equal_mass_nearest_neighbor_matches_analytic_folded_chain(self):
        grid = np.linspace(0, 1, 101)
        result = TriatomicBatchSolver(grid, 1).evaluate([[2.5, 2.5, 2.5]], [[1.7]])
        phases = (np.pi * grid[:, None] + 2 * np.pi * np.arange(3)) / 3
        expected_squared = np.sort(4 * 1.7 / 2.5 * np.sin(phases / 2) ** 2, axis=1)
        np.testing.assert_allclose(result.frequencies[0] ** 2, expected_squared, rtol=1e-12, atol=1e-14)

    def test_highest_interaction_matters(self):
        solver = TriatomicBatchSolver([0.37], 5)
        actual = solver.evaluate([[1, 2, 3], [1, 2, 3]], [[1, 2, 3, 4, 0], [1, 2, 3, 4, 5]])
        self.assertFalse(np.allclose(actual.frequencies[0], actual.frequencies[1]))

    def test_chunked_output_roundtrip_and_partial_last_chunk(self):
        rng = np.random.default_rng(41)
        masses = rng.uniform(0.2, 10, (7, 3))
        springs = rng.uniform(0, 10, (7, 5))
        solver = TriatomicBatchSolver([0.0, 0.1, 0.37, 1.0], 5)
        expected = solver.evaluate(masses, springs).frequencies
        for chunk_size in (1, 3, 10):
            with self.subTest(chunk_size=chunk_size), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "frequencies.npy"
                output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=expected.shape)
                seen = []
                for start, result in solver.iter_batches(masses, springs, chunk_size=chunk_size):
                    output[start:start + len(result.frequencies)] = result.frequencies
                    seen.append(start)
                output.flush()
                del output
                restored = np.load(path)
                np.testing.assert_allclose(restored, expected, rtol=1e-12, atol=1e-7)
                self.assertEqual(seen, list(range(0, 7, chunk_size)))

    def test_bad_inputs_rejected_without_first_chunk(self):
        solver = TriatomicBatchSolver([0, 1], 5)
        good_m = np.ones((3, 3))
        good_k = np.ones((3, 5))
        invalid = (
            (np.ones((3, 2)), good_k),
            (good_m, np.ones((3, 4))),
            (good_m, np.ones((2, 5))),
            (good_m[0], good_k),
            (good_m, good_k[0]),
            (good_m * 0, good_k),
            (good_m * np.inf, good_k),
            (good_m, -good_k),
            (good_m, good_k * np.nan),
            (good_m.astype(complex), good_k),
            (good_m, good_k.astype(complex)),
            (np.empty((0, 3)), np.empty((0, 5))),
        )
        for masses, springs in invalid:
            with self.subTest(masses=masses.shape, springs=springs.shape):
                with self.assertRaises(ValueError):
                    next(solver.iter_batches(masses, springs, chunk_size=1))
        late_invalid = good_m.copy()
        late_invalid[-1, 1] = 0
        with self.assertRaises(ValueError):
            next(solver.iter_batches(late_invalid, good_k, chunk_size=1))
        for size in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                next(solver.iter_batches(good_m, good_k, chunk_size=size))

    def test_grid_validation_and_copied_configuration(self):
        for grid in ([], [[0]], [-0.1], [1.1], [np.nan], [np.inf], [0j]):
            with self.assertRaises(ValueError):
                TriatomicBatchSolver(grid, 5)
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                TriatomicBatchSolver([0], count)
        grid = np.array([0.2, 0.1, 0.2])
        solver = TriatomicBatchSolver(grid, 5)
        grid[:] = 1
        returned = solver.q_hat_values
        returned[:] = 0
        np.testing.assert_array_equal(solver.q_hat_values, [0.2, 0.1, 0.2])
        result = solver.evaluate([[1, 2, 3]], [[1, 2, 3, 4, 5]])
        self.assertEqual(result.frequencies.shape, (1, 3, 3))
        self.assertEqual(result.frequencies.dtype, np.float64)
        np.testing.assert_array_equal(result.frequencies[:, 0], result.frequencies[:, 2])

    def test_numerical_failures_are_not_hidden(self):
        for matrix in (
            np.diag([-1e-6, 1, 2]).astype(complex),
            np.array([[0, 1, 0], [0, 1, 0], [0, 0, 2]], dtype=complex),
            np.diag([np.nan, 1, 2]).astype(complex),
        ):
            with self.assertRaises(ArithmeticError):
                _solve_dynamic_matrices(matrix[None, None])
        matrix = np.diag([-np.finfo(float).eps, 1, 2]).astype(complex)
        result = _solve_dynamic_matrices(matrix[None, None])
        self.assertEqual(result.negative_eigenvalues_clipped, 1)
        self.assertEqual(result.frequencies[0, 0, 0], 0)
        solver = TriatomicBatchSolver([0.37], 5)
        with self.assertRaises(ArithmeticError):
            solver.evaluate([[1e-300, 1e-300, 1e-300]], [[1e300] * 5])


if __name__ == "__main__":
    unittest.main()
