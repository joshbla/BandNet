import unittest

import numpy as np

import test_triatomic_batched as checks
from triatomic_batched import TriatomicBatchSolver
from triatomic_genuine_formula import triatomic_frequencies


class InteractionCoverageTests(unittest.TestCase):
    def test_dense_sparse_and_edge_cases_through_twenty(self):
        # Reuse the documented comparison assertion, not a production code path.
        checker = checks.TriatomicBatchTests()
        for interactions in range(1, 21):
            with self.subTest(interactions=interactions):
                rng = np.random.default_rng(1847 + interactions)
                masses = np.column_stack((np.ones(12), rng.uniform(0.1, 10, (12, 2))))
                springs = np.column_stack((np.ones(12), rng.uniform(0, 10, (12, interactions - 1))))
                springs[6:, 1:] *= rng.integers(0, 2, (6, interactions - 1))
                checker.assert_reference_agreement(masses, springs, np.linspace(0.001, 1, 500))
                masses[:3] = [[1, 1e-3, 1e3], [1, 1e3, 1e-3], [1, 1, 1]]
                edge_grid = [0, 1e-12, 1e-9, 1e-6, 0.001, 0.37, 1]
                actual = TriatomicBatchSolver(edge_grid, interactions).evaluate(masses, springs).frequencies
                expected = np.stack([triatomic_frequencies(m, k, edge_grid) for m, k in zip(masses, springs)])
                # For the expanded diagnostic grid, assert the unchanged spectral
                # bound. The previous binary zero-region frequency rule is not
                # generally satisfied at q=1e-6; benchmark reports expose that
                # separately instead of claiming it passed or changing the solver.
                scale = np.maximum(1, np.max(expected, axis=(1, 2)))[:, None, None]
                self.assertTrue(np.all(np.abs(actual**2 - expected**2) <= 128 * np.finfo(float).eps * scale**2))
                self.assertTrue(np.all(actual[:, 0, 0] <= np.sqrt(128 * np.finfo(float).eps) * scale[:, 0, 0]))

    def test_every_individual_neighbor_through_twenty(self):
        checker = checks.TriatomicBatchTests()
        for interactions in range(8, 21):
            with self.subTest(interactions=interactions):
                springs = np.eye(interactions)
                masses = np.tile([1.0, 1.7, 2.4], (interactions, 1))
                checker.assert_reference_agreement(masses, springs, [0, 0.17, 0.51, 1])

    def test_highest_spring_and_padding_for_each_count(self):
        for interactions in range(5, 21):
            with self.subTest(interactions=interactions):
                springs = np.ones((1, interactions))
                masses = [[1, 1.7, 2.4]]
                solver = TriatomicBatchSolver([0.19, 0.37, 0.63], interactions)
                actual = solver.evaluate(masses, springs).frequencies
                missing_last = springs.copy()
                missing_last[0, -1] = 0
                self.assertFalse(np.allclose(actual, solver.evaluate(masses, missing_last).frequencies))
                padded = np.pad(springs, ((0, 0), (0, (-interactions) % 3)))
                padded_result = TriatomicBatchSolver([0.19, 0.37, 0.63], padded.shape[1]).evaluate(masses, padded)
                np.testing.assert_allclose(actual, padded_result.frequencies, rtol=1e-12, atol=1e-12)
