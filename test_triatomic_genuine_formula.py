import unittest

import numpy as np

from triatomic_genuine_formula import (
    _validated_squared_frequencies,
    triatomic_force_matrix,
    triatomic_frequencies,
)


def _direct_neighbor_force_matrix(stiffnesses, q_hat):
    """Construct F directly from the documented physical site equation."""
    force_matrix = np.zeros((3, 3), dtype=np.complex128)
    Q = np.pi * q_hat

    for basis_index in range(3):
        for distance, stiffness in enumerate(stiffnesses, start=1):
            force_matrix[basis_index, basis_index] -= 2.0 * stiffness
            for direction in (-1, 1):
                neighbor_site = basis_index + direction * distance
                cell_shift, neighbor_basis = divmod(neighbor_site, 3)
                force_matrix[basis_index, neighbor_basis] += (
                    stiffness * np.exp(1j * cell_shift * Q)
                )

    return force_matrix


def _documented_five_spring_force_matrix(stiffnesses, q_hat):
    """Use the explicitly expanded N=5 equations from the derivation."""
    k1, k2, k3, k4, k5 = stiffnesses
    Q = np.pi * q_hat
    exp_positive_Q = np.exp(1j * Q)
    exp_negative_Q = np.exp(-1j * Q)

    A = -2.0 * np.sum(stiffnesses) + k3 * (
        exp_positive_Q + exp_negative_Q
    )
    B = (
        k1
        + k2 * exp_negative_Q
        + k4 * exp_positive_Q
        + k5 * np.exp(-2j * Q)
    )
    C = (
        k1 * exp_negative_Q
        + k2
        + k4 * np.exp(-2j * Q)
        + k5 * exp_positive_Q
    )

    return np.array(
        [
            [A, B, C],
            [exp_positive_Q * C, A, B],
            [exp_positive_Q * B, exp_positive_Q * C, A],
        ],
        dtype=np.complex128,
    )


class TriatomicGenuineFormulaTests(unittest.TestCase):
    def test_matches_direct_neighbor_equation(self):
        stiffness_sets = (
            np.array([1.2]),
            np.array([1.2, 0.7]),
            np.array([1.2, 0.7, 0.4]),
            np.array([1.2, 0.7, 0.4, 0.3]),
            np.array([1.2, 0.7, 0.4, 0.3, 0.9]),
            np.array([1.2, 0.7, 0.4, 0.3, 0.9, 0.2]),
            np.array([1.2, 0.7, 0.4, 0.3, 0.9, 0.2, 0.5]),
        )

        for stiffnesses in stiffness_sets:
            for q_hat in (0.0, 0.19, 0.63, 1.0):
                with self.subTest(count=len(stiffnesses), q_hat=q_hat):
                    np.testing.assert_allclose(
                        triatomic_force_matrix(stiffnesses, q_hat),
                        _direct_neighbor_force_matrix(stiffnesses, q_hat),
                        rtol=2e-15,
                        atol=2e-14,
                    )

    def test_matches_expanded_five_spring_equations(self):
        stiffnesses = np.array([1.0, 0.8, 0.6, 0.4, 0.2])

        for q_hat in (0.0, 0.13, 0.5, 0.87, 1.0):
            with self.subTest(q_hat=q_hat):
                np.testing.assert_allclose(
                    triatomic_force_matrix(stiffnesses, q_hat),
                    _documented_five_spring_force_matrix(stiffnesses, q_hat),
                    rtol=2e-15,
                    atol=2e-14,
                )

    def test_zone_center_has_rigid_translation_mode(self):
        stiffnesses = np.array([1.0, 0.9, 0.7, 0.5, 0.3])
        force_matrix = triatomic_force_matrix(stiffnesses, 0.0)
        frequencies = triatomic_frequencies(
            [1.0, 1.8, 2.6], stiffnesses, [0.0]
        )

        np.testing.assert_allclose(
            force_matrix @ np.ones(3), np.zeros(3), atol=2e-14
        )
        self.assertLessEqual(frequencies[0, 0], 1e-7)

    def test_passive_systems_are_hermitian_real_nonnegative_and_ordered(self):
        stiffnesses = np.array([1.0, 0.7, 0.3, 0.9, 0.2])
        q_hat_values = np.array([0.0, 0.11, 0.37, 0.72, 1.0])

        for q_hat in q_hat_values:
            force_matrix = triatomic_force_matrix(stiffnesses, q_hat)
            np.testing.assert_allclose(
                force_matrix,
                force_matrix.conj().T,
                rtol=2e-15,
                atol=2e-14,
            )

        frequencies = triatomic_frequencies(
            [1.0, 2.3, 0.8], stiffnesses, q_hat_values
        )
        self.assertEqual(frequencies.dtype, np.float64)
        self.assertTrue(np.all(np.isfinite(frequencies)))
        self.assertTrue(np.all(frequencies >= 0.0))
        self.assertTrue(np.all(np.diff(frequencies, axis=1) >= 0.0))

    def test_equal_mass_nearest_neighbor_limit_matches_folded_chain(self):
        mass = 2.5
        stiffness = 1.7
        q_hat_values = np.array([0.0, 0.2, 0.73, 1.0])
        actual = triatomic_frequencies(
            [mass, mass, mass], [stiffness], q_hat_values
        )

        expected = []
        for q_hat in q_hat_values:
            reduced_phase = np.pi * q_hat
            unfolded_phases = (
                reduced_phase + 2.0 * np.pi * np.arange(3)
            ) / 3.0
            squared_frequencies = (2.0 * stiffness / mass) * (
                1.0 - np.cos(unfolded_phases)
            )
            expected.append(np.sqrt(np.sort(squared_frequencies)))

        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-7)

    def test_highest_range_spring_changes_interior_dispersion(self):
        masses = [1.0, 1.7, 2.4]
        without_k5 = triatomic_frequencies(
            masses, [1.0, 0.8, 0.6, 0.4, 0.0], [0.37]
        )
        with_k5 = triatomic_frequencies(
            masses, [1.0, 0.8, 0.6, 0.4, 0.9], [0.37]
        )

        self.assertFalse(np.allclose(without_k5, with_k5, rtol=1e-12, atol=1e-12))

    def test_implicit_and_explicit_zero_padding_are_equivalent(self):
        masses = [1.0, 1.4, 2.1]
        q_hat_values = [0.0, 0.23, 0.61, 1.0]
        five_springs = [1.0, 0.8, 0.6, 0.4, 0.2]

        implicit_padding = triatomic_frequencies(
            masses, five_springs, q_hat_values
        )
        explicit_padding = triatomic_frequencies(
            masses, five_springs + [0.0], q_hat_values
        )

        np.testing.assert_array_equal(implicit_padding, explicit_padding)

    def test_output_shape_dtype_and_all_zero_stiffnesses(self):
        frequencies = triatomic_frequencies(
            [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.4, 1.0]
        )

        self.assertEqual(frequencies.shape, (3, 3))
        self.assertEqual(frequencies.dtype, np.float64)
        np.testing.assert_array_equal(frequencies, np.zeros((3, 3)))

    def test_invalid_inputs_are_rejected(self):
        invalid_calls = (
            lambda: triatomic_frequencies([1.0, 2.0], [1.0], [0.0]),
            lambda: triatomic_frequencies([[1.0, 2.0, 3.0]], [1.0], [0.0]),
            lambda: triatomic_frequencies([1.0, 0.0, 3.0], [1.0], [0.0]),
            lambda: triatomic_frequencies([1.0, np.inf, 3.0], [1.0], [0.0]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0j], [1.0], [0.0]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [], [0.0]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [-1.0], [0.0]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [np.nan], [0.0]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [1.0j], [0.0]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [[1.0]], [0.0]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [1.0], []),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [1.0], [-0.1]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [1.0], [1.1]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [1.0], [np.nan]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [1.0], [0.2j]),
            lambda: triatomic_frequencies([1.0, 2.0, 3.0], [1.0], [[0.2]]),
            lambda: triatomic_force_matrix([1.0], [0.2]),
            lambda: triatomic_force_matrix([1.0], 0.2j),
        )

        for invalid_call in invalid_calls:
            with self.subTest(call=invalid_call):
                with self.assertRaises(ValueError):
                    invalid_call()

    def test_roundoff_negative_is_clipped_but_material_negative_is_rejected(self):
        epsilon = np.finfo(np.float64).eps
        roundoff_matrix = np.diag([-epsilon, 1.0, 2.0]).astype(np.complex128)
        squared_frequencies = _validated_squared_frequencies(roundoff_matrix)
        np.testing.assert_array_equal(squared_frequencies, [0.0, 1.0, 2.0])

        material_negative_matrix = np.diag([-1e-6, 1.0, 2.0]).astype(
            np.complex128
        )
        with self.assertRaises(ArithmeticError):
            _validated_squared_frequencies(material_negative_matrix)

        non_hermitian_matrix = np.array(
            [[0.0, 1.0], [0.0, 1.0]], dtype=np.complex128
        )
        with self.assertRaises(ArithmeticError):
            _validated_squared_frequencies(non_hermitian_matrix)


if __name__ == "__main__":
    unittest.main()
