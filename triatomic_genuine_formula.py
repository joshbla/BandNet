"""Auditable float64 implementation of canonical BN-TRI equations.

The synchronized specification is in ``generated/derivations``. This module
implements BN-TRI-PADDING, BN-TRI-FORCE, BN-TRI-ABC, and BN-TRI-DYNAMIC.

Matrix and eigenvalue checks use 64 times float64 machine epsilon multiplied by
the relevant matrix scale. Only squared frequencies inside that roundoff bound
are clipped to zero.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray


_ROUND_OFF_FACTOR = 64.0


def _real_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    raw_values = np.asarray(values)
    if np.iscomplexobj(raw_values):
        raise ValueError(f"{name} must contain real values")

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _normalized_wave_number(q_hat: float) -> np.float64:
    raw_q_hat = np.asarray(q_hat)
    if np.iscomplexobj(raw_q_hat) or raw_q_hat.ndim != 0:
        raise ValueError("q_hat must be a real scalar")

    normalized_q = np.float64(q_hat)
    if not np.isfinite(normalized_q):
        raise ValueError("q_hat must be finite")
    if normalized_q < 0.0 or normalized_q > 1.0:
        raise ValueError("q_hat must be within the normalized interval [0, 1]")
    return normalized_q


def _hermitian_tolerance(matrix: NDArray[np.complex128]) -> np.float64:
    scale = max(1.0, float(np.max(np.abs(matrix))))
    return np.float64(
        _ROUND_OFF_FACTOR * np.finfo(np.float64).eps * scale
    )


def _require_hermitian(
    matrix: NDArray[np.complex128],
    name: str,
) -> None:
    residual = float(np.max(np.abs(matrix - matrix.conj().T)))
    tolerance = float(_hermitian_tolerance(matrix))
    if residual > tolerance:
        raise ArithmeticError(
            f"{name} is not Hermitian: residual {residual} exceeds {tolerance}"
        )


def _validated_squared_frequencies(
    dynamic_matrix: NDArray[np.complex128],
) -> NDArray[np.float64]:
    _require_hermitian(dynamic_matrix, "mass-normalized dynamic matrix")
    squared_frequencies = np.linalg.eigvalsh(dynamic_matrix)

    scale = max(1.0, float(np.linalg.norm(dynamic_matrix, ord=np.inf)))
    tolerance = _ROUND_OFF_FACTOR * np.finfo(np.float64).eps * scale
    minimum = float(squared_frequencies[0])
    if minimum < -tolerance:
        raise ArithmeticError(
            "materially negative squared frequency: "
            f"{minimum} is below {-tolerance}"
        )

    return np.maximum(squared_frequencies, 0.0)


def triatomic_force_matrix(
    stiffnesses: ArrayLike,
    q_hat: float,
) -> NDArray[np.complex128]:
    """Construct the documented Bloch force matrix at normalized wave number q_hat."""
    physical_stiffnesses = _real_vector(stiffnesses, "stiffnesses")
    if np.any(physical_stiffnesses < 0.0):
        raise ValueError("stiffnesses must be nonnegative")

    normalized_q = _normalized_wave_number(q_hat)
    interaction_groups = math.ceil(physical_stiffnesses.size / 3)
    padded_count = 3 * interaction_groups
    padded_stiffnesses = np.zeros(padded_count, dtype=np.float64)
    padded_stiffnesses[: physical_stiffnesses.size] = physical_stiffnesses

    Q = np.pi * normalized_q
    A = np.complex128(-2.0 * np.sum(padded_stiffnesses))
    B = np.complex128(0.0)
    C = np.complex128(0.0)

    # Keep the one-based group index and terms in the same order as the derivation.
    for j in range(1, interaction_groups + 1):
        k_3j_minus_2 = padded_stiffnesses[3 * j - 3]
        k_3j_minus_1 = padded_stiffnesses[3 * j - 2]
        k_3j = padded_stiffnesses[3 * j - 1]

        exp_positive_jQ = np.exp(1j * j * Q)
        exp_negative_jQ = np.exp(-1j * j * Q)
        exp_positive_previous_Q = np.exp(1j * (j - 1) * Q)

        A += k_3j * (exp_positive_jQ + exp_negative_jQ)
        B += (
            k_3j_minus_2 * exp_positive_previous_Q
            + k_3j_minus_1 * exp_negative_jQ
        )
        C += (
            k_3j_minus_2 * exp_negative_jQ
            + k_3j_minus_1 * exp_positive_previous_Q
        )

    exp_positive_Q = np.exp(1j * Q)
    force_matrix = np.array(
        [
            [A, B, C],
            [exp_positive_Q * C, A, B],
            [exp_positive_Q * B, exp_positive_Q * C, A],
        ],
        dtype=np.complex128,
    )
    _require_hermitian(force_matrix, "Bloch force matrix")
    return force_matrix


def triatomic_frequencies(
    masses: ArrayLike,
    stiffnesses: ArrayLike,
    q_hat_values: ArrayLike,
) -> NDArray[np.float64]:
    """Return the three ascending triatomic frequencies for each q_hat value."""
    mass_values = _real_vector(masses, "masses")
    if mass_values.size != 3:
        raise ValueError("masses must contain exactly three values")
    if np.any(mass_values <= 0.0):
        raise ValueError("masses must be strictly positive")

    physical_stiffnesses = _real_vector(stiffnesses, "stiffnesses")
    if np.any(physical_stiffnesses < 0.0):
        raise ValueError("stiffnesses must be nonnegative")

    normalized_wave_numbers = _real_vector(q_hat_values, "q_hat_values")
    if np.any((normalized_wave_numbers < 0.0) | (normalized_wave_numbers > 1.0)):
        raise ValueError(
            "q_hat_values must be within the normalized interval [0, 1]"
        )

    inverse_sqrt_mass = np.diag(1.0 / np.sqrt(mass_values)).astype(
        np.complex128
    )
    frequencies = np.empty((normalized_wave_numbers.size, 3), dtype=np.float64)

    for index, q_hat in enumerate(normalized_wave_numbers):
        force_matrix = triatomic_force_matrix(physical_stiffnesses, q_hat)
        dynamic_matrix = -inverse_sqrt_mass @ force_matrix @ inverse_sqrt_mass
        squared_frequencies = _validated_squared_frequencies(dynamic_matrix)
        frequencies[index] = np.sqrt(squared_frequencies)

    return frequencies
