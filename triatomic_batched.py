"""Independent float64 CPU candidate for canonical triatomic dispersion.

Construct positive stiffness from physical neighbor bonds, not the reference
solver's grouped A/B/C expressions. Phase factors are reused across examples;
Hermitian eigenvalue solves are batched across examples and wave numbers.
This module neither imports the reference nor uses the historical Core path.
"""

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray


SOLVER_VERSION = "bond-batched-float64-v1"
_ROUND_OFF_FACTOR = 64.0


@dataclass(frozen=True)
class BatchResult:
    frequencies: NDArray[np.float64]
    negative_eigenvalues_clipped: int
    minimum_raw_eigenvalue: float
    maximum_roundoff_tolerance: float
    maximum_hermitian_residual: float


def _real_array(values: ArrayLike, name: str, ndim: int) -> NDArray[np.float64]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values")
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != ndim or result.size == 0:
        raise ValueError(f"{name} must be a nonempty {ndim}-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _solve_dynamic_matrices(matrices: NDArray[np.complex128]) -> BatchResult:
    """Validate full matrices before eigvalsh reads its triangular storage."""
    if not np.all(np.isfinite(matrices)):
        raise ArithmeticError("nonfinite mass-normalized dynamic matrix")
    scale = np.maximum(1.0, np.max(np.sum(np.abs(matrices), axis=-1), axis=-1))
    tolerance = _ROUND_OFF_FACTOR * np.finfo(np.float64).eps * scale
    residual = np.max(
        np.abs(matrices - matrices.conj().swapaxes(-1, -2)), axis=(-2, -1)
    )
    if np.any(residual > tolerance):
        raise ArithmeticError("mass-normalized dynamic matrix is not Hermitian")
    eigenvalues = np.linalg.eigvalsh(matrices)
    if not np.all(np.isfinite(eigenvalues)):
        raise ArithmeticError("nonfinite squared frequency")
    if np.any(eigenvalues[..., 0] < -tolerance):
        raise ArithmeticError("materially negative squared frequency")
    clipped_count = int(np.count_nonzero(eigenvalues < 0.0))
    minimum = float(np.min(eigenvalues))
    np.maximum(eigenvalues, 0.0, out=eigenvalues)
    np.sqrt(eigenvalues, out=eigenvalues)
    return BatchResult(
        frequencies=eigenvalues,
        negative_eigenvalues_clipped=clipped_count,
        minimum_raw_eigenvalue=minimum,
        maximum_roundoff_tolerance=float(np.max(tolerance)),
        maximum_hermitian_residual=float(np.max(residual)),
    )


class TriatomicBatchSolver:
    """Reuse a fixed wave-number grid and physical interaction count.

    Inputs are masses (examples, 3) and stiffnesses (examples, interactions).
    Results are ascending float64 frequencies (examples, wave numbers, 3).
    q_hat is q*L/pi in [0, 1]. The generic solver also accepts k1=0 for physical
    limit checks; choosing the training domain is the caller's responsibility.
    """

    def __init__(self, q_hat_values: ArrayLike, interaction_count: int):
        _positive_integer(interaction_count, "interaction_count")
        grid = _real_array(q_hat_values, "q_hat_values", 1).copy()
        if np.any((grid < 0.0) | (grid > 1.0)):
            raise ValueError("q_hat_values must lie in [0, 1]")
        grid.setflags(write=False)
        self._grid = grid
        self.interaction_count = int(interaction_count)
        self._bond_stiffness = self._build_bond_stiffness()

    @property
    def q_hat_values(self) -> NDArray[np.float64]:
        return self._grid.copy()

    def _build_bond_stiffness(self) -> NDArray[np.complex128]:
        basis = np.zeros(
            (self.interaction_count, self._grid.size, 3, 3), dtype=np.complex128
        )
        phase = np.pi * self._grid
        # One positive-direction bond per site and physical neighbor distance.
        # Its energy is k*|u_i - exp(i*cell_shift*Q)*u_j|^2.
        for index in range(self.interaction_count):
            distance = index + 1
            for site in range(3):
                cell_shift, neighbor = divmod(site + distance, 3)
                if neighbor == site:
                    # Stable 2*(1-cos(theta)), including very small theta.
                    basis[index, :, site, site] += 4.0 * np.sin(
                        0.5 * cell_shift * phase
                    ) ** 2
                else:
                    coupling = np.exp(1j * cell_shift * phase)
                    basis[index, :, site, site] += 1.0
                    basis[index, :, neighbor, neighbor] += 1.0
                    basis[index, :, site, neighbor] -= coupling
                    basis[index, :, neighbor, site] -= coupling.conj()
        basis.setflags(write=False)
        return basis

    def _inputs(
        self, masses: ArrayLike, stiffnesses: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        mass_array = _real_array(masses, "masses", 2)
        spring_array = _real_array(stiffnesses, "stiffnesses", 2)
        if mass_array.shape[1] != 3:
            raise ValueError("masses must have exactly three columns")
        if spring_array.shape != (mass_array.shape[0], self.interaction_count):
            raise ValueError("stiffnesses must match example and interaction counts")
        if np.any(mass_array <= 0.0):
            raise ValueError("masses must be strictly positive")
        if np.any(spring_array < 0.0):
            raise ValueError("stiffnesses must be nonnegative")
        return mass_array, spring_array

    def _evaluate(
        self, masses: NDArray[np.float64], stiffnesses: NDArray[np.float64]
    ) -> BatchResult:
        # No Python loop over examples or wave numbers, and no padded spring count.
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            matrices = np.einsum(
                "bn,nqij->bqij", stiffnesses, self._bond_stiffness, optimize=True
            )
            inverse_mass = 1.0 / np.sqrt(masses)
            matrices *= inverse_mass[:, None, :, None]
            matrices *= inverse_mass[:, None, None, :]
            return _solve_dynamic_matrices(matrices)

    def evaluate(self, masses: ArrayLike, stiffnesses: ArrayLike) -> BatchResult:
        """Calculate one caller-sized batch; temporary matrix memory scales with it."""
        mass_array, spring_array = self._inputs(masses, stiffnesses)
        return self._evaluate(mass_array, spring_array)

    def iter_batches(
        self, masses: ArrayLike, stiffnesses: ArrayLike, *, chunk_size: int
    ) -> Iterator[tuple[int, BatchResult]]:
        """Yield (first example index, result) without allocating all output curves.

        The caller may write frequencies directly into a memory-mapped array.
        All input labels are checked before the first chunk is yielded.
        """
        _positive_integer(chunk_size, "chunk_size")
        mass_array, spring_array = self._inputs(masses, stiffnesses)
        for start in range(0, mass_array.shape[0], chunk_size):
            stop = start + chunk_size
            yield start, self._evaluate(mass_array[start:stop], spring_array[start:stop])
