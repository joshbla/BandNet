import Mathlib

namespace BandNet.Monoatomic

noncomputable section

/-- A finite family of spring stiffnesses for interaction ranges `1,...,N` (BN-MONO-EOM). -/
abbrev StiffnessFamily (N : ℕ) := Fin N → ℝ

/-- The finite-range monoatomic dispersion relation `ω²(q)` (BN-MONO-DISPERSION). -/
def omegaSq (N : ℕ) (k : StiffnessFamily N) (m q a : ℝ) : ℝ :=
  (2 / m) * ∑ n, k n * (1 - Real.cos (((n : ℕ) + 1 : ℝ) * q * a))

/-- The rigid translation has zero frequency (BN-MONO-DISPERSION). -/
theorem omegaSq_zero_phase (N : ℕ) (k : Fin N → ℝ) (m a : ℝ) :
    omegaSq N k m 0 a = 0 := by
  simp [omegaSq]

/-- Nonnegative springs and positive mass give nonnegative `ω²` (BN-MONO-DISPERSION). -/
theorem omegaSq_nonneg (N : ℕ) (k : Fin N → ℝ) (m q a : ℝ)
    (hm : 0 < m) (hk : ∀ n, 0 ≤ k n) : 0 ≤ omegaSq N k m q a := by
  apply mul_nonneg (div_nonneg (by norm_num) hm.le)
  exact Finset.sum_nonneg fun n _ ↦
    mul_nonneg (hk n) (sub_nonneg.mpr (Real.cos_le_one _))

/-- A Bloch factor plus its inverse is twice the cosine (BN-MONO-DISPERSION). -/
theorem bloch_factor_cosine (x : ℝ) :
    Complex.exp (Complex.I * x) + Complex.exp (-Complex.I * x) =
      2 * (Real.cos x : ℂ) := by
  rw [mul_comm Complex.I, Complex.exp_ofReal_mul_I]
  rw [show -Complex.I * (x : ℂ) = ((-x : ℝ) : ℂ) * Complex.I by push_cast; ring]
  rw [Complex.exp_ofReal_mul_I, Real.cos_neg, Real.sin_neg]
  push_cast
  ring

end
end BandNet.Monoatomic
