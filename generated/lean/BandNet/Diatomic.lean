import Mathlib

namespace BandNet.Diatomic

noncomputable section

/-- A finite family indexed by physical interaction distance `n+1` (BN-DIA-K). -/
abbrev StiffnessFamily (N : ℕ) := Fin N → ℝ

/-- Physical distance represented by a zero-based finite index (BN-DIA-K). -/
def physicalDistance {N : ℕ} (n : Fin N) : ℕ := (n : ℕ) + 1

/-- The sum of odd-distance couplings (BN-DIA-K). -/
def oddCouplingSum {N : ℕ} (k : StiffnessFamily N) : ℝ :=
  ∑ n with Odd (physicalDistance n), k n

/-- The diagonal scalar `K₀(q)` with even-distance filtering (BN-DIA-K). -/
def finiteK0 {N : ℕ} (k : StiffnessFamily N) (q d : ℝ) : ℝ :=
  (∑ n, k n) -
    ∑ n with Even (physicalDistance n), k n * Real.cos ((physicalDistance n : ℝ) * q * d)

/-- The off-diagonal scalar `K₁(q)` with odd-distance filtering (BN-DIA-K). -/
def finiteK1 {N : ℕ} (k : StiffnessFamily N) (q d : ℝ) : ℝ :=
  2 * ∑ n with Odd (physicalDistance n),
    k n * Real.cos ((physicalDistance n : ℝ) * q * d)

private theorem stiffness_sum_even_add_odd {N : ℕ} (k : StiffnessFamily N) :
    (∑ n, k n) =
      (∑ n with Even (physicalDistance n), k n) + oddCouplingSum k := by
  rw [oddCouplingSum, ← Finset.sum_filter_add_sum_filter_not Finset.univ
    (fun n : Fin N ↦ Even (physicalDistance n)) k]
  have hfilter :
      Finset.univ.filter (fun n : Fin N ↦ ¬Even (physicalDistance n)) =
        Finset.univ.filter (fun n : Fin N ↦ Odd (physicalDistance n)) := by
    ext n
    simp [Nat.not_even_iff_odd]
  rw [hfilter]

/-- At phase zero, `K₀` is the odd-distance coupling sum (BN-DIA-K). -/
theorem finiteK0_zero {N : ℕ} (k : StiffnessFamily N) (d : ℝ) :
    finiteK0 k 0 d = oddCouplingSum k := by
  simp only [finiteK0, mul_zero, zero_mul, Real.cos_zero, mul_one]
  rw [stiffness_sum_even_add_odd]
  ring

/-- At phase zero, `K₁` is twice the odd-distance coupling sum (BN-DIA-K). -/
theorem finiteK1_zero {N : ℕ} (k : StiffnessFamily N) (d : ℝ) :
    finiteK1 k 0 d = 2 * oddCouplingSum k := by
  simp [finiteK1, oddCouplingSum]

private theorem finiteK0_decomposition {N : ℕ} (k : StiffnessFamily N) (q d : ℝ) :
    finiteK0 k q d = oddCouplingSum k +
      ∑ n with Even (physicalDistance n),
        k n * (1 - Real.cos ((physicalDistance n : ℝ) * q * d)) := by
  rw [finiteK0, stiffness_sum_even_add_odd]
  calc
    (∑ n with Even (physicalDistance n), k n) + oddCouplingSum k -
        ∑ n with Even (physicalDistance n),
          k n * Real.cos ((physicalDistance n : ℝ) * q * d) =
      oddCouplingSum k + ((∑ n with Even (physicalDistance n), k n) -
        ∑ n with Even (physicalDistance n),
          k n * Real.cos ((physicalDistance n : ℝ) * q * d)) := by ring
    _ = oddCouplingSum k +
        ∑ n with Even (physicalDistance n),
          (k n - k n * Real.cos ((physicalDistance n : ℝ) * q * d)) := by
      rw [Finset.sum_sub_distrib]
    _ = oddCouplingSum k +
        ∑ n with Even (physicalDistance n),
          k n * (1 - Real.cos ((physicalDistance n : ℝ) * q * d)) := by
      congr 1
      apply Finset.sum_congr rfl
      intro n _
      ring

/-- Nonnegative stiffnesses make `K₀(q)` nonnegative (BN-DIA-K). -/
theorem finiteK0_nonneg {N : ℕ} (k : StiffnessFamily N) (q d : ℝ)
    (hk : ∀ n, 0 ≤ k n) : 0 ≤ finiteK0 k q d := by
  rw [finiteK0_decomposition]
  apply add_nonneg
  · exact Finset.sum_nonneg fun n hn ↦ hk n
  · exact Finset.sum_nonneg fun n hn ↦
      mul_nonneg (hk n) (sub_nonneg.mpr (Real.cos_le_one _))

/-- Passive finite couplings satisfy `|K₁(q)| ≤ 2 K₀(q)` (BN-DIA-K). -/
theorem finiteK1_abs_le_two_K0 {N : ℕ} (k : StiffnessFamily N) (q d : ℝ)
    (hk : ∀ n, 0 ≤ k n) : |finiteK1 k q d| ≤ 2 * finiteK0 k q d := by
  have hsum :
      |∑ n with Odd (physicalDistance n),
          k n * Real.cos ((physicalDistance n : ℝ) * q * d)| ≤ oddCouplingSum k := by
    calc
      _ ≤ ∑ n with Odd (physicalDistance n),
          |k n * Real.cos ((physicalDistance n : ℝ) * q * d)| :=
        Finset.abs_sum_le_sum_abs _ _
      _ ≤ ∑ n with Odd (physicalDistance n), k n := by
        apply Finset.sum_le_sum
        intro n hn
        rw [abs_mul, abs_of_nonneg (hk n)]
        exact mul_le_of_le_one_right (hk n) (Real.abs_cos_le_one _)
      _ = oddCouplingSum k := rfl
  have hodd_le : oddCouplingSum k ≤ finiteK0 k q d := by
    rw [finiteK0_decomposition]
    exact le_add_of_nonneg_right <| Finset.sum_nonneg fun n hn ↦
      mul_nonneg (hk n) (sub_nonneg.mpr (Real.cos_le_one _))
  rw [finiteK1, abs_mul, abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 2)]
  nlinarith

/-- The common center of the lower and upper branches (BN-DIA-BRANCHES). -/
def center (K₀ m₁ m₂ : ℝ) : ℝ := K₀ * (1 / m₁ + 1 / m₂)

/-- The radicand in the direct quadratic-formula expression (BN-DIA-BRANCHES). -/
def originalRadicand (K₀ K₁ m₁ m₂ : ℝ) : ℝ :=
  K₀ ^ 2 * (1 / m₁ + 1 / m₂) ^ 2 + (K₁ ^ 2 - 4 * K₀ ^ 2) / (m₁ * m₂)

/-- The manifestly nonnegative form of the radicand (BN-DIA-RADICAND). -/
def rewrittenRadicand (K₀ K₁ m₁ m₂ : ℝ) : ℝ :=
  K₀ ^ 2 * (1 / m₁ - 1 / m₂) ^ 2 + K₁ ^ 2 / (m₁ * m₂)

/-- The lower branch (BN-DIA-BRANCHES). -/
def lowerBranch (K₀ K₁ m₁ m₂ : ℝ) : ℝ :=
  center K₀ m₁ m₂ - Real.sqrt (rewrittenRadicand K₀ K₁ m₁ m₂)

/-- The upper branch (BN-DIA-BRANCHES). -/
def upperBranch (K₀ K₁ m₁ m₂ : ℝ) : ℝ :=
  center K₀ m₁ m₂ + Real.sqrt (rewrittenRadicand K₀ K₁ m₁ m₂)

/-- The generalized determinant polynomial in `ω²` (BN-DIA-DETERMINANT). -/
def determinantPolynomial (K₀ K₁ m₁ m₂ x : ℝ) : ℝ :=
  m₁ * m₂ * x ^ 2 - 2 * K₀ * (m₁ + m₂) * x + 4 * K₀ ^ 2 - K₁ ^ 2

/-- Expansion of the factorized generalized determinant (BN-DIA-DETERMINANT). -/
theorem determinant_expansion (K₀ K₁ m₁ m₂ x : ℝ) :
    (2 * K₀ - m₁ * x) * (2 * K₀ - m₂ * x) - K₁ ^ 2 =
      determinantPolynomial K₀ K₁ m₁ m₂ x := by
  simp [determinantPolynomial]
  ring

/-- The two displayed radicands are algebraically identical (BN-DIA-RADICAND). -/
theorem radicand_rewrite (K₀ K₁ m₁ m₂ : ℝ) (hm₁ : m₁ ≠ 0) (hm₂ : m₂ ≠ 0) :
    originalRadicand K₀ K₁ m₁ m₂ = rewrittenRadicand K₀ K₁ m₁ m₂ := by
  simp only [originalRadicand, rewrittenRadicand, one_div]
  field_simp [hm₁, hm₂]
  ring

/-- Positive masses make the rewritten radicand nonnegative (BN-DIA-RADICAND). -/
theorem rewrittenRadicand_nonneg (K₀ K₁ m₁ m₂ : ℝ) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    0 ≤ rewrittenRadicand K₀ K₁ m₁ m₂ := by
  exact add_nonneg (mul_nonneg (sq_nonneg _) (sq_nonneg _))
    (div_nonneg (sq_nonneg _) (mul_pos hm₁ hm₂).le)

/-- The lower branch never exceeds the upper branch (BN-DIA-BRANCHES). -/
theorem branch_order (K₀ K₁ m₁ m₂ : ℝ) :
    lowerBranch K₀ K₁ m₁ m₂ ≤ upperBranch K₀ K₁ m₁ m₂ := by
  rw [lowerBranch, upperBranch]
  linarith [Real.sqrt_nonneg (rewrittenRadicand K₀ K₁ m₁ m₂)]

/-- A passive coupling bound makes the lower branch nonnegative (BN-DIA-BRANCHES). -/
theorem lowerBranch_nonneg (K₀ K₁ m₁ m₂ : ℝ) (hK₀ : 0 ≤ K₀)
    (hK₁ : |K₁| ≤ 2 * K₀) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    0 ≤ lowerBranch K₀ K₁ m₁ m₂ := by
  have hK₁sq : K₁ ^ 2 ≤ 4 * K₀ ^ 2 := by
    have habs : |K₁| ^ 2 = K₁ ^ 2 := sq_abs K₁
    have hprod : 0 ≤ (2 * K₀ - |K₁|) * (2 * K₀ + |K₁|) :=
      mul_nonneg (sub_nonneg.mpr hK₁)
        (add_nonneg (mul_nonneg (by norm_num) hK₀) (abs_nonneg K₁))
    nlinarith
  have hc : 0 ≤ center K₀ m₁ m₂ :=
    mul_nonneg hK₀ (add_nonneg (one_div_nonneg.mpr hm₁.le) (one_div_nonneg.mpr hm₂.le))
  have hrad : rewrittenRadicand K₀ K₁ m₁ m₂ ≤ center K₀ m₁ m₂ ^ 2 := by
    rw [← radicand_rewrite K₀ K₁ m₁ m₂ hm₁.ne' hm₂.ne', originalRadicand]
    have hcenter :
        K₀ ^ 2 * (1 / m₁ + 1 / m₂) ^ 2 = center K₀ m₁ m₂ ^ 2 := by
      simp [center]
      ring
    rw [hcenter]
    have hterm : (K₁ ^ 2 - 4 * K₀ ^ 2) / (m₁ * m₂) ≤ 0 :=
      div_nonpos_of_nonpos_of_nonneg (sub_nonpos.mpr hK₁sq) (mul_pos hm₁ hm₂).le
    linarith
  have hsqrt : Real.sqrt (rewrittenRadicand K₀ K₁ m₁ m₂) ≤ center K₀ m₁ m₂ :=
    Real.sqrt_le_iff.mpr ⟨hc, hrad⟩
  rw [lowerBranch]
  linarith

/-- Nonnegative `K₀` makes the upper branch nonnegative (BN-DIA-BRANCHES). -/
theorem upperBranch_nonneg (K₀ K₁ m₁ m₂ : ℝ) (hK₀ : 0 ≤ K₀)
    (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    0 ≤ upperBranch K₀ K₁ m₁ m₂ := by
  rw [upperBranch]
  exact add_nonneg
    (mul_nonneg hK₀ (add_nonneg (one_div_nonneg.mpr hm₁.le) (one_div_nonneg.mpr hm₂.le)))
    (Real.sqrt_nonneg _)

/-- Finite nonnegative stiffnesses give a nonnegative lower branch (BN-DIA-K, BN-DIA-BRANCHES). -/
theorem finite_lowerBranch_nonneg {N : ℕ} (k : StiffnessFamily N) (q d m₁ m₂ : ℝ)
    (hk : ∀ n, 0 ≤ k n) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    0 ≤ lowerBranch (finiteK0 k q d) (finiteK1 k q d) m₁ m₂ :=
  lowerBranch_nonneg _ _ _ _ (finiteK0_nonneg k q d hk)
    (finiteK1_abs_le_two_K0 k q d hk) hm₁ hm₂

/-- Finite nonnegative stiffnesses give a nonnegative upper branch (BN-DIA-K, BN-DIA-BRANCHES). -/
theorem finite_upperBranch_nonneg {N : ℕ} (k : StiffnessFamily N) (q d m₁ m₂ : ℝ)
    (hk : ∀ n, 0 ≤ k n) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    0 ≤ upperBranch (finiteK0 k q d) (finiteK1 k q d) m₁ m₂ :=
  upperBranch_nonneg _ _ _ _ (finiteK0_nonneg k q d hk) hm₁ hm₂

private theorem branch_is_root (K₀ K₁ m₁ m₂ s : ℝ)
    (hm₁ : m₁ ≠ 0) (hm₂ : m₂ ≠ 0)
    (hs : s ^ 2 = rewrittenRadicand K₀ K₁ m₁ m₂) :
    determinantPolynomial K₀ K₁ m₁ m₂ (center K₀ m₁ m₂ + s) = 0 := by
  simp only [rewrittenRadicand, one_div] at hs
  field_simp [hm₁, hm₂] at hs
  simp only [determinantPolynomial, center, one_div]
  field_simp [hm₁, hm₂]
  nlinarith

/-- The lower branch solves the generalized determinant equation (BN-DIA-DETERMINANT). -/
theorem lowerBranch_is_root (K₀ K₁ m₁ m₂ : ℝ) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    determinantPolynomial K₀ K₁ m₁ m₂ (lowerBranch K₀ K₁ m₁ m₂) = 0 := by
  rw [lowerBranch, sub_eq_add_neg]
  apply branch_is_root K₀ K₁ m₁ m₂ _ hm₁.ne' hm₂.ne'
  rw [neg_sq, Real.sq_sqrt (rewrittenRadicand_nonneg K₀ K₁ m₁ m₂ hm₁ hm₂)]

/-- The upper branch solves the generalized determinant equation (BN-DIA-DETERMINANT). -/
theorem upperBranch_is_root (K₀ K₁ m₁ m₂ : ℝ) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    determinantPolynomial K₀ K₁ m₁ m₂ (upperBranch K₀ K₁ m₁ m₂) = 0 := by
  rw [upperBranch]
  apply branch_is_root K₀ K₁ m₁ m₂ _ hm₁.ne' hm₂.ne'
  exact Real.sq_sqrt (rewrittenRadicand_nonneg K₀ K₁ m₁ m₂ hm₁ hm₂)

private theorem zone_center_radicand (κ m₁ m₂ : ℝ) (hm₁ : m₁ ≠ 0) (hm₂ : m₂ ≠ 0) :
    rewrittenRadicand κ (2 * κ) m₁ m₂ = center κ m₁ m₂ ^ 2 := by
  rw [rewrittenRadicand, center]
  field_simp [hm₁, hm₂]
  ring

/-- At the zone center, nonnegative odd coupling gives the rigid lower mode (BN-DIA-BRANCHES). -/
theorem zone_center_lower (κ m₁ m₂ : ℝ) (hκ : 0 ≤ κ) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    lowerBranch κ (2 * κ) m₁ m₂ = 0 := by
  have hc : 0 ≤ center κ m₁ m₂ :=
    mul_nonneg hκ (add_nonneg (one_div_nonneg.mpr hm₁.le) (one_div_nonneg.mpr hm₂.le))
  rw [lowerBranch, zone_center_radicand κ m₁ m₂ hm₁.ne' hm₂.ne', Real.sqrt_sq hc]
  ring

/-- The zone-center upper value, without a strictness or uniqueness claim (BN-DIA-BRANCHES). -/
theorem zone_center_upper (κ m₁ m₂ : ℝ) (hκ : 0 ≤ κ) (hm₁ : 0 < m₁) (hm₂ : 0 < m₂) :
    upperBranch κ (2 * κ) m₁ m₂ = 2 * κ * (1 / m₁ + 1 / m₂) := by
  have hc : 0 ≤ center κ m₁ m₂ :=
    mul_nonneg hκ (add_nonneg (one_div_nonneg.mpr hm₁.le) (one_div_nonneg.mpr hm₂.le))
  rw [upperBranch, zone_center_radicand κ m₁ m₂ hm₁.ne' hm₂.ne', Real.sqrt_sq hc]
  simp [center]
  ring

end
end BandNet.Diatomic
