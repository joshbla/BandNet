import Mathlib

namespace BandNet.Triatomic

noncomputable section

open scoped ComplexConjugate ComplexOrder

/-- The phase `exp(i r Q)` used throughout the triatomic formulas (BN-TRI-ABC). -/
def phase (r : ℤ) (Q : ℝ) : ℂ := Complex.exp (Complex.I * ((r : ℝ) * Q))

/-- Conjugating a phase reverses its integer exponent (BN-TRI-HERMITIAN). -/
theorem phase_conj (r : ℤ) (Q : ℝ) : conj (phase r Q) = phase (-r) Q := by
  rw [phase, ← Complex.exp_conj]
  congr 1
  simp

/-- Products of Bloch phases add their integer exponents (BN-TRI-ABC). -/
theorem phase_add (r s : ℤ) (Q : ℝ) : phase r Q * phase s Q = phase (r + s) Q := by
  rw [phase, phase, phase, ← Complex.exp_add]
  congr 1
  push_cast
  ring

@[simp] theorem phase_zero_index (Q : ℝ) : phase 0 Q = 1 := by simp [phase]

@[simp] theorem phase_zero_wave (r : ℤ) : phase r 0 = 1 := by simp [phase]

@[simp] theorem phase_neg_mul_self (r : ℤ) (Q : ℝ) : phase (-r) Q * phase r Q = 1 := by
  rw [phase_add]
  simp

@[simp] theorem phase_neg_succ_mul_succ (j : ℕ) (Q : ℝ) :
    phase (-1 - (j : ℤ)) Q * phase (1 + (j : ℤ)) Q = 1 := by
  rw [phase_add]
  convert phase_zero_index Q
  omega

@[simp] theorem phase_succ_mul_neg_succ (j : ℕ) (Q : ℝ) :
    phase (1 + (j : ℤ)) Q * phase (-1 - (j : ℤ)) Q = 1 := by
  rw [phase_add]
  convert phase_zero_index Q
  omega

@[simp] theorem phase_one_mul_neg_succ (j : ℕ) (Q : ℝ) :
    phase 1 Q * phase (-1 - (j : ℤ)) Q = phase (-(j : ℤ)) Q := by
  rw [phase_add]
  congr 1
  omega

@[simp] theorem phase_one_mul_nat (j : ℕ) (Q : ℝ) :
    phase 1 Q * phase (j : ℤ) Q = phase (1 + (j : ℤ)) Q := by
  rw [phase_add]

@[simp] theorem phase_neg_one_sub_mul_one_add (r : ℤ) (Q : ℝ) :
    phase (-1 - r) Q * phase (1 + r) Q = 1 := by
  rw [phase_add]
  convert phase_zero_index Q
  ring

@[simp] theorem phase_one_add_mul_neg_one_sub (r : ℤ) (Q : ℝ) :
    phase (1 + r) Q * phase (-1 - r) Q = 1 := by
  rw [phase_add]
  convert phase_zero_index Q
  ring

@[simp] theorem phase_one_mul_neg_one_sub (r : ℤ) (Q : ℝ) :
    phase 1 Q * phase (-1 - r) Q = phase (-r) Q := by
  rw [phase_add]
  congr 1
  ring

@[simp] theorem phase_one_mul_int (r : ℤ) (Q : ℝ) :
    phase 1 Q * phase r Q = phase (1 + r) Q := by
  rw [phase_add]

@[simp] theorem mul_phase_neg_one_sub_mul_one_add (a : ℂ) (r : ℤ) (Q : ℝ) :
    a * phase (-1 - r) Q * phase (1 + r) Q = a := by
  rw [mul_assoc, phase_neg_one_sub_mul_one_add, mul_one]

@[simp] theorem mul_phase_one_mul_neg_one_sub (a : ℂ) (r : ℤ) (Q : ℝ) :
    a * phase 1 Q * phase (-1 - r) Q = a * phase (-r) Q := by
  rw [mul_assoc, phase_one_mul_neg_one_sub]

@[simp] theorem mul_phase_one_mul_int (a : ℂ) (r : ℤ) (Q : ℝ) :
    a * phase 1 Q * phase r Q = a * phase (1 + r) Q := by
  rw [mul_assoc, phase_one_mul_int]

@[simp] theorem phase_one_add_mul_neg_one_sub_mul (a : ℂ) (r : ℤ) (Q : ℝ) :
    phase (1 + r) Q * phase (-1 - r) Q * a = a := by
  rw [phase_one_add_mul_neg_one_sub, one_mul]

@[simp] theorem phase_pair_conj (r : ℤ) (Q : ℝ) :
    conj (phase r Q + phase (-r) Q) = phase r Q + phase (-r) Q := by
  simp [phase_conj, add_comm]

/-- Couplings at distances `3j-2`, `3j-1`, and `3j` (BN-TRI-PADDING). -/
structure Interaction where
  left : ℝ
  right : ℝ
  same : ℝ

/-- Every spring in a grouped interaction is passive. -/
def Interaction.Nonnegative (h : Interaction) : Prop :=
  0 ≤ h.left ∧ 0 ≤ h.right ∧ 0 ≤ h.same

/-- Total cross-sublattice coupling used at the zone center (BN-TRI-ABC). -/
def crossCoupling {P : ℕ} (g : Fin P → Interaction) : ℝ :=
  (∑ j, (g j).left) + ∑ j, (g j).right

/-- The diagonal force entry `A(q)` (BN-TRI-ABC). -/
def A {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) : ℂ :=
  -2 * ((∑ j, ((g j).left : ℂ)) + (∑ j, ((g j).right : ℂ)) +
    ∑ j, ((g j).same : ℂ)) +
    ∑ j, (g j).same * (phase ((j : ℕ) + 1) Q + phase (-((j : ℕ) + 1 : ℤ)) Q)

/-- The forward off-diagonal force entry `B(q)` (BN-TRI-ABC). -/
def B {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) : ℂ :=
  ∑ j, ((g j).left * phase (j : ℕ) Q +
    (g j).right * phase (-((j : ℕ) + 1 : ℤ)) Q)

/-- The backward off-diagonal force entry `C(q)` (BN-TRI-ABC). -/
def C {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) : ℂ :=
  ∑ j, ((g j).left * phase (-((j : ℕ) + 1 : ℤ)) Q +
    (g j).right * phase (j : ℕ) Q)

/-- The diagonal entry is real (BN-TRI-HERMITIAN). -/
theorem A_conj {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) : conj (A g Q) = A g Q := by
  have htwo : conj (2 : ℂ) = 2 := Complex.conj_ofReal 2
  simp [A, phase_conj]
  rw [htwo]
  apply congrArg₂ (· + ·)
  · rfl
  · apply Finset.sum_congr rfl
    intro j _
    ring

/-- `e^{iQ} C = B*` (BN-TRI-HERMITIAN). -/
theorem phase_mul_C {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) :
    phase 1 Q * C g Q = conj (B g Q) := by
  simp_rw [C, B, Finset.mul_sum, map_sum, map_add, map_mul, Complex.conj_ofReal, phase_conj]
  apply Finset.sum_congr rfl
  intro j _
  rw [mul_add]
  rw [show phase 1 Q * ((g j).left * phase (-((j : ℕ) + 1 : ℤ)) Q) =
    (g j).left * (phase 1 Q * phase (-((j : ℕ) + 1 : ℤ)) Q) by ring, phase_add]
  rw [show phase 1 Q * ((g j).right * phase (j : ℕ) Q) =
    (g j).right * (phase 1 Q * phase (j : ℕ) Q) by ring, phase_add]
  have h₁ : (1 : ℤ) + -((j : ℕ) + 1 : ℤ) = -(j : ℕ) := by omega
  have h₂ : (1 : ℤ) + (j : ℕ) = ((j : ℕ) + 1 : ℤ) := by omega
  rw [h₁, h₂]
  simp

/-- `e^{iQ} B = C*` (BN-TRI-HERMITIAN). -/
theorem phase_mul_B {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) :
    phase 1 Q * B g Q = conj (C g Q) := by
  simp_rw [C, B, Finset.mul_sum, map_sum, map_add, map_mul, Complex.conj_ofReal, phase_conj]
  apply Finset.sum_congr rfl
  intro j _
  rw [mul_add]
  rw [show phase 1 Q * ((g j).left * phase (j : ℕ) Q) =
    (g j).left * (phase 1 Q * phase (j : ℕ) Q) by ring, phase_add]
  rw [show phase 1 Q * ((g j).right * phase (-((j : ℕ) + 1 : ℤ)) Q) =
    (g j).right * (phase 1 Q * phase (-((j : ℕ) + 1 : ℤ)) Q) by ring, phase_add]
  have h₁ : (1 : ℤ) + (j : ℕ) = ((j : ℕ) + 1 : ℤ) := by omega
  have h₂ : (1 : ℤ) + -((j : ℕ) + 1 : ℤ) = -(j : ℕ) := by omega
  rw [h₁, h₂]
  simp

/-- The force matrix in the `U,V,W` basis (BN-TRI-FORCE). -/
def forceMatrix {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) : Matrix (Fin 3) (Fin 3) ℂ :=
  !![A g Q, B g Q, C g Q;
     phase 1 Q * C g Q, A g Q, B g Q;
     phase 1 Q * B g Q, phase 1 Q * C g Q, A g Q]

/-- Rank-one energy matrix associated with one complex edge-difference row. -/
def edgeEnergy (v : Fin 3 → ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  Matrix.vecMulVec (star v) v

private def leftEdge01 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![1, -phase j Q, 0]
private def leftEdge02 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![1, 0, -phase (-(j + 1 : ℤ)) Q]
private def leftEdge12 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![0, 1, -phase j Q]

private def rightEdge01 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![1, -phase (-(j + 1 : ℤ)) Q, 0]
private def rightEdge02 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![1, 0, -phase j Q]
private def rightEdge12 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![0, 1, -phase (-(j + 1 : ℤ)) Q]

private def sameEdge0 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![1 - phase (j + 1) Q, 0, 0]
private def sameEdge1 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![0, 1 - phase (j + 1) Q, 0]
private def sameEdge2 (j : ℕ) (Q : ℝ) : Fin 3 → ℂ := ![0, 0, 1 - phase (j + 1) Q]

/-- Sum-of-squares spring energy contributed by one grouped interaction (BN-TRI-EOM). -/
def interactionEnergy (h : Interaction) (j : ℕ) (Q : ℝ) : Matrix (Fin 3) (Fin 3) ℂ :=
  h.left • (edgeEnergy (leftEdge01 j Q) + edgeEnergy (leftEdge02 j Q) +
    edgeEnergy (leftEdge12 j Q)) +
  h.right • (edgeEnergy (rightEdge01 j Q) + edgeEnergy (rightEdge02 j Q) +
    edgeEnergy (rightEdge12 j Q)) +
  h.same • (edgeEnergy (sameEdge0 j Q) + edgeEnergy (sameEdge1 j Q) +
    edgeEnergy (sameEdge2 j Q))

/-- Finite spring-energy matrix for all grouped interactions (BN-TRI-EOM). -/
def springEnergyMatrix {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) :
    Matrix (Fin 3) (Fin 3) ℂ :=
  ∑ j, interactionEnergy (g j) j Q

private theorem energy_diag_neg (l r s pn pp : ℂ) (h : pn * pp = 1) :
    l * 2 + r * 2 + s - s * pn + s * pn * pp - s * pp =
      l * 2 + r * 2 + s * 2 - s * pn - s * pp := by
  rw [show s * pn * pp = s by rw [mul_assoc, h, mul_one]]
  ring

private theorem energy_diag_pos (l r s pn pp : ℂ) (h : pp * pn = 1) :
    l * 2 + r + r * pp * pn + pp * pn * s - pp * s - pn * s + s =
      l * 2 + r * 2 - pp * s - pn * s + s * 2 := by
  rw [show r * pp * pn = r by rw [mul_assoc, h, mul_one]]
  rw [show pp * pn * s = s by rw [h, one_mul]]
  ring

private theorem energy_diag_pos_all (l r s pn pp : ℂ) (h : pp * pn = 1) :
    l + l * pp * pn + pp * pn * r + pp * pn * s - pp * s - pn * s + r + s =
      l * 2 - pp * s - pn * s + r * 2 + s * 2 := by
  rw [show l * pp * pn = l by rw [mul_assoc, h, mul_one]]
  rw [show pp * pn * r = r by rw [h, one_mul]]
  rw [show pp * pn * s = s by rw [h, one_mul]]
  ring

private theorem energy_offdiag (l r pminus pplus p1 pn p : ℂ)
    (hm : p1 * pn = pminus) (hp : p1 * p = pplus) :
    -(l * pminus) - r * pplus = -(l * p1 * pn) - r * p1 * p := by
  rw [show l * p1 * pn = l * pminus by rw [mul_assoc, hm]]
  rw [show r * p1 * p = r * pplus by rw [mul_assoc, hp]]

/-- The spring-energy Gram matrix is exactly the negative canonical force matrix. -/
theorem springEnergyMatrix_eq_neg_forceMatrix {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) :
    springEnergyMatrix g Q = -forceMatrix g Q := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [springEnergyMatrix, interactionEnergy, edgeEnergy, leftEdge01, leftEdge02,
      leftEdge12, rightEdge01, rightEdge02, rightEdge12, sameEdge0, sameEdge1,
      sameEdge2, forceMatrix, A, B, C, Matrix.vecMulVec, Matrix.sum_apply,
      Matrix.of_apply, Matrix.vecHead, Matrix.vecTail, Function.comp_apply,
      Pi.smul_apply, Pi.add_apply, phase_conj]
  all_goals
    repeat rw [Finset.mul_sum]
    repeat rw [← Finset.sum_neg_distrib]
    repeat rw [← Finset.sum_add_distrib]
    repeat rw [Finset.mul_sum]
    repeat rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro x hx
    have hnp := phase_neg_succ_mul_succ (x : ℕ) Q
    have hpn := phase_succ_mul_neg_succ (x : ℕ) Q
    have hm := phase_one_mul_neg_succ (x : ℕ) Q
    have hp := phase_one_mul_nat (x : ℕ) Q
    have hc : phase ((x : ℤ) + 1) Q = phase (1 + (x : ℤ)) Q := by
      congr 1
      ring
    try rw [hc]
  · convert energy_diag_neg ((g x).left : ℂ) (g x).right (g x).same
      (phase (-1 - (x : ℤ)) Q) (phase (1 + (x : ℤ)) Q) hnp using 1 <;> ring
  · ring
  · ring
  · convert energy_offdiag ((g x).left : ℂ) (g x).right (phase (-(x : ℤ)) Q)
      (phase (1 + (x : ℤ)) Q) (phase 1 Q) (phase (-1 - (x : ℤ)) Q)
      (phase (x : ℤ) Q) hm hp using 1 <;> ring
  · convert energy_diag_pos ((g x).left : ℂ) (g x).right (g x).same
      (phase (-1 - (x : ℤ)) Q) (phase (1 + (x : ℤ)) Q) hpn using 1 <;> ring
  · ring
  · convert energy_offdiag ((g x).left : ℂ) (g x).right (phase (1 + (x : ℤ)) Q)
      (phase (-(x : ℤ)) Q) (phase 1 Q) (phase (x : ℤ) Q)
      (phase (-1 - (x : ℤ)) Q) hp hm using 1 <;> ring
  · convert energy_offdiag ((g x).left : ℂ) (g x).right (phase (-(x : ℤ)) Q)
      (phase (1 + (x : ℤ)) Q) (phase 1 Q) (phase (-1 - (x : ℤ)) Q)
      (phase (x : ℤ) Q) hm hp using 1 <;> ring
  · convert energy_diag_pos_all ((g x).left : ℂ) (g x).right (g x).same
      (phase (-1 - (x : ℤ)) Q) (phase (1 + (x : ℤ)) Q) hpn using 1 <;> ring

/-- Each complex edge difference contributes a positive semidefinite Gram matrix. -/
theorem edgeEnergy_posSemidef (v : Fin 3 → ℂ) : (edgeEnergy v).PosSemidef := by
  exact Matrix.posSemidef_vecMulVec_star_self v

/-- A passive grouped interaction contributes nonnegative spring energy. -/
theorem interactionEnergy_posSemidef (h : Interaction) (j : ℕ) (Q : ℝ)
    (hh : h.Nonnegative) : (interactionEnergy h j Q).PosSemidef := by
  have hleft := ((edgeEnergy_posSemidef (leftEdge01 j Q)).add
    (edgeEnergy_posSemidef (leftEdge02 j Q))).add (edgeEnergy_posSemidef (leftEdge12 j Q))
  have hright := ((edgeEnergy_posSemidef (rightEdge01 j Q)).add
    (edgeEnergy_posSemidef (rightEdge02 j Q))).add (edgeEnergy_posSemidef (rightEdge12 j Q))
  have hsame := ((edgeEnergy_posSemidef (sameEdge0 j Q)).add
    (edgeEnergy_posSemidef (sameEdge1 j Q))).add (edgeEnergy_posSemidef (sameEdge2 j Q))
  exact ((hleft.smul hh.1).add (hright.smul hh.2.1)).add (hsame.smul hh.2.2)

/-- The finite spring-energy matrix is positive semidefinite for passive springs. -/
theorem springEnergyMatrix_posSemidef {P : ℕ} (g : Fin P → Interaction) (Q : ℝ)
    (hg : ∀ j, (g j).Nonnegative) : (springEnergyMatrix g Q).PosSemidef := by
  rw [springEnergyMatrix]
  apply Matrix.posSemidef_sum Finset.univ
  intro j hj
  exact interactionEnergy_posSemidef (g j) j Q (hg j)

/-- The canonical force matrix is negative semidefinite for passive springs (BN-TRI-EOM). -/
theorem neg_forceMatrix_posSemidef {P : ℕ} (g : Fin P → Interaction) (Q : ℝ)
    (hg : ∀ j, (g j).Nonnegative) : (-forceMatrix g Q).PosSemidef := by
  rw [← springEnergyMatrix_eq_neg_forceMatrix]
  exact springEnergyMatrix_posSemidef g Q hg

/-- Three strictly positive masses in one triatomic cell (BN-TRI-GENERALIZED). -/
structure PositiveMasses where
  m₁ : ℝ
  m₂ : ℝ
  m₃ : ℝ
  m₁_pos : 0 < m₁
  m₂_pos : 0 < m₂
  m₃_pos : 0 < m₃

/-- The positive diagonal mass matrix (BN-TRI-GENERALIZED). -/
def massMatrix (m : PositiveMasses) : Matrix (Fin 3) (Fin 3) ℂ :=
  Matrix.diagonal ![(m.m₁ : ℂ), m.m₂, m.m₃]

/-- The diagonal square-root mass matrix. -/
def massSqrtMatrix (m : PositiveMasses) : Matrix (Fin 3) (Fin 3) ℂ :=
  Matrix.diagonal ![(Real.sqrt m.m₁ : ℂ), Real.sqrt m.m₂, Real.sqrt m.m₃]

/-- The diagonal inverse-square-root mass matrix. -/
def massInvSqrtMatrix (m : PositiveMasses) : Matrix (Fin 3) (Fin 3) ℂ :=
  Matrix.diagonal ![((1 / Real.sqrt m.m₁ : ℝ) : ℂ),
    ((1 / Real.sqrt m.m₂ : ℝ) : ℂ), ((1 / Real.sqrt m.m₃ : ℝ) : ℂ)]

/-- The mass-normalized dynamic matrix `-M⁻¹ᐟ² F M⁻¹ᐟ²` (BN-TRI-DYNAMIC). -/
def dynamicMatrix {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) (m : PositiveMasses) :
    Matrix (Fin 3) (Fin 3) ℂ :=
  massInvSqrtMatrix m * (-forceMatrix g Q) * massInvSqrtMatrix m

/-- `M¹ᐟ² M⁻¹ᐟ² = I` for positive diagonal masses. -/
theorem massSqrt_mul_massInvSqrt (m : PositiveMasses) :
    massSqrtMatrix m * massInvSqrtMatrix m = 1 := by
  have h₁ : (Real.sqrt m.m₁ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₁_pos).ne'
  have h₂ : (Real.sqrt m.m₂ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₂_pos).ne'
  have h₃ : (Real.sqrt m.m₃ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₃_pos).ne'
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [massSqrtMatrix, massInvSqrtMatrix, Matrix.mul_apply, Fin.sum_univ_three,
      h₁, h₂, h₃]

/-- `M⁻¹ᐟ² M¹ᐟ² = I` for positive diagonal masses. -/
theorem massInvSqrt_mul_massSqrt (m : PositiveMasses) :
    massInvSqrtMatrix m * massSqrtMatrix m = 1 := by
  have h₁ : (Real.sqrt m.m₁ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₁_pos).ne'
  have h₂ : (Real.sqrt m.m₂ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₂_pos).ne'
  have h₃ : (Real.sqrt m.m₃ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₃_pos).ne'
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [massSqrtMatrix, massInvSqrtMatrix, Matrix.mul_apply, Fin.sum_univ_three,
      h₁, h₂, h₃]

/-- Multiplying `M` by `M⁻¹ᐟ²` gives `M¹ᐟ²`. -/
theorem mass_mul_massInvSqrt (m : PositiveMasses) :
    massMatrix m * massInvSqrtMatrix m = massSqrtMatrix m := by
  have h₁ : (Real.sqrt m.m₁ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₁_pos).ne'
  have h₂ : (Real.sqrt m.m₂ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₂_pos).ne'
  have h₃ : (Real.sqrt m.m₃ : ℂ) ≠ 0 :=
    Complex.ofReal_ne_zero.mpr (Real.sqrt_pos.2 m.m₃_pos).ne'
  have hs₁ : (m.m₁ : ℂ) = (Real.sqrt m.m₁ : ℂ) * Real.sqrt m.m₁ := by
    norm_cast
    nlinarith [Real.sq_sqrt m.m₁_pos.le]
  have hs₂ : (m.m₂ : ℂ) = (Real.sqrt m.m₂ : ℂ) * Real.sqrt m.m₂ := by
    norm_cast
    nlinarith [Real.sq_sqrt m.m₂_pos.le]
  have hs₃ : (m.m₃ : ℂ) = (Real.sqrt m.m₃ : ℂ) * Real.sqrt m.m₃ := by
    norm_cast
    nlinarith [Real.sq_sqrt m.m₃_pos.le]
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [massMatrix, massSqrtMatrix, massInvSqrtMatrix, Matrix.mul_apply,
      Fin.sum_univ_three, h₁, h₂, h₃, hs₁, hs₂, hs₃]

/-- The mass-normalized dynamic matrix is positive semidefinite for passive springs. -/
theorem dynamicMatrix_posSemidef {P : ℕ} (g : Fin P → Interaction) (Q : ℝ)
    (m : PositiveMasses) (hg : ∀ j, (g j).Nonnegative) :
    (dynamicMatrix g Q m).PosSemidef := by
  have h := (neg_forceMatrix_posSemidef g Q hg).conjTranspose_mul_mul_same
    (massInvSqrtMatrix m)
  have hs : Matrix.conjTranspose (massInvSqrtMatrix m) = massInvSqrtMatrix m := by
    ext i j
    fin_cases i <;> fin_cases j <;>
      simp [Matrix.conjTranspose, massInvSqrtMatrix]
  simpa [dynamicMatrix, hs] using h

/-- The mass-normalized dynamic matrix is Hermitian for passive springs. -/
theorem dynamicMatrix_isHermitian {P : ℕ} (g : Fin P → Interaction) (Q : ℝ)
    (m : PositiveMasses) (hg : ∀ j, (g j).Nonnegative) :
    (dynamicMatrix g Q m).IsHermitian :=
  (dynamicMatrix_posSemidef g Q m hg).isHermitian

/-- Ordinary dynamic eigenvectors are algebraically equivalent to the signed generalized problem. -/
theorem dynamic_eigen_iff_generalized {P : ℕ} (g : Fin P → Interaction) (Q eig : ℝ)
    (m : PositiveMasses) (x : Fin 3 → ℂ) :
    (dynamicMatrix g Q m).mulVec x = eig • x ↔
      (-forceMatrix g Q).mulVec ((massInvSqrtMatrix m).mulVec x) =
        eig • (massSqrtMatrix m).mulVec x := by
  constructor
  · intro h
    have h' := congrArg (fun y ↦ (massSqrtMatrix m).mulVec y) h
    simpa [dynamicMatrix, Matrix.mulVec_mulVec, ← Matrix.mul_assoc,
      massSqrt_mul_massInvSqrt, Matrix.mulVec_smul] using h'
  · intro h
    have h' := congrArg (fun y ↦ (massInvSqrtMatrix m).mulVec y) h
    simpa [dynamicMatrix, Matrix.mulVec_mulVec, ← Matrix.mul_assoc,
      massInvSqrt_mul_massSqrt, Matrix.mulVec_smul] using h'

/-- The right side of the normalized equation is the canonical mass action on `u=M⁻¹ᐟ²x`. -/
theorem generalized_mass_action (m : PositiveMasses) (x : Fin 3 → ℂ) :
    (massMatrix m).mulVec ((massInvSqrtMatrix m).mulVec x) =
      (massSqrtMatrix m).mulVec x := by
  have h := congrArg (fun M : Matrix (Fin 3) (Fin 3) ℂ ↦ M.mulVec x)
    (mass_mul_massInvSqrt m)
  simpa [Matrix.mulVec_mulVec] using h

/-- Canonical signed form `F u = -ω² M u` under `u=M⁻¹ᐟ²x` (BN-TRI-GENERALIZED). -/
theorem dynamic_eigen_iff_signed_generalized {P : ℕ} (g : Fin P → Interaction)
    (Q eig : ℝ) (m : PositiveMasses) (x : Fin 3 → ℂ) :
    (dynamicMatrix g Q m).mulVec x = eig • x ↔
      (forceMatrix g Q).mulVec ((massInvSqrtMatrix m).mulVec x) =
        -(eig • (massMatrix m).mulVec ((massInvSqrtMatrix m).mulVec x)) := by
  rw [dynamic_eigen_iff_generalized]
  rw [generalized_mass_action]
  simp only [Matrix.neg_mulVec]
  exact neg_eq_iff_eq_neg

/-- The triatomic force matrix is Hermitian (BN-TRI-HERMITIAN). -/
theorem forceMatrix_isHermitian {P : ℕ} (g : Fin P → Interaction) (Q : ℝ) :
    (forceMatrix g Q).IsHermitian := by
  apply Matrix.IsHermitian.ext
  intro i j
  fin_cases i <;> fin_cases j <;>
    simp [forceMatrix, A_conj, phase_mul_C, phase_mul_B]

/-- At `Q=0`, `A=-2 Kcross` (BN-TRI-ABC). -/
theorem A_zero {P : ℕ} (g : Fin P → Interaction) :
    A g 0 = -2 * crossCoupling g := by
  simp [A, crossCoupling, ← Complex.ofReal_sum]
  rw [← Finset.sum_mul]
  push_cast
  ring

/-- At `Q=0`, `B=Kcross` (BN-TRI-ABC). -/
theorem B_zero {P : ℕ} (g : Fin P → Interaction) :
    B g 0 = crossCoupling g := by
  rw [B, crossCoupling, Finset.sum_add_distrib]
  simp [← Complex.ofReal_sum]

/-- At `Q=0`, `C=Kcross` (BN-TRI-ABC). -/
theorem C_zero {P : ℕ} (g : Fin P → Interaction) :
    C g 0 = crossCoupling g := by
  rw [C, crossCoupling, Finset.sum_add_distrib]
  simp [← Complex.ofReal_sum]

/-- The all-ones rigid vector lies in the zone-center force-matrix kernel (BN-TRI-FORCE). -/
theorem rigid_vector_kernel {P : ℕ} (g : Fin P → Interaction) :
    (forceMatrix g 0).mulVec (fun _ ↦ 1) = 0 := by
  funext i
  fin_cases i <;>
    simp [Matrix.mulVec, dotProduct, forceMatrix, A_zero, B_zero, C_zero, Fin.sum_univ_three] <;>
    ring

/-- The grouped family in which only the nearest-neighbor stiffness `k₁` is nonzero. -/
def nearestNeighborInteraction (k₁ : ℝ) : Fin 1 → Interaction :=
  ![⟨k₁, 0, 0⟩]

/-- Nearest-neighbor specialization `A=-2k₁` (BN-TRI-ABC). -/
theorem nearestNeighbor_A (k₁ Q : ℝ) :
    A (nearestNeighborInteraction k₁) Q = -2 * k₁ := by
  simp [A, nearestNeighborInteraction]

/-- Nearest-neighbor specialization `B=k₁` (BN-TRI-ABC). -/
theorem nearestNeighbor_B (k₁ Q : ℝ) :
    B (nearestNeighborInteraction k₁) Q = k₁ := by
  simp [B, nearestNeighborInteraction]

/-- Nearest-neighbor specialization `C=k₁e⁻ⁱQ` (BN-TRI-ABC). -/
theorem nearestNeighbor_C (k₁ Q : ℝ) :
    C (nearestNeighborInteraction k₁) Q = k₁ * phase (-1) Q := by
  simp [C, nearestNeighborInteraction]

/-- Standard nearest-neighbor three-site-cell force matrix. -/
theorem nearestNeighbor_forceMatrix (k₁ Q : ℝ) :
    forceMatrix (nearestNeighborInteraction k₁) Q =
      !![((-2 * k₁ : ℝ) : ℂ), k₁, k₁ * phase (-1) Q;
         k₁, -2 * k₁, k₁;
         k₁ * phase 1 Q, k₁, -2 * k₁] := by
  have hp : phase 1 Q * phase (-1) Q = 1 := by
    rw [phase_add]
    simp
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [forceMatrix, nearestNeighbor_A, nearestNeighbor_B, nearestNeighbor_C]
  all_goals
    first
    | linear_combination (k₁ : ℂ) * hp
    | linear_combination -(k₁ : ℂ) * hp
    | ring

/-- The grouped family for exactly five interactions, padded by a zero sixth coupling (BN-TRI-FIVE). -/
def fiveInteractions (k₁ k₂ k₃ k₄ k₅ : ℝ) : Fin 2 → Interaction
  := ![⟨k₁, k₂, k₃⟩, ⟨k₄, k₅, 0⟩]

/-- Exact five-interaction expansion of `A(q)` (BN-TRI-FIVE). -/
theorem five_A (k₁ k₂ k₃ k₄ k₅ Q : ℝ) :
    A (fiveInteractions k₁ k₂ k₃ k₄ k₅) Q =
      -2 * ((k₁ : ℂ) + k₂ + k₃ + k₄ + k₅) + k₃ * (phase 1 Q + phase (-1) Q) := by
  simp [A, fiveInteractions, Fin.sum_univ_two]
  ring

/-- Exact five-interaction expansion of `B(q)` (BN-TRI-FIVE). -/
theorem five_B (k₁ k₂ k₃ k₄ k₅ Q : ℝ) :
    B (fiveInteractions k₁ k₂ k₃ k₄ k₅) Q =
      k₁ + k₂ * phase (-1) Q + k₄ * phase 1 Q + k₅ * phase (-2) Q := by
  simp [B, fiveInteractions, Fin.sum_univ_two]
  ring

/-- Exact five-interaction expansion of `C(q)` (BN-TRI-FIVE). -/
theorem five_C (k₁ k₂ k₃ k₄ k₅ Q : ℝ) :
    C (fiveInteractions k₁ k₂ k₃ k₄ k₅) Q =
      k₁ * phase (-1) Q + k₂ + k₄ * phase (-2) Q + k₅ * phase 1 Q := by
  simp [C, fiveInteractions, Fin.sum_univ_two]
  ring

end
end BandNet.Triatomic
