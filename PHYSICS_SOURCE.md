# Physics Source

The single editable source for BandNet's adopted one-dimensional equations and
derivations is `BandNet-Docs/derivations/source`. This repository contains a
byte-for-byte mirror under [generated/derivations](generated/derivations/). Its
manifest records the SHA-256 digest of each canonical file. Do not edit the
mirror directly.

The implementation aligned with the canonical triatomic formulas is
`triatomic_genuine_formula.py`. Its tests compare the grouped `BN-TRI-ABC`
construction with the direct-neighbor equation and the explicit
`BN-TRI-FIVE` specialization.

`triatomic_batched.py` is an independent CPU candidate. It assembles positive
stiffness from physical neighbor-bond energies, mass-normalizes it, and uses
batched Hermitian eigenvalue solves. Its bounded numerical comparisons and
known-limit tests are in `test_triatomic_batched.py`. It does not replace the
reference as the correctness authority and is not connected to historical
training or inference. Local benchmark evidence and the near-zero numerical
acceptance clarification live in the workspace documentation's
`DATA_GENERATION_PERFORMANCE.md`.

`test_triatomic_interactions.py` extends the independent candidate's checks
through 20 physical interactions. `triatomic_execution.py` and
`generation_resources.py` select CPU execution settings using the explicit
reserves in `generation_policy.py` and preserve
input/output order without changing the numerical solver. The default native
500-point grid meets the strict frequency comparison in the expanded checks;
additional near-zero diagnostic limitations are recorded explicitly in the
performance document rather than changing the canonical physics.

The triatomic branches in `Core.py` and `inference_set.py` are historical code.
They contain the documented zero-based group-index defect and do not implement
the canonical triatomic derivation. They remain only to reproduce and audit the
historical workflow. New scientific results must use an implementation validated
against `triatomic_genuine_formula.py` and the canonical source.

The two-dimensional implementation is also historical. The 2D physical model is
open and is not part of the canonical or formally verified specification.
