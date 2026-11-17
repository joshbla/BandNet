# Lean Verification

This project verifies finite algebraic claims from the canonical BandNet
derivations. The theorem docstrings cite stable `BN-*` formula identifiers from
the canonical `equations.tex`, which the code repository mirrors in
`generated/derivations/`.

Run:

```text
lake exe cache get
lake build
```

The local `.lake/` build cache (about 7.5 GB, almost all Mathlib) is not kept
in the repository. `lake exe cache get` downloads Mathlib's prebuilt files for
the pinned version before the build; without it, `lake build` compiles Mathlib
from source.

The accepted proof boundary includes monoatomic zone-center behavior, the
diatomic determinant and branch algebra, triatomic Hermitian identities,
spring-energy positivity, mass-normalized dynamic-matrix properties, the
zone-center rigid vector, and the explicit five-interaction expansion. It does
not include empirical results, floating-point correctness, neural-network
results, the unresolved 2D model, a line-by-line encoding of the infinite-chain
equations of motion or their assembly into Bloch matrices, a sorted enumeration of the three triatomic eigenvalues,
or an explicit construction of the equal-mass zone-folded eigenvectors.

Completed formalization must compile without `sorry`, `admit`, or additional
axioms.
