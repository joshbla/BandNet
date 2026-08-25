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

The triatomic branches in `Core.py` and `inference_set.py` are historical code.
They contain the documented zero-based group-index defect and do not implement
the canonical triatomic derivation. They remain only to reproduce and audit the
historical workflow. New scientific results must use an implementation validated
against `triatomic_genuine_formula.py` and the canonical source.

The two-dimensional implementation is also historical. The 2D physical model is
open and is not part of the canonical or formally verified specification.
