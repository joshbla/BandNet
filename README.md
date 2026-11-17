# BandNet

Code for the paper "Neural-network inverse design of multiband dispersion in
nonlocal phononic crystals" (Smith, Deshmukh, Paul and Wang). A multilayer
perceptron maps sampled dispersion relations of nonlocal spring-mass lattices
to their masses and spring stiffnesses.

## Paper Release

- **One-band, two-band and 2D models:** `Core.py`, `1Band_2D.py`,
  `inference_set.py`, `inference_single_2D.py` and the plotting scripts are the
  historical implementation. The exact code version that produced the paper's
  one-band, two-band and 2D results was not preserved, so these scripts are not
  guaranteed to reproduce those numbers exactly. Their triatomic branches are
  superseded; see [PHYSICS_SOURCE.md](PHYSICS_SOURCE.md).
- **Corrected three-band study:** `triatomic_genuine_formula.py` is the
  reference forward model, `triatomic_batched.py`, `triatomic_data.py` and
  `triatomic_execution.py` generate data, `triatomic_learning.py` trains and
  evaluates, `score_completed_fits.py` scores saved fits, and
  `three_band_paper.py` rebuilds the paper's three-band figures and values from
  the evaluation records. The trained models and evaluation records are not
  included in this repository.
- **Equations:** [generated/derivations](generated/derivations/) mirrors the
  canonical one-dimensional derivations.
- **Formal verification:** [generated/lean](generated/lean/) is a Lean 4
  project that verifies algebraic properties of the one-dimensional dispersion
  relations. It does not cover floating-point computation, the neural networks
  or the 2D model; its README states the exact boundary.

### Computational Record

Hardware and run times are recorded here rather than in the paper. Generation
and training times measure offline work, not per-target inference.

| Model | Hardware | Training data | Epochs | Generation (s) | Training (s) |
| --- | --- | --- | --- | --- | --- |
| 1D monoatomic M5 | Nvidia RTX 3070, Intel Core i7-12700 | 100,000 | 100 | 77 | 514 |
| 1D diatomic M5 | Nvidia RTX 3070, Intel Core i7-12700 | 1,000,000 | not recorded | 752.58 | 5,101.22 |
| 2D monoatomic M5 | Nvidia RTX 3070, Intel Core i7-12700 | 100,000 | 100 | 63.49 | 454.07 |
| 1D triatomic main M5 | Nvidia H100 (cloud) | 2,500,000 | 5 | 2,007.05 | 893.58 |

The one-band, two-band and 2D times were reported during the original work;
their run records were not retained, so they cannot be checked against logs.
The triatomic generation time covers generating, writing and hashing the
training curves, and its training time excludes the validation after each
epoch.

## Environment

BandNet uses [UV](https://docs.astral.sh/uv/) with Python 3.12. Create the
locked lightweight environment used by the genuine-formula implementation and
its tests with:

```bash
uv sync --frozen
```

The lightweight dependencies are NumPy plus psutil for resource discovery and
threadpoolctl for supported native CPU thread-pool control. Corrected training
and figure export use the `training` extra:

```bash
uv sync --frozen --extra training
```

The historical training, plotting, and inference scripts require the larger
legacy dependency set:

```bash
uv sync --frozen --extra legacy
```

## Validation

Run the reference and independent batched-solver validation suites with:

```bash
uv run --frozen python -m unittest discover -v
```

The lightweight suite writes small temporary fixtures and arrays. With the
`training` extra it also runs tiny corrected training/checkpoint/inference tests;
it never invokes `Core.py`. Run the complete suite with:

```bash
uv run --frozen --extra training python -m unittest discover -v
```

Without PyTorch the corrected learning tests are explicitly skipped.

## Cloud Runs

The corrected three-band study was trained on rented cloud GPUs.
`production_launcher.ts`, `production_pod.ts`, `production_common.ts`,
`runpod_watchdog.ts` and `scoring_restore.ts` are the scripts used to launch,
monitor and save those runs. They require Node 22.18 or newer and local
account settings in `.env.local`; `.env.local.example` lists the settings.
