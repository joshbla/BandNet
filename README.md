# BandNet

## Environment

BandNet uses [UV](https://docs.astral.sh/uv/) with Python 3.12. Create the
locked lightweight environment used by the genuine-formula implementation and
its tests with:

```bash
uv sync --frozen
```

The historical training, plotting, and inference scripts require the larger
legacy dependency set:

```bash
uv sync --frozen --extra legacy
```

## Validation

Run the deterministic genuine-formula validation suite with:

```bash
uv run --frozen python -m unittest discover -v
```

The suite does not generate datasets, train models, or invoke `Core.py`.
