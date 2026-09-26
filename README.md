# BandNet

## Environment

BandNet uses [UV](https://docs.astral.sh/uv/) with Python 3.12. Create the
locked lightweight environment used by the genuine-formula implementation and
its tests with:

```bash
uv sync --frozen
```

The lightweight dependencies are NumPy plus psutil for resource discovery and
threadpoolctl for supported native CPU thread-pool control. Training dependencies
remain in the separate legacy extra.

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

The suite does not generate datasets, train models, or invoke `Core.py`.
It writes small temporary fixtures and arrays for resource-policy and output checks.

## Batched Three-Band CPU Candidate

`triatomic_batched.py` constructs stiffness from physical neighbor bonds and
solves Hermitian matrices in batches. It does not call the reference solver or
the historical training/generation path. `TriatomicBatchSolver` takes a fixed
normalized wave-number grid and physical interaction count. `evaluate` accepts
mass rows of shape `(N, 3)` and stiffness rows of shape `(N, K)` and returns
ascending float64 frequencies of shape `(N, Q, 3)` plus numerical health counts.
`iter_batches(..., chunk_size=128)` bounds temporary matrix storage and yields
each chunk with its first row index, allowing direct streaming to disk.

The candidate was compared with the independent reference on seeded dense,
sparse and difficult cases. Near acoustic zero, squared-frequency agreement is
checked because taking a square root magnifies eigenvalue roundoff. The exact
acceptance policy and benchmark evidence are in the workspace documentation's
`DATA_GENERATION_PERFORMANCE.md`. This is not yet an integrated training pipeline.

## Fixed-M5 Baseline Benchmark

Create `code/.env.local` using `.env.local.example` and set its three script
controls explicitly. The benchmark reads that local file, not process environment
variables. The checked-in example uses 10,000 curves, chunks of 128 and three
repetitions. The bounded scaling run uses 100,000 curves and one repetition.

From `code/`, in the locked environment:

```bash
uv run --frozen python benchmark_triatomic.py --report /absolute/existing/directory/new-report.json
```

The report path must be new and its parent directory must exist. The benchmark
runs correctness checks, a same-input reference timing, compute-only trials and
streamed-output trials. It records configuration, source hashes, software,
hardware, numerical errors, timings and process-lifetime peak RSS. It writes
temporary frequency-only float64 NPY files, checks mmap access, then removes
them. This format and its bounded input distribution are benchmark choices, not
final training decisions. It neither trains nor loads a model.

## Resource-Calibrated Benchmark and RunPod Preparation

Use `benchmark_triatomic_resources.py` for variable interaction counts and
automatic CPU execution selection. The `BANDNET_RESOURCE_BENCHMARK_*` section of
`.env.local.example` supplies the three required local controls: interaction
counts, curve count and repetitions. Batch size and worker count are chosen
automatically, rather than copied from the Mac's baseline settings.

Also set all three required `BANDNET_GENERATION_*` policy values in `.env.local`:
CPU reserve (logical CPU equivalents), RAM reserve (MiB of additional available
system memory), and startup tuning seconds. `generation_policy.py` reads these
explicit values without process-environment substitutions or missing-value
defaults. The example's one CPU, 1024 MiB and five seconds are local verification
settings, not approved RunPod production reserves. A one-CPU allocation with a
one-CPU reserve fails explicitly; choose appropriate reserves for the actual pod.

```bash
uv run --frozen python benchmark_triatomic_resources.py --report /absolute/existing/directory/new-resource-report.json
```

The runner discovers host available memory and CPU count. On Linux it also
intersects process CPU affinity with visible cgroup v1/v2 quotas and memory
headroom, walking visible parents. It subtracts the explicit CPU/RAM reserves,
then builds a geometric candidate search within the remaining resources and
calibration workload. There is no permanent eight-worker or 512 MiB ceiling.
With a controllable native BLAS library it considers worker counts through the
remaining CPU budget, including a non-power-of-two maximum. Chunk candidates
grow with available memory and workload rather than stopping at 512 rows.
Each BLAS worker is limited to one native thread to
avoid nested oversubscription. It chooses the smallest worker/memory footprint
within 5% of the fastest measured median on at most 1,024 representative examples.
Probe storage has its own allowance. The search tries serial and high-concurrency
settings early, checks the elapsed-time budget between candidates/trials and
reports how much of the feasible search was measured. An in-flight NumPy call
is not interrupted, so this is a soft deadline; overruns and incomplete searches
are explicit in the report. A selected plan must have an actual timed trial.

Startup calibration has memory and time costs, included separately in the
report. Resource budgets and reserves are rechecked before execution; the runner
stops on inadequate headroom or unavailable limits rather than catching an OOM
and silently switching settings. This is a startup snapshot and conservative
workspace estimate, not OS resource isolation, an OOM guarantee or proof of a
global speed optimum. It uses extra resources when measurement supports them,
not merely to fill RAM or force high utilization.

Apple Accelerate does not expose a pool to this installed threadpoolctl, so the
Mac branch keeps one outer worker and explicitly reports native threads as
backend-managed; it cannot enforce a native-thread CPU reserve on Accelerate.
Linux execution requires a detected controllable NumPy BLAS backend.
An unresolvable Linux cgroup membership or unsupported backend is an explicit
error. Ancestor limits hidden outside a container's cgroup namespace cannot be
discovered; verify the report against the actual pod allocation before scaling.

The low-level solver still accepts an explicit caller batch. For new application
integration, call `triatomic_execution.tune_execution(solver, masses, springs, policy)`
once per machine/workload
and use its plan with `execution_batches`, consuming or closing the iterator so
thread pools and native limits are released. Do not reuse a Mac calibration as
a RunPod calibration. The historical `Core.py` training path is not connected to
this corrected generator, and the new runner does not detect or use CUDA GPUs.

Validation covers dense/sparse native-grid cases for 1-20 interactions, individual
neighbor responses and padding through 20, and resource/ordering/failure cases.
The wider near-zero diagnostic set also records where the earlier frequency
tolerance does not hold despite squared-frequency agreement; see the numerical
qualification in `docs/DATA_GENERATION_PERFORMANCE.md` in the workspace.
Linux resource fixtures and threaded correctness tests are not a live RunPod
benchmark. Run this entry point and the tests on the selected pod before using
its measurements for a corrected training run.
