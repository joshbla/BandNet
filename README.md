# BandNet

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
it never invokes `Core.py`. Run all 52 checks with:

```bash
uv run --frozen --extra training python -m unittest discover -v
```

Without PyTorch the eight corrected learning tests are explicitly skipped.

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
`DATA_GENERATION_PERFORMANCE.md`. The separate corrected pilot below now connects
this solver to durable labels, batch-backed training and consistent evaluation.

## Corrected Local Training Pilot

The owner adopted bounded mass ratios `[0.1,10]`, passive spring ratios `[0,10]`,
fixed `m1=k1=1`, and band-based loss, and retained the M5-M20 study. The first
local integration run uses M5 only. Configure every `BANDNET_PILOT_*` and
`BANDNET_GENERATION_*` value in `.env.local`; the example defines a 1,024-example,
five-epoch CPU pilot. These commands read that file, not process environment
variables. Create an artifact parent directory first, then choose a new run path:

```bash
uv run --frozen --extra training python corrected_pilot.py --output artifacts/corrected-m5-pilot
```

The runner refuses to overwrite a run. Its output includes:

- Immutable source snapshots, an explicit protocol, preflight and calibration
  reports, and file SHA-256 identities.
- Float64 frequency arrays `(N,500,3)`, float64 generating labels `(N,K+1)`,
  one shared grid, and a manifest with chunk checksums and progress. Label order
  is `m2,m3,k2,...,kK`. Five-interaction showcase vectors retain their historical
  identities; higher-count versions pad the extra physical springs with zeros.
- Independent seeded train, dense/sparse validation and dense/sparse test
  populations, plus separate boundary and showcase populations. Training uses
  an exact half dense/half independently masked mixture. A masked example can
  happen to have no zeros. Masses and active springs use continuous uniforms;
  upper endpoints are covered by boundary cases rather than random draws.
- A small float64 network with two 128-wide tanh hidden layers, bounded sigmoid
  decoding, training-only per-band input scales, Adam, validation-selected
  checkpoints, and separate generation/training/evaluation timings.
- Full-precision predicted labels, reconstructed bands, per-band scores and
  separate parameter diagnostics, plus four-panel and validation-history figures.

The loss averages **squared** target-RMS-normalized band errors. Checkpoint
selection and reported primary scores average per-band normalized **RMSE**, with
equal band and example weights. No generating-label penalty is applied. This is
a new architecture/objective, not a historical checkpoint continuation. Sigmoid
decoding usually predicts interior springs, so zero-spring targets can be
approximated without claiming exact zero-support recovery. Failed designs remain
counted and make the population primary score undefined; they are not dropped.

`triatomic_data.generate_artifact(..., resume=True)` explicitly resumes an
interrupted dataset with identical source/configuration, verifying completed
prefixes before rewriting uncommitted rows. The command above creates new runs;
training resume is not implemented. `LabeledArtifact` rejects incomplete or
checksum-invalid datasets and reads dense frequency batches from read-only mmap
arrays. Parameter arrays and permutation indices remain pilot-sized in RAM.

### Standalone Corrected Inference

Load the saved model in a new process and supply the actual target grid:

```bash
uv run --frozen --extra training python corrected_inference.py --checkpoint artifacts/corrected-m5-pilot/training/best.pt --targets artifacts/corrected-m5-pilot/data/showcase.bands.npy --grid artifacts/corrected-m5-pilot/data/q_hat.npy --output artifacts/corrected-m5-pilot/standalone-showcase
```

This command streams output arrays and uses the same decoder, corrected CPU
forward solver and target-normalized scorer as pilot evaluation. Frequency-only
targets can be shared across models with different interaction counts, supporting
the later M5-M20 study. That study has not been executed by this pilot.

After standalone showcase inference, verify the retained files, source snapshots,
checkpoint identity, all final predictions/scores and integrated/standalone
agreement. The report path must be new in an existing directory:

```bash
uv run --frozen --extra training python verify_corrected_pilot.py --run artifacts/corrected-m5-pilot --report /absolute/existing/directory/corrected-pilot-audit.json
```

The audit exports the original raw run report together with verification evidence.
The numerical arrays, checkpoint and source snapshots remain under the run path.
Local `artifacts/` is Git-ignored. The workspace documentation retains the initial
pilot report and its limitations in `THREE_BAND_RERUN_MAP.md`. Neither command
starts cloud compute or determines appropriate RunPod reserve values.

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
