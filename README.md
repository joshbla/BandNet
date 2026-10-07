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
it never invokes `Core.py`. Run the complete suite with:

```bash
uv run --frozen --extra training python -m unittest discover -v
```

Without PyTorch the corrected learning tests are explicitly skipped.

## Full Corrected Production Experiment

### Current State: GPU Recheck Passed, Cleanup Complete

The subsequent approved ten-minute GPU recheck passed at candidate `0.0001`, as
did the on-pod independent audit. The bulk export hit its 150-second transfer
timeout. The owner approved three-minute retrieval, but the restart was refused
because the host had no free GPU. The owner then explicitly accepted incomplete
M5 export and authorized deletion. Pod `sdovg9ufwm1nzb` and its 200 GB storage were
deleted; live pod and network-volume inventories are empty. Full Mac checkpoint
auditing was not completed; the successful on-pod audit and exported evidence
remain retained with that limitation. The measured-rate projection is
$24.84-$27.78, or $34.73 with the explicit margin, plus setup/export. See
`../docs/THREE_BAND_RERUN_MAP.md`, Ten-Minute Candidate GPU Recheck.

The disposable H100 timing test and approved local learning/saving follow-up are
complete. Work has stopped at the owner's requested boundary. Exactly 35 full-width
local updates reproduced M5 saturation at Adam `0.001` and checked the single
candidate `0.0001` across the timing seed, two fixed M5 production seeds and M20.
All four candidate witnesses improved with varied predictions and live gradients;
this does not establish production accuracy or GPU learning. The real experiment
starts fresh only after owner approval. Test weights/data never start production.

Production protocol `corrected-full-three-band-v3` adopts Adam `0.0001` after
the short CPU/GPU checks. Production preflight v4 probes that same rate.
Short learning witnesses do not establish long-run accuracy or authorize production.
The cross-machine procedures below remain reference material, not the active task.

For a future explicitly approved GPU check, use a container-disk software environment/cache
and a credential-free `.env.local` with the documented controls. The timing mode
replaces CPU/RAM reserve and thread settings with choices for the actual machine:

```bash
python corrected_pilot.py --timing-check --output /workspace/artifacts/disposable-timing
python verify_corrected_pilot.py --timing --run /workspace/artifacts/disposable-timing --report /workspace/artifacts/timing-audit.json
```

Timing report `corrected-timing-check-v3` measures every M5..M20 count: two warmups,
five timed endpoint updates or three intermediate updates, full 4,096-row validation,
and a full recovery save per count. Numerical/reload checks remain at M5/M20.
Generation tuning runs at M5/M20; M6-M19 reuse the M5 plan to fit the allocation,
and the auditor charges them the slower endpoint's tuning time.
Within the 560-second cooperative limit, the remaining budget funds at most
120 seconds of main-M5 slice generation/read measurement, reserving 20 seconds for
finalization. `BANDNET_TIMING_IO_ROWS=250000` requests about 3 GB of real float64
bands; the report records any time-based reduction. One complete shuffled epoch
uses production mmap indexing, host copying and device transfer, without updates.
Page-cache eviction is advisory and never reported as guaranteed cold storage.

Export `report.json`, `report.sha256`, `sources/`, all `m*-timing-records/` and the
independent audit only. Exclude `bulk/`, which contains disposable checkpoints and
datasets. Each small evidence file is checksummed; `report.sha256` covers the report.
The portable audit checks sampled scores against the independent physics reference,
not checkpoint prediction reexecution. Historical timing v1/v2 audits remain supported.
Pass `--hourly-price` with the actual allocation price to the auditor for seconds,
hours, cost and an explicit 25-percent margin. Without that input it reports no cost.
All 17 fits use their measured count-specific rates plus measured shuffled-read
bandwidth. Reads are added conservatively to cached-read-inclusive updates, so some
I/O is counted twice. Setup/export and long-run learning remain unmeasured.

Use the Python executable from the frozen training environment. The first H100
attempt failed before timing on excessive CUDA eigensolver scratch allocation.
Evidence was exported and the pod/storage removed. The VRAM-bounded eigensolver
workaround has now passed actual H100 M5/M20 numerical, gradient, full-size update,
reload and CPU-scoring checks. Independent timing audits passed on the pod and
again on the Mac. The old schedule projected to 10.83-12.25 hours/$38-$43. The
implemented lean schedule reduces recovery saves from 1,967 to 52 and projects
to 6.81-8.17 hours/$23.97-$28.77 at those historical rates, or $35.96 with the
explicit 25-percent margin, plus setup/export. This is a schedule-only estimate,
not a measurement of the changed code, and still exceeds the owner's $11 ceiling.
See `../docs/THREE_BAND_RERUN_MAP.md`, Bounded Learning Diagnosis And Lean Recovery
Saving, for the diagnosis, source hashes, limitations and saved evidence.
Both retry pods/storage were removed. Current local verification: 78 Python tests
and nine shutdown-controller tests pass. No GPU timing v3 measurement was run locally.

The fixed CUDA runner restores the recorded five-hidden-layer ReLU architecture
(`5000,2500,5000,2500,5000`) with the adopted corrected contract: 1,500 scaled
frequency inputs, bounded physical outputs, float64 network/physics, and band
loss. M5 has 57,550,006 trainable parameters. The input q column is redundant on
the fixed grid and remains excluded. Adam uses the adopted production rate
`0.0001`, without the historical parameter-loss scheduler. This is a corrected
retraining, not historical checkpoint continuation.

The matrix is main M5 at 2,500,000 examples/five epochs plus M5-M20 at 100,000
examples/100 epochs each. All fits use batch size 1,024, independent 2,048-case
dense and sparse validation, and validation-only checkpoint selection. The main
final populations have 125,000 examples each; study own-count tests have 5,000
each. The common 20,000-target M5 set is 10,000 dense plus 10,000 sparse, stored
once with its own seed. All fits finish before any final population is scored.
No pilot data or checkpoints are reused.

**Paid launch requires owner approval.** CUDA execution and full-size throughput
were measured on the disposable H100; production learning and budget remain open.
Set `BANDNET_PRODUCTION_*` and the actual pod's `BANDNET_GENERATION_*` reserves
in local `.env.local`. There is no CPU substitution when CUDA is unavailable.
Use the locked environment on a host compatible with its CUDA 13 runtime.

On the approved GPU, from `code/`, with an existing persistent artifact parent:

```bash
uv run --frozen --extra training python -m unittest discover -v
uv run --frozen --extra training python corrected_pilot.py --gpu-preflight --output /workspace/artifacts/gpu-preflight
```

Preflight checks M5/M20 values, gradients, full-architecture optimizer updates,
CPU-scored predictions, CUDA/CPU checkpoint reload, resources and native CPU
thread control. New preflight v4 also requires learning progress, varied predictions
and live gradients; timing v3 reports the same learning evidence separately from
numerical/timing correctness. Failed learning stops before large checkpoint export
or the next endpoint, retaining diagnostic observations. It measures two warmup and five timed updates per endpoint with
batch size 1,024, full validation and checkpoint-writing/integrity-hashing overhead.
It also verifies serialized Adam continuation with two additional updates per
endpoint, separately from the throughput samples. This is not
an architecture or accuracy search. Its training-only projection excludes the
remaining stages; use its recorded stage timings to form the full-run budget.

After approving that measured production budget:

```bash
uv run --frozen --extra training python corrected_pilot.py --production --preflight /workspace/artifacts/gpu-preflight --output /workspace/artifacts/full-three-band
uv run --frozen --extra training python corrected_pilot.py --production --resume --preflight /workspace/artifacts/gpu-preflight --output /workspace/artifacts/full-three-band
uv run --frozen --extra training python verify_corrected_pilot.py --production --run /workspace/artifacts/full-three-band --report /workspace/artifacts/full-three-band-audit.json
```

The second command is for an interrupted run, not a second experiment. It checks
the frozen scientific source/configuration and artifact identities against a
successful preflight for the **current** machine. A replacement GPU, runtime,
region or filesystem path does not change the scientific protocol.
Training persists Adam state, deterministic row cursor and best-validation weights
initially, at the explicit interval, after terminal validation, and on deadline.
The approved cadence is `BANDNET_PRODUCTION_CHECKPOINT_STEPS=5000` in `.env.local`.
Every epoch still validates and selects best weights in memory. A crash can replay
roughly six-to-nine minutes at prior rates, including best improvements not yet
saved. The final recovery save pairs current weights with current Adam, keeping
selected-best weights separate. Lightweight history is reconciled to the durable
cursor on reopening. Uncommitted duration is not filled in. `MAX_SECONDS` is a cooperative
per-invocation limit, **not a RunPod billing stop**. An external pod stop deadline
must be armed for the approved budget. A changed GPU/driver/runtime or operational
configuration requires a fresh preflight; its successful report is admitted and
archived automatically during explicit resume.

Final records store predicted parameters, per-band errors and chunk checksums;
they reference retained target arrays instead of duplicating reconstructed curves.
The independent audit replays all predictions, reconstructs every design with the
CPU solver, checks checkpoint validation selection and verifies summaries. Final
figures and manuscript replacements follow that audit. The runner requires
150 GB initial free space and keeps a 20 GB disk reserve; a 200 GB persistent
network volume accommodates about 55.55 GB of dataset payload and 39.16 GB of
retained model/Adam payload plus preflight, temporary checkpoints and software.

### Moving Between Machines Or Regions

There is no A100-only, 80-GB-only or region whitelist in the runner. A compatible
CUDA GPU must execute the fixed float64 model and batch size and pass the numerical,
resource and throughput checks. CPU threads, generation reserves/tuning and
checkpoint interval are execution settings that can be chosen for each machine
in that checkout's `.env.local`. The architecture, sampling, seeds, learning rate,
epochs, batch size, precision, source files and lockfile remain fixed.

1. Pause the old writer and preserve the **entire** production directory, including
   datasets, checkpoints, `protocol.json`, `progress.json`, `sources/`, `executions/`,
   per-fit histories and partial evaluation records. Copy a quiescent snapshot,
   not files while another pod is still changing them. Verify the exported copy
   before deleting old storage. Network volumes are region-bound; cross-region
   moves require an actual copy through durable storage or the Mac.
2. Use the same source/lockfile contents on the new machine. The scientific
   identity is content-based; commit, platform, package/runtime versions and GPU
   identity are retained separately in execution evidence. Set the new machine's
   explicit operating limits, then create a **new** preflight output directory.
   Do not use an old GPU's report or price projection for a different GPU.
3. Run the first command below for the new qualification. Only after its measured
   budget is approved, use the second command to continue the copied run:

```bash
uv run --frozen --extra training python corrected_pilot.py --gpu-preflight --output /workspace/artifacts/replacement-preflight
uv run --frozen --extra training python corrected_pilot.py --production --resume --preflight /workspace/artifacts/replacement-preflight --output /workspace/artifacts/full-three-band
```

Resume checks the saved model/Adam/cursor checksum and source/data identity. Before
updating transferred weights, it checks predictions against a saved training-only
probe on both CPU and the new device, checks common CPU scores at `rtol=atol=1e-10`,
and checks finite gradients. Selected models get the same prediction/score check
before final inference. This establishes numerical agreement, not a promise of
bit-identical training trajectories across different hardware.

Every production invocation archives its admitted preflight report under
`executions/` and records its actual controls, reserves and resources. Generation
chunks, committed optimizer steps, validation selections and evaluation chunks
carry execution IDs. Completed evaluation records and originating summaries are
preserved on reopen. The independent audit verifies this lineage, final predictions,
selection and scores; aggregate floating-point comparisons use the existing
`1e-12` score tolerance while counts and provenance remain exact.

The archived preflight reports are self-contained historical qualification records.
The current invocation still verifies every file in its supplied preflight folder;
keep that complete folder until admission. Copying a production run does not
require reproducing an old absolute path or retaining a physical GPU. Source
snapshots remain in the run; disposable preflight models/data need not be copied
into every production execution archive.

This is the v2 production/checkpoint/evaluation format. There are no real v1
production runs to migrate; legacy production fixtures/preflights must be rebuilt,
not silently accepted as qualified. Existing local pilot/sizing evidence remains
historical. All **65 local Python tests pass**, including a three-machine
interruption/relocation fixture, saved-state corruption checks and the complete
preflight flow with a tiny CPU backend. Actual CUDA and cross-GPU qualification
remain required on rented hardware.

### Initial RunPod Allocation And Independent Shutdown

The next disposable allocation has an owner-set **15-minute maximum**, counted from the
allocation request, including provisioning, setup, checks, evidence and shutdown.
Use at most **560 seconds** for the inner timing-runner limit and reduce it
when setup leaves less time for audit/export. The allocation and temporary storage count
toward the same **$11 total ceiling**. Neither this procedure nor the example
configuration authorizes paid creation.

`runpod_watchdog.ts` is a stop-only controller running on this Mac, independently
of Python, SSH and the chat. It uses the documented RunPod v2 API and Node
22.18 or newer, with no additional packages. Put `RUNPOD_API_KEY` in the scripts
section at the bottom of local `.env.local`, make that file owner-only, and run:

```bash
node runpod_watchdog.ts check
node --test test_runpod_watchdog.ts
```

`check` authenticates a read and verifies that the prepared Mac public SSH key
is registered. A read cannot establish stop permission; verify that through the
approved live allocation. Never upload the controller's local `.env.local` to a
pod. Build a separate pod configuration containing only the experiment controls.

For a newly approved creation, capture the local request time in epoch
milliseconds immediately **before** issuing the create request. Immediately
after creation, preserve a JSON receipt under ignored `artifacts/` containing
only `podId`, the exact `createdAt` returned by that creation, and
`allocationRequestedAtMs`. Only this conversation's actual creation response
establishes permission to control the pod. A name or an existing pod-list entry
does not. The receipt is operational evidence, not an authorization mechanism.

```bash
node runpod_watchdog.ts arm artifacts/runpod-check-receipt.json
```

Do this as the first action after creation, before SSH/setup, and require the
arming acknowledgement. If receipt binding or arming fails, stop that new pod
immediately through the MCP. The controller is not protecting the interval
between the create request and successful arming, so a launch must not be left
unattended during this handoff. Authentication and local controller checks must
be complete before requesting a paid allocation.

The detached controller uses macOS `caffeinate -is` to inhibit idle sleep while
it runs. It sends the normal check's stop request at **840 seconds** of its 900-second allocation, leaving 60 seconds for
provider shutdown, and retries API failures. A monotonic clock prevents wall-clock
rollback from extending the running deadline. On success or check/setup failure,
request earlier shutdown:

```bash
node runpod_watchdog.ts stop artifacts/runpod-check-receipt.json
```

A separately approved retrieval-only restart uses the original creation receipt
plus an explicit `retrievalRequestedAtMs`, captured before its start request. It
retains original pod identity but has its own **180-second** allowance, stop at
**135 seconds**, and separate controller directory. The receipt is never created
to extend a running check automatically. The nine controller tests cover both
allowances and reject invalid retrieval timestamps and changed pod identity.

The controller requires a fresh GET showing `EXITED` or `TERMINATED`, null runtime
and no remaining stop action before writing `stopped.json`. The real H100 stop
retained its catalog hourly rate in `cost`, so zero price is not the criterion.
An accepted POST, missing pod, network error or
Python exit is not shutdown proof. Evidence lives beside the receipt under
`watchdog-<podId>/`; independently confirm shutdown with the MCP. The controller
never deletes a pod or volume. Keep all experiment outputs under the network
mount and verify exported copies before deleting storage.

This is a local, best-effort controller, not a provider-guaranteed spending cap.
Mac power loss, loss of connectivity or a provider outage can prevent timely
shutdown. Keep the Mac powered and online. If shutdown remains unverified at the allocation end
(900 seconds), it logs the missed deadline and continues retrying; it does not
silently extend training or declare success. Eight simulated controller tests
pass; local authentication, detached arming, direct SSH and authenticated early
stop worked on the H100. The corrected lifecycle/runtime verification rule is
locally tested against the observed provider response. Continue to independently
verify shutdown and cleanup through the MCP.

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
prefixes before rewriting uncommitted rows. The command above creates new pilot
runs; the production command implements training resume. `LabeledArtifact` rejects incomplete or
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

### Bounded Training Sizing

This exercise is complete. The commands below preserve its reproducibility;
they are not the next production task. The owner directed a full-size corrected
three-band rerun on RunPod. These small settings have not been established as
replacements for the historical dataset sizes or architecture. See `TODO.md`.

The fixed sizing protocol runs five CPU/float64 fits: M5 at 2,048/8,192 examples
with width 128, one 8,192-example width-256 comparison, a repeat initialization
of the validation-selected configuration, and one M20 endpoint check. Each fit
runs 60 epochs with a best-validation checkpoint. Batch size, learning rate,
seed, CPU threads and generation reserves come from local `.env.local`.

```bash
uv run --frozen --extra training python corrected_pilot.py --sizing --output artifacts/corrected-training-sizing
uv run --frozen --extra training python verify_corrected_pilot.py --sizing --run artifacts/corrected-training-sizing --report /absolute/existing/directory/corrected-training-sizing-audit.json
```

The comparisons use the same 512 dense and 512 sparse M5 validation targets.
M20 is also evaluated against these shared M5 targets. Boundary cases are
development diagnostics. Final random tests and showcases are not scored;
the eventual publication run needs fresh held-out tests. The protocol fixes
the selection rule before fitting and retains all histories, checkpoints,
development predictions, source snapshots and separately scoped timings.
This is a bounded sizing exercise, not the M5-M20 publication study.

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
