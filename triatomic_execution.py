"""Allocation-based CPU execution with explicit reserves and bounded calibration.

The solver and its numerical tolerances are unchanged. There is no fixed CPU or
RAM ceiling: candidate settings fit the detected allocation minus the supplied
reserve and the calibration workload. Startup has a soft elapsed-time budget;
an in-flight NumPy call is never interrupted or claimed to be timed out safely.
"""

import statistics
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

import numpy as np
from threadpoolctl import ThreadpoolController

from generation_policy import GenerationPolicy, remaining_budgets
from generation_resources import Resources, detect_resources
from triatomic_batched import TriatomicBatchSolver


POLICY_VERSION = "allocation-reserve-v1"


@dataclass(frozen=True)
class ExecutionPlan:
    chunk_size: int
    workers: int
    estimated_working_bytes: int
    memory_budget_bytes: int
    native_thread_policy: str
    policy: GenerationPolicy


def working_bytes(grid_size: int, interactions: int, chunk_size: int, workers: int) -> int:
    # Conservative workspace estimate, not an allocator guarantee or total RSS.
    return (16 * 1024**2 + 2 * interactions * grid_size * 9 * 16
            + chunk_size * workers * (1024 * grid_size + 8 * (interactions + 3)))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def candidate_plans(resources: Resources, grid_size: int, interactions: int,
                    backend_controlled: bool, policy: GenerationPolicy,
                    *, examples: int) -> list[ExecutionPlan]:
    for value, name in ((grid_size, "grid_size"), (interactions, "interactions"), (examples, "examples")):
        _positive_integer(value, name)
    cpu_budget, budget = remaining_budgets(resources, policy)
    maximum_workers = min(cpu_budget, examples) if backend_controlled else 1
    workers_to_try = [1]
    workers = 2
    while workers < maximum_workers:
        workers_to_try.append(workers)
        workers *= 2
    if maximum_workers > 1:
        workers_to_try.append(maximum_workers)
    # Compare serial with the largest feasible concurrency early, not only after
    # exhausting every small-worker chunk size on a large machine.
    workers_to_try = [1] + sorted(set(workers_to_try) - {1}, reverse=True)
    native_policy = "one BLAS thread per worker" if backend_controlled else "backend-managed; one outer worker"
    fixed = working_bytes(grid_size, interactions, 0, 0)
    per_curve = 1024 * grid_size + 8 * (interactions + 3)
    per_worker_plans = []
    for workers in workers_to_try:
        maximum_chunk = min((budget - fixed) // (workers * per_curve), examples // workers)
        if maximum_chunk < 1:
            continue
        chunks = []
        chunk = 32
        while chunk < maximum_chunk:
            chunks.append(chunk)
            chunk *= 4
        chunks.append(maximum_chunk)
        per_worker_plans.append([
            ExecutionPlan(chunk, workers, working_bytes(grid_size, interactions, chunk, workers),
                          budget, native_policy, policy)
            for chunk in chunks
        ])
    if not per_worker_plans:
        raise MemoryError("insufficient detected headroom after reserve for a generation plan")
    # Interleave worker counts so the time budget does not inherently exclude
    # large machines. Every selected plan will have been measured on this host.
    return [plans[index] for index in range(max(map(len, per_worker_plans)))
            for plans in per_worker_plans if index < len(plans)]


def _parallel_batches(solver, masses, springs, plan, controller):
    """Bound in-flight futures and preserve input order; propagate failures."""
    def limit_worker():
        controller.limit(limits=1, user_api="blas")
        if any(info["num_threads"] != 1 for info in controller.select(user_api="blas").info()):
            raise RuntimeError("worker native BLAS thread limit could not be verified")

    with ThreadPoolExecutor(max_workers=plan.workers, initializer=limit_worker) as pool:
        pending = deque()
        starts = iter(range(0, len(masses), plan.chunk_size))

        def submit(start):
            stop = start + plan.chunk_size
            return start, pool.submit(solver.evaluate, masses[start:stop], springs[start:stop])

        for _ in range(plan.workers):
            start = next(starts, None)
            if start is None:
                break
            pending.append(submit(start))
        while pending:
            start, future = pending.popleft()
            yield start, future.result()
            next_start = next(starts, None)
            if next_start is not None:
                pending.append(submit(next_start))


def execution_batches(solver, masses, springs, plan: ExecutionPlan):
    _positive_integer(plan.workers, "workers")
    _positive_integer(plan.chunk_size, "chunk_size")
    expected = working_bytes(len(solver.q_hat_values), solver.interaction_count, plan.chunk_size, plan.workers)
    if plan.estimated_working_bytes != expected or expected > plan.memory_budget_bytes:
        raise ValueError("execution plan does not match the solver or its memory budget")
    resources = detect_resources()
    cpus, memory = remaining_budgets(resources, plan.policy)
    if plan.workers > cpus:
        raise RuntimeError("CPU headroom changed; select a new execution plan")
    if expected > memory:
        raise MemoryError("RAM headroom changed; select a new execution plan")
    mass_array, spring_array = solver._inputs(masses, springs)
    controller = ThreadpoolController()
    blas = controller.select(user_api="blas").info()
    if plan.native_thread_policy == "backend-managed; one outer worker":
        if resources.platform.startswith("linux") or plan.workers != 1:
            raise ValueError("uncontrolled native execution is limited to one macOS outer worker")
        yield from solver.iter_batches(mass_array, spring_array, chunk_size=plan.chunk_size)
    elif plan.native_thread_policy == "one BLAS thread per worker":
        if not blas:
            raise RuntimeError("the selected native thread control is unavailable")
        with controller.limit(limits=1, user_api="blas"):
            if any(info["num_threads"] != 1 for info in controller.select(user_api="blas").info()):
                raise RuntimeError("native BLAS thread limit could not be verified")
            if plan.workers == 1:
                yield from solver.iter_batches(mass_array, spring_array, chunk_size=plan.chunk_size)
            else:
                yield from _parallel_batches(solver, mass_array, spring_array, plan, controller)
    else:
        raise ValueError("unknown native thread policy")


def tune_execution(solver: TriatomicBatchSolver, masses, springs,
                   policy: GenerationPolicy) -> tuple[ExecutionPlan, dict]:
    started = time.perf_counter()
    deadline = started + policy.tuning_seconds
    masses, springs = solver._inputs(masses, springs)
    resources = detect_resources()
    _, memory = remaining_budgets(resources, policy)
    controller = ThreadpoolController()
    blas = controller.select(user_api="blas").info()
    # A process may have loaded another BLAS library through a different package.
    # That must not be mistaken for control over NumPy's Apple Accelerate backend.
    numpy_blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]["name"]
    backend_controlled = bool(blas) and numpy_blas != "accelerate"
    if not backend_controlled and resources.platform.startswith("linux"):
        raise RuntimeError("no controllable NumPy BLAS backend detected; use a supported build")
    # Bound calibration outputs separately from batch workspace. Probe size is a
    # tuning cost limit, not a permanent worker or production memory ceiling.
    per_probe = 4 * len(solver.q_hat_values) * 3 * 8 + 8 * (solver.interaction_count + 3)
    probe_size = min(len(masses), 1024, memory // (4 * per_probe))
    if probe_size < 32:
        raise MemoryError("calibration needs headroom for at least 32 representative examples")
    indices = np.linspace(0, len(masses) - 1, probe_size, dtype=int)
    probe_masses, probe_springs = masses[indices], springs[indices]
    reference_output = np.empty((probe_size, len(solver.q_hat_values), 3), dtype=np.float64)
    # Account conservatively for the held probe arrays and comparison temporaries.
    calibration_bytes = probe_size * per_probe
    calibration_resources = Resources(
        resources.platform, resources.host_logical_cpus, resources.affinity_cpus,
        resources.cpu_budget, resources.host_available_memory_bytes,
        resources.available_memory_bytes - calibration_bytes, resources.cgroups,
    )
    plans = candidate_plans(calibration_resources, len(solver.q_hat_values), solver.interaction_count,
                            backend_controlled, policy, examples=probe_size)
    records = []
    for plan in plans:
        if records and time.perf_counter() >= deadline:
            break
        for first, result in execution_batches(solver, probe_masses, probe_springs, plan):
            stop = first + len(result.frequencies)
            if not records:
                reference_output[first:stop] = result.frequencies
            else:
                np.testing.assert_allclose(result.frequencies, reference_output[first:stop], rtol=1e-10, atol=1e-10)
        durations = []
        for _ in range(3):
            # Always finish one timed trial of the first candidate. A soft deadline
            # cannot interrupt an in-flight native call; report any overrun honestly.
            if durations and time.perf_counter() >= deadline:
                break
            tick = time.perf_counter()
            for _ in execution_batches(solver, probe_masses, probe_springs, plan):
                pass
            durations.append(time.perf_counter() - tick)
        records.append({"plan": asdict(plan), "seconds": durations,
                        "median_seconds": statistics.median(durations)})
    best_time = min(record["median_seconds"] for record in records)
    eligible = [i for i, record in enumerate(records) if record["median_seconds"] <= best_time * 1.05]
    winner = min(eligible, key=lambda i: (plans[i].workers, plans[i].estimated_working_bytes))
    elapsed = time.perf_counter() - started
    return plans[winner], {
        "policy_version": POLICY_VERSION, "policy": asdict(policy),
        "resources": resources.as_dict(), "native_blas_pools": blas,
        "numpy_blas": numpy_blas, "native_backend_controlled": backend_controlled,
        "probe_examples": probe_size, "calibration_array_allowance_bytes": calibration_bytes,
        "feasible_candidates": len(plans), "measured_candidates": len(records),
        "candidate_measurements": records, "selected_plan": asdict(plans[winner]),
        "elapsed_seconds": elapsed, "time_budget_exceeded": elapsed > policy.tuning_seconds,
        "search_complete": len(records) == len(plans) and all(len(record["seconds"]) == 3 for record in records),
        "selection": "smallest worker/memory footprint within 5 percent of fastest measured median",
        "limitations": "soft startup deadline; at most 1024 probe examples; finite geometric candidate set; no global optimum, GPU or process pool",
    }
