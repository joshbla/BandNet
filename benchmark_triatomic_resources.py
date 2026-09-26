"""Allocation-based CPU benchmark. Workload and explicit reserve policy come
from local .env.local. No pod, GPU, model training or historical checkpoint is run.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import psutil
import threadpoolctl

import benchmark_triatomic as baseline
from generation_policy import load_generation_policy, remaining_budgets
from generation_resources import detect_resources
from triatomic_batched import TriatomicBatchSolver
from triatomic_execution import execution_batches, tune_execution
from triatomic_genuine_formula import triatomic_frequencies


ROOT = Path(__file__).resolve().parent


def settings():
    names = {"BANDNET_RESOURCE_BENCHMARK_INTERACTIONS", "BANDNET_RESOURCE_BENCHMARK_CURVES", "BANDNET_RESOURCE_BENCHMARK_REPEATS"}
    found = {}
    for raw in (ROOT / ".env.local").read_text().splitlines():
        key, separator, value = raw.strip().partition("=")
        if key not in names:
            continue
        if not separator or key in found:
            raise ValueError(f"duplicate or malformed setting {key}")
        found[key] = value.strip()
    if set(found) != names:
        raise ValueError("all three BANDNET_RESOURCE_BENCHMARK settings must be explicit in .env.local")
    interactions = [int(item) for item in found["BANDNET_RESOURCE_BENCHMARK_INTERACTIONS"].split(",")]
    count = int(found["BANDNET_RESOURCE_BENCHMARK_CURVES"])
    repeats = int(found["BANDNET_RESOURCE_BENCHMARK_REPEATS"])
    if not interactions or len(set(interactions)) != len(interactions) or any(k < 1 or k > 20 for k in interactions):
        raise ValueError("interaction counts must be distinct values in [1,20]")
    if not 1024 <= count <= 100000 or not 1 <= repeats <= 3:
        raise ValueError("use 1024..100000 curves and 1..3 repetitions")
    return {"interactions": interactions, "curves": count, "repeats": repeats}


def cases(count, interactions):
    rng = np.random.default_rng(1847 + interactions)
    masses = np.column_stack((np.ones(count), rng.uniform(0.1, 10, (count, 2))))
    springs = np.column_stack((np.ones(count), rng.uniform(0, 10, (count, interactions - 1))))
    springs[count // 2:, 1:] *= rng.integers(0, 2, (count - count // 2, interactions - 1))
    return masses, springs


def check_edges(interactions):
    masses, springs = cases(12, interactions)
    masses[:3] = [[1, 1e-3, 1e3], [1, 1e3, 1e-3], [1, 1, 1]]
    grid = np.array([0, 1e-12, 1e-9, 1e-6, 0.001, 0.37, 1])
    actual = TriatomicBatchSolver(grid, interactions).evaluate(masses, springs).frequencies
    reference = np.stack([triatomic_frequencies(m, k, grid) for m, k in zip(masses, springs)])
    scale = np.maximum(1, np.max(reference, axis=(1, 2)))[:, None, None]
    squared_error = np.abs(actual**2 - reference**2)
    frequency_error = np.abs(actual - reference)
    limit = 128 * np.finfo(float).eps * scale**2
    near_zero = np.maximum(actual, reference)**2 <= limit
    old_policy_exceedance = (frequency_error > 1e-10 * scale) & ~near_zero
    if np.any(squared_error > limit):
        raise AssertionError("extended diagnostic spectral tolerance failed")
    return {
        "spectral_check_passed": True,
        "maximum_scaled_squared_error": float(np.max(squared_error / scale**2)),
        "maximum_absolute_frequency_error": float(np.max(frequency_error)),
        "previous_conditioned_frequency_policy_passed": not bool(np.any(old_policy_exceedance)),
        "previous_policy_exceedances": [
            {"sample": int(i), "q_hat": float(grid[j]), "band": int(b),
             "frequency_error": float(frequency_error[i, j, b]),
             "scaled_squared_error": float((squared_error / scale**2)[i, j, b])}
            for i, j, b in np.argwhere(old_policy_exceedance)
        ],
    }


def trial(solver, masses, springs, plan):
    output_bytes = len(masses) * len(baseline.GRID) * 3 * 8 + 128
    with tempfile.TemporaryDirectory(prefix="bandnet-resource-") as temporary:
        directory = Path(temporary)
        if psutil.disk_usage(str(directory)).free < output_bytes + 64 * 1024**2:
            raise OSError("insufficient temporary disk space for this benchmark")
        started = time.perf_counter()
        clipped = 0
        digest = hashlib.sha256()
        write_seconds = 0.0
        output_path = directory / "frequencies.npy"
        with output_path.open("xb") as output:
            np.lib.format.write_array_header_2_0(output, {
                "descr": np.dtype(np.float64).str, "fortran_order": False,
                "shape": (len(masses), 500, 3),
            })
            for _, result in execution_batches(solver, masses, springs, plan):
                tick = time.perf_counter()
                result.frequencies.tofile(output)
                write_seconds += time.perf_counter() - tick
                digest.update(result.frequencies.tobytes())
                clipped += result.negative_eigenvalues_clipped
            tick = time.perf_counter()
            output.flush()
            os.fsync(output.fileno())
            write_seconds += time.perf_counter() - tick
        seconds = time.perf_counter() - started
        saved = np.load(output_path, mmap_mode="r")
        selected = np.unique([0, len(masses) // 2, len(masses) - 1])
        expected = solver.evaluate(masses[selected], springs[selected]).frequencies
        np.testing.assert_allclose(saved[selected], expected, rtol=1e-10, atol=1e-10)
        del saved
        return {
            "seconds": seconds, "write_flush_fsync_seconds": write_seconds,
            "curves_per_second": len(masses) / seconds,
            "output_bytes": output_path.stat().st_size,
            "payload_sha256": digest.hexdigest(), "mmap_row_checks_passed": True,
            "negative_eigenvalues_clipped": clipped,
            "process_lifetime_peak_rss_bytes": baseline.peak_rss_bytes(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("report must be a new path in an existing directory")
    config = settings()
    policy = load_generation_policy(ROOT / ".env.local")
    runs = []
    for interactions in config["interactions"]:
        _, available = remaining_budgets(detect_resources(), policy)
        if config["curves"] * (interactions + 3) * 8 > available // 8:
            raise MemoryError("benchmark label allocation exceeds its memory allowance after reserve")
        small_masses, small_springs = cases(12, interactions)
        native = baseline.validate_cases("native_grid", small_masses, small_springs, baseline.GRID)
        if native["strict_frequency_exceedances"]:
            raise AssertionError("native-grid strict frequency tolerance failed")
        edges = check_edges(interactions)
        masses, springs = cases(config["curves"], interactions)
        solver = TriatomicBatchSolver(baseline.GRID, interactions)
        started = time.perf_counter()
        plan, tuning = tune_execution(solver, masses, springs, policy)
        tuning_seconds = time.perf_counter() - started
        trials = [trial(solver, masses, springs, plan) for _ in range(config["repeats"])]
        if len({item["payload_sha256"] for item in trials}) != 1:
            raise AssertionError("repeated trial output differs")
        runs.append({
            "interactions": interactions, "native_validation": native,
            "edge_validation": edges, "tuning": tuning, "tuning_seconds": tuning_seconds,
            "trials": trials, "median_streamed_seconds": statistics.median(item["seconds"] for item in trials),
            "masses_sha256": hashlib.sha256(masses.tobytes()).hexdigest(),
            "stiffnesses_sha256": hashlib.sha256(springs.tobytes()).hexdigest(),
        })
        print(json.dumps({"interactions": interactions, "plan": asdict(plan),
                          "median_streamed_seconds": runs[-1]["median_streamed_seconds"]}), flush=True)
    sources = (*baseline.SOURCE_FILES, "generation_resources.py", "generation_policy.py", "triatomic_execution.py",
               "benchmark_triatomic_resources.py", "test_generation_resources.py", "test_generation_policy.py", "test_triatomic_interactions.py")
    report = {
        "configuration": config, "policy": asdict(policy),
        "code_base_commit": baseline.command_output("git", "rev-parse", "HEAD"),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "numpy": np.__version__, "psutil": psutil.__version__, "threadpoolctl": threadpoolctl.__version__,
        "numpy_build": np.show_config(mode="dicts"), "platform": platform.platform(),
        "hardware_cpu": platform.processor(), "runs": runs,
        "input_protocol": "seed 1847+K; m1=k1=1; other masses U[.1,10), springs U[0,10); second half zeros each nonreference spring independently with p=.5; 500 q_hat points [.001,1]",
        "timing_scope": "streamed float64 frequency NPY with hashing, flush, fsync; solver setup and calibration separately/excluded; CPU only, no training",
        "memory_scope": "process-lifetime high-water RSS includes previous counts and tuning; allocation model is conservative, not a memory guarantee",
        "portability_scope": "local measurement; reserve values are explicit local test settings, not approved production pod reserves",
    }
    with args.report.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
