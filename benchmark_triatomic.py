"""Bounded local CPU benchmark; controls are read only from local .env.local.

Writes a new JSON report at --report. Numerical arrays live in a temporary
directory and are removed afterward. No historical model or training is run.
The streamed float64 frequency-only NPY is a benchmark format, not an adopted
training artifact schema. The report retains configuration and source hashes.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from triatomic_batched import SOLVER_VERSION, TriatomicBatchSolver
from triatomic_genuine_formula import triatomic_frequencies


ROOT = Path(__file__).resolve().parent
SEED = 1847
GRID = np.linspace(0.001, 1.0, 500)
SOURCE_FILES = (
    "triatomic_batched.py", "triatomic_genuine_formula.py",
    "test_triatomic_batched.py", "test_triatomic_genuine_formula.py",
    "benchmark_triatomic.py", "pyproject.toml", "uv.lock",
)


def configuration() -> dict[str, int]:
    names = {
        "BANDNET_BENCHMARK_CURVES": "curves",
        "BANDNET_BENCHMARK_CHUNK_SIZE": "chunk_size",
        "BANDNET_BENCHMARK_REPEATS": "repeats",
    }
    controls = {}
    for raw_line in (ROOT / ".env.local").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if key not in names:
            continue
        if not separator or names[key] in controls:
            raise ValueError(f"invalid or duplicate benchmark setting: {key}")
        controls[names[key]] = int(value.strip())
    if set(controls) != set(names.values()):
        raise ValueError(".env.local must explicitly define all three benchmark controls")
    if not 1 <= controls["curves"] <= 100_000:
        raise ValueError("benchmark curves must be in [1, 100000]")
    if not 1 <= controls["chunk_size"] <= 512:
        raise ValueError("benchmark chunk_size must be in [1, 512]")
    if not 1 <= controls["repeats"] <= 5:
        raise ValueError("benchmark repeats must be in [1, 5]")
    return controls


def parameter_cases(count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    masses = np.column_stack((np.ones(count), rng.uniform(0.1, 10.0, (count, 2))))
    springs = np.column_stack((np.ones(count), rng.uniform(0.0, 10.0, (count, 4))))
    sparse_count = count - count // 2
    springs[count // 2:, 1:] *= rng.integers(0, 2, (sparse_count, 4))
    return masses, springs


def validate_cases(name, masses, springs, grid) -> dict:
    result = TriatomicBatchSolver(grid, springs.shape[1]).evaluate(masses, springs)
    reference = np.stack([
        triatomic_frequencies(mass, stiffness, grid)
        for mass, stiffness in zip(masses, springs)
    ])
    actual = result.frequencies
    scale = np.maximum(1.0, np.max(reference, axis=(1, 2)))[:, None, None]
    squared_limit = 128 * np.finfo(float).eps * scale**2
    frequency_error = np.abs(actual - reference)
    squared_error = np.abs(actual**2 - reference**2)
    strict_failures = frequency_error > 1e-10 * scale
    near_zero = np.maximum(actual, reference)**2 <= squared_limit
    if np.any(squared_error > squared_limit):
        raise AssertionError(f"{name}: squared-frequency agreement failed")
    if np.any(strict_failures & ~near_zero):
        raise AssertionError(f"{name}: resolved-frequency agreement failed")
    return {
        "name": name,
        "examples": len(masses),
        "wave_numbers": len(grid),
        "maximum_absolute_frequency_error": float(np.max(frequency_error)),
        "maximum_scaled_frequency_error": float(np.max(frequency_error / scale)),
        "maximum_scaled_squared_frequency_error": float(np.max(squared_error / scale**2)),
        "strict_frequency_exceedances": int(np.count_nonzero(strict_failures)),
        "exceedances_outside_roundoff_zero_region": int(np.count_nonzero(strict_failures & ~near_zero)),
        "negative_eigenvalues_clipped": result.negative_eigenvalues_clipped,
        "maximum_hermitian_residual": result.maximum_hermitian_residual,
        "passed": True,
    }


def validation_report() -> list[dict]:
    masses, springs = parameter_cases(64)
    cases = [validate_cases("seeded_dense_sparse_native_grid", masses, springs, GRID)]
    cases.append(validate_cases(
        "seeded_dense_sparse_zero_and_near_zero", masses, springs,
        np.array([0, 1e-12, 1e-9, 1e-6, 0.001, 0.37, 1]),
    ))
    cases.append(validate_cases(
        "preserved_four_showcase_targets",
        np.array([[1, 8, 5], [1, 1, 7], [1, 4, 4], [1, 6, 1]], dtype=float),
        np.array([[1, 6, 6, 4, 8], [1, 2, 8, 1, 6], [1, 8, 4, 9, 4], [1, 3, 0, 7, 9]], dtype=float),
        GRID,
    ))
    cases.append(validate_cases(
        "extreme_mass_sparse_repeated_and_zero_limits",
        np.array([[1, 1e-3, 1e3], [1, 1e3, 1e-3], [1, 1, 1], [1, 2, 3]], dtype=float),
        np.array([[1, 10, 0, 10, 0], [1, 0, 10, 0, 10], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=float),
        np.array([0, 1e-12, 1e-9, 1e-6, 0.001, 0.37, 1]),
    ))
    return cases


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    if sys.platform.startswith("linux"):
        return int(value * 1024)
    raise RuntimeError("peak RSS unit is only specified here for macOS and Linux")


def compute_trial(masses, springs, chunk_size) -> dict:
    start = time.perf_counter()
    solver = TriatomicBatchSolver(GRID, 5)
    clipped = 0
    for _, result in solver.iter_batches(masses, springs, chunk_size=chunk_size):
        clipped += result.negative_eigenvalues_clipped
    seconds = time.perf_counter() - start
    return {
        "seconds": seconds,
        "curves_per_second": len(masses) / seconds,
        "negative_eigenvalues_clipped": clipped,
        "process_lifetime_peak_rss_bytes": peak_rss_bytes(),
    }


def streamed_trial(masses, springs, chunk_size) -> dict:
    with tempfile.TemporaryDirectory(prefix="bandnet-cpu-") as temporary:
        output_path = Path(temporary) / "frequencies.npy"
        start = time.perf_counter()
        solver = TriatomicBatchSolver(GRID, 5)
        iterator = solver.iter_batches(masses, springs, chunk_size=chunk_size)
        compute_seconds = 0.0
        write_seconds = 0.0
        clipped = 0
        payload_hash = hashlib.sha256()
        with output_path.open("xb") as output:
            np.lib.format.write_array_header_2_0(output, {
                "descr": np.dtype(np.float64).str,
                "fortran_order": False,
                "shape": (len(masses), len(GRID), 3),
            })
            for expected_start in range(0, len(masses), chunk_size):
                tick = time.perf_counter()
                first, result = next(iterator)
                compute_seconds += time.perf_counter() - tick
                if first != expected_start:
                    raise AssertionError("unexpected chunk order")
                tick = time.perf_counter()
                result.frequencies.tofile(output)
                write_seconds += time.perf_counter() - tick
                payload_hash.update(result.frequencies.tobytes())
                clipped += result.negative_eigenvalues_clipped
            tick = time.perf_counter()
            output.flush()
            os.fsync(output.fileno())
            write_seconds += time.perf_counter() - tick
        seconds = time.perf_counter() - start
        output_bytes = output_path.stat().st_size
        # Outside the timed region, verify a mmap consumer and saved row identity.
        restored = np.load(output_path, mmap_mode="r")
        if restored.shape != (len(masses), 500, 3) or restored.dtype != np.float64:
            raise AssertionError("written NPY shape/dtype mismatch")
        indices = np.unique([0, len(masses) // 2, len(masses) - 1])
        expected = solver.evaluate(masses[indices], springs[indices]).frequencies
        np.testing.assert_allclose(restored[indices], expected, rtol=1e-10, atol=1e-10)
        del restored
        return {
            "seconds": seconds,
            "compute_seconds": compute_seconds,
            "write_flush_fsync_seconds": write_seconds,
            "other_seconds_including_setup_and_hash": seconds - compute_seconds - write_seconds,
            "curves_per_second": len(masses) / seconds,
            "output_bytes": output_bytes,
            "payload_sha256": payload_hash.hexdigest(),
            "negative_eigenvalues_clipped": clipped,
            "process_lifetime_peak_rss_bytes": peak_rss_bytes(),
            "mmap_row_checks_passed": True,
        }


def command_output(*command: str) -> str:
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def run_benchmark(controls: dict[str, int]) -> dict:
    validation = validation_report()
    tick = time.perf_counter()
    masses, springs = parameter_cases(controls["curves"])
    parameter_seconds = time.perf_counter() - tick
    reference_masses, reference_springs = parameter_cases(32)
    solver = TriatomicBatchSolver(GRID, 5)
    solver.evaluate(reference_masses[:2], reference_springs[:2])
    triatomic_frequencies(reference_masses[0], reference_springs[0], GRID)
    reference_trials = []
    matching_batch_trials = []
    for _ in range(3):
        tick = time.perf_counter()
        for mass, springs_row in zip(reference_masses, reference_springs):
            triatomic_frequencies(mass, springs_row, GRID)
        reference_trials.append(time.perf_counter() - tick)
        tick = time.perf_counter()
        solver.evaluate(reference_masses, reference_springs)
        matching_batch_trials.append(time.perf_counter() - tick)
    computed = []
    streamed = []
    for _ in range(controls["repeats"]):
        computed.append(compute_trial(masses, springs, controls["chunk_size"]))
        streamed.append(streamed_trial(masses, springs, controls["chunk_size"]))
    if len({trial["payload_sha256"] for trial in streamed}) != 1:
        raise AssertionError("repeated output payloads differ")
    machine = {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        machine["cpu"] = command_output("sysctl", "-n", "machdep.cpu.brand_string")
        machine["physical_memory_bytes"] = int(command_output("sysctl", "-n", "hw.memsize"))
    return {
        "solver_version": SOLVER_VERSION,
        "code_base_commit": command_output("git", "rev-parse", "HEAD"),
        "working_tree_status": command_output("git", "status", "--short"),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        "python": sys.version,
        "numpy": np.__version__,
        "numpy_build": np.show_config(mode="dicts"),
        "hardware": machine,
        "configuration": controls,
        "benchmark_inputs": {
            "seed": SEED, "m1": 1, "k1": 1, "m2_m3_uniform": [0.1, 10.0],
            "k2_k5_uniform": [0.0, 10.0],
            "sparsity": "second half independently zeros each nonreference spring with probability 0.5",
            "q_hat": {"start": 0.001, "stop": 1.0, "count": 500},
            "masses_sha256": hashlib.sha256(masses.tobytes()).hexdigest(),
            "stiffnesses_sha256": hashlib.sha256(springs.tobytes()).hexdigest(),
            "not_a_training_distribution": True,
        },
        "validation": validation,
        "reference_comparison": {
            "curves": 32, "reference_seconds": reference_trials,
            "batched_seconds": matching_batch_trials,
            "median_same_inputs_speedup": statistics.median(reference_trials) / statistics.median(matching_batch_trials),
            "comparison": "correct literal reference versus preconfigured correct batch solver; not historical Core",
        },
        "parameter_preparation_seconds": parameter_seconds,
        "compute_only_trials": computed,
        "streamed_output_trials": streamed,
        "median_compute_seconds": statistics.median(trial["seconds"] for trial in computed),
        "median_streamed_seconds": statistics.median(trial["seconds"] for trial in streamed),
        "scope": {
            "precision": "float64 computation and storage",
            "temporary_output": "streamed frequency-only NPY, readable with mmap; removed after checks",
            "timing": "solver setup, input validation and curve generation; streaming includes write, flush, fsync and hash; excludes label preparation, post-write checks and JSON report",
            "memory": "process lifetime ru_maxrss, including validation and prior trials; not incremental allocation or sampled RSS",
            "storage": "local temporary filesystem; fsync measured, physical drive cache not independently measured",
            "device_transfer": "none; CPU only",
            "training": "not run",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or not args.report.parent.is_dir():
        raise ValueError("report must be a new file in an existing directory")
    report = run_benchmark(configuration())
    serialized = json.dumps(report, indent=2, allow_nan=False)
    with args.report.open("x") as output:
        output.write(serialized + "\n")
    print(serialized)


if __name__ == "__main__":
    main()
