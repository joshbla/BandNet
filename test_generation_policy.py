import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from generation_policy import GenerationPolicy, load_generation_policy, remaining_budgets
from generation_resources import combine_resources
from triatomic_batched import BatchResult, TriatomicBatchSolver
from triatomic_execution import candidate_plans, execution_batches, tune_execution


class PolicyTests(unittest.TestCase):
    def test_large_machine_candidates_extend_beyond_old_ceilings(self):
        resources = combine_resources("linux", 128, 128, 48 * 1024**3, ())
        policy = GenerationPolicy(2, 4 * 1024**3, 5)
        plans = candidate_plans(resources, 500, 20, True, policy, examples=16384)
        self.assertEqual(max(plan.workers for plan in plans), 126)
        self.assertTrue(any(plan.estimated_working_bytes > 512 * 1024**2 for plan in plans))
        self.assertTrue(all(plan.estimated_working_bytes <= 44 * 1024**3 for plan in plans))
        self.assertEqual(plans[0].workers, 1)
        self.assertEqual(plans[1].workers, 126)

    def test_explicit_reserves_and_exhausted_allocations(self):
        resources = combine_resources("linux", 32, 16, 8 * 1024**3, ())
        self.assertEqual(remaining_budgets(resources, GenerationPolicy(2, 2 * 1024**3, 5)), (14, 6 * 1024**3))
        with self.assertRaises(RuntimeError):
            remaining_budgets(resources, GenerationPolicy(16, 0, 5))
        with self.assertRaises(MemoryError):
            remaining_budgets(resources, GenerationPolicy(0, 8 * 1024**3, 5))
        for args in ((-1, 0, 5), (0, -1, 5), (True, 0, 5), (0, 0, 0), (0, 0, float("inf"))):
            with self.assertRaises(ValueError):
                GenerationPolicy(*args)

    def test_local_settings_required_no_environment_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env.local"
            path.write_text("BANDNET_GENERATION_CPU_RESERVE=2\nBANDNET_GENERATION_RAM_RESERVE_MIB=2048\nBANDNET_GENERATION_TUNING_SECONDS=4\n")
            with patch.dict("os.environ", {"BANDNET_GENERATION_CPU_RESERVE": "99"}):
                self.assertEqual(load_generation_policy(path), GenerationPolicy(2, 2 * 1024**3, 4))
            path.write_text("BANDNET_GENERATION_CPU_RESERVE=2\n")
            with self.assertRaises(ValueError):
                load_generation_policy(path)
            path.write_text("BANDNET_GENERATION_CPU_RESERVE=2\nBANDNET_GENERATION_CPU_RESERVE=3\n")
            with self.assertRaises(ValueError):
                load_generation_policy(path)

    def test_execution_keeps_reserve_after_memory_or_cpu_changes(self):
        resources = combine_resources("darwin", 16, 16, 4 * 1024**3, ())
        policy = GenerationPolicy(2, 1024**3, 5)
        plan = candidate_plans(resources, 2, 5, False, policy, examples=64)[0]
        solver = TriatomicBatchSolver([0.1, 1], 5)
        low_memory = replace(resources, available_memory_bytes=policy.ram_reserve_bytes + plan.estimated_working_bytes - 1)
        with patch("triatomic_execution.detect_resources", return_value=low_memory):
            with self.assertRaises(MemoryError):
                list(execution_batches(solver, np.ones((64, 3)), np.ones((64, 5)), plan))
        with patch("triatomic_execution.detect_resources", return_value=replace(resources, cpu_budget=2)):
            with self.assertRaises(RuntimeError):
                list(execution_batches(solver, np.ones((64, 3)), np.ones((64, 5)), plan))

    def test_soft_deadline_selects_only_measured_plans_and_reports_incomplete_search(self):
        resources = combine_resources("darwin", 16, 16, 4 * 1024**3, ())
        policy = GenerationPolicy(1, 1024**3, 0.1)
        solver = TriatomicBatchSolver([0.1, 1], 5)
        clock = [0.0]

        def fake_batches(solver, masses, springs, plan):
            clock[0] += 0.2
            for first in range(0, len(masses), plan.chunk_size):
                count = min(plan.chunk_size, len(masses) - first)
                yield first, BatchResult(np.zeros((count, 2, 3)), 0, 0, 0, 0)

        with (patch("triatomic_execution.detect_resources", return_value=resources),
              patch("triatomic_execution.time.perf_counter", side_effect=lambda: clock[0]),
              patch("triatomic_execution.execution_batches", side_effect=fake_batches)):
            plan, report = tune_execution(solver, np.ones((128, 3)), np.ones((128, 5)), policy)
        self.assertEqual(report["measured_candidates"], 1)
        self.assertFalse(report["search_complete"])
        self.assertTrue(report["time_budget_exceeded"])
        self.assertEqual(report["candidate_measurements"][0]["seconds"], [0.2])
        self.assertEqual(plan.chunk_size, report["candidate_measurements"][0]["plan"]["chunk_size"])

    def test_candidate_plans_fit_both_workload_and_memory(self):
        resources = combine_resources("linux", 64, 64, 512 * 1024**2, ())
        plans = candidate_plans(resources, 500, 20, True, GenerationPolicy(1, 400 * 1024**2, 5), examples=71)
        self.assertTrue(all(plan.workers * plan.chunk_size <= 71 for plan in plans))
        self.assertTrue(all(plan.estimated_working_bytes <= 112 * 1024**2 for plan in plans))
        self.assertTrue(all(plan.workers <= 63 for plan in plans))
