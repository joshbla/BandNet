"""Resource discovery and execution tests; starter-policy source is archived."""

import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from generation_policy import GenerationPolicy
from generation_resources import CgroupConstraint, cgroup_constraints, combine_resources
from triatomic_batched import TriatomicBatchSolver
from triatomic_execution import candidate_plans, execution_batches, _parallel_batches


class TestController:
    def limit(self, **kwargs):
        return nullcontext()

    def select(self, **kwargs):
        return self

    def info(self):
        return [{"num_threads": 1}]


def plans_for(resources, grid, interactions, controlled):
    return candidate_plans(resources, grid, interactions, controlled,
                           GenerationPolicy(0, 0, 5), examples=1024)


class ResourceTests(unittest.TestCase):
    def test_cpu_and_memory_constraints_are_intersected(self):
        limits = (CgroupConstraint("parent", 2.5, 300 * 1024**2),
                  CgroupConstraint("child", 8, 800 * 1024**2))
        resources = combine_resources("linux", 64, 6, 10 * 1024**3, limits)
        self.assertEqual(resources.cpu_budget, 2)
        self.assertEqual(resources.available_memory_bytes, 300 * 1024**2)
        fraction = combine_resources("linux", 64, 6, 100, (CgroupConstraint("tiny", 0.5, 0),))
        self.assertEqual(fraction.cpu_budget, 1)
        self.assertEqual(fraction.available_memory_bytes, 0)

    def test_v2_resolves_membership_ancestors_and_unlimited_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary) / "cgroup"
            leaf = mount / "job" / "worker"
            leaf.mkdir(parents=True)
            (mount / "job" / "cpu.max").write_text("250000 100000")
            (mount / "job" / "memory.max").write_text("1000")
            (mount / "job" / "memory.current").write_text("300")
            (leaf / "cpu.max").write_text("max 100000")
            (leaf / "memory.max").write_text("max")
            info = f"31 20 0:1 / {mount} rw - cgroup2 cgroup rw"
            resources = combine_resources("linux", 64, 12, 10000, cgroup_constraints("0::/job/worker", info))
            self.assertEqual(resources.cpu_budget, 2)
            self.assertEqual(resources.available_memory_bytes, 700)

    def test_v1_separate_mounts_and_nonroot_mount_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            cpu, memory = Path(temporary) / "cpu", Path(temporary) / "memory"
            (cpu / "worker").mkdir(parents=True)
            (memory / "worker").mkdir(parents=True)
            (cpu / "cpu.cfs_quota_us").write_text("400000")
            (cpu / "cpu.cfs_period_us").write_text("100000")
            (cpu / "worker" / "cpu.cfs_quota_us").write_text("-1")
            (cpu / "worker" / "cpu.cfs_period_us").write_text("100000")
            (memory / "worker" / "memory.limit_in_bytes").write_text("10000")
            (memory / "worker" / "memory.usage_in_bytes").write_text("4000")
            info = (f"31 20 0:1 /tenant {cpu} rw - cgroup cgroup rw,cpu,cpuacct\n"
                    f"32 20 0:2 /tenant {memory} rw - cgroup cgroup rw,memory")
            limits = cgroup_constraints("4:cpu,cpuacct:/tenant/worker\n5:memory:/tenant/worker", info)
            resources = combine_resources("linux", 64, 2, 100000, limits)
            self.assertEqual(resources.cpu_budget, 2)
            self.assertEqual(resources.available_memory_bytes, 6000)

    def test_unknown_or_malformed_cgroup_is_not_host_fallback(self):
        with self.assertRaises(RuntimeError):
            cgroup_constraints("0::/job", "")
        with self.assertRaises(ValueError):
            cgroup_constraints("0::/../job", "")
        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary)
            info = f"31 20 0:1 / {mount} rw - cgroup2 cgroup rw"
            (mount / "cpu.max").write_text("50000 0")
            with self.assertRaises(ValueError):
                cgroup_constraints("0::/", info)
            (mount / "cpu.max").write_text("max 100000")
            (mount / "memory.max").write_text("1000")
            with self.assertRaises(FileNotFoundError):
                cgroup_constraints("0::/", info)

    def test_memory_and_cpu_bound_plan_candidates(self):
        resources = combine_resources("linux", 128, 3, 256 * 1024**2, ())
        policy = GenerationPolicy(1, 192 * 1024**2, 5)
        plans = candidate_plans(resources, 500, 20, True, policy, examples=1024)
        self.assertTrue(all(plan.workers <= 2 for plan in plans))
        self.assertTrue(all(plan.estimated_working_bytes <= 64 * 1024**2 for plan in plans))
        self.assertEqual({plan.workers for plan in candidate_plans(resources, 500, 20, False, policy, examples=1024)}, {1})
        with self.assertRaises(MemoryError):
            plans_for(replace(resources, available_memory_bytes=1024), 500, 20, True)

    def test_execution_rejects_changed_or_forged_resources(self):
        resources = combine_resources("darwin", 8, 8, 4 * 1024**3, ())
        plan = plans_for(resources, 4, 20, False)[0]
        solver = TriatomicBatchSolver([0.1, 0.3, 0.7, 1], 20)
        with patch("triatomic_execution.detect_resources", return_value=resources):
            for bad in (replace(plan, workers=0), replace(plan, estimated_working_bytes=1)):
                with self.assertRaises(ValueError):
                    list(execution_batches(solver, np.ones((3, 3)), np.ones((3, 20)), bad))
        with patch("triatomic_execution.detect_resources", return_value=replace(resources, available_memory_bytes=1024)):
            with self.assertRaises(MemoryError):
                list(execution_batches(solver, np.ones((3, 3)), np.ones((3, 20)), plan))

    def test_parallel_chunk_order_partial_tail_and_numeric_identity(self):
        resources = combine_resources("linux", 4, 4, 4 * 1024**3, ())
        plan = next(plan for plan in plans_for(resources, 7, 20, True) if plan.workers == 2)
        rng = np.random.default_rng(7)
        masses, springs = rng.uniform(0.1, 10, (71, 3)), rng.uniform(0, 10, (71, 20))
        solver = TriatomicBatchSolver(np.linspace(0.001, 1, 7), 20)
        parts = list(_parallel_batches(solver, masses, springs, plan, TestController()))
        self.assertEqual([start for start, _ in parts], [0, 32, 64])
        np.testing.assert_allclose(np.concatenate([part.frequencies for _, part in parts]),
                                   solver.evaluate(masses, springs).frequencies, rtol=1e-10, atol=1e-10)

    def test_parallel_exception_propagates(self):
        resources = combine_resources("linux", 2, 2, 4 * 1024**3, ())
        plan = next(plan for plan in plans_for(resources, 2, 5, True) if plan.workers == 2)
        solver = TriatomicBatchSolver([0.1, 1], 5)
        springs = np.ones((40, 5))
        springs[-1, -1] = -1
        with self.assertRaises(ValueError):
            list(_parallel_batches(solver, np.ones((40, 3)), springs, plan, TestController()))

    def test_reduced_cpu_quota_rejects_previously_selected_workers(self):
        resources = combine_resources("linux", 4, 4, 4 * 1024**3, ())
        plan = next(plan for plan in plans_for(resources, 2, 5, True) if plan.workers == 2)
        solver = TriatomicBatchSolver([0.1, 1], 5)
        with patch("triatomic_execution.detect_resources", return_value=replace(resources, cpu_budget=1)):
            with self.assertRaises(RuntimeError):
                list(execution_batches(solver, np.ones((40, 3)), np.ones((40, 5)), plan))

    def test_unenforced_native_limit_is_reported_as_error(self):
        class IneffectiveController(TestController):
            def info(self):
                return [{"num_threads": 4}]

        resources = combine_resources("linux", 4, 4, 4 * 1024**3, ())
        plan = plans_for(resources, 2, 5, True)[0]
        solver = TriatomicBatchSolver([0.1, 1], 5)
        with (patch("triatomic_execution.detect_resources", return_value=resources),
              patch("triatomic_execution.ThreadpoolController", return_value=IneffectiveController())):
            with self.assertRaises(RuntimeError):
                list(execution_batches(solver, np.ones((40, 3)), np.ones((40, 5)), plan))
