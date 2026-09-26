"""Explicit CPU and memory limits for local or containerized CPU generation.

Linux limits combine CPU affinity with visible cgroup v1/v2 ancestor quotas,
and host available memory with visible cgroup memory headroom. Detection errors
are errors, not permission to substitute host-wide resources.
"""

import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import psutil


@dataclass(frozen=True)
class CgroupConstraint:
    directory: str
    cpu_quota: float | None
    memory_headroom_bytes: int | None


@dataclass(frozen=True)
class Resources:
    platform: str
    host_logical_cpus: int
    affinity_cpus: int
    cpu_budget: int
    host_available_memory_bytes: int
    available_memory_bytes: int
    cgroups: tuple[CgroupConstraint, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def _unescape_mount(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def _ancestry(leaf: Path, mount: Path):
    leaf.relative_to(mount)
    current = leaf
    while True:
        yield current
        if current == mount:
            return
        current = current.parent


def _constraint(directory: Path, version: int, controllers: set[str]) -> CgroupConstraint:
    quota = None
    headroom = None
    if version == 2:
        cpu_path = directory / "cpu.max"
        if cpu_path.exists():
            maximum, period_text = cpu_path.read_text().split()
            period = int(period_text)
            if period <= 0:
                raise ValueError(f"invalid CPU period: {cpu_path}")
            if maximum != "max":
                quota = int(maximum) / period
        memory_path = directory / "memory.max"
        if memory_path.exists():
            maximum = memory_path.read_text().strip()
            if maximum != "max":
                limit = int(maximum)
                used = int((directory / "memory.current").read_text())
                if min(limit, used) < 0:
                    raise ValueError(f"invalid cgroup memory values: {directory}")
                headroom = max(0, limit - used)
    else:
        cpu_path = directory / "cpu.cfs_quota_us"
        if "cpu" in controllers and cpu_path.exists():
            maximum = int(cpu_path.read_text())
            period = int((directory / "cpu.cfs_period_us").read_text())
            if period <= 0 or maximum < -1:
                raise ValueError(f"invalid cgroup CPU values: {directory}")
            if maximum != -1:
                quota = maximum / period
        memory_path = directory / "memory.limit_in_bytes"
        if "memory" in controllers and memory_path.exists():
            limit = int(memory_path.read_text())
            used = int((directory / "memory.usage_in_bytes").read_text())
            if min(limit, used) < 0:
                raise ValueError(f"invalid cgroup memory values: {directory}")
            headroom = max(0, limit - used)
    if quota is not None and quota <= 0:
        raise ValueError(f"nonpositive CPU quota: {directory}")
    return CgroupConstraint(str(directory), quota, headroom)


def cgroup_constraints(membership: str, mountinfo: str) -> tuple[CgroupConstraint, ...]:
    """Resolve process membership against mounted hierarchies, including parents.

    Text inputs make Linux namespace and hierarchy cases testable on macOS.
    Hidden ancestors beyond a namespace's mount root cannot be discovered here.
    """
    memberships = []
    for line in membership.splitlines():
        hierarchy, names, location = line.split(":", 2)
        controllers = set(names.split(",")) - {""}
        if hierarchy == "0" or controllers & {"cpu", "memory"}:
            member = PurePosixPath(location)
            if not member.is_absolute() or ".." in member.parts:
                raise ValueError("invalid cgroup membership path")
            memberships.append((2 if hierarchy == "0" else 1, controllers, member))
    mounts = []
    for line in mountinfo.splitlines():
        before, after = line.split(" - ", 1)
        fields, attributes = before.split(), after.split()
        if attributes[0] not in {"cgroup", "cgroup2"}:
            continue
        version = 2 if attributes[0] == "cgroup2" else 1
        mounts.append((
            version, set(attributes[2].split(",")),
            PurePosixPath(_unescape_mount(fields[3])),
            Path(_unescape_mount(fields[4])),
        ))
    found = {}
    for version, controllers, member in memberships:
        matched = False
        for mount_version, mount_controllers, root, mount in mounts:
            if version != mount_version:
                continue
            if version == 1 and not controllers & mount_controllers:
                continue
            if not member.is_relative_to(root):
                continue
            leaf = mount / str(member.relative_to(root))
            if not leaf.is_dir():
                raise RuntimeError(f"cgroup membership directory is unavailable: {leaf}")
            matched = True
            for directory in _ancestry(leaf, mount):
                constraint = _constraint(directory, version, controllers)
                found[(str(directory), version)] = constraint
        if not matched:
            raise RuntimeError(f"cannot resolve cgroup membership {member}; resource limits unknown")
    return tuple(found.values())


def combine_resources(platform_name, host_cpus, affinity_cpus, available_memory, constraints):
    if host_cpus is None or host_cpus < 1 or affinity_cpus < 1 or available_memory < 0:
        raise RuntimeError("invalid or unavailable host resources")
    cpu_limits = [float(host_cpus), float(affinity_cpus)]
    memory_limits = [int(available_memory)]
    for constraint in constraints:
        if constraint.cpu_quota is not None:
            cpu_limits.append(constraint.cpu_quota)
        if constraint.memory_headroom_bytes is not None:
            memory_limits.append(constraint.memory_headroom_bytes)
    # A sub-CPU quota still needs one execution thread; the kernel time-slices it.
    budget = max(1, math.floor(min(cpu_limits)))
    return Resources(platform_name, host_cpus, affinity_cpus, budget,
                     available_memory, min(memory_limits), tuple(constraints))


def detect_resources() -> Resources:
    host_cpus = os.cpu_count()
    if host_cpus is None:
        raise RuntimeError("host CPU count unavailable")
    available = int(psutil.virtual_memory().available)
    if sys.platform.startswith("linux"):
        affinity = len(os.sched_getaffinity(0))
        constraints = cgroup_constraints(
            Path("/proc/self/cgroup").read_text(),
            Path("/proc/self/mountinfo").read_text(),
        )
    elif sys.platform == "darwin":
        # macOS exposes no sched_getaffinity or Linux cgroup constraints.
        affinity = host_cpus
        constraints = ()
    else:
        raise RuntimeError("resource detection currently supports Linux and macOS")
    return combine_resources(sys.platform, host_cpus, affinity, available, constraints)
