"""Explicit CPU/RAM headroom and calibration policy, with no machine defaults."""

import math
from dataclasses import dataclass
from pathlib import Path

from generation_resources import Resources


@dataclass(frozen=True)
class GenerationPolicy:
    cpu_reserve: int
    ram_reserve_bytes: int
    tuning_seconds: float

    def __post_init__(self):
        for name in ("cpu_reserve", "ram_reserve_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be an explicit nonnegative integer")
        if isinstance(self.tuning_seconds, bool) or not math.isfinite(self.tuning_seconds) or self.tuning_seconds <= 0:
            raise ValueError("tuning_seconds must be finite and positive")


def load_generation_policy(path: Path) -> GenerationPolicy:
    keys = {"BANDNET_GENERATION_CPU_RESERVE", "BANDNET_GENERATION_RAM_RESERVE_MIB",
            "BANDNET_GENERATION_TUNING_SECONDS"}
    values = {}
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if key not in keys:
            continue
        if not separator or key in values:
            raise ValueError(f"duplicate or malformed generation policy setting: {key}")
        values[key] = value.strip()
    if set(values) != keys:
        raise ValueError(".env.local must explicitly define all three BANDNET_GENERATION policy settings")
    return GenerationPolicy(
        cpu_reserve=int(values["BANDNET_GENERATION_CPU_RESERVE"]),
        ram_reserve_bytes=int(values["BANDNET_GENERATION_RAM_RESERVE_MIB"]) * 1024**2,
        tuning_seconds=float(values["BANDNET_GENERATION_TUNING_SECONDS"]),
    )


def remaining_budgets(resources: Resources, policy: GenerationPolicy) -> tuple[int, int]:
    cpus = resources.cpu_budget - policy.cpu_reserve
    memory = resources.available_memory_bytes - policy.ram_reserve_bytes
    if cpus < 1:
        raise RuntimeError("CPU reserve leaves no generation worker within the detected allocation")
    if memory <= 0:
        raise MemoryError("RAM reserve leaves no generation workspace within detected headroom")
    return cpus, memory
