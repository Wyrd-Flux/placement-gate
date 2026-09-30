"""Stage 3A hardware observer: protocol, real assembly, and deterministic fake.

Facts only; no admission decisions live here (v2 §B layering rule).
Positive GPU-absence evidence arrives ONLY as operator-supplied typed facts;
probe failure yields UNKNOWN adapters, never ABSENT (v3 §6).
"""

from __future__ import annotations

from typing import Optional, Protocol

from .facts import (
    AdapterKind,
    DetectionStatus,
    GpuAdapterFact,
    HardwareProfile,
    MemoryFact,
)


class HardwareObserver(Protocol):
    def observe(self) -> HardwareProfile:  # pragma: no cover - protocol
        ...


def absence_fact(source: str) -> GpuAdapterFact:
    return GpuAdapterFact(
        kind=AdapterKind.ABSENT,
        detection_status=DetectionStatus.OK,
        probe_source=f"operator_enumeration:{source}",
    )


class RealHardwareObserver:
    """Assembles RAM facts + GPU facts into an immutable HardwareProfile."""

    def __init__(self, memory_probe, nvidia_adapter=None, gpu_absent_source=None):
        self._mem = memory_probe
        self._nv = nvidia_adapter
        self._absent_source = gpu_absent_source

    def observe(self) -> HardwareProfile:
        ram: MemoryFact = self._mem.observe()
        adapters: list[GpuAdapterFact] = []
        if self._nv is not None:
            adapters.extend(self._nv.probe())  # [] == no conclusion, not absence
        if self._absent_source:
            # presence is NEVER suppressed by other evidence: if both a
            # discrete probe and positive absence evidence exist, BOTH are
            # recorded; the profile then reports UNKNOWN/contradictory and
            # admission fails closed (v3 review §2 — no "discrete wins").
            adapters.append(absence_fact(self._absent_source))
        return HardwareProfile(ram=ram, adapters=tuple(adapters))


class FakeHardwareObserver:
    """Deterministic injected profile (Stage 3A / tests / CLI proof)."""

    def __init__(self, profile: HardwareProfile):
        self._profile = profile
        self.calls = 0

    def observe(self) -> HardwareProfile:
        self.calls += 1
        return self._profile
