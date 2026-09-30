"""Shared Stage 3A fixtures: injected hardware facts, profiles, evidence.

Deterministic and offline ONLY (fake-backed). GiB/MiB use exact scaled
integers; no fixture encodes real host truth.
"""

from __future__ import annotations

import copy

from pgate_demo.placement.topology import (
    AttributionStatus,
    ExecutionTopology,
    HostGpuEvidence,
    HostRamEvidence,
    ResidencyReport,
    TopologyEvidence,
)
from pgate_demo.placement.hardware.facts import (
    AdapterKind,
    DetectionStatus,
    GpuAdapterFact,
    HardwareProfile,
    MemoryFact,
)
from pgate_demo.placement.policy.admission import AdmissionPolicy
from pgate_demo.placement.policy.model_profile import ModelProfile

KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
NOW = "2026-08-30T00:00:00+00:00"


# --------------------------------------------------------------------- #
# hardware profiles
# --------------------------------------------------------------------- #
def ram_fact(total=64 * GIB, avail=50 * GIB, status=DetectionStatus.OK):
    return MemoryFact(
        physical_total_bytes=total, observed_available_bytes=avail,
        detection_status=status,
    )


def discrete(vram_total=8 * GIB, free=2 * GIB, foreign=6 * GIB):
    return GpuAdapterFact(
        kind=AdapterKind.DISCRETE, vendor="NVIDIA", name="Fake RTX",
        vram_total_bytes=vram_total, observed_free_vram_bytes=free,
        foreign_usage_bytes=foreign, detection_status=DetectionStatus.OK,
        probe_source="fixture", probe_tool_version="fixture-driver",
    )


def coherent(total=128 * GIB, free=100 * GIB):
    return GpuAdapterFact(
        kind=AdapterKind.COHERENT_UNIFIED, vendor="NVIDIA", name="Fake GB10",
        vram_total_bytes=total, observed_free_vram_bytes=free,
        detection_status=DetectionStatus.OK, probe_source="fixture",
    )


def integrated():
    return GpuAdapterFact(
        kind=AdapterKind.INTEGRATED_SHARED, vendor="IGPU", name="Fake iGPU",
        detection_status=DetectionStatus.OK, probe_source="fixture",
    )


def absent(source="fixture-enumeration"):
    return GpuAdapterFact(
        kind=AdapterKind.ABSENT, detection_status=DetectionStatus.OK,
        probe_source=f"operator_enumeration:{source}",
    )


def hw_discrete(ram_kwargs=None, gpu_kwargs=None):
    return HardwareProfile(
        ram=ram_fact(**(ram_kwargs or {})),
        adapters=(discrete(**(gpu_kwargs or {})),),
        observed_at_utc=NOW,
    )


def hw_cpu_only():
    return HardwareProfile(ram=ram_fact(), adapters=(absent(),), observed_at_utc=NOW)


def hw_unified():
    return HardwareProfile(ram=ram_fact(), adapters=(coherent(),), observed_at_utc=NOW)


def hw_unknown():
    return HardwareProfile(ram=ram_fact(), adapters=(), observed_at_utc=NOW)


def hw_integrated_only():
    return HardwareProfile(
        ram=ram_fact(), adapters=(integrated(),), observed_at_utc=NOW
    )


# --------------------------------------------------------------------- #
# model profiles (digest-bound)
# --------------------------------------------------------------------- #
def profile_large(digest=DIGEST_A, observed=12 * GIB, **kw):
    base = dict(
        digest=digest, architecture="fake-llm", layer_count=32,
        context_limit=8192, kv_bytes_per_token=5 * KIB,
        bytes_per_weight_milli=562, quantization_level="Q4_K_M",
        observed_artifact_bytes=observed, provenance="fixture",
    )
    base.update(kw)
    return ModelProfile(**base)


def profile_param_only(digest=DIGEST_A, **kw):
    base = dict(
        digest=digest, layer_count=32, context_limit=8192,
        kv_bytes_per_token=5 * KIB, bytes_per_weight_milli=562,
        provenance="fixture",
    )
    base.update(kw)
    return ModelProfile(**base)


def profile_geometry_unknown(digest=DIGEST_A, **kw):
    base = dict(
        digest=digest, observed_artifact_bytes=12 * GIB, provenance="fixture"
    )
    base.update(kw)
    return ModelProfile(**base)


def policy_discrete(partial=True, **kw):
    base = dict(
        weight_margin_bps=500,
        required_free_ram_bytes=8 * GIB,
        required_free_vram_bytes=1 * GIB,
        runtime_overhead_bytes=2 * GIB,
        partial_placement_supported=partial,
    )
    base.update(kw)
    return AdmissionPolicy(**base)


def policy_unified(**kw):
    base = dict(
        weight_margin_bps=500,
        required_free_ram_bytes=8 * GIB,
        required_free_unified_bytes=8 * GIB,
        runtime_overhead_bytes=2 * GIB,
    )
    base.update(kw)
    return AdmissionPolicy(**base)


def policy_cpu_only(**kw):
    base = dict(
        weight_margin_bps=500,
        required_free_ram_bytes=8 * GIB,
        runtime_overhead_bytes=2 * GIB,
    )
    base.update(kw)
    return AdmissionPolicy(**base)


# --------------------------------------------------------------------- #
# attestation evidence channel builders
# --------------------------------------------------------------------- #
def ev_frontload_agree(size=15 * GIB, vram=6 * GIB, host_vram=6 * GIB,
                       host_ram=9 * GIB):
    return TopologyEvidence(
        requested_topology=ExecutionTopology.CPU_FRONTLOADED_GPU_ASSISTED,
        ollama_reported=ResidencyReport(
            schema_supported=True, size_bytes=size, size_vram_bytes=vram
        ),
        host_gpu=HostGpuEvidence(
            attributable_model_vram_bytes=host_vram, detection_status="OK"
        ),
        host_ram=HostRamEvidence(
            attributable_model_ram_bytes=host_ram, detection_status="OK"
        ),
        attribution=AttributionStatus.ATTRIBUTED,
    )


def ev_with(**updates):
    """Agreeing frontload evidence with per-channel overrides (deep copy)."""
    ev = ev_frontload_agree()
    data = ev.model_dump()
    for key, value in updates.items():
        data[key] = None if value is None else (
            value.model_dump() if hasattr(value, "model_dump") else value
        )
    return TopologyEvidence.model_validate(data)
