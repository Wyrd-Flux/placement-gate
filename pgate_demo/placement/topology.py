"""Stage 3A execution-topology types and multi-source attestation model.

Frozen Stage 1 PlacementClass semantics are NOT altered here:
ExecutionTopology is a SEPARATELY VERSIONED CURRENT EXECUTION DECISION axis.
historical_placement_class remains an immutable historical policy
classification; CPU_REQUIRED is never equated with CPU_RESIDENT.

Attestation is OCCURRENCE-based: one admission may accumulate many
immutable TopologyAttestation records over time; attestation never mutates
an AdmissionDecision.
"""

from __future__ import annotations

import enum
import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator


class ExecutionTopology(str, enum.Enum):
    GPU_RESIDENT = "GPU_RESIDENT"
    CPU_FRONTLOADED_GPU_ASSISTED = "CPU_FRONTLOADED_GPU_ASSISTED"
    CPU_RESIDENT = "CPU_RESIDENT"
    UNIFIED_MEMORY_ACCELERATED = "UNIFIED_MEMORY_ACCELERATED"
    PROHIBITED = "PROHIBITED"


class AttestationState(str, enum.Enum):
    CONFIRMED = "CONFIRMED"
    MISMATCH = "MISMATCH"
    UNCONFIRMED = "UNCONFIRMED"
    PROHIBITED = "PROHIBITED"


class AttributionStatus(str, enum.Enum):
    ATTRIBUTED = "ATTRIBUTED"
    UNATTRIBUTED = "UNATTRIBUTED"
    FOREIGN_PRESENT = "FOREIGN_PRESENT"


class ResidencyReport(BaseModel):
    """Channel 2: Ollama-REPORTED residency evidence (never physical truth).

    Only machine-readable API fields (e.g. /api/ps size / size_vram where the
    installed schema supports them) belong here; the human CLI PROCESSOR
    presentation is not an API contract.
    """

    model_config = ConfigDict(frozen=True)

    schema_supported: bool
    size_bytes: Optional[int] = None
    size_vram_bytes: Optional[int] = None
    reported_by: str = "ollama_api_ps"

    @field_validator("size_bytes", "size_vram_bytes")
    @classmethod
    def _nonneg(cls, v):
        if v is not None and v < 0:
            raise ValueError("reported residency sizes must be nonnegative")
        return v


class HostGpuEvidence(BaseModel):
    """Channel 3: host-observed GPU memory attributed to the load."""

    model_config = ConfigDict(frozen=True)

    attributable_model_vram_bytes: Optional[int] = None
    detection_status: str  # OK | UNKNOWN


class HostRamEvidence(BaseModel):
    """Channel 4: host-observed system-memory backing attributed to the load."""

    model_config = ConfigDict(frozen=True)

    attributable_model_ram_bytes: Optional[int] = None
    detection_status: str  # OK | UNKNOWN


class TopologyEvidence(BaseModel):
    """Five independently preserved attestation channels (v2 §G / v3 §5).

    Channel 1 (requested_topology) comes from the immutable admission record;
    channels 2-5 are later observations. None means the channel is MISSING —
    missing required evidence must never become CONFIRMED.
    """

    model_config = ConfigDict(frozen=True)

    requested_topology: ExecutionTopology
    ollama_reported: Optional[ResidencyReport] = None
    host_gpu: Optional[HostGpuEvidence] = None
    host_ram: Optional[HostRamEvidence] = None
    attribution: Optional[AttributionStatus] = None

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def _coherent_int(v) -> bool:
    return v is not None and v >= 0


def derive_attestation(
    ev: TopologyEvidence, tau_vram_bytes: int = 0
) -> AttestationState:
    """Structural multi-source derivation of an attestation occurrence.

    Rules (frozen by v3 §5/G-3AT):
      - any channel required by the requested topology that is missing or
        UNKNOWN  -> UNCONFIRMED
      - channels that agree with the requested topology's structural shape
        and none contradicts it -> CONFIRMED
      - any two present channels contradict the requested shape -> MISMATCH
      - PROHIBITED requested -> PROHIBITED (short-circuit)
    tau_vram_bytes is MEASUREMENT UNCERTAINTY only (versioned policy input);
    it is never a semantic topology threshold.
    """
    t = ev.requested_topology
    if t is ExecutionTopology.PROHIBITED:
        return AttestationState.PROHIBITED

    rep = ev.ollama_reported
    gpu = ev.host_gpu
    ram = ev.host_ram
    att = ev.attribution

    # channel presence
    if rep is None or gpu is None or ram is None or att is None:
        return AttestationState.UNCONFIRMED
    if not rep.schema_supported:
        return AttestationState.UNCONFIRMED
    if gpu.detection_status != "OK" or ram.detection_status != "OK":
        return AttestationState.UNCONFIRMED

    size = rep.size_bytes
    vram = rep.size_vram_bytes
    g_bytes = gpu.attributable_model_vram_bytes
    r_bytes = ram.attributable_model_ram_bytes
    if not all(_coherent_int(x) for x in (size, vram, g_bytes, r_bytes)):
        return AttestationState.UNCONFIRMED

    split = (size - vram) > tau_vram_bytes  # material system-memory backing
    fully_gpu = vram >= (size - tau_vram_bytes)
    zero_vram = vram == 0 and g_bytes == 0

    if t is ExecutionTopology.GPU_RESIDENT:
        if fully_gpu and g_bytes > 0:
            return AttestationState.CONFIRMED
        if _contradicts_gpu_resident(size, vram, g_bytes, tau_vram_bytes):
            return AttestationState.MISMATCH
        return AttestationState.UNCONFIRMED

    if t is ExecutionTopology.CPU_FRONTLOADED_GPU_ASSISTED:
        if split and g_bytes > 0 and r_bytes > 0 and att is AttributionStatus.ATTRIBUTED:
            return AttestationState.CONFIRMED
        # contradiction: report/host show NO gpu subset at all, or report/host
        # show full GPU residency when frontload was requested
        if (vram == 0 and g_bytes == 0) or fully_gpu:
            return AttestationState.MISMATCH
        if att is AttributionStatus.UNATTRIBUTED and g_bytes == 0:
            return AttestationState.MISMATCH
        return AttestationState.UNCONFIRMED

    if t is ExecutionTopology.CPU_RESIDENT:
        if zero_vram and g_bytes == 0 and r_bytes > 0:
            return AttestationState.CONFIRMED
        if vram > 0 or g_bytes > 0:
            return AttestationState.MISMATCH
        return AttestationState.UNCONFIRMED

    if t is ExecutionTopology.UNIFIED_MEMORY_ACCELERATED:
        # coherent unified: single-pool backing with acceleration active.
        # Ollama-reported VRAM is not meaningful on this branch; confirmation
        # requires attributed host evidence and attribution, and NO evidence
        # of a separate unsatisfied discrete-VRAM requirement.
        if (
            att is AttributionStatus.ATTRIBUTED
            and r_bytes > 0
            and rep.schema_supported
        ):
            return AttestationState.CONFIRMED
        if att is AttributionStatus.UNATTRIBUTED and r_bytes == 0:
            return AttestationState.MISMATCH
        return AttestationState.UNCONFIRMED

    return AttestationState.UNCONFIRMED


def _contradicts_gpu_resident(size, vram, g_bytes, tau) -> bool:
    # present channels directly asserting split/CPU-only residency
    if vram == 0 or g_bytes == 0:
        return True
    if size - vram > tau:
        return True
    return False
