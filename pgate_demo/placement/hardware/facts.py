"""Stage 3A hardware facts: pure observations, never decisions (v2 §C).

HardwareProfile is serialized canonically and stored content-addressed;
OCCURRENCE identity (record rows) is separate from CONTENT identity
(canonical bytes / content hash) per the Stage 2 lesson.
"""

from __future__ import annotations

import enum
import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

PROVENANCE_VERSION = "stage3a.1"


class AdapterKind(str, enum.Enum):
    DISCRETE = "DISCRETE"
    INTEGRATED_SHARED = "INTEGRATED_SHARED"
    COHERENT_UNIFIED = "COHERENT_UNIFIED"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


class DetectionStatus(str, enum.Enum):
    OK = "OK"
    UNKNOWN = "UNKNOWN"


class MemoryFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    physical_total_bytes: int
    observed_available_bytes: int
    detection_status: DetectionStatus = DetectionStatus.OK

    @field_validator("physical_total_bytes", "observed_available_bytes")
    @classmethod
    def _nn(cls, v: int) -> int:
        if v < 0:
            raise ValueError("memory facts cannot be negative")
        return v

    @model_validator(mode="after")
    def _status_coherence(self) -> "MemoryFact":
        if self.detection_status is DetectionStatus.OK:
            if self.physical_total_bytes <= 0:
                raise ValueError("OK memory fact requires positive physical total")
            if self.observed_available_bytes > self.physical_total_bytes:
                raise ValueError("available cannot exceed physical total")
        else:  # UNKNOWN carries no usable numbers
            object.__setattr__(self, "physical_total_bytes", 0)
            object.__setattr__(self, "observed_available_bytes", 0)
        return self


class GpuAdapterFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: AdapterKind
    vendor: str = ""
    name: str = ""
    vram_total_bytes: Optional[int] = None
    observed_free_vram_bytes: Optional[int] = None
    foreign_usage_bytes: Optional[int] = None  # evidence only; see accounting
    detection_status: DetectionStatus = DetectionStatus.OK
    probe_source: str = ""
    probe_tool_version: str = ""


class GpuUtilizationFact(BaseModel):
    """Point-in-time utilization evidence; basis points avoid float drift."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gpu_utilization_bps: int
    memory_utilization_bps: int
    probe_source: str = ""

    @field_validator("gpu_utilization_bps", "memory_utilization_bps")
    @classmethod
    def _percent_bps(cls, value: int) -> int:
        if isinstance(value, bool) or not 0 <= value <= 10_000:
            raise ValueError("utilization must be within 0..10000 basis points")
        return value


class HardwareProfile(BaseModel):
    """Immutable hardware evidence used by exactly one admission decision.

    coherent_unified requires POSITIVE classification; integrated/shared
    memory alone NEVER implies coherent unified memory (v2 §H / v3 §2).
    """

    model_config = ConfigDict(frozen=True)

    ram: MemoryFact
    adapters: tuple[GpuAdapterFact, ...] = ()
    observed_at_utc: str = ""
    profile_version: str = PROVENANCE_VERSION

    @property
    def is_contradictory(self) -> bool:
        """Conflicting adapter classes remain VISIBLE (v3 review §2).

        Contradictory facts are never silently normalized into a preferred
        class: DISCRETE+COHERENT_UNIFIED or ABSENT+(DISCRETE|COHERENT) are
        contradictions, and admission fails closed on them.
        """
        kinds = {a.kind for a in self.adapters}
        has_presence = bool(kinds & {AdapterKind.DISCRETE, AdapterKind.COHERENT_UNIFIED})
        if AdapterKind.ABSENT in kinds and has_presence:
            return True
        if (
            AdapterKind.DISCRETE in kinds
            and AdapterKind.COHERENT_UNIFIED in kinds
        ):
            return True
        return False

    @property
    def detection_status(self) -> DetectionStatus:
        if self.ram.detection_status is DetectionStatus.UNKNOWN:
            return DetectionStatus.UNKNOWN
        if self.is_contradictory:
            return DetectionStatus.UNKNOWN
        kinds = {a.kind for a in self.adapters}
        if not kinds or kinds == {AdapterKind.UNKNOWN}:
            return DetectionStatus.UNKNOWN
        return DetectionStatus.OK

    @property
    def is_coherent_unified(self) -> bool:
        return any(a.kind is AdapterKind.COHERENT_UNIFIED for a in self.adapters)

    @property
    def has_discrete_gpu(self) -> bool:
        return any(
            a.kind is AdapterKind.DISCRETE and a.detection_status is DetectionStatus.OK
            for a in self.adapters
        )

    @property
    def gpu_absence_proven(self) -> bool:
        # positive proof of no usable GPU (operator-supplied enumeration
        # evidence); probe failure alone can never yield this (v3 §6).
        return bool(self.adapters) and all(
            a.kind is AdapterKind.ABSENT for a in self.adapters
        )

    def total_vram(self) -> Optional[int]:
        vals = [
            a.vram_total_bytes
            for a in self.adapters
            if a.kind is AdapterKind.DISCRETE and a.vram_total_bytes is not None
        ]
        return sum(vals) if vals and len(vals) == len(
            [a for a in self.adapters if a.kind is AdapterKind.DISCRETE]
        ) else None

    def free_vram(self) -> Optional[int]:
        vals = [
            a.observed_free_vram_bytes
            for a in self.adapters
            if a.kind is AdapterKind.DISCRETE and a.observed_free_vram_bytes is not None
        ]
        return sum(vals) if vals and len(vals) == len(
            [a for a in self.adapters if a.kind is AdapterKind.DISCRETE]
        ) else None

    def unified_capacity(self) -> Optional[int]:
        vals = [
            a.vram_total_bytes
            for a in self.adapters
            if a.kind is AdapterKind.COHERENT_UNIFIED and a.vram_total_bytes is not None
        ]
        return sum(vals) if vals else None

    def canonical_bytes(self) -> bytes:
        """Deterministic canonical serialization for content addressing."""
        d = self.model_dump(mode="json")
        d["adapters"] = [a for a in d["adapters"]]
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "HardwareProfile":
        return cls.model_validate(json.loads(raw.decode("utf-8")))
