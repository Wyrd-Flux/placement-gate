"""Stage 3A resource admission: hardware-profile-first branching + budgets.

Decisions only; observation lives in hardware/. Parameter count is never
memory size; parameter-derived weight bytes are explicitly approximate with
a conservative versioned margin (v2 §F, v3 §2).

Budget equations (v3 §3 review / v2 §F2) preserve raw facts and subtract
each quantity exactly once:

    admissible_new_ram  = max(0, observed_available  - required_free_reserve
                              - outstanding_ram_reservations)
    admissible_new_vram = max(0, observed_free_vram  - required_vram_reserve
                              - outstanding_vram_reservations)

Foreign GPU usage that is already reflected inside observed_free_vram /
observed_available is NEVER subtracted again (it remains attribution
evidence only).
"""

from __future__ import annotations

import enum
import json
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from ..topology import ExecutionTopology
from ..hardware.facts import AdapterKind, DetectionStatus, HardwareProfile
from .model_profile import (
    ModelProfile,
    ParameterCountAmbiguous,
    WeightSizeEstimate,
    kv_cache_bytes,
    weight_size_from_profile,
)

POLICY_VERSION = "hw-admission-v3a.1"


class AdmitReason(str, enum.Enum):
    ADMITTED = "ADMITTED"
    HARDWARE_UNKNOWN = "HARDWARE_UNKNOWN"
    GEOMETRY_UNKNOWN = "GEOMETRY_UNKNOWN"
    EXCEEDS_RAM_BUDGET = "EXCEEDS_RAM_BUDGET"
    PARTIAL_PLACEMENT_UNSUPPORTED = "PARTIAL_PLACEMENT_UNSUPPORTED"
    UNIFIED_CPU_FALLBACK_NOT_SELECTED = "UNIFIED_CPU_FALLBACK_NOT_SELECTED"
    ESTIMATE_UNAVAILABLE = "ESTIMATE_UNAVAILABLE"


def admissible_new(
    observed_free_bytes: Optional[int],
    required_reserve_bytes: int,
    outstanding_reservation_bytes: int,
) -> Optional[int]:
    """Shared non-double-counting budget equation. None input -> None budget
    (unknown: caller must fail closed; unknown is never coerced to 0)."""
    if observed_free_bytes is None:
        return None
    value = (
        observed_free_bytes - required_reserve_bytes - outstanding_reservation_bytes
    )
    return max(0, value)


class AdmissionPolicy(BaseModel):
    """Versioned policy inputs. Reserves/margins are explicit inputs, never
    arbitrary hidden constants."""

    model_config = ConfigDict(frozen=True)

    version: str = POLICY_VERSION
    weight_margin_bps: int = 0
    required_free_ram_bytes: int = 0
    required_free_vram_bytes: int = 0
    required_free_unified_bytes: int = 0
    runtime_overhead_bytes: int = 0
    overhead_assumptions: str = "caller-supplied measured/justified overhead"
    partial_placement_supported: bool = False  # capability-gated, verified
    unified_cpu_fallback_allowed: bool = False
    tau_vram_bytes: int = 0  # measurement uncertainty ONLY (versioned)

    @field_validator("weight_margin_bps", "required_free_ram_bytes",
                     "required_free_vram_bytes", "required_free_unified_bytes",
                     "runtime_overhead_bytes", "tau_vram_bytes")
    @classmethod
    def _nn(cls, v: int) -> int:
        if v < 0:
            raise ValueError("policy quantities must be nonnegative")
        return v


class AdmissionPlan(BaseModel):
    """The full deterministic result of one admission computation."""

    model_config = ConfigDict(frozen=True)

    admitted: bool
    requested_topology: ExecutionTopology
    reason: AdmitReason
    detail: str = ""
    weights: Optional[WeightSizeEstimate] = None
    kv_bytes: Optional[int] = None
    total_required_bytes: int = 0
    budget_ram: Optional[int] = None
    budget_vram: Optional[int] = None
    budget_unified: Optional[int] = None
    policy_version: str = POLICY_VERSION


class AdmissionDecision(BaseModel):
    """IMMUTABLE decision at admission time (v3 §5).

    Never carries a mutable attestation field; attestations are separate
    append-only occurrences referencing decision_id. It pins the EXACT
    evidence hashes used — later probes/profiles cannot alter this record.
    """

    model_config = ConfigDict(frozen=True)

    decision_id: str  # OCCURRENCE identity (uuid hex), never content-derived
    session_id: str
    identity_id: Optional[str] = None
    model_digest: str
    model_profile_hash: str
    hardware_profile_hash: str
    requested_topology: ExecutionTopology
    reservation_ram_bytes: int = 0
    reservation_vram_bytes: int = 0
    total_required_bytes: int
    policy_version: str = POLICY_VERSION
    admitted: bool
    reason: AdmitReason
    detail: str = ""
    occurred_at_utc: str

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")


def _prohibited(reason: AdmitReason, detail: str, **ctx) -> AdmissionPlan:
    return AdmissionPlan(
        admitted=False,
        requested_topology=ExecutionTopology.PROHIBITED,
        reason=reason,
        detail=detail,
        **ctx,
    )


def admit(
    profile: ModelProfile,
    hardware: HardwareProfile,
    *,
    parameter_count: int,
    requested_context_tokens: int,
    policy: AdmissionPolicy,
    outstanding_ram_bytes: int = 0,
    outstanding_vram_bytes: int = 0,
    kv_envelope_bytes: Optional[int] = None,
) -> AdmissionPlan:
    """Hardware-profile-first admission (v3 §2). Branch precedes arithmetic."""
    budgets = {}
    if hardware.detection_status is DetectionStatus.UNKNOWN:
        return _prohibited(
            AdmitReason.HARDWARE_UNKNOWN,
            "hardware evidence incomplete/unknown; fail closed",
        )

    try:
        weights = weight_size_from_profile(
            profile, parameter_count, policy_margin_bps=policy.weight_margin_bps
        )
    except ParameterCountAmbiguous as e:
        return _prohibited(AdmitReason.ESTIMATE_UNAVAILABLE, str(e))

    kv = kv_cache_bytes(profile, requested_context_tokens)
    if kv is None:
        kv = kv_envelope_bytes  # explicit operator envelope, if provided
    if kv is None:
        return _prohibited(
            AdmitReason.GEOMETRY_UNKNOWN,
            "required KV geometry unknown and no operator envelope supplied",
        )

    total = weights.estimated_bytes + kv + policy.runtime_overhead_bytes
    ram_budget = admissible_new(
        hardware.ram.observed_available_bytes if hardware.ram.detection_status
        is DetectionStatus.OK else None,
        policy.required_free_ram_bytes,
        outstanding_ram_bytes,
    )

    # ---------- branch 1: positively classified coherent unified host ------
    if hardware.is_coherent_unified:
        coherent = [
            a for a in hardware.adapters
            if a.kind is AdapterKind.COHERENT_UNIFIED
        ]
        free_vals = [a.observed_free_vram_bytes for a in coherent]
        # MISSING availability is NEVER substituted by total capacity (v3
        # review §1): existing consumption cannot disappear. UNKNOWN stays
        # UNKNOWN -> fail closed.
        if not coherent or any(v is None for v in free_vals):
            ctx = {
                "weights": weights, "kv_bytes": kv,
                "total_required_bytes": total,
            }
            return _prohibited(
                AdmitReason.HARDWARE_UNKNOWN,
                "coherent unified AVAILABLE memory unobserved; capacity "
                "substitution is prohibited",
                **ctx,
            )
        obs_free = int(sum(free_vals))
        unified_budget = admissible_new(
            obs_free, policy.required_free_unified_bytes, outstanding_ram_bytes
        )
        ctx = {
            "weights": weights, "kv_bytes": kv, "total_required_bytes": total,
            "budget_unified": unified_budget,
        }
        if unified_budget is None:
            return _prohibited(
                AdmitReason.HARDWARE_UNKNOWN,
                "unified pool availability unknown; fail closed", **ctx,
            )
        if total <= unified_budget:
            return AdmissionPlan(
                admitted=True,
                requested_topology=ExecutionTopology.UNIFIED_MEMORY_ACCELERATED,
                reason=AdmitReason.ADMITTED,
                **ctx,
            )
        if policy.unified_cpu_fallback_allowed and ram_budget is not None \
                and total <= ram_budget:
            # separately supported / policy-selected mode ONLY (v3 §2)
            return AdmissionPlan(
                admitted=True,
                requested_topology=ExecutionTopology.CPU_RESIDENT,
                reason=AdmitReason.ADMITTED,
                detail="policy-selected CPU mode on unified host",
                **ctx,
            )
        return _prohibited(
            AdmitReason.UNIFIED_CPU_FALLBACK_NOT_SELECTED
            if not policy.unified_cpu_fallback_allowed
            else AdmitReason.EXCEEDS_RAM_BUDGET,
            "exceeds coherent unified budget and no policy-selected CPU mode",
            **ctx,
        )

    # ---------- branch 2: discrete GPU host --------------------------------
    if hardware.has_discrete_gpu:
        vram_budget = admissible_new(
            hardware.free_vram(), policy.required_free_vram_bytes,
            outstanding_vram_bytes,
        )
        ctx = {
            "weights": weights, "kv_bytes": kv, "total_required_bytes": total,
            "budget_ram": ram_budget, "budget_vram": vram_budget,
        }
        if ram_budget is None or vram_budget is None:
            return _prohibited(
                AdmitReason.HARDWARE_UNKNOWN, "budget inputs unknown; fail closed",
                **ctx,
            )
        if total <= vram_budget:
            return AdmissionPlan(
                admitted=True,
                requested_topology=ExecutionTopology.GPU_RESIDENT,
                reason=AdmitReason.ADMITTED, **ctx,
            )
        if total <= ram_budget:
            if policy.partial_placement_supported:
                return AdmissionPlan(
                    admitted=True,
                    requested_topology=ExecutionTopology.CPU_FRONTLOADED_GPU_ASSISTED,
                    reason=AdmitReason.ADMITTED,
                    detail="working set exceeds VRAM budget; admitted GPU subset",
                    **ctx,
                )
            return _prohibited(
                AdmitReason.PARTIAL_PLACEMENT_UNSUPPORTED,
                "fits RAM but partial placement capability not verified", **ctx,
            )
        return _prohibited(
            AdmitReason.EXCEEDS_RAM_BUDGET,
            "working set exceeds admissible system RAM budget", **ctx,
        )

    # ---------- branch 3: no usable GPU, positively proven -----------------
    if hardware.gpu_absence_proven:
        ctx = {
            "weights": weights, "kv_bytes": kv, "total_required_bytes": total,
            "budget_ram": ram_budget,
        }
        if ram_budget is None:
            return _prohibited(AdmitReason.HARDWARE_UNKNOWN, "ram budget unknown", **ctx)
        if total <= ram_budget:
            return AdmissionPlan(
                admitted=True,
                requested_topology=ExecutionTopology.CPU_RESIDENT,
                reason=AdmitReason.ADMITTED, **ctx,
            )
        return _prohibited(
            AdmitReason.EXCEEDS_RAM_BUDGET, "working set exceeds admissible RAM", **ctx
        )

    # ---------- branch 4: everything else is UNKNOWN -----------------------
    return _prohibited(
        AdmitReason.HARDWARE_UNKNOWN,
        "no positively classified topology branch applies; fail closed",
    )
