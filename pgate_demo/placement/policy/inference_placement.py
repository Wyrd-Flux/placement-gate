"""Load-boundary resource-aware inference placement planning.

This module is deliberately additive.  It does not alter the frozen historical
``PlacementClass`` axis and it does not claim that Ollama's ``num_gpu`` input is
an exact layer-placement control.  Plans are requests which require separate
post-load evidence.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ..topology import (
    AttestationState,
    ExecutionTopology,
    TopologyEvidence,
    derive_attestation,
)
from ..hardware.facts import DetectionStatus, HardwareProfile
from .admission import AdmissionDecision, AdmitReason, admissible_new
from .model_profile import (
    ModelProfile,
    ParameterCountAmbiguous,
    WeightSizeEstimate,
    kv_cache_bytes,
    weight_size_from_profile,
)

PLACEMENT_POLICY_VERSION = "resource-placement-v1"
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class ComputeDomain(str, enum.Enum):
    CPU = "CPU"
    CUDA = "CUDA"


class MemoryDomain(str, enum.Enum):
    HOST_LOCAL = "HOST_LOCAL"
    DEVICE_LOCAL = "DEVICE_LOCAL"
    HOST_GPU_ADDRESSABLE = "HOST_GPU_ADDRESSABLE"


class PlacementPolicy(str, enum.Enum):
    DEVICE_ONLY = "DEVICE_ONLY"
    HOST_ONLY = "HOST_ONLY"
    HYBRID_STATIC = "HYBRID_STATIC"
    MANAGED_FALLBACK = "MANAGED_FALLBACK"


class PlacementVerificationState(str, enum.Enum):
    REQUESTED = "REQUESTED"
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIABLE = "UNVERIFIABLE"
    FAILED = "FAILED"


class PlacementReason(str, enum.Enum):
    PLANNED = "PLANNED"
    HARDWARE_UNKNOWN = "HARDWARE_UNKNOWN"
    GEOMETRY_UNKNOWN = "GEOMETRY_UNKNOWN"
    ESTIMATE_UNAVAILABLE = "ESTIMATE_UNAVAILABLE"
    POLICY_UNSUPPORTED = "POLICY_UNSUPPORTED"
    CAPABILITY_UNVERIFIED = "CAPABILITY_UNVERIFIED"
    EXCEEDS_RESOURCE_BUDGET = "EXCEEDS_RESOURCE_BUDGET"


class PlacementPlanningPolicy(BaseModel):
    """Explicit operator/reviewer inputs; no hidden size threshold exists."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    placement: PlacementPolicy
    required_free_ram_bytes: int = 0
    required_free_vram_bytes: int = 0
    runtime_overhead_bytes: int = 0
    weight_margin_bps: int = 0
    hybrid_supported: bool = False
    managed_fallback_supported: bool = False
    managed_fallback_order: tuple[PlacementPolicy, ...] = ()
    requested_device_bytes: Optional[int] = None
    requested_num_gpu: Optional[int] = None
    policy_version: str = PLACEMENT_POLICY_VERSION

    @field_validator(
        "required_free_ram_bytes", "required_free_vram_bytes",
        "runtime_overhead_bytes", "weight_margin_bps",
    )
    @classmethod
    def _nonnegative(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("policy quantities must be nonnegative integers")
        return value

    @field_validator("requested_device_bytes", "requested_num_gpu")
    @classmethod
    def _optional_nonnegative(cls, value: Optional[int]) -> Optional[int]:
        if value is not None and (isinstance(value, bool) or value < 0):
            raise ValueError("requested values must be nonnegative integers")
        return value

    @model_validator(mode="after")
    def _coherent(self) -> "PlacementPlanningPolicy":
        if self.placement is PlacementPolicy.MANAGED_FALLBACK:
            if not self.managed_fallback_supported:
                raise ValueError("managed fallback requires verified capability")
            if not self.managed_fallback_order:
                raise ValueError("managed fallback requires an explicit order")
            if PlacementPolicy.MANAGED_FALLBACK in self.managed_fallback_order:
                raise ValueError("managed fallback order cannot recurse")
        elif self.managed_fallback_order:
            raise ValueError("fallback order is valid only for MANAGED_FALLBACK")
        return self


class ResourceClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_domain: MemoryDomain
    bytes: int
    physical_claim: bool
    note: str = ""

    @field_validator("bytes")
    @classmethod
    def _claim_nonnegative(cls, value: int) -> int:
        if isinstance(value, bool) or value < 0:
            raise ValueError("resource claim must be nonnegative")
        return value


class ModelAdmissionPlan(BaseModel):
    """Canonical inspectable result of one load-boundary calculation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    admitted: bool
    reason: PlacementReason
    detail: str = ""
    model_digest: str
    model_id: str = ""
    model_manifest_digest: Optional[str] = None
    weights_digest: Optional[str] = None
    model_profile_hash: str
    hardware_profile_hash: str
    model_architecture: str
    quantization_level: str
    parameter_count_metadata: int
    requested_context_tokens: int
    weights: Optional[WeightSizeEstimate] = None
    kv_bytes: Optional[int] = None
    runtime_overhead_bytes: int
    total_required_bytes: int = 0
    observed_available_ram_bytes: Optional[int] = None
    physical_ram_total_bytes: int
    observed_free_vram_bytes: Optional[int] = None
    dedicated_vram_total_bytes: Optional[int] = None
    observed_free_unified_bytes: Optional[int] = None
    host_gpu_addressable_bytes: Optional[int] = None
    gpu_utilization_bps: Optional[int] = None
    gpu_memory_utilization_bps: Optional[int] = None
    required_free_ram_bytes: int
    required_free_vram_bytes: int
    outstanding_ram_bytes: int
    outstanding_vram_bytes: int
    budget_ram_bytes: Optional[int] = None
    budget_vram_bytes: Optional[int] = None
    budget_unified_bytes: Optional[int] = None
    requested_policy: PlacementPolicy
    selected_policy: Optional[PlacementPolicy] = None
    requested_topology: ExecutionTopology = ExecutionTopology.PROHIBITED
    compute_domains: tuple[ComputeDomain, ...] = ()
    memory_domains: tuple[MemoryDomain, ...] = ()
    claims: tuple[ResourceClaim, ...] = ()
    requested_num_gpu: Optional[int] = None
    verification_state: PlacementVerificationState = (
        PlacementVerificationState.REQUESTED
    )
    policy_version: str = PLACEMENT_POLICY_VERSION
    recalculation_boundary: str = "LOAD_BOUNDARY_ONLY"

    @field_validator("model_manifest_digest", "weights_digest")
    @classmethod
    def _optional_digest(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and _SHA256.fullmatch(value) is None:
            raise ValueError("explicit model identities must be 64-hex SHA-256")
        return value.lower() if value is not None else None

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


def _candidate_order(policy: PlacementPlanningPolicy) -> tuple[PlacementPolicy, ...]:
    if policy.placement is PlacementPolicy.MANAGED_FALLBACK:
        return policy.managed_fallback_order
    return (policy.placement,)


def plan_inference_placement(
    profile: ModelProfile,
    hardware: HardwareProfile,
    *,
    parameter_count: int,
    requested_context_tokens: int,
    policy: PlacementPlanningPolicy,
    outstanding_ram_bytes: int = 0,
    outstanding_vram_bytes: int = 0,
    kv_envelope_bytes: Optional[int] = None,
    host_gpu_addressable_bytes: Optional[int] = None,
    gpu_utilization_bps: Optional[int] = None,
    gpu_memory_utilization_bps: Optional[int] = None,
    model_id: str = "",
    model_manifest_digest: Optional[str] = None,
    weights_digest: Optional[str] = None,
) -> ModelAdmissionPlan:
    """Create a plan without loading a model or changing a running placement."""
    common = dict(
        model_digest=profile.digest,
        model_id=model_id,
        model_manifest_digest=model_manifest_digest,
        weights_digest=weights_digest,
        model_profile_hash=profile.content_hash(),
        hardware_profile_hash=hashlib.sha256(
            hardware.canonical_bytes()
        ).hexdigest(),
        model_architecture=profile.architecture,
        quantization_level=profile.quantization_level,
        parameter_count_metadata=parameter_count,
        requested_context_tokens=requested_context_tokens,
        runtime_overhead_bytes=policy.runtime_overhead_bytes,
        required_free_ram_bytes=policy.required_free_ram_bytes,
        required_free_vram_bytes=policy.required_free_vram_bytes,
        outstanding_ram_bytes=outstanding_ram_bytes,
        outstanding_vram_bytes=outstanding_vram_bytes,
        requested_policy=policy.placement,
        requested_num_gpu=policy.requested_num_gpu,
        policy_version=policy.policy_version,
        physical_ram_total_bytes=hardware.ram.physical_total_bytes,
        dedicated_vram_total_bytes=hardware.total_vram(),
        gpu_utilization_bps=gpu_utilization_bps,
        gpu_memory_utilization_bps=gpu_memory_utilization_bps,
    )
    if any(isinstance(v, bool) or v < 0 for v in (
        parameter_count, requested_context_tokens, outstanding_ram_bytes,
        outstanding_vram_bytes,
    )):
        raise ValueError("planning quantities must be nonnegative integers")
    if hardware.detection_status is DetectionStatus.UNKNOWN:
        return ModelAdmissionPlan(
            admitted=False, reason=PlacementReason.HARDWARE_UNKNOWN,
            detail="hardware evidence is unknown or contradictory", **common,
            verification_state=PlacementVerificationState.FAILED,
        )
    if host_gpu_addressable_bytes is not None:
        if (isinstance(host_gpu_addressable_bytes, bool)
                or host_gpu_addressable_bytes < 0
                or host_gpu_addressable_bytes > hardware.ram.physical_total_bytes):
            raise ValueError("GPU-addressable host aperture must be a RAM subset")
    for label, value in (
        ("gpu_utilization_bps", gpu_utilization_bps),
        ("gpu_memory_utilization_bps", gpu_memory_utilization_bps),
    ):
        if value is not None and (
            isinstance(value, bool) or value < 0 or value > 10_000
        ):
            raise ValueError(f"{label} must be within 0..10000")
    try:
        weights = weight_size_from_profile(
            profile, parameter_count, policy.weight_margin_bps
        )
    except ParameterCountAmbiguous as exc:
        return ModelAdmissionPlan(
            admitted=False, reason=PlacementReason.ESTIMATE_UNAVAILABLE,
            detail=str(exc), **common,
            verification_state=PlacementVerificationState.FAILED,
        )
    kv = kv_cache_bytes(profile, requested_context_tokens)
    if kv is None:
        kv = kv_envelope_bytes
    if kv is None:
        return ModelAdmissionPlan(
            admitted=False, reason=PlacementReason.GEOMETRY_UNKNOWN,
            detail="KV geometry unknown and no explicit envelope supplied",
            weights=weights, **common,
            verification_state=PlacementVerificationState.FAILED,
        )
    total = weights.estimated_bytes + kv + policy.runtime_overhead_bytes
    observed_ram = hardware.ram.observed_available_bytes
    observed_vram = hardware.free_vram() if hardware.has_discrete_gpu else None
    unified_free = None
    if hardware.is_coherent_unified:
        values = [
            adapter.observed_free_vram_bytes for adapter in hardware.adapters
            if adapter.kind.value == "COHERENT_UNIFIED"
        ]
        if values and all(value is not None for value in values):
            unified_free = sum(values)
    ram_budget = admissible_new(
        observed_ram, policy.required_free_ram_bytes, outstanding_ram_bytes
    )
    vram_budget = admissible_new(
        observed_vram, policy.required_free_vram_bytes, outstanding_vram_bytes
    )
    unified_budget = admissible_new(
        unified_free, policy.required_free_ram_bytes, outstanding_ram_bytes
    )
    base = dict(
        weights=weights, kv_bytes=kv, total_required_bytes=total,
        observed_available_ram_bytes=observed_ram,
        observed_free_vram_bytes=observed_vram,
        observed_free_unified_bytes=unified_free,
        host_gpu_addressable_bytes=host_gpu_addressable_bytes,
        budget_ram_bytes=ram_budget, budget_vram_bytes=vram_budget,
        budget_unified_bytes=unified_budget, **common,
    )

    failures: list[str] = []
    for candidate in _candidate_order(policy):
        if candidate is PlacementPolicy.DEVICE_ONLY:
            if policy.requested_num_gpu == 0:
                failures.append("DEVICE_ONLY cannot request num_gpu=0")
                continue
            if not hardware.has_discrete_gpu or vram_budget is None:
                failures.append("DEVICE_ONLY requires known discrete CUDA VRAM")
                continue
            if total > vram_budget:
                failures.append("DEVICE_ONLY exceeds VRAM budget")
                continue
            return ModelAdmissionPlan(
                admitted=True, reason=PlacementReason.PLANNED,
                selected_policy=candidate,
                requested_topology=ExecutionTopology.GPU_RESIDENT,
                compute_domains=(ComputeDomain.CUDA,),
                memory_domains=(MemoryDomain.DEVICE_LOCAL,),
                claims=(ResourceClaim(memory_domain=MemoryDomain.DEVICE_LOCAL,
                                      bytes=total, physical_claim=True),),
                **base,
            )
        if candidate is PlacementPolicy.HOST_ONLY:
            if policy.requested_num_gpu not in (None, 0):
                failures.append("HOST_ONLY requires num_gpu=0 when specified")
                continue
            if ram_budget is None or total > ram_budget:
                failures.append("HOST_ONLY exceeds or lacks RAM budget")
                continue
            return ModelAdmissionPlan(
                admitted=True, reason=PlacementReason.PLANNED,
                selected_policy=candidate,
                requested_topology=ExecutionTopology.CPU_RESIDENT,
                compute_domains=(ComputeDomain.CPU,),
                memory_domains=(MemoryDomain.HOST_LOCAL,),
                claims=(ResourceClaim(memory_domain=MemoryDomain.HOST_LOCAL,
                                      bytes=total, physical_claim=True),),
                **base,
            )
        if candidate is PlacementPolicy.HYBRID_STATIC:
            if not policy.hybrid_supported:
                failures.append("HYBRID_STATIC capability is unverified")
                continue
            if policy.requested_num_gpu == 0:
                failures.append("HYBRID_STATIC cannot request num_gpu=0")
                continue
            if hardware.is_coherent_unified:
                usable = None if ram_budget is None or unified_budget is None \
                    else min(ram_budget, unified_budget)
                if (usable is None or total > usable
                        or host_gpu_addressable_bytes is None
                        or total > host_gpu_addressable_bytes):
                    failures.append(
                        "coherent HYBRID_STATIC exceeds or lacks shared-pool evidence"
                    )
                    continue
                return ModelAdmissionPlan(
                    admitted=True, reason=PlacementReason.PLANNED,
                    selected_policy=candidate,
                    requested_topology=ExecutionTopology.UNIFIED_MEMORY_ACCELERATED,
                    compute_domains=(ComputeDomain.CPU, ComputeDomain.CUDA),
                    memory_domains=(MemoryDomain.HOST_LOCAL,
                                    MemoryDomain.HOST_GPU_ADDRESSABLE),
                    claims=(
                        ResourceClaim(memory_domain=MemoryDomain.HOST_LOCAL,
                                      bytes=total, physical_claim=True),
                        ResourceClaim(
                            memory_domain=MemoryDomain.HOST_GPU_ADDRESSABLE,
                            bytes=total, physical_claim=False,
                            note="same physical host pool; accessibility only",
                        ),
                    ),
                    **base,
                )
            device = policy.requested_device_bytes
            if (not hardware.has_discrete_gpu or vram_budget is None
                    or ram_budget is None or device is None or device <= 0
                    or device >= total):
                failures.append("HYBRID_STATIC requires an explicit valid split")
                continue
            host = total - device
            if device > vram_budget or host > ram_budget:
                failures.append("HYBRID_STATIC split exceeds a resource budget")
                continue
            domains = [MemoryDomain.HOST_LOCAL, MemoryDomain.DEVICE_LOCAL]
            claims = [
                ResourceClaim(memory_domain=MemoryDomain.HOST_LOCAL,
                              bytes=host, physical_claim=True),
                ResourceClaim(memory_domain=MemoryDomain.DEVICE_LOCAL,
                              bytes=device, physical_claim=True),
            ]
            if host_gpu_addressable_bytes is not None:
                domains.append(MemoryDomain.HOST_GPU_ADDRESSABLE)
                claims.append(ResourceClaim(
                    memory_domain=MemoryDomain.HOST_GPU_ADDRESSABLE,
                    bytes=min(host, host_gpu_addressable_bytes),
                    physical_claim=False,
                    note="accessibility aperture into HOST_LOCAL; not capacity",
                ))
            return ModelAdmissionPlan(
                admitted=True, reason=PlacementReason.PLANNED,
                selected_policy=candidate,
                requested_topology=ExecutionTopology.CPU_FRONTLOADED_GPU_ASSISTED,
                compute_domains=(ComputeDomain.CPU, ComputeDomain.CUDA),
                memory_domains=tuple(domains), claims=tuple(claims), **base,
            )
        failures.append(f"unsupported candidate {candidate.value}")

    reason = PlacementReason.CAPABILITY_UNVERIFIED if any(
        "unverified" in item for item in failures
    ) else PlacementReason.EXCEEDS_RESOURCE_BUDGET
    return ModelAdmissionPlan(
        admitted=False, reason=reason, detail="; ".join(failures), **base,
        verification_state=PlacementVerificationState.FAILED,
    )


def verify_placement(
    plan: ModelAdmissionPlan,
    evidence: Optional[TopologyEvidence],
    *,
    tau_vram_bytes: int = 0,
) -> PlacementVerificationState:
    """Conservatively map existing topology evidence to plan verification."""
    if not plan.admitted or plan.requested_topology is ExecutionTopology.PROHIBITED:
        return PlacementVerificationState.FAILED
    if evidence is None or evidence.requested_topology is not plan.requested_topology:
        return PlacementVerificationState.UNVERIFIABLE
    state = derive_attestation(evidence, tau_vram_bytes=tau_vram_bytes)
    if state is AttestationState.CONFIRMED:
        return PlacementVerificationState.VERIFIED
    if state in (AttestationState.MISMATCH, AttestationState.PROHIBITED):
        return PlacementVerificationState.FAILED
    report = evidence.ollama_reported
    if (report is not None and report.schema_supported
            and report.size_bytes is not None
            and report.size_vram_bytes is not None):
        size = report.size_bytes
        vram = report.size_vram_bytes
        topology = plan.requested_topology
        if topology is ExecutionTopology.CPU_RESIDENT:
            return (
                PlacementVerificationState.PARTIALLY_VERIFIED
                if vram == 0 else PlacementVerificationState.FAILED
            )
        if topology is ExecutionTopology.GPU_RESIDENT:
            return (
                PlacementVerificationState.PARTIALLY_VERIFIED
                if vram > 0 and size - vram <= tau_vram_bytes
                else PlacementVerificationState.FAILED
            )
        if topology is ExecutionTopology.CPU_FRONTLOADED_GPU_ASSISTED:
            return (
                PlacementVerificationState.PARTIALLY_VERIFIED
                if 0 < vram < size - tau_vram_bytes
                else PlacementVerificationState.FAILED
            )
    channels = (
        evidence.ollama_reported, evidence.host_gpu,
        evidence.host_ram, evidence.attribution,
    )
    if all(channel is not None for channel in channels):
        return PlacementVerificationState.PARTIALLY_VERIFIED
    return PlacementVerificationState.UNVERIFIABLE


def decision_from_placement_plan(
    plan: ModelAdmissionPlan,
    *,
    decision_id: str,
    session_id: str,
    identity_id: Optional[str],
    occurred_at_utc: str,
) -> AdmissionDecision:
    """Bridge a plan into the existing admission/reservation authority model.

    Only physical claims become reservation quantities.  In particular,
    HOST_GPU_ADDRESSABLE claims are excluded because they describe access to
    already-claimed host memory, not a third pool.
    """
    if not plan.admitted or plan.selected_policy is None:
        raise ValueError("only an admitted placement plan can become a decision")
    if not decision_id or not session_id or not occurred_at_utc:
        raise ValueError("decision occurrence identity and time are required")
    ram = sum(
        claim.bytes for claim in plan.claims
        if claim.physical_claim
        and claim.memory_domain is MemoryDomain.HOST_LOCAL
    )
    vram = sum(
        claim.bytes for claim in plan.claims
        if claim.physical_claim
        and claim.memory_domain is MemoryDomain.DEVICE_LOCAL
    )
    if ram + vram != plan.total_required_bytes:
        raise ValueError("physical placement claims do not cover the working set")
    return AdmissionDecision(
        decision_id=decision_id,
        session_id=session_id,
        identity_id=identity_id,
        model_digest=plan.model_manifest_digest or plan.model_digest,
        model_profile_hash=plan.model_profile_hash,
        hardware_profile_hash=plan.hardware_profile_hash,
        requested_topology=plan.requested_topology,
        reservation_ram_bytes=ram,
        reservation_vram_bytes=vram,
        total_required_bytes=plan.total_required_bytes,
        policy_version=plan.policy_version,
        admitted=True,
        reason=AdmitReason.ADMITTED,
        detail=(
            f"resource-aware placement plan {plan.content_hash()} "
            f"selected {plan.selected_policy.value}"
        ),
        occurred_at_utc=occurred_at_utc,
    )
