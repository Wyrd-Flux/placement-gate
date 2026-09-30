"""Stage 3A digest-bound model profiles and EXPLICITLY APPROXIMATE estimates.

Invariants (v2 §D / v3 §7, §8 / G-3P):
  - profiles bind to an EXACT model digest; no model-name lookup tables exist;
  - a profile's identity is its content hash; revising bytes yields a new
    hash; old admissions keep referencing the old hash (never "current");
  - parameter-derived weight size is an ESTIMATE carrying source, precision,
    and assumptions, and may never be represented as exact;
  - unknown required geometry fails closed;
  - integer arithmetic only (milli-bytes) — no float drift.
"""

from __future__ import annotations

import enum
import hashlib
import json
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from ..topology import ExecutionTopology

DIGEST_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PROFILE_VERSION = "stage3a.1"


class ProfileSource(str, enum.Enum):
    INJECTED_FIXTURE = "INJECTED_FIXTURE"          # Stage 3A tests / CLI proof
    OPERATOR_SUPPLIED = "OPERATOR_SUPPLIED"        # source order (2)
    LOCAL_ARTIFACT_METADATA = "LOCAL_ARTIFACT_METADATA"  # source order (1)
    REVIEWED_READ_ONLY_SOURCE = "REVIEWED_READ_ONLY_SOURCE"  # source order (3)


class ModelProfile(BaseModel):
    """Digest-bound model geometry facts. All sizes are integer bytes.

    kv_bytes_per_token / bytes_per_weight_milli missing (None) means the
    geometry is UNKNOWN: topologies that require it must fail closed.
    bits-per-weight is represented as milli-bytes-per-weight
    (e.g. 4.5 bits = 0.5625 B = 562 milli-bytes) — scaled integer, no floats.
    """

    model_config = ConfigDict(frozen=True)

    digest: str
    architecture: str = ""
    layer_count: Optional[int] = None
    context_limit: Optional[int] = None
    kv_bytes_per_token: Optional[int] = None
    bytes_per_weight_milli: Optional[int] = None
    quantization_level: str = ""
    observed_artifact_bytes: Optional[int] = None  # directly observed size
    ollama_reported_size_bytes: Optional[int] = None  # reported, not truth
    source: ProfileSource = ProfileSource.INJECTED_FIXTURE
    provenance: str = ""
    precision: str = "APPROXIMATE_LABEL"
    confidence: str = "MEDIUM"
    profile_version: str = PROFILE_VERSION

    @field_validator("digest")
    @classmethod
    def _digest(cls, v: str) -> str:
        if not DIGEST_RE.match(v or ""):
            raise ValueError("profile requires an exact 64-hex model digest")
        return v.lower()

    @field_validator(
        "layer_count", "context_limit", "kv_bytes_per_token",
        "bytes_per_weight_milli", "observed_artifact_bytes",
        "ollama_reported_size_bytes",
    )
    @classmethod
    def _positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("profile quantities must be positive when present")
        return v

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    @classmethod
    def from_canonical_bytes(cls, raw: bytes) -> "ModelProfile":
        return cls.model_validate(json.loads(raw.decode("utf-8")))


class SizeSource(str, enum.Enum):
    OBSERVED_ARTIFACT_BYTES = "OBSERVED_ARTIFACT_BYTES"
    OLLAMA_REPORTED_SIZE = "OLLAMA_REPORTED_SIZE"
    PARAMETER_DERIVED = "PARAMETER_DERIVED"


class WeightSizeEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    estimated_bytes: int
    source: SizeSource
    precision: str  # EXACT_OBSERVED only for artifact bytes; else APPROXIMATE
    assumptions: str
    policy_margin_bps: int = 0  # basis points already APPLIED, versioned

    def __init__(self, **data):
        super().__init__(**data)
        if self.source is not SizeSource.OBSERVED_ARTIFACT_BYTES:
            if self.precision == "EXACT_OBSERVED":
                raise ValueError("only observed artifact bytes may be exact")
        if self.precision == "EXACT_OBSERVED" and (
            self.source is not SizeSource.OBSERVED_ARTIFACT_BYTES
        ):
            raise ValueError("exactness reserved for observed artifact bytes")


class ParameterCountAmbiguous(Exception):
    """Required geometry missing and no operator envelope supplied."""


def weight_size_from_profile(
    profile: ModelProfile,
    parameter_count: int,
    policy_margin_bps: int = 0,
) -> WeightSizeEstimate:
    """Derive the best weight-size evidence available (never as exact).

    Preference (v2 §D): observed artifact bytes > Ollama-reported size >
    parameter-derived estimate (with milli-byte bpw math + conservative
    margin). Parameter-derived requires bytes_per_weight_milli; missing =>
    fail closed unless the caller provides an observed/reported size.
    """
    if profile.observed_artifact_bytes is not None:
        return WeightSizeEstimate(
            estimated_bytes=profile.observed_artifact_bytes,
            source=SizeSource.OBSERVED_ARTIFACT_BYTES,
            precision="EXACT_OBSERVED",
            assumptions="directly observed model artifact byte size",
        )
    if profile.ollama_reported_size_bytes is not None:
        return WeightSizeEstimate(
            estimated_bytes=profile.ollama_reported_size_bytes,
            source=SizeSource.OLLAMA_REPORTED_SIZE,
            precision="APPROXIMATE_REPORTED",
            assumptions="Ollama-reported size field; evidence, not physical truth",
        )
    if parameter_count <= 0 or profile.bytes_per_weight_milli is None:
        raise ParameterCountAmbiguous(
            "no observed/reported size and bytes_per_weight_milli unknown; "
            "parameter-derived estimation cannot proceed"
        )
    if policy_margin_bps < 0:
        raise ValueError("policy margin must be nonnegative")
    base = parameter_count * profile.bytes_per_weight_milli // 1000
    with_margin = base * (10_000 + policy_margin_bps) // 10_000
    return WeightSizeEstimate(
        estimated_bytes=with_margin,
        source=SizeSource.PARAMETER_DERIVED,
        precision="APPROXIMATE_LABEL",
        assumptions=(
            "parameter_count x bytes-per-weight(milli) scaled-integer product; "
            "ignores quantization metadata, tensor alignment, non-weight "
            "structure; conservative policy margin applied (bps="
            f"{policy_margin_bps})"
        ),
        policy_margin_bps=policy_margin_bps,
    )


def kv_cache_bytes(
    profile: ModelProfile, requested_context_tokens: int
) -> Optional[int]:
    """KV estimate from digest-bound geometry; None when unknown (fail-closed
    decisions belong to the admission layer)."""
    if profile.kv_bytes_per_token is None or requested_context_tokens <= 0:
        return None
    return profile.kv_bytes_per_token * requested_context_tokens
