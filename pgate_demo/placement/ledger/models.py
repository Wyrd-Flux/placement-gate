"""Canonical typed CONTROL/domain semantics.

The Controller's canonical CONTROL semantics are typed and authoritative.
Content representations are separately typed and provenance-addressable and
never become the semantic identity of Controller state.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


class PlacementClass(str, Enum):
    GPU_ELIGIBLE = "GPU_ELIGIBLE"
    CPU_REQUIRED = "CPU_REQUIRED"
    PROHIBITED = "PROHIBITED"


class ResourceClass(str, Enum):
    NORMAL = "NORMAL"
    HEAVY = "HEAVY"


class Visibility(str, Enum):
    PUBLIC = "public"
    OPERATOR = "operator"
    SEALED = "sealed"


class SessionState(str, Enum):
    NEW_BOOTSTRAP_REQUIRED = "NEW_BOOTSTRAP_REQUIRED"
    READY = "READY"
    GENERATING = "GENERATING"
    ACTIONS_PENDING = "ACTIONS_PENDING"
    EXECUTING = "EXECUTING"
    SUSPENDED = "SUSPENDED"
    INVALID = "INVALID"
    CLOSED = "CLOSED"


class TurnState(str, Enum):
    CREATED = "CREATED"
    REQUEST_MANIFEST_COMMITTED = "REQUEST_MANIFEST_COMMITTED"
    TRANSMISSION_STARTED = "TRANSMISSION_STARTED"
    STREAMING = "STREAMING"
    RESPONSE_COMMITTED = "RESPONSE_COMMITTED"
    COMPLETE = "COMPLETE"
    STOP_INVALID = "STOP_INVALID"
    CRASH_INVALID = "CRASH_INVALID"
    MALFORMED_INVALID = "MALFORMED_INVALID"
    MODEL_MISMATCH_INVALID = "MODEL_MISMATCH_INVALID"


TERMINAL_TURN_STATES = {
    TurnState.COMPLETE,
    TurnState.STOP_INVALID,
    TurnState.CRASH_INVALID,
    TurnState.MALFORMED_INVALID,
    TurnState.MODEL_MISMATCH_INVALID,
}


class LeaseState(str, Enum):
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"
    EXHAUSTED = "EXHAUSTED"


class ActionState(str, Enum):
    PROPOSED = "PROPOSED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    CANCELED_INVALID = "CANCELED_INVALID"
    DISPATCHED = "DISPATCHED"
    RECEIPT = "RECEIPT"
    INDETERMINATE = "INDETERMINATE"


class CheckpointState(str, Enum):
    PROPOSED = "PROPOSED"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"


class VerificationState(str, Enum):
    PENDING = "PENDING"
    EQUIVALENCE_VERIFIED = "EQUIVALENCE_VERIFIED"
    NOT_EQUIVALENT = "NOT_EQUIVALENT"
    INCONSISTENT = "INCONSISTENT"
    UNSUPPORTED = "UNSUPPORTED"


class BackendMode(str, Enum):
    MANAGED_STRICT = "MANAGED_STRICT"
    ATTACH_RELAXED = "ATTACH_RELAXED"


class Identity(BaseModel):
    """Digest-bound identity. A model/digest/policy change requires a NEW identity."""

    identity_id: str
    session_id: str
    model_requested: str
    model_digest: str
    parameter_count: int
    placement_class: PlacementClass
    resource_class: ResourceClass
    backend_id: str
    bootstrap_payload_ref: str
    bootstrap_sha256: str
    request_schema_version: str
    action_policy_version: str
    visibility_policy_version: str


class Bootstrap(BaseModel):
    """Immutable, content-addressed frozen bootstrap."""

    bootstrap_id: str
    identity_id: str
    content_hash: str
    content: str


class Turn(BaseModel):
    turn_id: str
    session_id: str
    identity_id: str
    ordinal: int
    state: TurnState = TurnState.CREATED
    request_hash: Optional[str] = None
    actual_backend_model: Optional[str] = None
    actual_backend_digest: Optional[str] = None
    lease_id: Optional[str] = None


class Lease(BaseModel):
    lease_id: str
    turn_id: str
    capability_nonce_hash: str
    state: LeaseState = LeaseState.ACTIVE


class AuthorizationRecord(BaseModel):
    """Source-of-authority record. Trusted ingress only — never via untrusted IPC."""

    auth_id: str
    reviewer_type: str
    reviewer_id: str
    authority_class: str
    decision: str
    subject_hash: str
    permitted_scope: list[str] = Field(default_factory=list)
    constraints: dict = Field(default_factory=dict)
    auth_sha256: str
    issued_at_utc: str
    expires_at_utc: Optional[str] = None
    evidence_refs: list[str] = Field(default_factory=list)
    supersedes: Optional[str] = None


class ScopeSnapshot(BaseModel):
    """Derived constraint from an accepted AuthorizationRecord. Enforces, never manufactures."""

    scope_id: str
    derived_from_auth_id: str
    read_scope: list[str] = Field(default_factory=list)
    write_scope: list[str] = Field(default_factory=list)
    network_scope: list[str] = Field(default_factory=list)
    policy_version: str


class ActionProposal(BaseModel):
    action_id: str
    turn_id: str
    lease_id: str
    auth_id: str
    scope_id: str
    class_: str
    adapter: str
    typed_arguments_payload: dict = Field(default_factory=dict)
    proposal_sha256: str
    scope_snapshot_sha256: str
    state: ActionState = ActionState.PROPOSED


class Representation(BaseModel):
    """Content is multi-representational. Independently typed and provenance-addressable."""

    representation_id: str
    content_hash: str
    representation_type: str
    representation_version: str
    payload_reference: str
    visibility: Visibility
    source_representation_id: list[str] = Field(default_factory=list)
    transformation_id: Optional[str] = None
    verification_state: VerificationState = VerificationState.PENDING


class Transformation(BaseModel):
    transformation_id: str
    transformation_type: str
    source_representation_ids: list[str] = Field(default_factory=list)
    derived_representation_id: str
    parameters: dict = Field(default_factory=dict)


class Verification(BaseModel):
    verification_id: str
    subject_representation_ids: list[str] = Field(default_factory=list)
    disposition: VerificationState
    verifier_type: str
    verifier_id: str
    notes: str = ""


class Checkpoint(BaseModel):
    checkpoint_id: str
    identity_id: str
    covers_from_seq: int
    covers_through_seq: int
    covered_head_hash: str
    summary_payload_ref: str
    summary_sha256: str
    state: CheckpointState = CheckpointState.PROPOSED
    verified_by: Optional[str] = None


class RequestManifest(BaseModel):
    """Canonical request manifest, durably committed BEFORE any transmission."""

    turn_id: str
    identity_id: str
    session_id: str
    model_requested: str
    model_digest: str
    messages_hash: str
    ordered_source_event_ids: list[str] = Field(default_factory=list)
    options: dict = Field(default_factory=dict)
    manifest_hash: str

    @model_validator(mode="after")
    def _check_manifest_hash(self) -> "RequestManifest":
        # manifest_hash must be derived from ALL manifest fields, incl. options
        # (canonical durable intent must fully bind the committed options).
        options_json = __import__("json").dumps(
            self.options, sort_keys=True, separators=(",", ":")
        )
        expected = sha256_text(
            "|".join(
                [
                    self.turn_id,
                    self.identity_id,
                    self.session_id,
                    self.model_requested,
                    self.model_digest,
                    self.messages_hash,
                    ",".join(self.ordered_source_event_ids),
                    options_json,
                ]
            )
        )
        if self.manifest_hash != expected:
            raise ValueError("manifest_hash does not match manifest fields")
        return self
