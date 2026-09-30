"""Controller.place_and_load_model() — method-level execution tests.

The existing placement tests cover the PLANNING layer: plan_inference_placement
and decision_from_placement_plan called directly. They never enter
Controller.place_and_load_model(), which is the method that performs the load,
observes residency, verifies, and writes the ledger event.

These tests close that gap. They execute the method itself against a fake
residency-capable backend, so they are deterministic and need no Ollama server.
The live behaviour is qualified separately; see
RESOURCE_AWARE_PLACEMENT_STAGE.md.
"""

from __future__ import annotations

import sqlite3

import pytest

from pgate_demo.placement.backends.backends import ChatSendError
from pgate_demo.placement.topology import ExecutionTopology
from pgate_demo.placement.policy.inference_placement import (
    AdmitReason,
    PlacementPolicy,
    PlacementPlanningPolicy,
    PlacementReason,
    PlacementVerificationState,
    plan_inference_placement,
)
from pgate_demo.placement.policy.model_profile import ModelProfile

from ._fixtures import GIB, hw_discrete, hw_unknown, profile_large

MODEL = "demo-model:8b"
DIGEST = "d" * 64
CTX = 8192
SESSION_ID = "session-fixture"
IDENTITY_ID = "identity-fixture"


# --------------------------------------------------------------------------- #
# fake backend: residency-capable, deterministic, no network
# --------------------------------------------------------------------------- #


class FakeResidentBackend:
    """Stands in for OllamaChatAdapter on the one capability placement needs.

    place_and_load_model requires a backend that can serve ("POST", "/api/chat")
    to reach the load boundary and ("GET", "/api/ps") to observe it. A backend
    without that capability must be refused, which is itself a tested path.
    """

    def __init__(self, *, size_bytes: int, size_vram_bytes: int,
                 digest: str = DIGEST, model: str = MODEL,
                 fail_load: bool = False, ps_available: bool = True):
        self.size_bytes = size_bytes
        self.size_vram_bytes = size_vram_bytes
        self.digest = digest
        self.model = model
        self.fail_load = fail_load
        self.ps_available = ps_available
        self.calls: list[tuple[str, str]] = []

    def endpoint_id(self) -> str:
        return "fake-resident"

    def send_chat(self, model, messages, **kwargs):
        raise AssertionError("place_and_load_model must not generate tokens")

    def _request(self, method: str, path: str, body=None) -> dict:
        self.calls.append((method, path))
        if path == "/api/chat":
            if self.fail_load:
                raise ChatSendError("fake load refused")
            return {"done": True, "model": self.model, "done_reason": "load"}
        if path == "/api/ps":
            if not self.ps_available:
                return {}
            return {
                "models": [
                    {
                        "name": self.model,
                        "model": self.model,
                        "digest": self.digest,
                        "size": self.size_bytes,
                        "size_vram": self.size_vram_bytes,
                        "context_length": CTX,
                    }
                ]
            }
        raise ChatSendError(f"unexpected operation {method} {path}")


class NonResidentBackend(FakeResidentBackend):
    """A backend that cannot serve /api/ps at all."""

    def _request(self, method: str, path: str, body=None) -> dict:
        raise ChatSendError("operation is not admitted by the chat adapter")


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def controller(tmp_path):
    from pgate_demo.placement.controller.placement import PlacementRunner
    from pgate_demo.placement.ledger.ledger import Ledger

    return PlacementRunner(Ledger(tmp_path / "placement.db"))


@pytest.fixture
def model_profile() -> ModelProfile:
    return ModelProfile(
        digest=DIGEST, architecture="demo-arch", layer_count=32,
        context_limit=CTX, kv_bytes_per_token=5 * 1024,
        bytes_per_weight_milli=562, quantization_level="Q4_K_M",
        observed_artifact_bytes=6 * GIB, provenance="fixture",
        source="INJECTED_FIXTURE",
    )


def admit(controller, model_profile, hardware, *, policy, parameter_count, ctx=CTX):
    """Build a session + identity and return them with the plan."""
    plan = plan_inference_placement(
        model_profile, hardware, parameter_count=parameter_count,
        requested_context_tokens=ctx, policy=policy,
        model_id=MODEL, model_manifest_digest=DIGEST,
    )
    # A public placement library takes identifiers as strings. The controller
    # this was extracted from minted them through a session store; that store is
    # estate machinery and the seam never used it for anything but the id.
    return plan, SESSION_ID, IDENTITY_ID


def events(controller, kind: str) -> list[sqlite3.Row]:
    conn = controller.ledger.connect()
    try:
        return conn.execute(
            "SELECT * FROM event WHERE kind=?", (kind,)
        ).fetchall()
    finally:
        conn.close()


def admissions(controller) -> list[sqlite3.Row]:
    conn = controller.ledger.connect()
    try:
        return conn.execute("SELECT * FROM admission_record").fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# the method executes
# --------------------------------------------------------------------------- #


def test_method_reaches_and_passes_the_load_boundary(controller, model_profile):
    """A fully-GPU-resident load must be ADMITTED and VERIFIED.

    This is the test the planning layer could not provide: it proves the method
    issues the load, observes residency, and reaches a verified decision.
    """
    backend = FakeResidentBackend(size_bytes=6 * GIB, size_vram_bytes=6 * GIB)
    controller.set_backend(backend)
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        parameter_count=8_000_000_000,
    )
    assert plan.admitted

    decision = controller.place_and_load_model(
        plan, hardware, model_profile,
        session_id=session, identity_id=identity,
        occurred_at_utc="2026-09-30T12:00:00+00:00", keep_alive="30s",
    )

    # the load was actually requested, then residency observed
    assert ("POST", "/api/chat") in backend.calls
    assert ("GET", "/api/ps") in backend.calls
    # the load request carried the plan's identity and the requested offload
    assert backend.calls.index(("POST", "/api/chat")) < backend.calls.index(
        ("GET", "/api/ps")
    )
    assert decision.admitted
    assert decision.reason is AdmitReason.ADMITTED
    assert decision.requested_topology is ExecutionTopology.GPU_RESIDENT
    assert decision.model_digest == DIGEST


def test_method_records_verified_placement_in_the_ledger(controller, model_profile):
    backend = FakeResidentBackend(size_bytes=6 * GIB, size_vram_bytes=6 * GIB)
    controller.set_backend(backend)
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        parameter_count=8_000_000_000,
    )
    controller.place_and_load_model(
        plan, hardware, model_profile, session_id=session, identity_id=identity,
        occurred_at_utc="2026-09-30T12:00:00+00:00", keep_alive="30s",
    )
    assert len(events(controller, "placement_verified")) == 1
    # the immutable admission record is written exactly once
    assert len(admissions(controller)) == 1
    assert len(events(controller, "placement_failed")) == 0


# --------------------------------------------------------------------------- #
# fail-closed paths -- each must refuse, and record why
# --------------------------------------------------------------------------- #


def test_partial_gpu_residency_fails_closed(controller, model_profile):
    """Requested GPU_RESIDENT but only part of the model reached VRAM.

    This is the live-observed case: requesting num_gpu=1 puts ONE layer on the
    GPU, which is not GPU residency. The gate must refuse rather than accept.
    """
    backend = FakeResidentBackend(size_bytes=6 * GIB, size_vram_bytes=1 * GIB)
    controller.set_backend(backend)
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=1
        ),
        parameter_count=8_000_000_000,
    )
    assert plan.admitted, "planning accepted; the failure must come from evidence"

    with pytest.raises(ChatSendError, match="verification failed"):
        controller.place_and_load_model(
            plan, hardware, model_profile, session_id=session,
            identity_id=identity, occurred_at_utc="2026-09-30T12:00:00+00:00",
        )
    assert len(events(controller, "placement_failed")) == 1
    assert len(events(controller, "placement_verified")) == 0
    # the immutable admission record still exists: refusal does not erase history
    assert len(admissions(controller)) == 1


def test_model_absent_from_ps_fails_closed(controller, model_profile):
    """Load requested, model not resident -> refuse. Absence is not residency."""
    backend = FakeResidentBackend(size_bytes=6 * GIB, size_vram_bytes=6 * GIB)
    backend.ps_available = False
    controller.set_backend(backend)
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        parameter_count=8_000_000_000,
    )
    with pytest.raises(ChatSendError, match="model not found"):
        controller.place_and_load_model(
            plan, hardware, model_profile, session_id=session,
            identity_id=identity, occurred_at_utc="2026-09-30T12:00:00+00:00",
        )
    assert len(events(controller, "placement_failed")) == 1


def test_backend_without_residency_capability_is_refused(controller, model_profile):
    """A backend that cannot read /api/ps cannot verify placement."""
    controller.set_backend(NonResidentBackend(size_bytes=1, size_vram_bytes=1))
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        parameter_count=8_000_000_000,
    )
    with pytest.raises(ChatSendError, match="cannot read residency|not admitted"):
        controller.place_and_load_model(
            plan, hardware, model_profile, session_id=session,
            identity_id=identity, occurred_at_utc="2026-09-30T12:00:00+00:00",
        )
    assert len(events(controller, "placement_failed")) == 1


def test_failed_load_request_fails_closed(controller, model_profile):
    backend = FakeResidentBackend(
        size_bytes=6 * GIB, size_vram_bytes=6 * GIB, fail_load=True
    )
    controller.set_backend(backend)
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        parameter_count=8_000_000_000,
    )
    with pytest.raises(ChatSendError, match="fake load refused"):
        controller.place_and_load_model(
            plan, hardware, model_profile, session_id=session,
            identity_id=identity, occurred_at_utc="2026-09-30T12:00:00+00:00",
        )
    assert ("GET", "/api/ps") not in backend.calls, "must not observe after a failed load"
    assert len(events(controller, "placement_failed")) == 1


def test_wrong_digest_residency_does_not_satisfy_the_plan(controller, model_profile):
    """Residency for different weights must not verify this plan.

    The tag matches but the digest does not. Placement is a claim about specific
    weights, so this must fail closed rather than trust the tag.
    """
    backend = FakeResidentBackend(
        size_bytes=6 * GIB, size_vram_bytes=6 * GIB, digest="e" * 64
    )
    controller.set_backend(backend)
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        parameter_count=8_000_000_000,
    )
    with pytest.raises(ChatSendError, match="digest does not match"):
        controller.place_and_load_model(
            plan, hardware, model_profile, session_id=session,
            identity_id=identity, occurred_at_utc="2026-09-30T12:00:00+00:00",
        )
    assert len(events(controller, "placement_verified")) == 0
    assert len(events(controller, "placement_failed")) == 1


# --------------------------------------------------------------------------- #
# unknown hardware must never reach the load boundary
# --------------------------------------------------------------------------- #


def test_unknown_hardware_never_attempts_a_load(controller, model_profile):
    hardware = hw_unknown()
    plan = plan_inference_placement(
        model_profile, hardware, parameter_count=8_000_000_000,
        requested_context_tokens=CTX,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        model_id=MODEL, model_manifest_digest=DIGEST,
    )
    assert not plan.admitted
    assert plan.reason is PlacementReason.HARDWARE_UNKNOWN
    assert plan.requested_topology is ExecutionTopology.PROHIBITED


def test_host_only_requests_zero_gpu(controller, model_profile):
    """HOST_ONLY must ask for num_gpu=0; that is the CPU-isolation invariant."""
    backend = FakeResidentBackend(size_bytes=6 * GIB, size_vram_bytes=0)
    controller.set_backend(backend)
    hardware = hw_discrete(ram_kwargs={}, gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.HOST_ONLY, requested_num_gpu=0
        ),
        parameter_count=8_000_000_000,
    )
    decision = controller.place_and_load_model(
        plan, hardware, model_profile, session_id=session, identity_id=identity,
        occurred_at_utc="2026-09-30T12:00:00+00:00",
    )
    # the load request itself must have carried num_gpu=0
    assert decision.requested_topology is ExecutionTopology.CPU_RESIDENT
    assert ("POST", "/api/chat") in backend.calls


def test_method_never_generates_tokens(controller, model_profile):
    """Placement is an observation. The fake raises if generation is attempted."""
    backend = FakeResidentBackend(size_bytes=6 * GIB, size_vram_bytes=6 * GIB)
    controller.set_backend(backend)
    hardware = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB})
    plan, session, identity = admit(
        controller, model_profile, hardware,
        policy=PlacementPlanningPolicy(
            placement=PlacementPolicy.DEVICE_ONLY, requested_num_gpu=99
        ),
        parameter_count=8_000_000_000,
    )
    controller.place_and_load_model(
        plan, hardware, model_profile, session_id=session, identity_id=identity,
        occurred_at_utc="2026-09-30T12:00:00+00:00",
    )
    assert "send_chat" not in [c[0] for c in backend.calls]
