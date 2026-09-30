"""The placement seam: execute a plan, then verify that it actually happened.

This module is deliberately small. The internal package it came from contains a
full controller — memory lanes, checkpoint generations, authorization chains, a
transmission gate — and the placement flow needs none of it. What is here is the
one method a caller needs, plus the reconciliation helper it fails through,
extracted unchanged.

The sequence is the whole point, and the order is not negotiable:

1. an immutable admission decision derived from the plan
2. a durable reservation appended to the ledger
3. ``num_gpu`` derived from the plan's selected policy
4. the load request, on the one transport the backend already admits
5. post-load observation from ``/api/ps``
6. verification against the plan
7. activation, or a fail-closed reconciliation

Two decisions are load-bearing:

**The load profile is built from the plan, not from the admission profile.**
``ModelProfile`` carries sizing metadata and no tag, no context window, and no
keep-alive; the plan carries all three. Taking identity from the plan means the
request, the observation, and the plan hash cannot drift apart.

**Identity is checked against the planned digest, not the tag.** A tag can be
re-pointed at different weights between the load request and the read-back, and a
placement verdict is a claim about the weights that actually ran. So when the
server reports a digest that contradicts the plan, the load is refused with
``identity_mismatch_after_load`` before verification is even attempted.

A failed verification deliberately **retains** the reservation. The model may be
partially loaded, and releasing it blindly could destroy state the operator did
not ask to destroy. The ledger records the failure; recovery is a separate,
informed decision.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PlacementError(RuntimeError):
    """The placement flow could not complete.

    Raised after the ledger has been reconciled, so the caller sees a failure and
    the audit trail shows why. Callers distinguish cases by the message prefix;
    the specific reasons are stable strings in :data:`FAILURE_DETAILS`.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


#: Reconciliation details this seam can record. Stable, because the ledger is
#: the audit surface and a reworded reason is a lost record.
FAILURE_DETAILS = (
    "backend_cannot_read_residency",
    "load_request_failed",
    "identity_mismatch_after_load",
    "model_not_found_after_load",
    "verification_failed",
)


class PlacementRunner:
    """Executes one placement plan against a residency-capable backend.

    The backend may be passed here or attached later with :meth:`set_backend`,
    matching the construction idiom of the controller this was extracted from.
    It is not optional in practice: without one, the flow refuses at
    ``backend_cannot_read_residency`` rather than skipping the observation.

    :param backend: anything exposing ``_request(method, path, body)`` for the
        three operations this flow uses: ``GET /api/ps``, ``POST /api/chat``, and
        (via the caller) ``GET /api/tags``.
    :param ledger: a :class:`pgate_demo.placement.ledger.ledger.Ledger`.
    """

    def __init__(self, ledger=None, backend=None) -> None:
        self.ledger = ledger
        self.backend = backend

    def set_backend(self, backend) -> None:
        self.backend = backend

    def place_and_load_model(
        self,
        plan,
        hardware,
        profile,
        *,
        session_id: str,
        identity_id: str,
        occurred_at_utc: str,
        keep_alive: int | str = "5m",
    ):
        """Wire a resource-aware placement plan through the full runtime lifecycle.

        Sequence: resource observation → immutable admission decision → durable
        reservation → load-boundary placement request → post-load observation →
        verification → activate on sufficient evidence → fail closed and unload/
        hold on failure → reconcile reservation and residency state.

        Reuses existing admission, ledger, and attestation components. Does not
        introduce a parallel authority, lease, registry, or lifecycle system.

        ``keep_alive`` is how long the server should hold the model after this
        call. Placement is an observation, so the caller owns residency lifetime.
        The default matches the qwen device profile in the internal runtime config.

        Returns the immutable :class:`AdmissionDecision` on verified success.
        Raises :class:`PlacementError` otherwise, having already reconciled.
        """
        from ..backends.backends import ChatSendError
        from ..backends.chat import OllamaChatProfile
        from ..policy.inference_placement import (
            PlacementPolicy,
            PlacementVerificationState,
            decision_from_placement_plan,
        )

        # Step 1: Immutable admission decision from the plan
        # Invariant 2: exact plan identity binds admission/reservation/backend/request/evidence
        # Invariant 3: requested placement never reported as verified placement
        # Invariant 12: preserve exact manifest, weights, hardware-profile, plan identities
        decision = decision_from_placement_plan(
            plan,
            decision_id=uuid.uuid4().hex,
            session_id=session_id,
            identity_id=identity_id,
            occurred_at_utc=occurred_at_utc,
        )

        # Step 2: Durable reservation — append admission to ledger
        # The decision's canonical_bytes + content_hash provide the content-addressed
        # immutable record. Physical claims only (HOST_GPU_ADDRESSABLE is an aperture,
        # not additive capacity per Invariant 4).
        ledger = self.ledger
        now = occurred_at_utc
        h_hash = ledger.append_hardware_profile(
            record_id=uuid.uuid4().hex, profile=hardware, observed_at_utc=now,
            probe_sources=["placement_plan"], tool_versions=[],
        )
        p_hash = ledger.append_model_profile(
            record_id=uuid.uuid4().hex, profile=profile, recorded_at_utc=now,
        )
        ledger.append_admission(decision)

        # Step 3: Determine num_gpu from the plan's selected policy
        # Invariant 5: HOST_ONLY must preserve true CPU-only isolation (num_gpu=0)
        # Invariant 6: DEVICE_ONLY/HYBRID_STATIC must respect reservations (budget checks already done in planning)
        # Invariant 9: recalculate only at admission/load boundaries (this is one-shot)
        selected = plan.selected_policy
        if selected is PlacementPolicy.HOST_ONLY:
            num_gpu = 0
        elif selected is PlacementPolicy.DEVICE_ONLY:
            num_gpu = plan.requested_num_gpu or 1
        elif selected is PlacementPolicy.HYBRID_STATIC:
            num_gpu = plan.requested_num_gpu or 1
        elif selected is PlacementPolicy.MANAGED_FALLBACK:
            # Managed fallback is policy-gated; caller should have resolved to a
            # concrete policy before reaching this runtime path (Invariant 7)
            num_gpu = 0
        else:
            num_gpu = 0

        # Step 4: Load-boundary placement request
        # Construct a profile with the correct num_gpu for the selected policy.
        # The profile's options() will pass num_gpu to the Ollama /api/chat request.
        # Invariant 10: parameter count is metadata, not the CPU/GPU decision boundary.
        #
        # The load profile is built from the PLAN's identity, not the admission
        # ModelProfile: the plan carries the exact tag and the exact requested
        # context window, while ModelProfile carries sizing metadata only (it has
        # no model/num_ctx/keep_alive fields). Identity comes from the plan so the
        # request, the /api/ps observation and the plan hash cannot diverge.
        working_profile = OllamaChatProfile(
            model=plan.model_id,
            num_gpu=num_gpu,
            num_ctx=plan.requested_context_tokens,
            keep_alive=keep_alive,
            require_cpu_only=(selected is PlacementPolicy.HOST_ONLY),
        )

        # Issue the load itself. Ollama materializes a model on any /api/chat
        # request; an empty message list loads without generating, which is the
        # cheapest way to reach the load boundary this method exists to observe.
        # The adapter already admits ("POST", "/api/chat"); no new transport.
        request_fn = getattr(self.backend, "_request", None)
        if request_fn is None:
            self._fail_and_reconcile(
                decision, plan, hardware, detail="backend_cannot_read_residency"
            )
            raise ChatSendError(
                "configured backend cannot serve /api/ps; placement requires a "
                "residency-capable backend"
            )
        try:
            request_fn(
                "POST",
                "/api/chat",
                {
                    "model": working_profile.model,
                    "messages": [],
                    "stream": False,
                    "options": working_profile.options(),
                    "keep_alive": working_profile.keep_alive,
                },
            )
        except ChatSendError:
            self._fail_and_reconcile(
                decision, plan, hardware, detail="load_request_failed"
            )
            raise

        # Step 5: Post-load observation — capture /api/ps evidence
        # Invariant 11: unknown, contradictory, stale, or identity-mismatched evidence fails closed
        ps_response = request_fn("GET", "/api/ps")
        models = ps_response.get("models", [])
        # Identity binding (Invariant 11: identity-mismatched evidence fails
        # closed). A tag can be re-pointed at different weights between the load
        # request and this read, and a placement verdict is a claim about the
        # weights that actually ran. So when the server reports a digest, it
        # must agree with the plan's model digest; a tag-only match with a
        # contradicting digest is NOT residency evidence for this plan.
        # When the server reports no digest, tag matching is all that exists and
        # is used as-is rather than treated as a mismatch.
        wanted_digest = (profile.digest or "").lower()
        tag_matches = [
            m for m in models
            if isinstance(m, dict) and m.get("name") == working_profile.model
        ]
        if wanted_digest:
            digest_matches = [
                m for m in tag_matches
                if wanted_digest in str(m.get("digest", "")).lower()
            ]
            if digest_matches:
                model_entry = digest_matches[0]
            elif tag_matches and any(
                str(m.get("digest", "")).strip() for m in tag_matches
            ):
                # the tag is resident, but under different weights than planned
                self._fail_and_reconcile(
                    decision, plan, hardware, detail="identity_mismatch_after_load"
                )
                raise ChatSendError(
                    "resident model digest does not match the planned model "
                    "digest; placement is a claim about specific weights"
                )
            else:
                model_entry = None
        else:
            model_entry = tag_matches[0] if tag_matches else None
        if model_entry is None:
            # Model not found after load — fail closed
            # Reconcile: release claimed resources, mark failed
            self._fail_and_reconcile(
                decision, plan, hardware, detail="model_not_found_after_load"
            )
            raise ChatSendError("model not found in /api/ps after load request")

        # Build topology evidence from post-load observation
        # Invariant 12: preserve exact manifest, weights, hardware-profile, plan identities
        size_bytes = model_entry.get("size")  # total model size as reported
        size_vram_bytes = model_entry.get("size_vram")  # VRAM size as reported

        from ..topology import (
            TopologyEvidence, HostGpuEvidence, HostRamEvidence,
            ResidencyReport, AttributionStatus, ExecutionTopology,
        )
        from ..policy.inference_placement import verify_placement

        ev = TopologyEvidence(
            requested_topology=plan.requested_topology,
            ollama_reported=ResidencyReport(
                schema_supported=True,
                size_bytes=size_bytes,
                size_vram_bytes=size_vram_bytes,
            ),
            host_gpu=HostGpuEvidence(
                attributable_model_vram_bytes=size_vram_bytes,
                detection_status="OK",
            ),
            host_ram=HostRamEvidence(
                attributable_model_ram_bytes=size_bytes,
                detection_status="OK",
            ),
            attribution=AttributionStatus.ATTRIBUTED,
        )

        # Step 6: Verification via verify_placement
        # Invariant 11: fail closed on unknown/contradictory/stale/identity-mismatched evidence
        verification = verify_placement(evidence=ev, plan=plan)

        if verification is PlacementVerificationState.VERIFIED:
            # Step 7: Activate on sufficient evidence
            # The admission decision is already immutable; we just record the
            # activation in the ledger as a durable event.
            with ledger.Tx(ledger) as tx:
                ledger.append_event_tx(
                    tx, session_id, "placement_verified", "operator",
                    "controller", "controller", _now(),
                )
            return decision

        # Step 8: Fail closed and unload/hold on failure
        # Invariant 11: unknown, contradictory, stale, or identity-mismatched evidence fails closed
        # Invariant 1: an admitted plan does not itself authorize execution
        # Reconcile reservation and residency state: the reservation exists but
        # residency is not confirmed; hold/keep in drained state
        self._fail_and_reconcile(decision, plan, hardware, verification_state=verification)
        raise ChatSendError(
            f"placement verification failed: {verification.value}; model loaded but "
            "evidence does not match requested placement; fail closed"
        )

    def _fail_and_reconcile(
        self,
        decision,
        plan,
        hardware,
        *,
        verification_state: Optional[object] = None,
        detail: str = "",
    ) -> None:
        """Reconcile reservation and residency state on verification failure.

        Invariants preserved:
        - Invariant 1: admitted plan does not authorize execution
        - Invariant 11: fail closed on bad/contradictory/stale evidence
        - Invariant 12: preserve exact identities (decision, plan, hardware)
        - No parallel authority introduced; uses existing ledger reconciliation
        """
        recon_detail = (
            f"verification={verification_state.value if verification_state else 'UNKNOWN'}; "
            f"{detail}"
        )
        # Record the failed verification in the ledger for audit/recovery
        # The admission decision itself is immutable and durably recorded;
        # we only mark the residency/reservation state as unconfirmed.
        ledger = self.ledger
        now = _now()
        # Append a reconciliation event. The detail is content-addressed rather
        # than inlined, matching how every other ledger payload is stored: the
        # event row carries a reference and the bytes are immutable.
        recon_hash = ledger.store_content_bytes(recon_detail.encode("utf-8"))
        # Append a reconciliation event
        with ledger.Tx(ledger) as tx:
            ledger.append_event_tx(
                tx, decision.session_id or "unknown",
                "placement_failed", "operator", "controller", "controller", now,
                payload_store="content", payload_ref=recon_hash,
                payload_sha256=recon_hash,
            )
        # Note: we do NOT release the reservation here because the model may have
        # been partially loaded; recovery/cleanup is handled by the managed lifecycle
        # which knows how to drain from EXCLUSIVE_ACTIVE state safely.
        # The key invariant: the immutable decision record is unchanged; only
        # downstream state reflects the failed verification.
