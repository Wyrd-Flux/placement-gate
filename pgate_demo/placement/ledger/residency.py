"""Stage 3B-2B-I: durable backend-residency occurrence models.

Separation of concerns (frozen by review):

    admission            immutable decision at a point in time
    turn reservation     lifecycle events locking budget for an INTENT window
    backend residency    OBSERVED state that may outlive any reservation
    placement attestation occurrences judging evidence about one placement

Residency is an OCCURRENCE stream (event identity != content identity, the
Stage 2 lesson): identical observations remain separate events. The
projection is disposable and rebuildable; it is NEVER the only record that a
model was resident.

expires_at is stored EVIDENCE ONLY. It is never a timer and can never release
residency by elapsing. Only a durable ABSENT_OBSERVED event (from a
successful full /api/ps census lacking the digest) or an explicit
RECOVERY_CLEAR-equivalent terminal event may end conservative consumption.

Accounting discipline (review correction): Ollama-reported `size` and
`size_vram` are kept as separate channels TOTAL_REPORTED / GPU_REPORTED.
The host-RAM subset is NOT assumed to be `size - size_vram` (installed
semantics unproven): RAM consumption uses the conservative TOTAL while the
model is RESIDENT. If channels contradict (size_vram > size, negatives) the
row goes UNKNOWN and the budget becomes UNKNOWN (None) => admission fails
closed. No double subtraction in any single pool.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, field_validator

RESIDENCY_POLICY_VERSION = "3b2b-i.1"


class ResidencyKind(str, enum.Enum):
    RESIDENT_OBSERVED = "RESIDENT_OBSERVED"
    ABSENT_OBSERVED = "ABSENT_OBSERVED"
    RECOVERY_HOLD = "RECOVERY_HOLD"


class ResidencyState(str, enum.Enum):
    RESIDENT = "RESIDENT"
    ABSENT = "ABSENT"
    UNKNOWN = "UNKNOWN"


class ResidencyBudget(BaseModel):
    """Aggregate consumption attributable to residency, by pool.

    None in either dimension means UNKNOWN: admission must fail closed on
    that pool, never coerce to zero.
    """

    model_config = ConfigDict(frozen=True)

    ram_bytes_consumed: int | None = 0
    vram_bytes_consumed: int | None = 0
    consuming_rows: int = 0


def is_contradictory(total: int | None, gpu: int | None) -> bool:
    if total is not None and total < 0:
        return True
    if gpu is not None and gpu < 0:
        return True
    if total is not None and gpu is not None and gpu > total:
        return True
    return False


class ResidencyEvent(BaseModel):
    """One append-only residency lifecycle event (occurrence identity)."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    kind: ResidencyKind
    digest: str  # manifest-bridge 64-hex identity observed for the model
    evidence_content_hash: str  # raw /api/ps payload stored content-addressed
    observed_at_utc: str
    reason: str
    decision_id: str | None = None
    reservation_id: str | None = None
    session_id: str | None = None
    identity_id: str | None = None
    reported_total_bytes: int | None = None
    reported_gpu_bytes: int | None = None
    expires_at_evidence: str | None = None  # EVIDENCE ONLY, never interpreted
    policy_version: str = RESIDENCY_POLICY_VERSION

    @field_validator("digest", "evidence_content_hash")
    @classmethod
    def _hex(cls, v: str) -> str:
        if len(v) != 64 or any(c not in "0123456789abcdef" for c in v.lower()):
            raise ValueError("residency identities must be exact 64-hex")
        return v.lower()


def fold_residency(events: list[ResidencyEvent], *, recovery: bool = False) -> dict[str, dict]:
    """Deterministic ordered replay of residency events into the current
    projection, per digest. recovery=True converts an unresolved RESIDENT
    into conservative UNKNOWN (which still consumes budget)."""
    state: dict[str, dict] = {}
    # replay order: caller supplies ledger-durable append order (ORDER BY seq)
    for ev in events:
        d = ev.digest
        if ev.kind is ResidencyKind.ABSENT_OBSERVED:
            state[d] = {
                "state": ResidencyState.ABSENT.value,
                "reported_total_bytes": ev.reported_total_bytes,
                "reported_gpu_bytes": ev.reported_gpu_bytes,
                "last_event_id": ev.event_id,
                "updated_at_utc": ev.observed_at_utc,
            }
            continue
        if ev.kind is ResidencyKind.RECOVERY_HOLD:
            cur = state.get(d)
            if cur is None or cur["state"] != ResidencyState.ABSENT.value:
                state[d] = {
                    "state": ResidencyState.UNKNOWN.value,
                    "reported_total_bytes": ev.reported_total_bytes,
                    "reported_gpu_bytes": ev.reported_gpu_bytes,
                    "last_event_id": ev.event_id,
                    "updated_at_utc": ev.observed_at_utc,
                }
            continue
        # RESIDENT_OBSERVED
        contradictory = is_contradictory(
            ev.reported_total_bytes, ev.reported_gpu_bytes
        )
        state[d] = {
            "state": (
                ResidencyState.UNKNOWN.value if contradictory
                else ResidencyState.RESIDENT.value
            ),
            "reported_total_bytes": ev.reported_total_bytes,
            "reported_gpu_bytes": ev.reported_gpu_bytes,
            "last_event_id": ev.event_id,
            "updated_at_utc": ev.observed_at_utc,
        }
    if recovery:
        for row in state.values():
            if row["state"] == ResidencyState.RESIDENT.value:
                # uncertain backend state after restart: conservative UNKNOWN
                row["state"] = ResidencyState.UNKNOWN.value
    return state


def residency_budget_from_projection(projection: dict[str, dict]) -> ResidencyBudget:
    ram: int | None = 0
    vram: int | None = 0
    rows = 0
    for row in projection.values():
        st = row["state"]
        if st == ResidencyState.ABSENT.value:
            continue
        rows += 1
        # RESIDENT or UNKNOWN both consume; UNKNOWN => both pools UNKNOWN
        total = row.get("reported_total_bytes")
        gpu = row.get("reported_gpu_bytes")
        if st == ResidencyState.UNKNOWN.value:
            ram = None
            vram = None
            continue
        if total is None or gpu is None:
            # a RESIDENT row must carry both channels; absence is UNKNOWN
            ram = None
            vram = None
            continue
        ram = None if ram is None else ram + total  # conservative TOTAL, not
        vram = None if vram is None else vram + gpu  # total-minus-gpu (RAM pool)
    return ResidencyBudget(
        ram_bytes_consumed=ram, vram_bytes_consumed=vram, consuming_rows=rows
    )
