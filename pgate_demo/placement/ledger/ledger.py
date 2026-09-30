"""Append-only event ledger with per-session hash chains.

The ledger is the canonical store. Current-state tables are disposable
projections. SQLite runs in WAL mode with synchronous=FULL, foreign keys on,
a single writer, and append-only triggers rejecting UPDATE/DELETE on canonical
rows.

Hash-chain topology:
  - Global SQLite `seq` (AUTOINCREMENT) provides total insertion order.
  - Each session holds its own chain + head.
  - event.prev_event_hash references the previous canonical event in the SAME
    session chain.
  - Interleaved events from other sessions never become prev_event_hash for
    this session.
  - Chain verification is deterministic per session.
  - Genesis event (session_created) has prev_event_hash = NULL.
  - Intentionally global events are stored in a separate `global_event` table
    with its own chain, never mixed into session chains.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Iterable, Optional

from .models import sha256_text

# Append-only invariant applies ONLY to true provenance tables. Current-state
# tables (turn, lease, action_proposal, identity, scope_snapshot, ...) are
# DISPOSABLE PROJECTIONS that the controller legitimately rewrites to advance
# derived state; the source of truth is always the `event` ledger.
# Stage 2 added backend_observation. Stage 3A adds (ALL additive; existing
# Stage 1/2 entries and semantics unchanged): hardware_profile_record,
# model_profile_record, admission_record, resource_reservation_event,
# topology_attestation. resource_reservation is a DISPOSABLE PROJECTION whose
# authority is the append-only lifecycle event history (v3 §3).
APPEND_ONLY_TABLES = [
    "event",
    "global_event",
    "content",  # content-addressed immutable payload store
    "backend_observation",  # Stage 2: provenance-bearing read-only observations
    "hardware_profile_record",  # Stage 3A: immutable hardware evidence (§6)
    "model_profile_record",  # Stage 3A: immutable digest-bound profiles (§7)
    "admission_record",  # Stage 3A: immutable AdmissionDecision (§5)
    "resource_reservation_event",  # Stage 3A: authoritative lifecycle (§3)
    "topology_attestation",  # Stage 3A: append-only attestations (§5)
    "backend_residency_event",  # Stage 3B-2B-I: durable residency lifecycle
    "execution_attempt",  # Stage 3B-2B-I: append-only attempt records
    "identity_admission",  # Stage 3B-2B-I: immutable admission bindings
]


class LedgerIntegrityError(Exception):
    """Raised when the ledger hash chain or schema fails verification."""


class AppendOnlyViolation(Exception):
    """Raised when code attempts to UPDATE/DELETE a canonical append-only row."""


class ReservationHistoryError(Exception):
    """Raised when the authoritative reservation event history is invalid
    (double acquisition, terminal without acquisition, amount drift)."""


def _candidate_key(row: sqlite3.Row) -> str:
    d = dict(row)
    d.pop("event_hash", None)
    d.pop("prev_event_hash", None)
    d.pop("payload_sha256", None)
    return json.dumps(d, sort_keys=True, default=str)


class Ledger:
    """Thread-safe append-only store. Single writer enforced via a lock."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ #
    # Schema / connection
    # ------------------------------------------------------------------ #
    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        conn = self.connect()
        try:
            cur = conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS global_event(
                  seq INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_id TEXT UNIQUE NOT NULL,
                  kind TEXT NOT NULL,
                  payload_sha256 TEXT NOT NULL,
                  prev_event_hash TEXT,
                  event_hash TEXT UNIQUE NOT NULL,
                  occurred_at_utc TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS content(
                  content_hash TEXT PRIMARY KEY,
                  content BLOB NOT NULL
                );

                CREATE TABLE IF NOT EXISTS event(
                  seq INTEGER PRIMARY KEY AUTOINCREMENT,
                  event_id TEXT UNIQUE NOT NULL,
                  session_id TEXT NOT NULL,
                  identity_id TEXT,
                  turn_id TEXT,
                  action_id TEXT,
                  kind TEXT NOT NULL,
                  visibility TEXT NOT NULL CHECK (visibility IN ('public','operator','sealed')),
                  actor_type TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  payload_store TEXT,
                  payload_ref TEXT,
                  payload_sha256 TEXT,
                  prev_event_hash TEXT,
                  event_hash TEXT UNIQUE NOT NULL,
                  occurred_at_utc TEXT NOT NULL,
                  controller_version TEXT NOT NULL
                );


















                CREATE TABLE IF NOT EXISTS backend_observation(
                  seq INTEGER PRIMARY KEY AUTOINCREMENT,
                  observation_id TEXT UNIQUE NOT NULL,
                  operation TEXT NOT NULL,
                  backend_mode TEXT NOT NULL,
                  endpoint TEXT NOT NULL,
                  ollama_version TEXT,
                  observed_at_utc TEXT NOT NULL,
                  raw_content_hash TEXT NOT NULL,
                  status TEXT NOT NULL,
                  disposition TEXT NOT NULL,
                  visibility TEXT NOT NULL CHECK
                    (visibility IN ('public','operator','sealed')),
                  failure TEXT,
                  visible_model_digests TEXT,
                  extracted_facts_sha256 TEXT
                );
                """
            )
            conn.executescript(self._STAGE3_SCHEMA)
            conn.executescript(self._STAGE3B2B_SCHEMA)
            conn.commit()
            self._ensure_trigger(conn)
        finally:
            conn.close()

    def _ensure_trigger(self, conn: sqlite3.Connection) -> None:
        for table in APPEND_ONLY_TABLES:
            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS trg_no_update_{table}
                BEFORE UPDATE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, 'append-only: UPDATE on {table} forbidden');
                END
                """
            )
            conn.execute(
                f"""
                CREATE TRIGGER IF NOT EXISTS trg_no_delete_{table}
                BEFORE DELETE ON {table}
                BEGIN
                    SELECT RAISE(ABORT, 'append-only: DELETE on {table} forbidden');
                END
                """
            )
        conn.commit()

    # ------------------------------------------------------------------ #
    # Primitive appends
    # ------------------------------------------------------------------ #
    def _session_head(self, conn: sqlite3.Connection, session_id: str) -> Optional[str]:
        row = conn.execute(
            "SELECT event_hash FROM event WHERE session_id=? ORDER BY seq DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return row["event_hash"] if row else None

    def append_event(
        self,
        session_id: str,
        kind: str,
        visibility: str,
        actor_type: str,
        actor_id: str,
        occurred_at_utc: str,
        *,
        event_id: Optional[str] = None,
        identity_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        action_id: Optional[str] = None,
        payload_store: Optional[str] = None,
        payload_ref: Optional[str] = None,
        payload_sha256: Optional[str] = None,
        controller_version: str = "stage1",
        commit: bool = True,
    ) -> tuple[str, int]:
        """Return (event_id, seq). If commit=False the caller manages txn."""
        conn = self._get_or_create_local()
        with self._lock:
            event_id = event_id or uuid.uuid4().hex
            prev = self._session_head(conn, session_id)
            canonical = "|".join(
                [
                    event_id,
                    session_id,
                    kind,
                    visibility,
                    actor_type,
                    actor_id,
                    payload_ref or "",
                    prev or "",
                ]
            )
            event_hash = sha256_text(canonical)
            conn.execute(
                """
                INSERT INTO event(
                  event_id, session_id, identity_id, turn_id, action_id, kind,
                  visibility, actor_type, actor_id, payload_store, payload_ref,
                  payload_sha256, prev_event_hash, event_hash, occurred_at_utc,
                  controller_version)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    event_id,
                    session_id,
                    identity_id,
                    turn_id,
                    action_id,
                    kind,
                    visibility,
                    actor_type,
                    actor_id,
                    payload_store,
                    payload_ref,
                    payload_sha256,
                    prev,
                    event_hash,
                    occurred_at_utc,
                    controller_version,
                ),
            )
            if commit:
                conn.commit()
            row = conn.execute(
                "SELECT seq FROM event WHERE event_id=?", (event_id,)
            ).fetchone()
            return event_id, row["seq"]

    def append_global_event(
        self,
        kind: str,
        payload_sha256: str,
        occurred_at_utc: str,
        event_id: Optional[str] = None,
    ) -> str:
        conn = self.connect()
        with self._lock, conn:
            event_id = event_id or uuid.uuid4().hex
            prev = conn.execute(
                "SELECT event_hash FROM global_event ORDER BY seq DESC LIMIT 1"
            ).fetchone()
            prev_hash = prev["event_hash"] if prev else None
            event_hash = sha256_text("|".join([event_id, kind, prev_hash or ""]))
            conn.execute(
                """
                INSERT INTO global_event(seq, event_id, kind, payload_sha256,
                  prev_event_hash, event_hash, occurred_at_utc)
                VALUES(NULL,?,?,?,?,?,?)
                """,
                (event_id, kind, payload_sha256, prev_hash, event_hash, occurred_at_utc),
            )
            return event_id

    def head_seq(self) -> int:
        conn = self.connect()
        try:
            row = conn.execute("SELECT COALESCE(MAX(seq),0) AS m FROM event").fetchone()
            return int(row["m"])
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Content-addressed payload store (durable-content layer for G-1B/E)
    # ------------------------------------------------------------------ #
    def store_content_tx(
        self, tx: "Ledger.Tx", content: bytes
    ) -> str:
        """Idempotently store content bytes; returns the content hash.

        Content is content-addressed (hash = sha256 of bytes) and immutable.
        Re-storing the same bytes is a no-op that returns the same hash.
        """
        content_hash = sha256_text(content.decode("utf-8", "replace"))
        tx.conn.execute(
            "INSERT OR IGNORE INTO content(content_hash, content) VALUES(?,?)",
            (content_hash, content),
        )
        return content_hash

    def get_content(self, content_hash: str) -> Optional[bytes]:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT content FROM content WHERE content_hash=?", (content_hash,)
            ).fetchone()
            return row["content"] if row else None
        finally:
            conn.close()

    def store_content_bytes(self, content: bytes) -> str:
        """Public content-addressed append (idempotent by hash)."""
        content_hash = sha256_text(content.decode("utf-8", "replace"))
        conn = self.connect()
        with self._lock, conn:
            conn.execute(
                "INSERT OR IGNORE INTO content(content_hash, content) VALUES(?,?)",
                (content_hash, content),
            )
        return content_hash

    # ------------------------------------------------------------------ #
    # Stage 2: read-only observation provenance (single writer authority)
    # ------------------------------------------------------------------ #
    def store_observation(
        self,
        *,
        observation_id: str,
        operation: str,
        backend_mode: str,
        endpoint: str,
        ollama_version: Optional[str],
        observed_at_utc: str,
        raw_bytes: bytes,
        status: str,
        disposition: str,
        visibility: str = "operator",
        failure: Optional[str] = None,
        visible_model_digests: Optional[list[str]] = None,
        extracted_facts_sha256: Optional[str] = None,
    ) -> str:
        """Append a provenance-bearing read-only observation (append-only).

        OBSERVATION IDENTITY != CONTENT IDENTITY: observation_id names the
        occurrence (unique per append); raw_content_hash names the bytes
        (content-addressed, deduplicated across occurrences). Identical raw
        bytes observed twice create two observation rows and one content row.

        ONE canonical write path: the raw response bytes are stored in the
        existing content-addressed `content` table (raw_content_hash), and the
        `backend_observation` row carries ONLY that content-hash reference plus
        observation metadata + explicit visibility. The raw evidence never
        becomes continuity/request content merely because it exists in content
        storage. This is the ledger (the single writer), not StoreManager.

        Runs in its own transaction (crash-safe, append-only).
        """
        raw_hash = sha256_text(raw_bytes.decode("utf-8", "replace"))
        with self.Tx(self) as tx:
            self.store_content_tx(tx, raw_bytes)
            tx.conn.execute(
                """
                INSERT INTO backend_observation(
                  observation_id, operation, backend_mode, endpoint, ollama_version,
                  observed_at_utc, raw_content_hash, status, disposition, visibility,
                  failure, visible_model_digests, extracted_facts_sha256)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    observation_id, operation, backend_mode, endpoint,
                    ollama_version, observed_at_utc, raw_hash, status, disposition,
                    visibility, failure,
                    json.dumps(visible_model_digests or []),
                    extracted_facts_sha256,
                ),
            )
        return raw_hash

    def get_observation(self, observation_id: str) -> Optional[dict]:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM backend_observation WHERE observation_id=?",
                (observation_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def all_observations(self) -> list[dict]:
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM backend_observation ORDER BY seq"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Stage 3A: hardware/profile/admission provenance (additive)
    #
    # OCCURRENCE identity (record_id/decision_id UUIDs + seq ordering) is
    # separate from CONTENT identity (canonical bytes -> content hash).
    # These tables are append-only; evidence referenced by an older decision
    # can never be mutated by a later probe or revised profile (v3 §5-§7).
    # ------------------------------------------------------------------ #
    def append_hardware_profile(
        self, *, record_id: str, profile, observed_at_utc: str,
        probe_sources: list[str], tool_versions: list[str],
    ) -> str:
        """Store an immutable HardwareProfile occurrence; returns its content
        hash. Duck-typed on .canonical_bytes()/.detection_status to avoid a
        ledger->hardware import cycle."""
        raw = profile.canonical_bytes()
        status = getattr(profile.detection_status, "value", str(profile.detection_status))
        conn = self.connect()
        with self._lock, conn:
            content_hash = self._store_content_raw(conn, raw)
            conn.execute(
                "INSERT INTO hardware_profile_record(record_id, profile_content_hash,"
                " detection_status, probe_sources, observed_at_utc, tool_versions)"
                " VALUES(?,?,?,?,?,?)",
                (record_id, content_hash, status, json.dumps(probe_sources),
                 observed_at_utc, json.dumps(tool_versions)),
            )
        return content_hash

    def append_model_profile(
        self, *, record_id: str, profile, recorded_at_utc: str,
    ) -> str:
        raw = profile.canonical_bytes()
        conn = self.connect()
        with self._lock, conn:
            content_hash = self._store_content_raw(conn, raw)
            conn.execute(
                "INSERT INTO model_profile_record(record_id, digest,"
                " profile_content_hash, profile_version, source, provenance,"
                " recorded_at_utc) VALUES(?,?,?,?,?,?,?)",
                (record_id, profile.digest, content_hash,
                 profile.profile_version, profile.source.value,
                 profile.provenance, recorded_at_utc),
            )
        return content_hash

    def get_profile_bytes(self, content_hash: str) -> Optional[bytes]:
        return self.get_content(content_hash)

    def append_admission(self, decision) -> None:
        """Persist an immutable AdmissionDecision (duck-typed: canonical_bytes,
        model_dump). Historical attestation NEVER mutates this row."""
        raw = decision.canonical_bytes()
        d = decision.model_dump(mode="json")
        conn = self.connect()
        with self._lock, conn:
            content_hash = self._store_content_raw(conn, raw)
            conn.execute(
                "INSERT INTO admission_record(decision_id, session_id, identity_id,"
                " model_digest, model_profile_hash, hardware_profile_hash,"
                " requested_topology, reservation_ram_bytes, reservation_vram_bytes,"
                " total_required_bytes, policy_version, admitted, reason, detail,"
                " decision_content_hash, occurred_at_utc)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    d["decision_id"], d["session_id"], d["identity_id"],
                    d["model_digest"], d["model_profile_hash"],
                    d["hardware_profile_hash"], d["requested_topology"],
                    d["reservation_ram_bytes"], d["reservation_vram_bytes"],
                    d["total_required_bytes"], d["policy_version"],
                    1 if d["admitted"] else 0, d["reason"], d["detail"],
                    content_hash, d["occurred_at_utc"],
                ),
            )

    def get_admission(self, decision_id: str) -> Optional[dict]:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM admission_record WHERE decision_id=?", (decision_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Stage 3A: authoritative reservation lifecycle + disposable projection
    # ------------------------------------------------------------------ #
    RESERVATION_KINDS = (
        "RESERVATION_ACQUIRED",
        "RESERVATION_RELEASED",
        "RESERVATION_INVALIDATED",
        "RESERVATION_RECOVERY_HOLD",
        "RESERVATION_RECOVERY_CLEARED",
    )
    _TERMINAL = (
        "RESERVATION_RELEASED",
        "RESERVATION_INVALIDATED",
        "RESERVATION_RECOVERY_CLEARED",
    )

    def append_reservation_event(
        self, *, event_id: str, reservation_id: str, kind: str,
        occurred_at_utc: str, reason: str,
        ram_bytes: int = 0, vram_bytes: int = 0,
        decision_id: Optional[str] = None,
        session_id: Optional[str] = None,
        identity_id: Optional[str] = None,
        hardware_profile_hash: str = "",
        model_profile_hash: str = "",
        policy_version: str = "",
    ) -> str:
        """Append one lifecycle event and advance the projection.

        Single-writer transaction: event (authority) + projection row (mirror).
        """
        if kind not in self.RESERVATION_KINDS:
            raise ReservationHistoryError(f"unknown reservation event kind {kind!r}")
        payload = json.dumps(
            {
                "event_id": event_id, "reservation_id": reservation_id, "kind": kind,
                "ram_bytes": ram_bytes, "vram_bytes": vram_bytes,
                "decision_id": decision_id, "session_id": session_id,
                "identity_id": identity_id,
                "hardware_profile_hash": hardware_profile_hash,
                "model_profile_hash": model_profile_hash,
                "policy_version": policy_version, "reason": reason,
                "occurred_at_utc": occurred_at_utc,
            },
            sort_keys=True,
            default=str,
        )
        state = {
            "RESERVATION_ACQUIRED": "ACTIVE",
            "RESERVATION_RELEASED": "RELEASED",
            "RESERVATION_INVALIDATED": "INVALIDATED",
            "RESERVATION_RECOVERY_HOLD": "RECOVERY_HOLD",
            "RESERVATION_RECOVERY_CLEARED": "CLEARED",
        }[kind]
        conn = self.connect()
        with self._lock, conn:
            conn.execute(
                "INSERT INTO resource_reservation_event(event_id, reservation_id,"
                " kind, decision_id, session_id, identity_id, ram_bytes, vram_bytes,"
                " hardware_profile_hash, model_profile_hash, policy_version, reason,"
                " occurred_at_utc, payload_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, reservation_id, kind, decision_id, session_id, identity_id,
                 ram_bytes, vram_bytes, hardware_profile_hash, model_profile_hash,
                 policy_version, reason, occurred_at_utc,
                 sha256_text(payload)),
            )
            if kind == "RESERVATION_ACQUIRED":
                # occurrence identity: two identical-amount reservations are
                # independent rows; never content-deduplicated (v3 §8)
                conn.execute(
                    "INSERT INTO resource_reservation(reservation_id, decision_id,"
                    " session_id, identity_id, state, ram_bytes, vram_bytes,"
                    " hardware_profile_hash, model_profile_hash, policy_version,"
                    " acquired_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (reservation_id, decision_id, session_id, identity_id, state,
                     ram_bytes, vram_bytes, hardware_profile_hash,
                     model_profile_hash, policy_version, occurred_at_utc),
                )
            else:
                cur = conn.execute(
                    "SELECT reservation_id FROM resource_reservation"
                    " WHERE reservation_id=?", (reservation_id,),
                ).fetchone()
                if cur is None:
                    raise ReservationHistoryError(
                        f"terminal event for unknown reservation {reservation_id}"
                    )
                conn.execute(
                    "UPDATE resource_reservation SET state=? WHERE reservation_id=?",
                    (state, reservation_id),
                )
        return event_id

    def rebuild_reservation_projection(
        self, *, recovery: bool = False
    ) -> dict:
        """Deterministic, idempotent pure replay of the authoritative event
        history. Projection loss can never release resources (v3 §4).

        recovery=False: unclosed acquisition -> ACTIVE (normal runtime).
        recovery=True:  unclosed acquisition -> RECOVERY_HOLD (post-restart
        uncertainty; still consumes the FULL budget until durable evidence
        clears it).
        """
        conn = self.connect()
        self._lock.acquire()
        try:
            events = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM resource_reservation_event ORDER BY seq"
                ).fetchall()
            ]
            folded: dict[str, dict] = {}
            for ev in events:
                rid = ev["reservation_id"]
                prev = folded.get(rid)
                if ev["kind"] == "RESERVATION_ACQUIRED":
                    if prev is not None and prev["state"] in (
                        "ACTIVE", "RECOVERY_HOLD",
                    ):
                        raise ReservationHistoryError(
                            f"double acquisition for reservation {rid}"
                        )
                    state = "RECOVERY_HOLD" if recovery else "ACTIVE"
                    folded[rid] = {
                        "reservation_id": rid,
                        "decision_id": ev["decision_id"],
                        "session_id": ev["session_id"],
                        "identity_id": ev["identity_id"],
                        "state": state,
                        "ram_bytes": ev["ram_bytes"],
                        "vram_bytes": ev["vram_bytes"],
                        "hardware_profile_hash": ev["hardware_profile_hash"],
                        "model_profile_hash": ev["model_profile_hash"],
                        "policy_version": ev["policy_version"],
                        "acquired_at_utc": ev["occurred_at_utc"],
                    }
                    continue
                if prev is None:
                    raise ReservationHistoryError(
                        f"terminal event for unknown reservation {rid}"
                    )
                for amount in ("ram_bytes", "vram_bytes"):
                    if prev[amount] != ev[amount]:
                        raise ReservationHistoryError(
                            f"reservation {rid} amount drift on {amount}"
                        )
                prev["state"] = {
                    "RESERVATION_RELEASED": "RELEASED",
                    "RESERVATION_INVALIDATED": "INVALIDATED",
                    "RESERVATION_RECOVERY_HOLD": "RECOVERY_HOLD",
                    "RESERVATION_RECOVERY_CLEARED": "CLEARED",
                }[ev["kind"]]
            conn.execute("DELETE FROM resource_reservation")  # projection only
            for row in folded.values():
                conn.execute(
                    "INSERT INTO resource_reservation(reservation_id, decision_id,"
                    " session_id, identity_id, state, ram_bytes, vram_bytes,"
                    " hardware_profile_hash, model_profile_hash, policy_version,"
                    " acquired_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    tuple(row[k] for k in (
                        "reservation_id", "decision_id", "session_id",
                        "identity_id", "state", "ram_bytes", "vram_bytes",
                        "hardware_profile_hash", "model_profile_hash",
                        "policy_version", "acquired_at_utc",
                    )),
                )
            conn.commit()
            active = [r for r in folded.values()
                      if r["state"] in ("ACTIVE", "RECOVERY_HOLD")]
            return {
                "replayed_events": len(events),
                "reservations": len(folded),
                "active_or_hold": len(active),
                "budget_ram_bytes": sum(r["ram_bytes"] for r in active),
                "budget_vram_bytes": sum(r["vram_bytes"] for r in active),
            }
        finally:
            conn.close()
            self._lock.release()

    def reservation_totals(self) -> dict:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(ram_bytes),0) AS ram,"
                " COALESCE(SUM(vram_bytes),0) AS vram,"
                " COUNT(*) AS n FROM resource_reservation"
                " WHERE state IN ('ACTIVE','RECOVERY_HOLD')"
            ).fetchone()
            return {"ram_bytes": int(row["ram"]), "vram_bytes": int(row["vram"]),
                    "count": int(row["n"])}
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    # Stage 3A: append-only topology attestation occurrences (v3 §5)
    # ------------------------------------------------------------------ #
    def append_attestation(
        self, *, attestation_id: str, decision_id: str, evidence, state: str,
        requested_topology: str, hardware_profile_hash: str, tau_version: str,
        observed_at_utc: str,
    ) -> str:
        """Persist one immutable attestation occurrence. The AdmissionDecision
        row is never touched; multiple attestations per admission accumulate."""
        raw = evidence.canonical_bytes()
        conn = self.connect()
        with self._lock, conn:
            content_hash = self._store_content_raw(conn, raw)
            conn.execute(
                "INSERT INTO topology_attestation(attestation_id, decision_id,"
                " state, evidence_content_hash, requested_topology,"
                " hardware_profile_hash, tau_version, observed_at_utc)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (attestation_id, decision_id, state, content_hash,
                 requested_topology, hardware_profile_hash, tau_version,
                 observed_at_utc),
            )
        return content_hash

    def attestations_for(self, decision_id: str) -> list[dict]:
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM topology_attestation WHERE decision_id=?"
                " ORDER BY seq",
                (decision_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def latest_attestation(self, decision_id: str) -> Optional[dict]:
        rows = self.attestations_for(decision_id)
        return rows[-1] if rows else None

    # ------------------------------------------------------------------ #
    # Schema: Stage 3A tables (additive; existing Stage 1/2 schema untouched)
    # ------------------------------------------------------------------ #
    _STAGE3_SCHEMA = """
    CREATE TABLE IF NOT EXISTS hardware_profile_record(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      record_id TEXT UNIQUE NOT NULL,
      profile_content_hash TEXT NOT NULL,
      detection_status TEXT NOT NULL,
      probe_sources TEXT NOT NULL,
      observed_at_utc TEXT NOT NULL,
      tool_versions TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS model_profile_record(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      record_id TEXT UNIQUE NOT NULL,
      digest TEXT NOT NULL,
      profile_content_hash TEXT NOT NULL,
      profile_version TEXT NOT NULL,
      source TEXT NOT NULL,
      provenance TEXT NOT NULL,
      recorded_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS admission_record(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      decision_id TEXT UNIQUE NOT NULL,
      session_id TEXT NOT NULL,
      identity_id TEXT,
      model_digest TEXT NOT NULL,
      model_profile_hash TEXT NOT NULL,
      hardware_profile_hash TEXT NOT NULL,
      requested_topology TEXT NOT NULL,
      reservation_ram_bytes INTEGER NOT NULL,
      reservation_vram_bytes INTEGER NOT NULL,
      total_required_bytes INTEGER NOT NULL,
      policy_version TEXT NOT NULL,
      admitted INTEGER NOT NULL,
      reason TEXT NOT NULL,
      detail TEXT NOT NULL,
      decision_content_hash TEXT NOT NULL,
      occurred_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS resource_reservation_event(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      event_id TEXT UNIQUE NOT NULL,
      reservation_id TEXT NOT NULL,
      decision_id TEXT,
      kind TEXT NOT NULL CHECK (kind IN (
        'RESERVATION_ACQUIRED','RESERVATION_RELEASED',
        'RESERVATION_INVALIDATED','RESERVATION_RECOVERY_HOLD',
        'RESERVATION_RECOVERY_CLEARED')),
      session_id TEXT,
      identity_id TEXT,
      ram_bytes INTEGER NOT NULL,
      vram_bytes INTEGER NOT NULL,
      hardware_profile_hash TEXT NOT NULL,
      model_profile_hash TEXT NOT NULL,
      policy_version TEXT NOT NULL,
      reason TEXT NOT NULL,
      occurred_at_utc TEXT NOT NULL,
      payload_sha256 TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS resource_reservation(
      reservation_id TEXT PRIMARY KEY,
      decision_id TEXT,
      session_id TEXT,
      identity_id TEXT,
      state TEXT NOT NULL CHECK (state IN (
        'ACTIVE','RECOVERY_HOLD','RELEASED','INVALIDATED','CLEARED')),
      ram_bytes INTEGER NOT NULL,
      vram_bytes INTEGER NOT NULL,
      hardware_profile_hash TEXT NOT NULL,
      model_profile_hash TEXT NOT NULL,
      policy_version TEXT NOT NULL,
      acquired_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS topology_attestation(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      attestation_id TEXT UNIQUE NOT NULL,
      decision_id TEXT NOT NULL,
      state TEXT NOT NULL,
      evidence_content_hash TEXT NOT NULL,
      requested_topology TEXT NOT NULL,
      hardware_profile_hash TEXT NOT NULL,
      tau_version TEXT NOT NULL,
      observed_at_utc TEXT NOT NULL
    );
    """

    # ------------------------------------------------------------------ #
    # Stage 3B-2B-I schema (additive; all prior schema/semantics untouched)
    # ------------------------------------------------------------------ #
    _STAGE3B2B_SCHEMA = """
    CREATE TABLE IF NOT EXISTS backend_residency_event(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      event_id TEXT UNIQUE NOT NULL,
      kind TEXT NOT NULL CHECK (kind IN (
        'RESIDENT_OBSERVED','ABSENT_OBSERVED','RECOVERY_HOLD')),
      digest TEXT NOT NULL,
      decision_id TEXT,
      reservation_id TEXT,
      session_id TEXT,
      identity_id TEXT,
      reported_total_bytes INTEGER,
      reported_gpu_bytes INTEGER,
      evidence_content_hash TEXT NOT NULL,
      expires_at_evidence TEXT,
      observed_at_utc TEXT NOT NULL,
      reason TEXT NOT NULL,
      policy_version TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS backend_residency(
      digest TEXT PRIMARY KEY,
      state TEXT NOT NULL CHECK (state IN ('RESIDENT','ABSENT','UNKNOWN')),
      reported_total_bytes INTEGER,
      reported_gpu_bytes INTEGER,
      last_event_id TEXT NOT NULL,
      updated_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS identity_admission(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      identity_id TEXT UNIQUE NOT NULL,
      session_id TEXT NOT NULL,
      decision_id TEXT NOT NULL,
      reservation_id TEXT NOT NULL,
      bound_at_utc TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS execution_attempt(
      seq INTEGER PRIMARY KEY AUTOINCREMENT,
      attempt_id TEXT UNIQUE NOT NULL,
      turn_id TEXT NOT NULL,
      identity_id TEXT NOT NULL,
      decision_id TEXT NOT NULL,
      reservation_id TEXT NOT NULL,
      manifest_sha256 TEXT NOT NULL,
      expected_digest TEXT NOT NULL,
      started_at_utc TEXT NOT NULL
    );
    """

    # ------------------------------------------------------------------ #
    # Stage 3B-2B-I: residency lifecycle (durable events + rebuildable
    # projection). expires_at is stored as evidence only; it NEVER
    # releases residency. Contradictory channel facts => UNKNOWN (fail
    # closed), never coerced or zero-filled.
    # ------------------------------------------------------------------ #
    def append_residency_event(
        self, *, event_id: str, kind: str, digest: str,
        evidence_content_hash: str, observed_at_utc: str, reason: str,
        decision_id=None, reservation_id=None, session_id=None,
        identity_id=None, reported_total_bytes=None, reported_gpu_bytes=None,
        expires_at_evidence=None, policy_version="3b2b-i.1",
    ) -> str:
        if kind not in ("RESIDENT_OBSERVED", "ABSENT_OBSERVED", "RECOVERY_HOLD"):
            raise ReservationHistoryError(f"unknown residency kind {kind!r}")
        conn = self.connect()
        with self._lock, conn:
            conn.execute(
                "INSERT INTO backend_residency_event(event_id, kind, digest,"
                " decision_id, reservation_id, session_id, identity_id,"
                " reported_total_bytes, reported_gpu_bytes,"
                " evidence_content_hash, expires_at_evidence,"
                " observed_at_utc, reason, policy_version)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (event_id, kind, digest.lower(), decision_id, reservation_id,
                 session_id, identity_id, reported_total_bytes,
                 reported_gpu_bytes, evidence_content_hash.lower(),
                 expires_at_evidence, observed_at_utc, reason, policy_version),
            )
            self._project_residency(
                conn, digest.lower(), kind, reported_total_bytes,
                reported_gpu_bytes, event_id, observed_at_utc,
            )
        return event_id

    def _project_residency(self, conn, digest, kind, total, gpu, event_id, at):
        from .residency import is_contradictory, ResidencyState

        if kind == "ABSENT_OBSERVED":
            state = ResidencyState.ABSENT.value
        elif kind == "RECOVERY_HOLD":
            state = ResidencyState.UNKNOWN.value
        else:
            state = (
                ResidencyState.UNKNOWN.value
                if is_contradictory(total, gpu)
                else ResidencyState.RESIDENT.value
            )
        conn.execute(
            "INSERT INTO backend_residency(digest, state, reported_total_bytes,"
            " reported_gpu_bytes, last_event_id, updated_at_utc)"
            " VALUES(?,?,?,?,?,?)"
            " ON CONFLICT(digest) DO UPDATE SET state=excluded.state,"
            " reported_total_bytes=excluded.reported_total_bytes,"
            " reported_gpu_bytes=excluded.reported_gpu_bytes,"
            " last_event_id=excluded.last_event_id,"
            " updated_at_utc=excluded.updated_at_utc",
            (digest, state, total, gpu, event_id, at),
        )

    def residency_events(self, digest: str) -> list[dict]:
        conn = self.connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM backend_residency_event WHERE digest=?"
                    " ORDER BY seq", (digest.lower(),)
                ).fetchall()
            ]
        finally:
            conn.close()

    def residency_projection(self) -> list[dict]:
        conn = self.connect()
        try:
            return [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM backend_residency ORDER BY digest"
                ).fetchall()
            ]
        finally:
            conn.close()

    def rebuild_residency_projection(self, *, recovery: bool = False) -> dict:
        """Deterministic idempotent replay from the authoritative event
        stream (single per-digest fold; last event wins; recovery converts
        unresolved RESIDENT into conservative UNKNOWN)."""
        from .residency import is_contradictory

        conn = self.connect()
        try:
            rows = [
                dict(r)
                for r in conn.execute(
                    "SELECT * FROM backend_residency_event ORDER BY seq"
                ).fetchall()
            ]
            folded: dict[str, dict] = {}
            for ev in rows:
                d = ev["digest"]
                k = ev["kind"]
                if k == "ABSENT_OBSERVED":
                    folded[d] = {"state": "ABSENT", "r": ev}
                elif k == "RECOVERY_HOLD":
                    if d not in folded or folded[d]["state"] != "ABSENT":
                        folded[d] = {"state": "UNKNOWN", "r": ev}
                else:
                    contra = is_contradictory(
                        ev["reported_total_bytes"], ev["reported_gpu_bytes"]
                    )
                    folded[d] = {
                        "state": "UNKNOWN" if contra else "RESIDENT", "r": ev
                    }
            if recovery:
                for f in folded.values():
                    if f["state"] == "RESIDENT":
                        f["state"] = "UNKNOWN"  # uncertain after restart
            conn.execute("DELETE FROM backend_residency")  # projection only
            for d, f in folded.items():
                ev = f["r"]
                conn.execute(
                    "INSERT INTO backend_residency(digest, state,"
                    " reported_total_bytes, reported_gpu_bytes,"
                    " last_event_id, updated_at_utc) VALUES(?,?,?,?,?,?)",
                    (d, f["state"], ev["reported_total_bytes"],
                     ev["reported_gpu_bytes"], ev["event_id"],
                     ev["observed_at_utc"]),
                )
            conn.commit()
            states = [f["state"] for f in folded.values()]
            return {
                "replayed_events": len(rows), "digests": len(folded),
                "resident": states.count("RESIDENT"),
                "unknown": states.count("UNKNOWN"),
                "absent": states.count("ABSENT"),
            }
        finally:
            conn.close()

    def residency_budget(self) -> dict:
        """Conservative per-pool consumption from the current projection.
        None == UNKNOWN => callers must fail closed (no zero coercion).
        RAM uses TOTAL_REPORTED (never total-minus-gpu; unproven semantics);
        VRAM uses GPU_REPORTED; pools are distinct, so no double subtraction
        occurs within a pool."""
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM backend_residency WHERE state != 'ABSENT'"
                " ORDER BY digest"
            ).fetchall()
        finally:
            conn.close()
        ram: int | None = 0
        vram: int | None = 0
        for r in rows:
            if r["state"] == "UNKNOWN":
                ram = None
                vram = None
                continue
            if r["reported_total_bytes"] is None or r["reported_gpu_bytes"] is None:
                ram = None
                vram = None
                continue
            if ram is not None:
                ram += int(r["reported_total_bytes"])
            if vram is not None:
                vram += int(r["reported_gpu_bytes"])
        return {"ram_bytes": ram, "vram_bytes": vram, "rows": len(rows)}

    # ------------------------------------------------------------------ #
    # Stage 3B-2B-I: identity<->admission binding + execution attempts
    # ------------------------------------------------------------------ #
    def bind_identity_admission(
        self, *, identity_id: str, session_id: str, decision_id: str,
        reservation_id: str, bound_at_utc: str,
    ) -> None:
        conn = self.connect()
        with self._lock, conn:
            conn.execute(
                "INSERT INTO identity_admission(identity_id, session_id,"
                " decision_id, reservation_id, bound_at_utc)"
                " VALUES(?,?,?,?,?)",
                (identity_id, session_id, decision_id, reservation_id,
                 bound_at_utc),
            )

    def get_identity_admission(self, identity_id: str) -> Optional[dict]:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM identity_admission WHERE identity_id=?",
                (identity_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def append_execution_attempt(
        self, *, attempt_id: str, turn_id: str, identity_id: str,
        decision_id: str, reservation_id: str, manifest_sha256: str,
        expected_digest: str, started_at_utc: str,
    ) -> None:
        conn = self.connect()
        with self._lock, conn:
            conn.execute(
                "INSERT INTO execution_attempt(attempt_id, turn_id,"
                " identity_id, decision_id, reservation_id, manifest_sha256,"
                " expected_digest, started_at_utc)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (attempt_id, turn_id, identity_id, decision_id,
                 reservation_id, manifest_sha256, expected_digest,
                 started_at_utc),
            )

    def latest_execution_attempt(self, identity_id: str) -> Optional[dict]:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM execution_attempt WHERE identity_id=?"
                " ORDER BY seq DESC LIMIT 1", (identity_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def reservation_state(self, reservation_id: str) -> Optional[str]:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT state FROM resource_reservation WHERE reservation_id=?",
                (reservation_id,),
            ).fetchone()
            return row["state"] if row else None
        finally:
            conn.close()

    # ------------------------------------------------------------------ #
    def _store_content_raw(self, conn: sqlite3.Connection, content: bytes) -> str:
        """Same content addressing as store_content_tx (hash = sha256 of the
        text-decoded content), usable inside an already-held connection."""
        content_hash = sha256_text(content.decode("utf-8", "replace"))
        conn.execute(
            "INSERT OR IGNORE INTO content(content_hash, content) VALUES(?,?)",
            (content_hash, content),
        )
        return content_hash


    # ------------------------------------------------------------------ #
    # Readers
    # ------------------------------------------------------------------ #
    def get_event(self, event_id: str) -> Optional[dict]:
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT * FROM event WHERE event_id=?", (event_id,)
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def get_session_events(self, session_id: str) -> list[dict]:
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM event WHERE session_id=? ORDER BY seq",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def iterate_session_chain(self, session_id: str) -> Iterable[dict]:
        yield from self.get_session_events(session_id)

    # ------------------------------------------------------------------ #
    # Integrity verification
    # ------------------------------------------------------------------ #
    def verify_session_chain(self, session_id: str) -> None:
        """Deterministically verify the session chain. Raises if broken."""
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM event WHERE session_id=? ORDER BY seq",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
        prev_hash: Optional[str] = None
        for i, row in enumerate(rows):
            r = dict(row)
            if i == 0:
                if r["prev_event_hash"] is not None:
                    raise LedgerIntegrityError(
                        f"session genesis event {r['event_id']} has non-NULL prev_event_hash"
                    )
            else:
                if r["prev_event_hash"] != prev_hash:
                    raise LedgerIntegrityError(
                        f"chain break at event {r['event_id']}: expected prev "
                        f"{prev_hash}, got {r['prev_event_hash']}"
                    )
            canonical = "|".join(
                [
                    r["event_id"],
                    r["session_id"],
                    r["kind"],
                    r["visibility"],
                    r["actor_type"],
                    r["actor_id"],
                    r["payload_ref"] or "",
                    r["prev_event_hash"] or "",
                ]
            )
            expected = sha256_text(canonical)
            if expected != r["event_hash"]:
                raise LedgerIntegrityError(
                    f"hash mismatch at event {r['event_id']}"
                )
            prev_hash = expected

    def verify_global_chain(self) -> None:
        conn = self.connect()
        try:
            rows = conn.execute(
                "SELECT * FROM global_event ORDER BY seq"
            ).fetchall()
        finally:
            conn.close()
        prev_hash: Optional[str] = None
        for i, row in enumerate(rows):
            r = dict(row)
            if i == 0:
                if r["prev_event_hash"] is not None:
                    raise LedgerIntegrityError("global genesis has non-NULL prev")
            else:
                if r["prev_event_hash"] != prev_hash:
                    raise LedgerIntegrityError("global chain break")
            expected = sha256_text(
                "|".join([r["event_id"], r["kind"], r["prev_event_hash"] or ""])
            )
            if expected != r["event_hash"]:
                raise LedgerIntegrityError("global hash mismatch")
            prev_hash = expected

    def verify_all(self) -> None:
        conn = self.connect()
        try:
            session_ids = [
                r["session_id"]
                for r in conn.execute("SELECT DISTINCT session_id FROM event").fetchall()
            ]
        finally:
            conn.close()
        for sid in session_ids:
            self.verify_session_chain(sid)
        self.verify_global_chain()

    # ------------------------------------------------------------------ #
    # Crash / corruption fixtures
    # ------------------------------------------------------------------ #
    def raw_conn(self) -> sqlite3.Connection:
        """Expose a raw connection for explicit corruption fixtures only."""
        return self.connect()

    def integrity_check(self) -> str:
        """Run SQLite PRAGMA integrity_check. Returns 'ok' or the report."""
        conn = self.connect()
        try:
            row = conn.execute("PRAGMA integrity_check").fetchone()
            return str(row[0])
        finally:
            conn.close()

    def _get_or_create_local(self) -> sqlite3.Connection:
        return self.connect()

    # ------------------------------------------------------------------ #
    # Explicit transaction support (durable intent vs external effect)
    # ------------------------------------------------------------------ #
    class Tx:
        """A transaction handle granting explicit commit control.

        Crash semantics:
          - if no commit() occurs, the txn rolls back and events do not exist;
          - a committed event fully exists after recovery;
          - an ordinary pre-commit crash is NOT ledger corruption.
        """

        def __init__(self, ledger: "Ledger"):
            self.ledger = ledger
            self.conn = ledger.connect()
            self._in_txn = False

        def __enter__(self) -> "Ledger.Tx":
            ledger = self.ledger
            ledger._lock.acquire()
            self.conn.execute("BEGIN")
            self._in_txn = True
            return self

        def __exit__(self, exc_type, exc, tb):
            try:
                if exc_type is not None:
                    self.conn.rollback()
                elif self._in_txn:
                    self.conn.commit()
            finally:
                self._in_txn = False
                self.conn.close()
                self.ledger._lock.release()
            return False

        def rollback(self) -> None:
            if self._in_txn:
                self.conn.rollback()
                self._in_txn = False

    def append_event_tx(
        self,
        tx: "Ledger.Tx",
        session_id: str,
        kind: str,
        visibility: str,
        actor_type: str,
        actor_id: str,
        occurred_at_utc: str,
        *,
        event_id: Optional[str] = None,
        identity_id: Optional[str] = None,
        turn_id: Optional[str] = None,
        action_id: Optional[str] = None,
        payload_store: Optional[str] = None,
        payload_ref: Optional[str] = None,
        payload_sha256: Optional[str] = None,
        controller_version: str = "stage1",
    ) -> tuple[str, int]:
        """Append within an explicit transaction (commit controlled by caller)."""
        conn = tx.conn
        event_id = event_id or uuid.uuid4().hex
        prev = self._session_head(conn, session_id)
        canonical = "|".join(
            [
                event_id,
                session_id,
                kind,
                visibility,
                actor_type,
                actor_id,
                payload_ref or "",
                prev or "",
            ]
        )
        event_hash = sha256_text(canonical)
        conn.execute(
            """
            INSERT INTO event(
              event_id, session_id, identity_id, turn_id, action_id, kind,
              visibility, actor_type, actor_id, payload_store, payload_ref,
              payload_sha256, prev_event_hash, event_hash, occurred_at_utc,
              controller_version)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                event_id,
                session_id,
                identity_id,
                turn_id,
                action_id,
                kind,
                visibility,
                actor_type,
                actor_id,
                payload_store,
                payload_ref,
                payload_sha256,
                prev,
                event_hash,
                occurred_at_utc,
                controller_version,
            ),
        )
        row = conn.execute("SELECT seq FROM event WHERE event_id=?", (event_id,)).fetchone()
        return event_id, row["seq"]
