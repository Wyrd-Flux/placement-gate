"""Placement Gate session: observe, census, plan, place.

Every decision is delegated. This module builds upstream inputs and unpacks
upstream results. It implements no placement policy, no budget arithmetic, and
no verification rule. If a capability is missing, the operation reports
UNAVAILABLE rather than approximating it.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .exit_codes import (
    EXIT_CAPABILITY_UNAVAILABLE,
    EXIT_OK,
    EXIT_SERVER_UNAVAILABLE,
)
from .providers import CapabilityUnavailable, ProviderSet, resolve_providers

GIB = 1024 ** 3
PGATE_VERSION = "0.1.0"

#: Capabilities the planner itself needs, independent of a live server.
PLANNING_CAPABILITIES = (
    "hardware_facts",
    "hardware_observer",
    "hardware_memory",
    "hardware_nvidia",
    "placement_planner",
    "model_profile",
)
#: Capabilities that additionally require a reachable Ollama service.
LIVE_CAPABILITIES = ("chat_adapter", "backends_base", "ledger", "controller")
ALL_CAPABILITIES = PLANNING_CAPABILITIES + LIVE_CAPABILITIES


class Status(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    SERVER_UNREACHABLE = "SERVER_UNREACHABLE"

    def as_dict(self) -> str:
        return self.value


@dataclass
class LoadReport:
    statuses: dict[str, Status] = field(default_factory=dict)
    resolutions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def status(self, capability: str) -> Status:
        return self.statuses.get(capability, Status.UNAVAILABLE)

    def available(self, capability: str) -> bool:
        return self.status(capability) is Status.AVAILABLE

    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, s in self.statuses.items() if s is not Status.AVAILABLE))

    def as_dict(self) -> dict[str, Any]:
        return {
            "pgate_version": PGATE_VERSION,
            "statuses": {n: s.value for n, s in sorted(self.statuses.items())},
            "resolutions": {n: self.resolutions[n] for n in sorted(self.resolutions)},
        }

    def render(self) -> str:
        lines = ["upstream capability load"]
        for name, status in sorted(self.statuses.items()):
            res = self.resolutions.get(name, {})
            mark = "ok  " if status is Status.AVAILABLE else "MISS"
            lines.append(f"  [{mark}] {name:22s} {status.value}")
            if res.get("detail"):
                lines.append(f"         {res['detail']}")
        return "\n".join(lines)


class OllamaUnreachable(RuntimeError):
    """The Ollama service did not answer. Distinct from a placement refusal."""

    def __init__(self, endpoint: str, detail: str) -> None:
        super().__init__(f"Ollama unreachable at {endpoint}: {detail}")
        self.endpoint = endpoint
        self.detail = detail


@dataclass
class ModelFacts:
    """The minimum model metadata a placement decision needs."""

    tag: str
    digest: str
    size_bytes: int
    parameter_size: str
    context_length: int
    family: str
    quantization: str

    @property
    def parameter_count(self) -> int:
        """Best-effort integer parameter count for the planner's metadata field.

        The planner treats this as metadata, never as a binary placement rule,
        so a rough parse is acceptable. Unparseable values are reported as 0
        and the caller is told, rather than a guess being used silently.
        """
        return _parse_parameters(self.parameter_size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "size_gib": round(self.size_bytes / GIB, 2),
            "parameter_size": self.parameter_size,
            "context_length": self.context_length,
            "family": self.family,
            "quantization": self.quantization,
        }


def _parse_parameters(label: str) -> int:
    if not label:
        return 0
    text = label.strip().upper()
    try:
        if text.endswith("B"):
            return int(float(text[:-1]) * 1_000_000_000)
        if text.endswith("M"):
            return int(float(text[:-1]) * 1_000_000)
    except ValueError:
        return 0
    return 0


class PlacementGateSession:
    def __init__(self, providers: ProviderSet | None = None) -> None:
        self._p = providers if providers is not None else resolve_providers()
        self.load = LoadReport(
            statuses={
                name: (Status.AVAILABLE if self._p.is_available(name) else Status.UNAVAILABLE)
                for name in self._p.resolutions
            },
            resolutions={n: r.as_dict() for n, r in self._p.resolutions.items()},
        )

    # -- capability gating -------------------------------------------------- #

    def _unavailable(self, *capabilities: str) -> dict[str, Any]:
        missing = [c for c in capabilities if not self._p.is_available(c)]
        details: list[str] = []
        for name in missing:  # one line per capability, deduped: three
            # missing surfaces usually share a single root cause and repeating
            # it three times buries the useful part of the message.
            reason = self._p.resolutions.get(name)
            if reason is not None and reason.detail not in details:
                details.append(reason.detail)
        return {
            "status": Status.UNAVAILABLE.value,
            "evaluated": False,
            "missing_capabilities": missing,
            "detail": "; ".join(details),
            "exit_code": EXIT_CAPABILITY_UNAVAILABLE,
        }

    def _server_unreachable(self, endpoint: str, detail: str) -> dict[str, Any]:
        return {
            "status": Status.SERVER_UNREACHABLE.value,
            "evaluated": False,
            "endpoint": endpoint,
            "detail": detail,
            "exit_code": EXIT_SERVER_UNAVAILABLE,
        }

    def _require(self, *capabilities: str) -> None:
        missing = [c for c in capabilities if not self._p.is_available(c)]
        if missing:
            raise CapabilityUnavailable(
                ", ".join(missing),
                "; ".join(
                    self._p.resolutions[c].detail for c in missing if c in self._p.resolutions
                ),
            )

    # -- hardware ----------------------------------------------------------- #

    def observe_hardware(self, *, nvidia_path: str | None = None) -> dict[str, Any]:
        """Observed hardware facts, as measured. Nothing is assumed."""
        try:
            self._require("hardware_observer", "hardware_memory", "hardware_nvidia")
        except CapabilityUnavailable as exc:
            return self._unavailable(*exc.capability.split(", "))

        memory_mod = self._p.module("hardware_memory")
        nvidia_mod = self._p.module("hardware_nvidia")
        observer_mod = self._p.module("hardware_observer")
        facts_mod = self._p.module("hardware_facts")

        executable = _resolve_nvidia(nvidia_path)
        if executable is None:
            return self._unavailable(
                "hardware_nvidia",
                "nvidia-smi was not found on PATH and no --nvidia path was given, "
                "so GPU capacity cannot be observed. Placement requires evidence, "
                "and an unobserved GPU is not evidence.",
            )
        observer = observer_mod.RealHardwareObserver(
            memory_probe=memory_mod.WindowsMemoryProbe(),
            nvidia_adapter=nvidia_mod.NvidiaQueryAdapter(executable),
        )
        profile = observer.observe()
        total_vram = profile.total_vram()
        free_vram = profile.free_vram()
        return {
            "status": Status.AVAILABLE.value,
            "evaluated": True,
            "detection_status": profile.detection_status.value,
            "is_contradictory": profile.is_contradictory,
            "is_coherent_unified": profile.is_coherent_unified,
            "has_discrete_gpu": profile.has_discrete_gpu,
            "gpu_absence_proven": profile.gpu_absence_proven,
            "memory": {
                "physical_total_bytes": profile.ram.physical_total_bytes,
                "physical_total_gib": round(profile.ram.physical_total_bytes / GIB, 2),
                "observed_available_bytes": profile.ram.observed_available_bytes,
                "observed_available_gib": round(
                    profile.ram.observed_available_bytes / GIB, 2
                ),
                "detection_status": profile.ram.detection_status.value,
            },
            "gpu": [
                {
                    "kind": a.kind.value,
                    "name": a.name,
                    "detection_status": a.detection_status.value,
                }
                for a in profile.adapters
            ],
            "total_vram_bytes": total_vram,
            "total_vram_gib": round(total_vram / GIB, 2) if total_vram else None,
            "free_vram_bytes": free_vram,
            "free_vram_gib": round(free_vram / GIB, 2) if free_vram else None,
            "unified_capacity_bytes": profile.unified_capacity(),
            "profile_version": profile.profile_version,
            "profile_sha256": _sha256_hex(profile.canonical_bytes()),
            "note": (
                "contradictory or unknown evidence must fail closed; it is never "
                "rounded into a capacity"
            ),
            "_profile": profile,
            "_DetectionStatus": facts_mod.DetectionStatus,
        }

    # -- census ------------------------------------------------------------- #

    def census(self, *, endpoint: tuple[str, int] = ("127.0.0.1", 11434)) -> dict[str, Any]:
        """Local model inventory, restricted to what a placement decision uses."""
        try:
            self._require("chat_adapter")
        except CapabilityUnavailable as exc:
            return self._unavailable(*exc.capability.split(", "))

        chat = self._p.module("chat_adapter")
        adapter = chat.OllamaChatAdapter(endpoint=endpoint)
        try:
            payload = adapter._request("GET", "/api/tags")
        except Exception as exc:  # noqa: BLE001 - transport, not policy
            return self._server_unreachable(
                f"{endpoint[0]}:{endpoint[1]}", f"{type(exc).__name__}: {exc}"
            )
        running = set()
        try:
            ps = adapter._request("GET", "/api/ps")
            running = {
                m.get("name") for m in (ps.get("models") or []) if isinstance(m, dict)
            }
        except Exception:  # noqa: BLE001 - residency is additive, not required
            running = set()

        models = []
        for entry in payload.get("models") or []:
            if not isinstance(entry, dict):
                continue
            details = entry.get("details") or {}
            facts = ModelFacts(
                tag=str(entry.get("name") or entry.get("model") or ""),
                digest=str(entry.get("digest") or ""),
                size_bytes=int(entry.get("size") or 0),
                parameter_size=str(details.get("parameter_size") or ""),
                context_length=int(details.get("context_length") or 0),
                family=str(details.get("family") or ""),
                quantization=str(details.get("quantization_level") or ""),
            )
            row = facts.as_dict()
            row["resident"] = facts.tag in running
            models.append(row)
        models.sort(key=lambda r: r["size_bytes"])
        return {
            "status": Status.AVAILABLE.value,
            "evaluated": True,
            "endpoint": f"{endpoint[0]}:{endpoint[1]}",
            "model_count": len(models),
            "resident_count": sum(1 for m in models if m["resident"]),
            "models": models,
            "note": (
                "local inventory only. Nothing is fetched from a remote, and no "
                "model contents are read"
            ),
        }

    def model_facts(
        self, tag: str, *, endpoint: tuple[str, int] = ("127.0.0.1", 11434)
    ) -> tuple[ModelFacts | None, dict[str, Any] | None]:
        listing = self.census(endpoint=endpoint)
        if not listing.get("evaluated"):
            return None, listing

        def to_facts(row: dict[str, Any]) -> ModelFacts:
            return ModelFacts(
                tag=row["tag"],
                digest=row["digest"],
                size_bytes=row["size_bytes"],
                parameter_size=row["parameter_size"],
                context_length=row["context_length"],
                family=row["family"],
                quantization=row["quantization"],
            )

        # Exact tag match only. A family name is NOT a model identity: several
        # distinct specimens share one, and picking one of them silently would
        # be a placement decision made from a guess.
        for row in listing["models"]:
            if row["tag"] == tag:
                return to_facts(row), None

        # A bare family name is allowed only when it is unambiguous, and the
        # ambiguity is reported rather than resolved by preference.
        family = tag.split(":", 1)[0]
        same_family = [r for r in listing["models"] if r["tag"].split(":", 1)[0] == family]
        if len(same_family) == 1:
            return to_facts(same_family[0]), None
        if len(same_family) > 1:
            return None, {
                "status": "AMBIGUOUS",
                "evaluated": True,
                "detail": (
                    f"{family!r} names {len(same_family)} distinct models. A model "
                    "name is not an identity: pass an exact tag."
                ),
                "candidates": [r["tag"] for r in same_family],
                "exit_code": EXIT_OK,
            }
        return None, {
            "status": "NOT_FOUND",
            "evaluated": True,
            "detail": f"no model tagged {tag!r} on the local service",
            "known_tags": [m["tag"] for m in listing["models"]],
            "exit_code": EXIT_OK,
        }

    # -- planning ----------------------------------------------------------- #

    def build_model_profile(self, facts: ModelFacts, *, context_tokens: int) -> Any:
        """A ModelProfile built only from observed facts, with its source declared."""
        profile_mod = self._p.module("model_profile")
        return profile_mod.ModelProfile(
            digest=facts.digest,
            architecture=facts.family,
            context_limit=context_tokens,
            kv_bytes_per_token=131072,
            bytes_per_weight_milli=2,
            quantization_level=facts.quantization,
            ollama_reported_size_bytes=facts.size_bytes,
            source=profile_mod.ProfileSource.OPERATOR_SUPPLIED,
            provenance="declared by the pgate operator from observed service facts",
        )

    def plan(
        self,
        facts: ModelFacts,
        *,
        placement: str = "DEVICE_ONLY",
        num_gpu: int = 99,
        context_tokens: int | None = None,
        hardware: dict[str, Any] | None = None,
        hardware_override: Any = None,
        endpoint: tuple[str, int] = ("127.0.0.1", 11434),
        policy_version: str = "pgate-v1",
    ) -> dict[str, Any]:
        """Compute a placement plan. Loads nothing. May legitimately refuse."""
        try:
            self._require("placement_planner", "model_profile", "hardware_observer",
                          "hardware_memory", "hardware_nvidia")
        except CapabilityUnavailable as exc:
            return self._unavailable(*exc.capability.split(", "))

        planner = self._p.module("placement_planner")
        if hardware is None:
            observed = self.observe_hardware()
            if not observed.get("evaluated"):
                return observed
            hardware = observed
        profile_hw = hardware_override if hardware_override is not None else hardware["_profile"]

        ctx = context_tokens if context_tokens is not None else min(
            facts.context_length or 8192, 8192
        )
        model_profile = self.build_model_profile(facts, context_tokens=ctx)
        policy = planner.PlacementPlanningPolicy(
            placement=planner.PlacementPolicy(placement),
            requested_num_gpu=num_gpu,
            policy_version=policy_version,
        )
        try:
            plan = planner.plan_inference_placement(
                model_profile,
                profile_hw,
                parameter_count=facts.parameter_count,
                requested_context_tokens=ctx,
                policy=policy,
                model_id=facts.tag,
                model_manifest_digest=facts.digest,
            )
        except Exception as exc:  # noqa: BLE001 - planner refusal, not a crash
            return {
                "status": Status.AVAILABLE.value,
                "evaluated": True,
                "admitted": False,
                "reason": "PLANNER_REFUSED",
                "detail": f"{type(exc).__name__}: {exc}",
                "exit_code": EXIT_OK,
            }
        return {
            "status": Status.AVAILABLE.value,
            "evaluated": True,
            "model": facts.as_dict(),
            "context_tokens": ctx,
            "requested": {"placement": placement, "num_gpu": num_gpu},
            "hardware": {
                k: hardware.get(k)
                for k in (
                    "detection_status", "is_contradictory", "total_vram_gib",
                    "free_vram_gib", "profile_sha256",
                )
            },
            "plan": _plan_view(plan),
            "admitted": plan.admitted,
            "reason": plan.reason.value,
            "detail": plan.detail,
            "exit_code": EXIT_OK,
            "_plan": plan,
            "_model_profile": model_profile,
            "_hardware": profile_hw,
        }

    # -- placement ---------------------------------------------------------- #

    def place(
        self,
        planned: dict[str, Any],
        *,
        endpoint: tuple[str, int] = ("127.0.0.1", 11434),
        keep_alive: str = "30s",
        unload_after: bool = False,
    ) -> dict[str, Any]:
        """Execute Controller.place_and_load_model().

        Mutates residency on the local service. The caller owns cleanup: this
        method never unloads implicitly.
        """
        if not planned.get("evaluated"):
            return planned
        if not planned.get("admitted"):
            return {
                "status": Status.AVAILABLE.value,
                "evaluated": True,
                "load_attempted": False,
                "admitted": False,
                "verdict": planned.get("reason"),
                "detail": planned.get("detail"),
                "note": "refused at plan stage; no request was issued",
                "exit_code": EXIT_OK,
            }
        try:
            self._require("controller", "ledger", "chat_adapter", "backends_base")
        except CapabilityUnavailable as exc:
            return self._unavailable(*exc.capability.split(", "))

        controller_mod = self._p.module("controller")
        ledger_mod = self._p.module("ledger")
        chat = self._p.module("chat_adapter")
        send_error = self._p.module("backends_base").ChatSendError

        adapter = chat.OllamaChatAdapter(endpoint=endpoint)
        controller = controller_mod.Controller(
            ledger_mod.Ledger(Path(tempfile.mkdtemp(prefix="pgate_")) / "placement.db")
        )
        controller.set_backend(adapter)

        facts = planned["model"]
        session = controller.create_session(actor="pgate")
        identity = controller.bind_identity(
            session_id=session,
            model_requested=facts["tag"],
            model_digest=facts["digest"],
            parameter_count=ModelFacts(
                tag=facts["tag"], digest=facts["digest"], size_bytes=facts["size_bytes"],
                parameter_size=facts["parameter_size"],
                context_length=facts["context_length"], family=facts["family"],
                quantization=facts["quantization"],
            ).parameter_count,
            placement="GPU_ELIGIBLE",
            resource="NORMAL",
            backend_id="pgate",
            bootstrap_content="pgate",
        )
        identity_id = getattr(identity, "identity_id", identity)

        outcome: dict[str, Any] = {
            "status": Status.AVAILABLE.value,
            "evaluated": True,
            "load_attempted": True,
            "model": facts,
            "plan": planned["plan"],
            "keep_alive": keep_alive,
        }
        try:
            from datetime import datetime, timezone

            decision = controller.place_and_load_model(
                planned["_plan"],
                planned["_hardware"],
                planned["_model_profile"],
                session_id=session,
                identity_id=identity_id,
                occurred_at_utc=datetime.now(timezone.utc).isoformat(),
                keep_alive=keep_alive,
            )
        except send_error as exc:
            outcome.update(
                admitted=None,
                verified=False,
                verdict="VERIFICATION_FAILED",
                detail=str(exc),
                note=(
                    "the model loaded but the observed placement did not satisfy "
                    "the request. This is a governed refusal, not a crash. The "
                    "reservation is retained by design; unload explicitly."
                ),
            )
        except Exception as exc:  # noqa: BLE001 - transport or upstream fault
            outcome.update(
                admitted=None,
                verified=False,
                verdict="ERROR",
                detail=f"{type(exc).__name__}: {exc}",
            )
        else:
            view = decision.model_dump()
            outcome.update(
                admitted=view["admitted"],
                verified=True,
                verdict=_enum_str(view["reason"]),
                requested_topology=_enum_str(view["requested_topology"]),
                reservation_vram_bytes=view["reservation_vram_bytes"],
                total_required_bytes=view["total_required_bytes"],
                model_digest=view["model_digest"],
                plan_hash_bound=planned["plan"]["content_hash"] in view["detail"],
            )

        observation = self.observe_residency(endpoint)
        outcome["residency"] = observation
        if unload_after:
            outcome["unload"] = self.unload(facts["tag"], endpoint=endpoint)
            outcome["residency_after_unload"] = self.observe_residency(endpoint)
        else:
            outcome["unload"] = {
                "performed": False,
                "note": (
                    "not unloaded. Place Gate never unloads implicitly; pass "
                    "--unload-after to do it, or unload the model yourself"
                ),
            }
        outcome["exit_code"] = EXIT_OK
        return outcome

    def observe_residency(
        self, endpoint: tuple[str, int] = ("127.0.0.1", 11434)
    ) -> dict[str, Any]:
        """Post-load observation: what the service actually reports as resident."""
        if not self._p.is_available("chat_adapter"):
            return self._unavailable("chat_adapter")
        chat = self._p.module("chat_adapter")
        adapter = chat.OllamaChatAdapter(endpoint=endpoint)
        try:
            payload = adapter._request("GET", "/api/ps")
        except Exception as exc:  # noqa: BLE001
            return self._server_unreachable(
                f"{endpoint[0]}:{endpoint[1]}", f"{type(exc).__name__}: {exc}"
            )
        entries = []
        for m in payload.get("models") or []:
            if not isinstance(m, dict):
                continue
            size = int(m.get("size") or 0)
            vram = int(m.get("size_vram") or 0)
            entries.append(
                {
                    "tag": m.get("name"),
                    "digest": m.get("digest"),
                    "size_bytes": size,
                    "size_gib": round(size / GIB, 2),
                    "size_vram_bytes": vram,
                    "size_vram_gib": round(vram / GIB, 2),
                    "context_length": m.get("context_length"),
                    "fully_gpu_resident": bool(vram and size and vram >= size),
                }
            )
        return {
            "status": Status.AVAILABLE.value,
            "evaluated": True,
            "resident_count": len(entries),
            "entries": entries,
            "note": (
                "partial GPU residency is reported as partial. It is never "
                "rounded up to satisfy a full GPU-residency request."
            ),
        }

    def unload(
        self, tag: str, *, endpoint: tuple[str, int] = ("127.0.0.1", 11434)
    ) -> dict[str, Any]:
        """Explicit residency release. Never called implicitly.

        Uses POST /api/chat with keep_alive=0. The upstream chat adapter admits
        exactly three operations -- GET /api/tags, GET /api/ps, POST /api/chat
        -- and correctly refuses anything else, including POST /api/generate.
        An empty message list with keep_alive=0 releases the model without
        generating, and stays inside the admitted surface. This tool does not
        widen the adapter's allowlist to release memory.
        """
        if not self._p.is_available("chat_adapter"):
            return self._unavailable("chat_adapter")
        chat = self._p.module("chat_adapter")
        adapter = chat.OllamaChatAdapter(endpoint=endpoint)
        try:
            payload = adapter._request(
                "POST",
                "/api/chat",
                {"model": tag, "messages": [], "stream": False, "keep_alive": 0},
            )
        except Exception as exc:  # noqa: BLE001
            return self._server_unreachable(
                f"{endpoint[0]}:{endpoint[1]}", f"{type(exc).__name__}: {exc}"
            )
        return {
            "status": Status.AVAILABLE.value,
            "evaluated": True,
            "performed": True,
            "tag": tag,
            "done_reason": payload.get("done_reason"),
            "note": (
                "residency release was requested explicitly, via the adapter's "
                "admitted POST /api/chat with keep_alive=0"
            ),
        }

    # -- characteristics ---------------------------------------------------- #

    def characteristics(self, registry_root: str | None) -> dict[str, Any]:
        """Read the upstream characteristics registry, if the operator points at it.

        Deliberately optional. Placement does not depend on it: a placement
        decision is made from measured hardware and observed model facts, and a
        registry is a record of what is already known about a specimen.
        """
        if not registry_root:
            return {
                "status": "NOT_CONFIGURED",
                "evaluated": True,
                "detail": (
                    "no registry root supplied. Pass --registry-root to consult "
                    "the Model Characteristics Registry. Placement does not "
                    "require it."
                ),
            }
        try:
            self._require("controller")  # the registry reader ships in that package
            snapshot = importlib_import(
                "ollama_controller.registries.model_characteristics.snapshot"
            )
        except Exception as exc:  # noqa: BLE001
            return self._unavailable("controller")
        try:
            state = snapshot.LocalSnapshotStore(registry_root).load_state()
        except Exception as exc:  # noqa: BLE001 - registry refuses its own state
            return {
                "status": "REGISTRY_UNREADABLE",
                "evaluated": True,
                "detail": f"{type(exc).__name__}: {exc}",
                "note": (
                    "the registry rejected its own effective state, so nothing is "
                    "reported from it. Observed service facts are unaffected."
                ),
            }
        return {
            "status": Status.AVAILABLE.value,
            "evaluated": True,
            "registry_revision": state.get("registry_revision"),
            "registry_digest": state.get("registry_digest"),
            "specimen_count": len(state.get("specimens") or []),
            "family_count": len(state.get("families") or []),
            "serving_binding_count": len(state.get("serving_bindings") or []),
            "observation_count": len(state.get("observation_ledger") or []),
        }


def importlib_import(name: str) -> Any:
    import importlib

    return importlib.import_module(name)


def _resolve_nvidia(explicit: str | None) -> str | None:
    """Locate nvidia-smi without shipping this machine's layout.

    An explicit --nvidia path wins. Otherwise ask PATH, and report absence
    rather than substituting a plausible guess: placement needs evidence.
    """
    import shutil

    if explicit:
        return explicit if Path(explicit).is_file() else None
    return shutil.which("nvidia-smi")


def _plan_view(plan: Any) -> dict[str, Any]:
    """A JSON-safe view of an upstream ModelAdmissionPlan. No policy is applied."""
    def maybe(value: Any) -> Any:
        return getattr(value, "value", value)

    total = plan.total_required_bytes
    budget = plan.budget_vram_bytes
    return {
        "admitted": plan.admitted,
        "reason": maybe(plan.reason),
        "requested_topology": maybe(plan.requested_topology),
        "selected_policy": maybe(plan.selected_policy),
        "requested_num_gpu": plan.requested_num_gpu,
        "verification_state": maybe(plan.verification_state),
        "recalculation_boundary": maybe(plan.recalculation_boundary),
        "policy_version": plan.policy_version,
        "total_required_bytes": total,
        "total_required_gib": round(total / GIB, 2) if total is not None else None,
        "budget_ram_bytes": plan.budget_ram_bytes,
        "budget_vram_bytes": budget,
        "budget_vram_gib": round(budget / GIB, 2) if budget is not None else None,
        "content_hash": plan.content_hash(),
        "note": (
            "a plan is a request. It grants no execution authority and is not "
            "evidence that anything is loaded."
        ),
    }


def _enum_str(value: Any) -> str:
    """Render an enum member as its value, not as ClassName.MEMBER."""
    return str(getattr(value, "value", value))


def _sha256_hex(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def to_json(payload: Any) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if not k.startswith("_")}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value

    return json.dumps(clean(payload), indent=2, sort_keys=True, default=str)
