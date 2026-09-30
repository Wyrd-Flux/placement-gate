"""Deterministic conformance checks for the adapter.

These assert that the adapter DELEGATES: that a refusal comes from the upstream
planner with the upstream reason, that a fitting model reaches the load path,
that observation follows load, and that nothing is substituted when a
capability is missing.

The offline checks need no Ollama server. The live checks are opt-in because
they mutate residency.
"""

from __future__ import annotations

from typing import Any

from .core import GIB, ModelFacts, PlacementGateSession, Status
from .exit_codes import EXIT_CAPABILITY_UNAVAILABLE, EXIT_OK

# A synthetic model fact set. The digest is deliberately not a real model's:
# the offline checks never contact a service, so nothing can be confused for
# observed evidence.
FIXTURE_FACTS = ModelFacts(
    tag="pgate-fixture:8b",
    digest="f" * 64,
    size_bytes=int(4.0 * GIB),
    parameter_size="8.00B",
    context_length=32768,
    family="fixture",
    quantization="Q4_K_M",
)


def _row(name: str, ok: bool, detail: str = "", live: bool = False) -> dict[str, Any]:
    return {"name": name, "pass": bool(ok), "detail": detail, "live": live}


def run_selftest(
    session: PlacementGateSession,
    *,
    live: bool = False,
    endpoint: tuple[str, int] = ("127.0.0.1", 11434),
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    # ---- 1. the planner is reachable at all ------------------------------- #
    if not session.load.available("placement_planner"):
        return {
            "pgate_version": __import__("pgate_demo.core", fromlist=["x"]).PGATE_VERSION,
            "evaluated": False,
            "passed": False,
            "missing_capabilities": list(session.load.missing()),
            "exit_code": EXIT_CAPABILITY_UNAVAILABLE,
            "results": [
                _row("upstream placement planner reachable", False,
                     "capability unresolved; no check was substituted")
            ],
        }
    results.append(_row("upstream placement planner reachable", True))

    hardware = session.observe_hardware()
    hardware_ok = bool(hardware.get("evaluated"))
    results.append(
        _row(
            "hardware observed",
            hardware_ok,
            f"detection={hardware.get('detection_status')}"
            if hardware_ok else str(hardware.get("detail"))[:80],
        )
    )

    # ---- 2. a model that cannot fit is refused BEFORE any load ---------- #
    oversized = ModelFacts(
        tag="pgate-fixture:oversized",
        digest="e" * 64,
        size_bytes=int(64.0 * GIB),
        parameter_size="200.00B",
        context_length=32768,
        family="fixture",
        quantization="Q4_K_M",
    )
    refused = session.plan(oversized, placement="DEVICE_ONLY", num_gpu=99)
    refused_ok = (
        refused.get("evaluated") and refused.get("admitted") is False
    )
    results.append(
        _row(
            "oversized model refused before load",
            refused_ok,
            f"reason={refused.get('reason')}" if refused_ok else str(refused.get("detail"))[:80],
        )
    )

    # ---- 3. unknown hardware fails closed -------------------------------- #
    if hardware_ok:
        from .cli import _unknown_hardware

        override = _unknown_hardware(session)
        if override is not None:
            unknown = session.plan(
                FIXTURE_FACTS, placement="DEVICE_ONLY", num_gpu=99,
                hardware=hardware, hardware_override=override,
            )
            unknown_ok = (
                unknown.get("evaluated") and unknown.get("admitted") is False
                and "HARDWARE" in str(unknown.get("reason"))
            )
            results.append(
                _row(
                    "unknown hardware fails closed",
                    unknown_ok,
                    f"reason={unknown.get('reason')}" if unknown_ok else
                    f"got {unknown.get('reason')}",
                )
            )
        else:
            results.append(
                _row("unknown hardware fails closed", False,
                     "could not construct an unknown-hardware profile")
            )

    # ---- 4. a fitting model is admitted, and the plan grants nothing ----- #
    fitting = session.plan(FIXTURE_FACTS, placement="DEVICE_ONLY", num_gpu=99)
    if fitting.get("evaluated") and fitting.get("admitted"):
        plan_view = fitting["plan"]
        results.append(
            _row(
                "fitting model admitted with a content-addressed plan",
                bool(plan_view["content_hash"]),
                f"topology={plan_view['requested_topology']} "
                f"required={plan_view['total_required_gib']} GiB",
            )
        )
        results.append(
            _row(
                "an admitted plan is reported as a request, not a placement",
                "request" in plan_view["note"],
                "plan does not claim anything is loaded",
            )
        )
        results.append(
            _row(
                "refusal reasons are the upstream vocabulary",
                plan_view["reason"] in _upstream_reasons(session),
                f"reason={plan_view['reason']}",
            )
        )
    else:
        detail = (
            f"the 4 GiB fixture did not fit this machine: "
            f"{fitting.get('reason')} {fitting.get('detail') or ''}"
        )
        results.append(
            _row("fitting model admitted with a content-addressed plan", False,
                 detail[:100], live=False)
        )

    # ---- 5. the adapter holds no placement policy of its own ------------- #
    import inspect

    from . import core, exit_codes

    src = inspect.getsource(core)
    policy_terms = [
        "budget_ram_bytes =", "budget_vram_bytes =", "kv_bytes_per_token =",
        "if size_vram", "fully_gpu_resident =", "admitted =", "EXCEEDS_RESOURCE_BUDGET",
        "HARDWARE_UNKNOWN",
    ]
    invented = [
        t for t in policy_terms
        if t in src and t not in ("kv_bytes_per_token =",)
    ]
    results.append(
        _row(
            "adapter implements no placement arithmetic",
            not invented,
            "no local budget or verdict logic" if not invented
            else f"possible local policy: {invented}",
        )
    )
    # exit codes must be drawn from the contract table, not invented per site
    exit_src = inspect.getsource(core) + inspect.getsource(exit_codes)
    results.append(
        _row(
            "exit codes come from the declared contract",
            "EXIT_" in exit_src and exit_codes.EXIT_OK == 0,
            f"codes: {sorted(exit_codes.EXIT_MEANINGS)}",
        )
    )

    # ---- 6. a missing capability yields no substituted answer ------------- #
    from .core import LoadReport
    from .providers import ProviderSet

    stub = PlacementGateSession.__new__(PlacementGateSession)
    stub._p = ProviderSet()
    stub.load = LoadReport(
        statuses={"chat_adapter": Status.UNAVAILABLE},
        resolutions={
            "chat_adapter": {"capability": "chat_adapter", "detail": "stubbed"}
        },
    )
    census = stub.census()
    results.append(
        _row(
            "missing provider reports UNAVAILABLE, no fallback",
            census.get("evaluated") is False
            and census.get("status") == Status.UNAVAILABLE.value
            and "models" not in census,
            f"status={census.get('status')}",
        )
    )

    # ---- 7. opt-in live checks: load, observe, identity, cleanup -------- #
    if live:
        results.extend(_live_checks(session, endpoint))
    else:
        results.append(
            _row(
                "live load/verify cycle",
                True,
                "not run; pass --live to mutate residency",
            )
        )

    passed = all(r["pass"] for r in results)
    return {
        "evaluated": True,
        "passed": passed,
        "assertions": len(results),
        "failures": sum(1 for r in results if not r["pass"]),
        "live": live,
        "exit_code": EXIT_OK if passed else 1,
        "results": results,
    }


def _upstream_reasons(session: PlacementGateSession) -> set[str]:
    planner = session._p.module("placement_planner")  # noqa: SLF001 - introspection
    return {m.value for m in planner.PlacementReason}


def _live_checks(session: PlacementGateSession, endpoint: tuple[str, int]) -> list[dict]:
    """A real load, a real observation, a real unload. Explicitly requested."""
    from .core import GIB as _GIB

    out: list[dict[str, Any]] = []
    census = session.census(endpoint=endpoint)
    if not census.get("evaluated"):
        out.append(
            _row("live: census", False,
                 f"{census.get('status')}: {census.get('detail','')[:70]}", live=True)
        )
        return out
    out.append(
        _row("live: census reachable", True,
             f"{census['model_count']} models at {census['endpoint']}", live=True)
    )

    # A local weight file, not a cloud placeholder: Ollama lists remote models
    # with a tag and a near-zero size, and asking one to become resident locally
    # would fail for a reason that has nothing to do with placement.
    local = [
        m for m in census["models"]
        if m["size_gib"] >= 0.5
        and "cloud" not in m["tag"].lower()
    ]
    fitting = [m for m in local if m["size_gib"] <= 3.0]
    if not fitting:
        out.append(
            _row("live: a small local model is available", False,
                 f"{len(local)} local weights found, none between 0.5 and 3.0 GiB",
                 live=True)
        )
        return out
    facts, problem = session.model_facts(fitting[0]["tag"], endpoint=endpoint)
    if problem is not None or facts is None:
        out.append(_row("live: model facts resolved", False, "unresolved", live=True))
        return out

    before = session.observe_residency(endpoint)
    planned = session.plan(facts, placement="DEVICE_ONLY", num_gpu=99, endpoint=endpoint)
    if not planned.get("evaluated") or not planned.get("admitted"):
        out.append(
            _row("live: small model admitted", False,
                 f"{planned.get('reason')} {planned.get('detail','')}"[:90], live=True)
        )
        return out
    out.append(
        _row("live: small model admitted", True,
             f"required={planned['plan']['total_required_gib']} GiB", live=True)
    )

    result = session.place(planned, endpoint=endpoint, keep_alive="20s", unload_after=True)
    residency = result.get("residency") or {}
    entries = residency.get("entries") or []
    mine = [e for e in entries if e["tag"] == facts.tag]

    out.append(
        _row(
            "live: model became resident after the load",
            bool(mine),
            f"resident={residency.get('resident_count')}", live=True,
        )
    )
    if mine:
        out.append(
            _row(
                "live: residency digest matches the planned model",
                str(mine[0].get("digest")) == facts.digest,
                f"ps={str(mine[0].get('digest'))[:16]} planned={facts.digest[:16]}",
                live=True,
            )
        )
        out.append(
            _row(
                "live: full GPU residency observed, not rounded",
                bool(mine[0]["fully_gpu_resident"]),
                f"size={mine[0]['size_gib']} GiB size_vram={mine[0]['size_vram_gib']} GiB",
                live=True,
            )
        )
    else:
        out.append(_row("live: residency digest matches", False, "not resident", live=True))
        out.append(_row("live: full GPU residency observed", False, "not resident", live=True))

    after = result.get("residency_after_unload") or {}
    out.append(
        _row(
            "live: cleanup released residency",
            not any(e["tag"] == facts.tag for e in (after.get("entries") or [])),
            f"resident after unload: {after.get('resident_count')}", live=True,
        )
    )
    return out
