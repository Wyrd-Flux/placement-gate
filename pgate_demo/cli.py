"""Command-line interface for Placement Gate.

    pgate doctor                 what upstream capabilities resolved, and how
    pgate hardware               observed hardware facts
    pgate census                the local service's models, minimally described
    pgate plan <model>           compute a plan without loading
    pgate place <model>          execute Controller.place_and_load_model()
    pgate selftest               deterministic checks; --live adds a real load
    pgate exit-codes             print the contract this CLI implements

`place` and `selftest --live` mutate Ollama residency. Nothing is ever unloaded
implicitly; pass --unload-after to ask for it.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from .core import PGATE_VERSION, PlacementGateSession, Status, to_json
from .exit_codes import (
    EXIT_ASSERTION_FAILED,
    EXIT_BAD_INVOCATION,
    EXIT_CAPABILITY_UNAVAILABLE,
    EXIT_OK,
    EXIT_SERVER_UNAVAILABLE,
    render_contracts,
)
from .selftest import run_selftest

DEFAULT_ENDPOINT = ("127.0.0.1", 11434)


def _endpoint(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError(
            f"endpoint must look like host:port, got {text!r}"
        )
    return host, int(port)


def _emit(payload: Any, as_text: bool, text: str | None = None) -> None:
    if as_text and text is not None:
        print(text)
    else:
        print(to_json(payload))


def _exit_for(payload: dict[str, Any]) -> int:
    """Execution status only. A domain verdict never moves the exit code."""
    if payload.get("evaluated") is False:
        return int(payload.get("exit_code") or EXIT_CAPABILITY_UNAVAILABLE)
    status = payload.get("status")
    if status == Status.UNAVAILABLE.value:
        return EXIT_CAPABILITY_UNAVAILABLE
    if status == Status.SERVER_UNREACHABLE.value:
        return EXIT_SERVER_UNAVAILABLE
    return EXIT_OK


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def _cmd_doctor(session: PlacementGateSession, args: argparse.Namespace) -> int:
    report = session.load
    payload = report.as_dict()
    payload["evaluated"] = True
    payload["exit_code"] = EXIT_OK if not report.missing() else EXIT_CAPABILITY_UNAVAILABLE
    _emit(payload, args.text, report.render())
    return int(payload["exit_code"])


def _cmd_hardware(session: PlacementGateSession, args: argparse.Namespace) -> int:
    result = session.observe_hardware(nvidia_path=args.nvidia)
    if not result.get("evaluated"):
        _emit(result, args.text, _unavailable_text(result))
        return _exit_for(result)
    lines = [
        f"detection status : {result['detection_status']}"
        f"   contradictory={result['is_contradictory']}",
        f"memory           : {result['memory']['physical_total_gib']} GiB total,"
        f" {result['memory']['observed_available_gib']} GiB available",
    ]
    for gpu in result["gpu"]:
        lines.append(
            f"gpu              : {gpu['kind']} {gpu['name']} ({gpu['detection_status']})"
        )
    lines.append(
        f"vram             : {result['total_vram_gib']} GiB total,"
        f" {result['free_vram_gib']} GiB free"
    )
    lines.append(f"profile sha256   : {result['profile_sha256']}")
    lines.append(f"note             : {result['note']}")
    _emit(result, args.text, "\n".join(lines))
    return _exit_for(result)


def _cmd_census(session: PlacementGateSession, args: argparse.Namespace) -> int:
    result = session.census(endpoint=args.endpoint)
    if getattr(args, "registry_root", None):
        # A registry that refuses its own state is reported, never repaired and
        # never bypassed. Census stays useful without it.
        result["characteristics_registry"] = session.characteristics(args.registry_root)
    if not result.get("evaluated"):
        _emit(result, args.text, _unavailable_text(result))
        return _exit_for(result)
    if args.json:
        _emit(result, args.text)
        return _exit_for(result)
    lines = [
        f"endpoint   : {result['endpoint']}",
        f"models     : {result['model_count']}  resident: {result['resident_count']}",
        "",
        f"{'TAG':<30s} {'SIZE GiB':>9s} {'CTX':>8s} {'PARAMS':>10s}  RES",
    ]
    for row in result["models"]:
        lines.append(
            f"{row['tag']:<30s} {row['size_gib']:>9.2f} {row['context_length']:>8d}"
            f" {row['parameter_size']:>10s}  {'yes' if row['resident'] else '-'}"
        )
    lines.append("")
    lines.append(f"note       : {result['note']}")
    lines += _registry_text(result.get("characteristics_registry"))
    _emit(result, args.text, "\n".join(lines))
    return _exit_for(result)


def _registry_text(report: Any) -> list[str]:
    if not report:
        return []
    if report.get("status") == Status.AVAILABLE.value:
        return [
            "characteristics registry",
            f"  revision     : {report.get('registry_revision')}",
            f"  families     : {report.get('family_count')}",
            f"  observations : {report.get('observation_count')}",
            "",
        ]
    return [
        "characteristics registry",
        f"  UNREADABLE   : {report.get('detail')}",
        "  effect       : optional. Placement uses measured hardware, not a "
        "record of it.",
        "",
    ]


def _cmd_plan(session: PlacementGateSession, args: argparse.Namespace) -> int:
    facts, problem = _resolve_facts(session, args)
    if problem is not None:
        _emit(problem, args.text, _unavailable_text(problem))
        return _exit_for(problem)
    hardware_override = None
    if args.unknown_hardware:
        hardware_override = _unknown_hardware(session)
        if hardware_override is None:
            payload = {
                "status": Status.UNAVAILABLE.value,
                "evaluated": False,
                "detail": "cannot construct an unknown-hardware profile",
                "exit_code": EXIT_CAPABILITY_UNAVAILABLE,
            }
            _emit(payload, args.text, _unavailable_text(payload))
            return _exit_for(payload)
    planned = session.plan(
        facts,
        placement=args.placement,
        num_gpu=args.num_gpu,
        context_tokens=args.context,
        hardware_override=hardware_override,
        endpoint=args.endpoint,
    )
    if not planned.get("evaluated"):
        _emit(planned, args.text, _unavailable_text(planned))
        return _exit_for(planned)

    plan = planned["plan"]
    lines = [
        f"model      : {facts.tag}  ({planned['model']['size_gib']} GiB, "
        f"digest {facts.digest[:16]})",
        f"hardware   : detection={planned['hardware']['detection_status']}"
        f"  vram_total={planned['hardware']['total_vram_gib']} GiB"
        f"  free={planned['hardware']['free_vram_gib']} GiB",
        f"requested  : {planned['requested']['placement']} num_gpu="
        f"{planned['requested']['num_gpu']}  ctx={planned['context_tokens']}",
        "",
        f"ADMITTED   : {planned['admitted']}",
        f"REASON     : {planned['reason']}",
        f"TOPOLOGY   : {plan['requested_topology']}",
        f"REQUIRED   : {plan['total_required_gib']} GiB"
        f"   budget_vram: {plan['budget_vram_gib']} GiB",
        f"STATE      : {plan['verification_state']}"
        f"   recalc at: {plan['recalculation_boundary']}",
        f"PLAN HASH  : {plan['content_hash']}",
    ]
    if planned.get("detail"):
        lines.append(f"DETAIL     : {planned['detail']}")
    lines.append(f"note       : {plan['note']}")
    _emit(planned, args.text, "\n".join(lines))
    return _exit_for(planned)


def _cmd_place(session: PlacementGateSession, args: argparse.Namespace) -> int:
    facts, problem = _resolve_facts(session, args)
    if problem is not None:
        _emit(problem, args.text, _unavailable_text(problem))
        return _exit_for(problem)
    planned = session.plan(
        facts,
        placement=args.placement,
        num_gpu=args.num_gpu,
        context_tokens=args.context,
        endpoint=args.endpoint,
    )
    if not planned.get("evaluated"):
        _emit(planned, args.text, _unavailable_text(planned))
        return _exit_for(planned)
    if not args.execute_anyway and not planned["admitted"]:
        payload = {
            "status": planned["status"],
            "evaluated": True,
            "load_attempted": False,
            "admitted": False,
            "verdict": planned["reason"],
            "detail": planned.get("detail"),
            "note": "refused at plan stage; no request was issued, nothing loaded",
            "exit_code": EXIT_OK,
        }
        lines = [
            f"model      : {facts.tag}",
            f"ADMITTED   : False",
            f"REASON     : {planned['reason']}",
            f"DETAIL     : {planned.get('detail') or '-'}",
            "LOAD       : not attempted (refused before any request)",
            f"note       : {payload['note']}",
        ]
        _emit(payload, args.text, "\n".join(lines))
        return EXIT_OK

    result = session.place(
        planned, endpoint=args.endpoint, keep_alive=args.keep_alive,
        unload_after=args.unload_after,
    )
    if not result.get("evaluated"):
        _emit(result, args.text, _unavailable_text(result))
        return _exit_for(result)

    lines = [f"model      : {facts.tag}  digest {facts.digest[:16]}"]
    lines.append(f"unload-after: {args.unload_after}")
    if result.get("admitted") is False or result.get("admitted") is None:
        lines.append(f"VERDICT    : {result.get('verdict')}")
        if result.get("requested_topology"):
            lines.append(f"TOPOLOGY   : {result['requested_topology']}")
        if result.get("detail"):
            lines.append(f"DETAIL     : {result['detail']}")
    else:
        lines.append(f"VERDICT    : {result.get('verdict')}")
        lines.append(f"ADMITTED   : {result.get('admitted')}")
        lines.append(f"RESERVED   : {result.get('reservation_vram_bytes', 0)/1024**3:.2f} GiB VRAM")
        lines.append(f"PLAN BOUND : {result.get('plan_hash_bound')}")
    residency = result.get("residency") or {}
    for entry in residency.get("entries", []) or []:
        lines.append(
            f"RESIDENCY  : {entry['tag']}  size={entry['size_gib']} GiB"
            f"  size_vram={entry['size_vram_gib']} GiB"
            f"  fully_gpu={entry['fully_gpu_resident']}"
        )
        if entry["tag"] == facts.tag:
            lines.append(
                f"IDENTITY   : digest match="
                f"{str(entry.get('digest')) == facts.digest}  "
                f"tag match={entry['tag'] == facts.tag}"
            )
    if not residency.get("entries"):
        lines.append("RESIDENCY  : nothing resident")
    if residency.get("note"):
        lines.append(f"note       : {residency['note']}")
    unload = result.get("unload") or {}
    if unload.get("performed"):
        after = result.get("residency_after_unload") or {}
        still = [e["tag"] for e in (after.get("entries") or []) if e["tag"] == facts.tag]
        lines.append(
            f"UNLOADED   : {facts.tag} still resident: {bool(still)}"
            f"  (service resident count now {after.get('resident_count', '?')})"
        )
    elif unload.get("status") in (
        Status.UNAVAILABLE.value, Status.SERVER_UNREACHABLE.value
    ):
        lines.append(f"UNLOAD     : FAILED -- {unload.get('detail','')}")
    else:
        lines.append(f"UNLOAD     : not performed. {unload.get('note','')}")
    _emit(result, args.text, "\n".join(lines))
    return _exit_for(result)


def _cmd_selftest(session: PlacementGateSession, args: argparse.Namespace) -> int:
    result = run_selftest(session, live=args.live, endpoint=args.endpoint)
    lines = []
    for row in result["results"]:
        mark = "PASS" if row["pass"] else "FAIL"
        scope = "live" if row.get("live") else "offline"
        lines.append(f"  [{mark}] {row['name']:38s} {scope:6s} {row.get('detail','')}")
    lines.append(
        f"  selftest {'passed' if result['passed'] else 'FAILED'}"
        + ("" if result["evaluated"] else " (upstream capability unavailable)")
    )
    _emit(result, args.text, "\n".join(lines))
    if not result["evaluated"]:
        return EXIT_CAPABILITY_UNAVAILABLE
    return EXIT_OK if result["passed"] else EXIT_ASSERTION_FAILED


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _unavailable_text(payload: dict[str, Any]) -> str:
    status = payload.get("status", "UNAVAILABLE")
    lines = [f"status     : {status}"]
    if payload.get("missing_capabilities"):
        lines.append(f"missing    : {', '.join(payload['missing_capabilities'])}")
        lines.append("no result was substituted for the missing capability")
    if payload.get("candidates"):
        lines.append(f"candidates : {', '.join(payload['candidates'][:8])}")
    if payload.get("known_tags"):
        lines.append(f"known tags : {len(payload['known_tags'])} on this service")
    if payload.get("endpoint"):
        lines.append(f"endpoint   : {payload['endpoint']}")
    if payload.get("detail"):
        lines.append(f"detail     : {payload['detail']}")
    return "\n".join(lines)


def _resolve_facts(session: PlacementGateSession, args: argparse.Namespace):
    size_bytes = getattr(args, "size_bytes", None)
    parameters = getattr(args, "parameters", None)
    if size_bytes and parameters:
        from .core import ModelFacts

        return (
            ModelFacts(
                tag=args.model,
                digest="0" * 64,
                size_bytes=size_bytes,
                parameter_size=f"{parameters / 1e9:.2f}B",
                context_length=args.context or 8192,
                family="",
                quantization="",
            ),
            None,
        )
    if not session.load.available("chat_adapter"):
        return None, {
            "status": Status.UNAVAILABLE.value,
            "evaluated": False,
            "missing_capabilities": ["chat_adapter"],
            "detail": session.load.resolutions.get("chat_adapter", {}).get("detail", ""),
            "exit_code": EXIT_CAPABILITY_UNAVAILABLE,
        }
    facts, problem = session.model_facts(args.model, endpoint=args.endpoint)
    if problem is not None:
        if problem.get("status") == Status.SERVER_UNREACHABLE.value:
            return None, problem
        return None, {
            "status": "NOT_FOUND",
            "evaluated": True,
            "detail": problem.get("detail"),
            "known_tags": problem.get("known_tags", []),
            "exit_code": EXIT_OK,
        }
    return facts, None


def _absent_probe() -> str:
    """An absolute path to a binary that does not exist.

    NvidiaQueryAdapter requires an absolute path and turns a missing one into
    a ValueError rather than an UNKNOWN fact, so the absent-evidence case must
    be a real absolute path pointing at nothing. Derived from the running
    interpreter so it is correct on any platform.
    """
    import pathlib
    import sys

    root = pathlib.Path(sys.prefix)
    return str(root / "pgate_absent_probe" / "nvidia-smi")


def _unknown_hardware(session: PlacementGateSession) -> Any:
    """A profile whose evidence is absent, built through the real observer."""
    facts = session.load
    if not all(
        facts.available(c)
        for c in ("hardware_observer", "hardware_memory", "hardware_nvidia", "hardware_facts")
    ):
        return None
    from .providers import resolve_providers

    p = resolve_providers()
    memory_mod = p.module("hardware_memory")
    nvidia_mod = p.module("hardware_nvidia")
    observer_mod = p.module("hardware_observer")
    facts_mod = p.module("hardware_facts")

    class _UnknownMemory:
        def observe(self):
            return facts_mod.MemoryFact(
                physical_total_bytes=0,
                observed_available_bytes=0,
                detection_status=facts_mod.DetectionStatus.UNKNOWN,
            )

    return observer_mod.RealHardwareObserver(
        memory_probe=_UnknownMemory(),
        nvidia_adapter=nvidia_mod.NvidiaQueryAdapter(_absent_probe()),
        gpu_absent_source="pgate_selftest_enumeration",
    ).observe()


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that exits with the documented bad-invocation code.

    argparse hardcodes exit 2, which would collide with
    EXIT_CAPABILITY_UNAVAILABLE and make a typo indistinguishable from a
    missing dependency.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        print(f"{self.prog}: error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_BAD_INVOCATION)


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="pgate",
        description=(
            "Placement Gate: inspect a machine's hardware budget, refuse a model "
            "that will not fit, load one that should, and verify that the "
            "expected model became resident under the expected identity."
        ),
        epilog=(
            "This tool delegates every decision to the upstream Ollama_Controller "
            "implementation. It contains no placement policy of its own."
        ),
    )
    parser.add_argument("--version", action="version", version=PGATE_VERSION)
    parser.add_argument("--text", action="store_true", help="human-readable output")
    sub = parser.add_subparsers(
        dest="command", required=True, parser_class=_Parser
    )

    sub.add_parser("doctor", help="report which upstream capabilities resolved")
    sub.add_parser("exit-codes", help="print the exit-code contract")

    hw = sub.add_parser("hardware", help="observed hardware facts")
    hw.add_argument("--nvidia", default=None, help="explicit nvidia-smi path")

    census = sub.add_parser("census", help="models visible to the local service")
    census.add_argument("--endpoint", type=_endpoint, default=DEFAULT_ENDPOINT)
    census.add_argument("--json", action="store_true", help="full JSON listing")
    census.add_argument("--registry-root", default=None,
                        help="optional characteristics registry to consult")

    for name, helptext in (("plan", "compute a plan without loading"),
                           ("place", "execute the placement path")):
        cmd = sub.add_parser(name, help=helptext)
        cmd.add_argument("model")
        cmd.add_argument("--endpoint", type=_endpoint, default=DEFAULT_ENDPOINT)
        cmd.add_argument(
            "--placement", default="DEVICE_ONLY",
            help="DEVICE_ONLY, HOST_ONLY, HYBRID_STATIC or MANAGED_FALLBACK",
        )
        cmd.add_argument(
            "--num-gpu", type=int, default=99,
            help="Ollama num_gpu. Note Ollama reads small values as a LAYER "
                 "count, not a device count; 99 requests full offload",
        )
        cmd.add_argument("--context", type=int, default=None)
        cmd.add_argument("--registry-root", default=None)
        if name == "plan":
            cmd.add_argument(
                "--size-bytes", type=int, default=None,
                help="supply the model size instead of reading the service",
            )
            cmd.add_argument(
                "--parameters", type=int, default=None,
                help="supply the parameter count instead of reading the service",
            )
            cmd.add_argument(
                "--unknown-hardware", action="store_true",
                help="plan against absent hardware evidence; must fail closed",
            )
        else:
            cmd.add_argument(
                "--keep-alive", default="30s",
                help="how long the service should hold the model after this call",
            )
            cmd.add_argument(
                "--unload-after", action="store_true",
                help="explicitly release residency afterwards. Never implicit.",
            )
            cmd.add_argument(
                "--execute-anyway", action="store_true",
                help="attempt the load even when the plan is refused",
            )

    st = sub.add_parser("selftest", help="deterministic checks against the bound surfaces")
    st.add_argument("--live", action="store_true", help="include a real load and unload")
    st.add_argument("--endpoint", type=_endpoint, default=DEFAULT_ENDPOINT)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "exit-codes":
        print(render_contracts())
        return EXIT_OK
    session = PlacementGateSession()
    handlers = {
        "doctor": _cmd_doctor,
        "hardware": _cmd_hardware,
        "census": _cmd_census,
        "plan": _cmd_plan,
        "place": _cmd_place,
        "selftest": _cmd_selftest,
    }
    handler = handlers[args.command]
    if args.command in ("plan", "place") and getattr(args, "registry_root", None):
        args.registry_report = session.characteristics(args.registry_root)
    return int(handler(session, args))


if __name__ == "__main__":
    raise SystemExit(main())
