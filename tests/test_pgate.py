"""Conformance tests for the Placement Gate adapter.

These assert that the adapter DELEGATES. The placement policy, the budget
arithmetic and the verification rules all live upstream; what is tested here is
that this package asks correctly, reports upstream verdicts verbatim, refuses to
substitute anything for a missing capability, and keeps process success separate
from domain refusal.

The upstream suites remain the authority on placement behaviour:
  Ollama_Controller/tests/test_inference_placement.py   the planning layer
  Ollama_Controller/tests/test_placement_runtime.py     the method itself
No upstream test logic is duplicated here.
"""

from __future__ import annotations

import inspect
import json

import pytest

from pgate_demo import PGATE_VERSION
from pgate_demo.cli import main
from pgate_demo.core import GIB, ModelFacts, PlacementGateSession, Status, to_json
from pgate_demo.exit_codes import (
    EXIT_BAD_INVOCATION,
    EXIT_CAPABILITY_UNAVAILABLE,
    EXIT_MEANINGS,
    EXIT_OK,
    EXIT_SERVER_UNAVAILABLE,
    render_contracts,
)
from pgate_demo.providers import (
    CapabilityUnavailable,
    ProviderSet,
    REQUIREMENTS,
    resolve_providers,
)
from pgate_demo.selftest import FIXTURE_FACTS, run_selftest

UPSTREAM = "ollama_controller"
TRIPLE = '"' * 3
NEWLINE = chr(10)


def upstream_available() -> bool:
    return resolve_providers(("placement_planner",)).is_available("placement_planner")


needs_upstream = pytest.mark.skipif(
    not upstream_available(),
    reason="the upstream ollama_controller package is not available",
)


@pytest.fixture(scope="module")
def session() -> PlacementGateSession:
    return PlacementGateSession()


def _requires(session: PlacementGateSession, *caps: str) -> None:
    missing = [c for c in caps if not session.load.available(c)]
    if missing:
        pytest.skip(f"upstream capability unavailable: {', '.join(missing)}")


def _stub_session(*unavailable: str) -> PlacementGateSession:
    """A session in which the named capabilities did not resolve."""
    from pgate_demo.core import LoadReport

    stub = PlacementGateSession.__new__(PlacementGateSession)
    stub._p = ProviderSet()
    report = LoadReport()
    for name in unavailable:
        report.statuses[name] = Status.UNAVAILABLE
        report.resolutions[name] = {
            "capability": name, "module": name, "source": "unresolved",
            "detail": "stubbed for test", "resolved": False, "missing_attributes": [],
        }
    stub.load = report
    return stub


def _code_only(source: str) -> str:
    """Strip docstrings and comments so policy checks inspect code, not prose."""
    without_docstrings = TRIPLE.join(source.split(TRIPLE)[0::2])
    return NEWLINE.join(
        line for line in without_docstrings.split(NEWLINE)
        if not line.strip().startswith("#")
    )


# --------------------------------------------------------------------------- #
# packaging
# --------------------------------------------------------------------------- #


def test_version_exported() -> None:
    assert PGATE_VERSION == "0.1.0"


def test_console_script_is_declared_and_dependencies_are_empty() -> None:
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["pgate"] == "pgate_demo.cli:main"
    assert data["project"]["dependencies"] == []


def test_shipped_provider_config_has_no_paths() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    cfg = json.loads(
        (root / "pgate_demo" / "pgate.providers.json").read_text(encoding="utf-8")
    )
    assert cfg["search_paths"] == [], "no machine-specific path may be shipped"


# --------------------------------------------------------------------------- #
# lazy provider resolution
# --------------------------------------------------------------------------- #


def test_every_capability_names_an_upstream_module() -> None:
    for name, (_env, module, required) in REQUIREMENTS.items():
        assert module.startswith(UPSTREAM + "."), f"{name} does not name upstream"
        assert required, f"{name} declares no required attribute"


def test_import_binds_nothing() -> None:
    """Importing the adapter must not import the upstream package."""
    import subprocess
    import sys

    code = (
        "import sys, pgate_demo, pgate_demo.providers;"
        "print([m for m in sys.modules if m.startswith('ollama_controller')])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"eager import leaked: {out.stdout}"


def test_resolution_is_not_a_filesystem_probe() -> None:
    """The provider layer must never enumerate a tree."""
    from pgate_demo import providers

    code = _code_only(inspect.getsource(providers))
    for forbidden in (
        "os.walk", "pkgutil", "iter_modules", "glob.glob", "iglob",
        "rglob", "scandir", "listdir",
    ):
        assert forbidden not in code, f"provider layer enumerates via {forbidden}"


def test_only_requested_capabilities_are_resolved() -> None:
    got = resolve_providers(capabilities=("model_profile",))
    assert set(got.resolutions) == {"model_profile"}


def test_unresolved_capability_raises_rather_than_substituting() -> None:
    providers = ProviderSet()
    assert not providers.is_available("model_profile")
    with pytest.raises(CapabilityUnavailable):
        providers.require("model_profile")


def test_unrequested_capability_is_not_available() -> None:
    providers = resolve_providers(capabilities=("model_profile",))
    assert not providers.is_available("controller")


# --------------------------------------------------------------------------- #
# exit-code contract
# --------------------------------------------------------------------------- #


def test_contract_table_renders() -> None:
    text = render_contracts()
    for command in ("hardware", "census", "plan", "place", "selftest"):
        assert command in text
    assert "separate from domain verdict" in text


def test_every_documented_code_is_distinct() -> None:
    codes = list(EXIT_MEANINGS)
    assert len(codes) == len(set(codes))
    assert EXIT_OK == 0
    assert EXIT_BAD_INVOCATION == 64


@pytest.mark.parametrize("argv", [["nosuchcommand"], ["plan", "m", "--bogus"]])
def test_parser_reports_bad_invocation_as_64(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == EXIT_BAD_INVOCATION, argv


def test_bad_invocation_does_not_collide_with_capability_failure() -> None:
    """A typo must be distinguishable from a missing dependency."""
    with pytest.raises(SystemExit) as exc:
        main(["nosuchcommand"])
    assert exc.value.code == EXIT_BAD_INVOCATION
    assert exc.value.code != EXIT_CAPABILITY_UNAVAILABLE


# --------------------------------------------------------------------------- #
# a missing capability yields no substituted answer
# --------------------------------------------------------------------------- #


def test_census_without_adapter_reports_unavailable() -> None:
    stub = _stub_session("chat_adapter")
    result = stub.census()
    assert result["evaluated"] is False
    assert result["status"] == Status.UNAVAILABLE.value
    assert "models" not in result
    assert result["exit_code"] == EXIT_CAPABILITY_UNAVAILABLE


def test_hardware_without_observer_reports_unavailable() -> None:
    stub = _stub_session("hardware_observer", "hardware_memory", "hardware_nvidia")
    result = stub.observe_hardware()
    assert result["evaluated"] is False
    assert result["status"] == Status.UNAVAILABLE.value


@needs_upstream
def test_unreachable_service_is_distinct_from_a_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dead port is SERVER_UNREACHABLE, not UNAVAILABLE and not a refusal.

    The provider IS resolvable here; only the transport fails. Conflating the
    three would make a stopped service look like a missing dependency.
    """
    import ollama_controller.backends.chat as chat_mod

    class _Dead:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _request(self, method, path, body=None):
            raise OSError("connection refused")

    monkeypatch.setattr(chat_mod, "OllamaChatAdapter", _Dead)
    result = PlacementGateSession().census(endpoint=("127.0.0.1", 1))
    assert result["evaluated"] is False
    assert result["status"] == Status.SERVER_UNREACHABLE.value
    assert result["exit_code"] == EXIT_SERVER_UNAVAILABLE
    assert "models" not in result


# --------------------------------------------------------------------------- #
# the adapter holds no placement policy
# --------------------------------------------------------------------------- #


def test_no_local_budget_or_verdict_arithmetic() -> None:
    """Placement decisions must come from upstream, not from this package."""
    from pgate_demo import core

    code = _code_only(inspect.getsource(core))
    for invented in (
        "EXCEEDS_RESOURCE_BUDGET",
        "HARDWARE_UNKNOWN",
        "PLANNED",
        "GPU_RESIDENT",
        "PROHIBITID",
        "budget_vram =",
        "budget_ram =",
        "if vram",
        "if size_vram",
        "fully_gpu_resident =",
    ):
        assert invented not in code, f"adapter appears to implement policy: {invented}"


def test_adapter_delegates_to_the_upstream_planner() -> None:
    src = inspect.getsource(PlacementGateSession.plan)
    assert "plan_inference_placement" in src


def test_adapter_delegates_placement_to_the_controller() -> None:
    src = inspect.getsource(PlacementGateSession.place)
    assert "place_and_load_model" in src


def test_adapter_never_generates_tokens() -> None:
    """Placement is an observation. No send_chat anywhere in the adapter."""
    from pgate_demo import core

    code = _code_only(inspect.getsource(core))
    assert "send_chat" not in code
    assert "stream_chat" not in code


def test_adapter_does_not_widen_the_adapter_allowlist() -> None:
    """Only the three operations upstream already admits may be used."""
    from pgate_demo import core

    code = _code_only(inspect.getsource(core))
    for path in ("/api/tags", "/api/ps", "/api/chat"):
        assert path in code, f"expected upstream operation {path} in use"
    # POST /api/generate is NOT admitted upstream. Prose may name it; code may not.
    assert "/api/generate" not in code, "adapter used an unadmitted operation"


# --------------------------------------------------------------------------- #
# model identity
# --------------------------------------------------------------------------- #


def test_parameter_count_parsing_is_honest_about_failure() -> None:
    def facts(label: str) -> ModelFacts:
        return ModelFacts("t", "d" * 64, 1, label, 1, "", "")

    assert facts("8.00B").parameter_count == 8_000_000_000
    assert facts("873.44M").parameter_count == 873_440_000
    assert facts("unknown").parameter_count == 0
    assert facts("").parameter_count == 0


def _fake_census(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "status": Status.AVAILABLE.value, "evaluated": True,
        "model_count": len(rows), "resident_count": 0, "models": rows,
    }


def _row(tag: str, digest_char: str) -> dict[str, object]:
    return {
        "tag": tag, "digest": digest_char * 64, "size_bytes": 1, "size_gib": 0.0,
        "parameter_size": "1B", "context_length": 8, "family": tag.split(":")[0],
        "quantization": "Q4", "resident": False,
    }


def test_a_family_name_is_not_a_model_identity() -> None:
    """Ambiguous families must be reported, never resolved by preference."""
    session = PlacementGateSession()
    session.census = lambda **kw: _fake_census([_row("fam:a", "1"), _row("fam:b", "2")])
    facts, problem = session.model_facts("fam")
    assert facts is None
    assert problem["status"] == "AMBIGUOUS"
    assert set(problem["candidates"]) == {"fam:a", "fam:b"}


def test_exact_tag_resolves_without_fallback() -> None:
    session = PlacementGateSession()
    session.census = lambda **kw: _fake_census([_row("fam:a", "1"), _row("fam:b", "2")])
    facts, problem = session.model_facts("fam:b")
    assert problem is None
    assert facts is not None and facts.tag == "fam:b"
    assert facts.digest == "2" * 64


def test_unambiguous_family_resolves() -> None:
    session = PlacementGateSession()
    session.census = lambda **kw: _fake_census([_row("solo:q4", "1")])
    facts, problem = session.model_facts("solo")
    assert problem is None and facts is not None and facts.tag == "solo:q4"


def test_unknown_tag_reports_not_found() -> None:
    session = PlacementGateSession()
    session.census = lambda **kw: _fake_census([_row("fam:a", "1")])
    facts, problem = session.model_facts("absent:model")
    assert facts is None
    assert problem["status"] == "NOT_FOUND"
    assert problem["known_tags"] == ["fam:a"]


# --------------------------------------------------------------------------- #
# planning: refused before load, and a refusal is not a failure
# --------------------------------------------------------------------------- #


OVERSIZED = ModelFacts(
    tag="pgate-test:oversized", digest="e" * 64,
    size_bytes=int(64 * GIB), parameter_size="200.00B",
    context_length=32768, family="test", quantization="Q4_K_M",
)


@needs_upstream
def test_oversized_model_is_refused_before_any_load() -> None:
    planned = PlacementGateSession().plan(OVERSIZED, placement="DEVICE_ONLY", num_gpu=99)
    assert planned["evaluated"] is True
    assert planned["admitted"] is False
    assert planned["reason"] == "EXCEEDS_RESOURCE_BUDGET"
    assert planned["exit_code"] == EXIT_OK
    assert "load_attempted" not in planned, "planning must not report a load attempt"


@needs_upstream
def test_place_on_a_refused_plan_issues_no_request() -> None:
    session = PlacementGateSession()
    planned = session.plan(OVERSIZED, placement="DEVICE_ONLY", num_gpu=99)
    outcome = session.place(planned, unload_after=False)
    assert outcome["load_attempted"] is False
    assert outcome["admitted"] is False
    assert outcome["exit_code"] == EXIT_OK


@needs_upstream
def test_refusal_reasons_are_the_upstream_vocabulary() -> None:
    session = PlacementGateSession()
    planner = session._p.module("placement_planner")  # noqa: SLF001
    upstream = {m.value for m in planner.PlacementReason}
    planned = session.plan(OVERSIZED, placement="DEVICE_ONLY", num_gpu=99)
    assert planned["reason"] in upstream


@needs_upstream
def test_unknown_hardware_fails_closed() -> None:
    from pgate_demo.cli import _unknown_hardware

    session = PlacementGateSession()
    _requires(session, "hardware_observer", "hardware_memory", "hardware_nvidia")
    hardware = session.observe_hardware()
    override = _unknown_hardware(session)
    if override is None:
        pytest.skip("cannot construct an unknown-hardware profile here")
    planned = session.plan(
        FIXTURE_FACTS, placement="DEVICE_ONLY", num_gpu=99,
        hardware=hardware, hardware_override=override,
    )
    assert planned["evaluated"] is True
    assert planned["admitted"] is False
    assert planned["reason"] == "HARDWARE_UNKNOWN"


@needs_upstream
def test_an_admitted_plan_is_never_reported_as_a_placement() -> None:
    planned = PlacementGateSession().plan(FIXTURE_FACTS, placement="DEVICE_ONLY", num_gpu=99)
    if not planned.get("admitted"):
        pytest.skip("fixture did not fit this machine")
    note = planned["plan"]["note"]
    assert "request" in note and "loaded" in note
    assert "load_attempted" not in planned


# --------------------------------------------------------------------------- #
# the load path is actually reached, and observation follows it
# --------------------------------------------------------------------------- #


@needs_upstream
def test_load_is_attempted_and_observation_follows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert the ordering the architecture depends on.

    If residency were observed before the load, every verdict would be
    meaningless, and no amount of correct arithmetic would catch it.
    """
    from ollama_controller.backends.chat import OllamaChatAdapter
    from ollama_controller.controller.controller import Controller

    session = PlacementGateSession()
    planned = session.plan(FIXTURE_FACTS, placement="DEVICE_ONLY", num_gpu=99)
    if not planned.get("admitted"):
        pytest.skip("fixture did not fit this machine")

    order: list[str] = []

    class _FakeDecision:
        def model_dump(self):
            return {
                "admitted": True, "reason": "ADMITTED",
                "requested_topology": "GPU_RESIDENT",
                "reservation_vram_bytes": 1, "total_required_bytes": 1,
                "model_digest": FIXTURE_FACTS.digest,
                "detail": planned["plan"]["content_hash"],
            }

    def fake_place_and_load(self, *a, **kw):
        order.append("load")
        return _FakeDecision()

    class _FakeAdapter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def _request(self, method, path, body=None):
            order.append("read:" + path)
            if path == "/api/ps":
                return {"models": [{
                    "name": FIXTURE_FACTS.tag, "digest": FIXTURE_FACTS.digest,
                    "size": int(4 * GIB), "size_vram": int(4 * GIB),
                    "context_length": 8192,
                }]}
            return {"models": []}

    monkeypatch.setattr(Controller, "place_and_load_model", fake_place_and_load)
    monkeypatch.setattr(OllamaChatAdapter, "_request", _FakeAdapter()._request, raising=False)
    monkeypatch.setattr(
        "ollama_controller.backends.chat.OllamaChatAdapter", _FakeAdapter
    )

    outcome = session.place(planned, unload_after=False)
    assert outcome["load_attempted"] is True
    assert "load" in order, "the upstream placement method was never called"
    assert order.index("load") < order.index("read:/api/ps"), (
        "residency must be observed after the load, never before"
    )
    assert outcome["verified"] is True
    assert outcome["plan_hash_bound"] is True
    assert outcome["residency"]["entries"][0]["digest"] == FIXTURE_FACTS.digest


@needs_upstream
def test_partial_residency_is_not_rounded_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """size_vram > 0 is not GPU residency, and must not be reported as such.

    This is the live finding the demo exists to preserve: requesting full
    offload and getting one layer back is a refusal, not a success.
    """
    import ollama_controller.backends.chat as chat_mod

    cases = [
        # name, size, size_vram, expected fully_gpu_resident
        ("full offload", int(4 * GIB), int(4 * GIB), True),
        ("partial offload", int(4 * GIB), int(1 * GIB), False),
        ("no vram", int(4 * GIB), 0, False),
    ]
    for name, size, vram, expected in cases:
        class _Adapter:
            def __init__(self, **kwargs):
                pass

            def _request(self, method, path, body=None):
                if path == "/api/ps":
                    return {"models": [{
                        "name": FIXTURE_FACTS.tag, "digest": FIXTURE_FACTS.digest,
                        "size": size, "size_vram": vram, "context_length": 8192,
                    }]}
                return {"models": []}

        monkeypatch.setattr(chat_mod, "OllamaChatAdapter", _Adapter)
        observed = PlacementGateSession().observe_residency()
        assert observed["entries"][0]["fully_gpu_resident"] is expected, name
        assert observed["entries"][0]["size_vram_bytes"] == vram, name


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #


def test_selftest_reports_unavailable_without_upstream() -> None:
    stub = _stub_session("placement_planner", "hardware_observer")
    result = run_selftest(stub)
    assert result["evaluated"] is False
    assert result["passed"] is False
    assert result["exit_code"] == EXIT_CAPABILITY_UNAVAILABLE


@needs_upstream
def test_selftest_passes_offline() -> None:
    result = run_selftest(PlacementGateSession(), live=False)
    failures = [r["name"] for r in result["results"] if not r["pass"]]
    assert result["passed"], failures
    assert result["live"] is False


@needs_upstream
def test_selftest_does_not_mutate_residency_offline() -> None:
    session = PlacementGateSession()
    before = session.observe_residency()
    run_selftest(session, live=False)
    after = session.observe_residency()
    assert before.get("entries") == after.get("entries"), (
        "an offline selftest must not change residency"
    )


@needs_upstream
def test_selftest_cli_offline_exits_zero() -> None:
    assert main(["selftest"]) == EXIT_OK


@needs_upstream
def test_selftest_cli_needs_no_service() -> None:
    """The offline selftest must not require a reachable service."""
    assert main(["selftest", "--endpoint", "127.0.0.1:1"]) == EXIT_OK


# --------------------------------------------------------------------------- #
# output hygiene
# --------------------------------------------------------------------------- #


def test_json_output_drops_private_keys() -> None:
    payload = {"ok": True, "_plan": object(), "_hardware": object(), "nested": {"_x": 1}}
    assert json.loads(to_json(payload)) == {"ok": True, "nested": {}}


@needs_upstream
def test_hardware_report_carries_no_path_or_identity() -> None:
    hardware = PlacementGateSession().observe_hardware()
    assert hardware["evaluated"] is True
    blob = json.dumps(
        {k: v for k, v in hardware.items() if not k.startswith("_")}, default=str
    )
    assert "C:" not in blob and "session_id" not in blob and "identity_id" not in blob


def test_census_reports_only_placement_relevant_metadata() -> None:
    from pgate_demo import core

    code = _code_only(inspect.getsource(PlacementGateSession.census))
    for leaked in ("modified_at", "parent_model", "blob", "file", "path"):
        assert leaked not in code, f"census exposes {leaked}"
    assert "ModelFacts" in inspect.getsource(core)


# --------------------------------------------------------------------------- #
# no machine layout in the adapter
# --------------------------------------------------------------------------- #

def test_adapter_ships_no_hardcoded_executable_paths() -> None:
    """The adapter must discover nvidia-smi, not remember where it was found."""
    from pgate_demo import core

    code = _code_only(inspect.getsource(core._resolve_nvidia))
    assert "shutil.which" in code, "nvidia-smi is located on PATH"
    for machine_path in (r"C:\Windows", "/usr/bin/", "/usr/local/bin/", "System32"):
        assert machine_path not in code, f"adapter hardcodes {machine_path}"


def test_absent_nvidia_is_reported_not_guessed() -> None:
    """An unobserved GPU is not evidence, so absence must surface as UNAVAILABLE."""
    from pgate_demo import core

    assert core._resolve_nvidia(None) is not None, "nvidia-smi is present here"
    assert core._resolve_nvidia("/definitely/not/nvidia-smi") is None
    assert core._resolve_nvidia(r"C:\definitely\not\nvidia-smi.exe") is None


# --------------------------------------------------------------------------- #
# optional characteristics registry
# --------------------------------------------------------------------------- #

def test_census_consults_a_registry_only_when_asked() -> None:
    from pgate_demo import cli

    parser = cli.build_parser()
    assert parser.parse_args(["census"]).registry_root is None
    parsed = parser.parse_args(["census", "--registry-root", "/some/registry"])
    assert parsed.registry_root == "/some/registry"


def test_unreadable_registry_is_reported_not_repaired_or_hidden() -> None:
    """A registry that rejects its own state must not be bypassed or rewritten."""
    from pgate_demo import cli

    report = {
        "status": "UNAVAILABLE",
        "evaluated": False,
        "detail": "ModelCharacteristicsRegistryError: effective state digest "
                  "does not match registry_digest",
    }
    rendered = "\n".join(cli._registry_text(report))
    assert "UNREADABLE" in rendered
    assert "optional" in rendered
    assert cli._registry_text(None) == []
    # it must never claim to have read state it could not verify
    for word in ("families", "observations"):
        assert word not in rendered
