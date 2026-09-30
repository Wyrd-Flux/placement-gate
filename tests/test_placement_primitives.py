"""Tests for wyrd-placement-core.

``test_placement_seam.py`` is the upstream method-level placement suite, carried
across with only its import paths and its session-plumbing helper adapted. Those
ten tests are the behavioural contract for the seam.

This file covers the extracted surface that had no upstream test of its own:
hardware observation, the planner closure, the transport allowlist, and the
properties that make this a publishable dependency.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

import pgate_demo.placement
from pgate_demo.placement import topology
from pgate_demo.placement.backends import backends, chat
from pgate_demo.placement.hardware import facts, observer
from pgate_demo.placement.ledger import ledger as ledger_mod
from pgate_demo.placement.policy import admission, inference_placement, model_profile

from ._fixtures import GIB, NOW, hw_discrete, hw_unknown, profile_large, ram_fact

PACKAGE_DIR = Path(pgate_demo.placement.__file__).parent


def _plan(profile=None, hardware=None, **overrides):
    """A DEVICE_ONLY plan for a model that fits, unless overridden."""
    kwargs = dict(
        parameter_count=8_000_000_000,
        requested_context_tokens=8192,
        policy=inference_placement.PlacementPlanningPolicy(
            placement=inference_placement.PlacementPolicy.DEVICE_ONLY,
            requested_num_gpu=99,
        ),
        model_id="demo:8b",
        model_manifest_digest="a" * 64,
    )
    kwargs.update(overrides)
    return inference_placement.plan_inference_placement(
        profile if profile is not None else profile_large(observed=6 * GIB),
        hardware if hardware is not None else hw_discrete(
            gpu_kwargs={"vram_total": 8 * GIB, "free": 7 * GIB, "foreign": GIB}
        ),
        **kwargs,
    )


def _evidence(plan, size=6 * GIB, size_vram=6 * GIB, gpu_status="OK", ram_status="OK",
              attribution=topology.AttributionStatus.ATTRIBUTED):
    return topology.TopologyEvidence(
        requested_topology=plan.requested_topology,
        ollama_reported=topology.ResidencyReport(
            schema_supported=True, size_bytes=size, size_vram_bytes=size_vram,
        ),
        host_gpu=topology.HostGpuEvidence(
            attributable_model_vram_bytes=size_vram, detection_status=gpu_status,
        ),
        host_ram=topology.HostRamEvidence(
            attributable_model_ram_bytes=max(size - size_vram, 0),
            detection_status=ram_status,
        ),
        attribution=attribution,
    )


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #


def test_a_coherent_profile_is_not_contradictory() -> None:
    profile = hw_discrete(gpu_kwargs={"vram_total": 8 * GIB, "free": 6 * GIB, "foreign": 2 * GIB})
    assert profile.is_contradictory is False


def test_a_discrete_and_an_integrated_class_at_once_is_contradictory() -> None:
    """Conflicting adapter classes stay visible and fail closed.

    Contradiction is about *class* disagreement, not arithmetic: the profile must
    not silently normalise two different adapter classes into one.
    """
    from pgate_demo.placement.hardware.facts import (
        AdapterKind, GpuAdapterFact, HardwareProfile,
    )

    def gpu(kind, name):
        return GpuAdapterFact(
            kind=kind, vendor="test", name=name,
            vram_total_bytes=8 * GIB, observed_free_vram_bytes=6 * GIB,
            foreign_usage_bytes=2 * GIB, detection_status=facts.DetectionStatus.OK,
            probe_source="fixture", probe_tool_version="fixture",
        )

    both = HardwareProfile(
        ram=ram_fact(),
        adapters=(gpu(AdapterKind.DISCRETE, "discrete"),
                  gpu(AdapterKind.COHERENT_UNIFIED, "unified")),
        observed_at_utc=NOW,
    )
    assert both.is_contradictory is True


def test_unknown_detection_status_is_not_ok() -> None:
    assert facts.DetectionStatus.OK.value == "OK"
    assert facts.DetectionStatus.UNKNOWN.value != "OK"


def test_the_observer_uses_only_the_probes_it_was_given() -> None:
    """A probe it was not given must not be silently consulted."""
    calls: list[str] = []

    class Memory:
        def observe(self):
            calls.append("memory")
            return ram_fact()

    class Nvidia:
        def probe(self):
            calls.append("nvidia")
            return []

    real = observer.RealHardwareObserver(memory_probe=Memory(), nvidia_adapter=Nvidia())
    real.observe()
    assert calls == ["memory", "nvidia"]


def test_a_machine_with_no_gpu_probes_still_produces_a_profile() -> None:
    class Memory:
        def observe(self):
            return ram_fact()

    profile = observer.RealHardwareObserver(
        memory_probe=Memory(), nvidia_adapter=None
    ).observe()
    assert profile.adapters == (), "absence is recorded, not invented"
    assert profile.ram is not None


def test_presence_is_never_suppressed_by_absence_evidence() -> None:
    """Both facts are recorded; the profile then fails closed.

    A discrete probe finding a GPU and a separate source reporting none must not
    resolve to "there is no GPU" by precedence. That is how a real machine gets
    planned against zero capacity.
    """

    class Memory:
        def observe(self):
            return ram_fact()

    class Nvidia:
        def probe(self):
            return [
                facts.GpuAdapterFact(
                    kind=facts.AdapterKind.DISCRETE, vendor="NVIDIA", name="Fake",
                    vram_total_bytes=8 * GIB, observed_free_vram_bytes=6 * GIB,
                    foreign_usage_bytes=2 * GIB,
                    detection_status=facts.DetectionStatus.OK,
                    probe_source="fixture", probe_tool_version="fixture",
                )
            ]

    profile = observer.RealHardwareObserver(
        memory_probe=Memory(),
        nvidia_adapter=Nvidia(),
        gpu_absent_source="pci-scan",
    ).observe()
    assert len(profile.adapters) == 2, "both the presence and the absence are kept"
    assert profile.is_contradictory is True


# --------------------------------------------------------------------------- #
# planning
# --------------------------------------------------------------------------- #


def test_a_model_that_fits_is_admitted_with_a_reason() -> None:
    plan = _plan()
    assert plan.admitted is True
    assert plan.reason is inference_placement.PlacementReason.PLANNED


def test_an_oversized_model_is_refused_with_a_budget_reason() -> None:
    plan = _plan(profile=profile_large(observed=40 * GIB), parameter_count=40_000_000_000)
    assert plan.admitted is False
    assert plan.reason is inference_placement.PlacementReason.EXCEEDS_RESOURCE_BUDGET


def test_unknown_hardware_never_produces_an_admission() -> None:
    """Fail closed. This is the one that must not be optimisable away."""
    plan = _plan(hardware=hw_unknown())
    assert plan.admitted is False
    assert plan.reason is inference_placement.PlacementReason.HARDWARE_UNKNOWN


def test_class_contradictory_hardware_never_produces_an_admission() -> None:
    """A discrete and a coherent-unified adapter at once must fail closed."""
    from pgate_demo.placement.hardware.facts import (
        AdapterKind, GpuAdapterFact, HardwareProfile,
    )

    def gpu(kind, name):
        return GpuAdapterFact(
            kind=kind, vendor="test", name=name,
            vram_total_bytes=8 * GIB, observed_free_vram_bytes=7 * GIB,
            foreign_usage_bytes=GIB, detection_status=facts.DetectionStatus.OK,
            probe_source="fixture", probe_tool_version="fixture",
        )

    contradictory = HardwareProfile(
        ram=ram_fact(),
        adapters=(gpu(AdapterKind.DISCRETE, "discrete"),
                  gpu(AdapterKind.COHERENT_UNIFIED, "unified")),
        observed_at_utc=NOW,
    )
    plan = _plan(hardware=contradictory)
    assert plan.admitted is False, (
        "two conflicting adapter classes must not resolve to the larger one"
    )


def test_a_plan_is_content_addressed_and_deterministic() -> None:
    assert _plan().content_hash() == _plan().content_hash()


def test_a_different_context_window_is_a_different_plan() -> None:
    """Otherwise the plan hash would not identify the request."""
    assert (
        _plan(requested_context_tokens=8192).content_hash()
        != _plan(requested_context_tokens=32768).content_hash()
    )


def test_a_different_digest_is_a_different_plan() -> None:
    assert (
        _plan(model_manifest_digest="a" * 64).content_hash()
        != _plan(model_manifest_digest="c" * 64).content_hash()
    )


def test_partial_residency_does_not_verify_as_full_residency() -> None:
    """The live finding this whole library exists to preserve."""
    plan = _plan()
    full = inference_placement.verify_placement(
        evidence=_evidence(plan), plan=plan
    )
    partial = inference_placement.verify_placement(
        evidence=_evidence(plan, size=6 * GIB, size_vram=GIB), plan=plan
    )
    assert full is inference_placement.PlacementVerificationState.VERIFIED
    assert partial is not inference_placement.PlacementVerificationState.VERIFIED, (
        "1 GiB of 6 GiB on the GPU is not full GPU residency"
    )


def test_unattributed_evidence_does_not_verify() -> None:
    """Missing evidence must be UNCONFIRMED, never a pass."""
    plan = _plan()
    result = inference_placement.verify_placement(
        evidence=_evidence(
            plan, gpu_status="UNKNOWN", ram_status="UNKNOWN",
            size_vram=0,
            attribution=topology.AttributionStatus.UNATTRIBUTED,
        ),
        plan=plan,
    )
    assert result is not inference_placement.PlacementVerificationState.VERIFIED, (
        "evidence that cannot be attributed must never verify, and must not "
        "silently become a pass"
    )
    assert result is not inference_placement.PlacementVerificationState.PARTIALLY_VERIFIED


def test_an_admitted_plan_is_not_a_verification() -> None:
    """A plan grants nothing. It must not read as evidence anything happened."""
    plan = _plan()
    assert plan.admitted is True
    assert plan.verification_state is not (
        inference_placement.PlacementVerificationState.VERIFIED
    )


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #


def test_the_adapter_admits_exactly_three_operations() -> None:
    """A closed allowlist. Anything wider is a widening nobody chose."""
    source = Path(chat.__file__).read_text(encoding="utf8")
    admitted = {
        f"{method} {path}"
        for method, path in re.findall(r'\("(GET|POST)",\s*"([^"]+)"', source)
    }
    assert admitted == {"GET /api/tags", "GET /api/ps", "POST /api/chat"}, admitted


def test_generation_is_not_reachable_through_this_adapter() -> None:
    """Placement is an observation. No path here may produce tokens."""
    source = Path(chat.__file__).read_text(encoding="utf8")
    assert "/api/generate" not in source
    for generator in ("def generate(", "def stream_generate("):
        assert generator not in source, f"{generator} must not exist here"


def test_an_operation_outside_the_allowlist_is_refused() -> None:
    adapter = chat.OllamaChatAdapter(endpoint=("127.0.0.1", 1))
    with pytest.raises(Exception) as exc:
        adapter._request("POST", "/api/generate", {"model": "x", "prompt": "y"})
    message = str(exc.value).lower()
    assert "generate" in message or "not admitted" in message or "refus" in message


def test_an_unreachable_service_raises_rather_than_returning_empty() -> None:
    """An unreachable service must not look like an empty inventory."""
    adapter = chat.OllamaChatAdapter(endpoint=("127.0.0.1", 1), timeout=2.0)
    with pytest.raises(backends.ChatSendError):
        adapter._request("GET", "/api/ps")


def test_a_chat_profile_passes_num_gpu_and_keep_alive_through() -> None:
    profile = chat.OllamaChatProfile(
        model="demo:8b", num_gpu=99, num_ctx=8192, keep_alive="30s",
    )
    options = profile.options()
    assert options["num_gpu"] == 99
    assert options["num_ctx"] == 8192
    assert profile.keep_alive == "30s"


def test_cpu_only_is_requested_explicitly() -> None:
    """Silence would let the service choose, which is not the same as CPU."""
    profile = chat.OllamaChatProfile(
        model="demo:8b", num_gpu=0, num_ctx=8192, keep_alive="5m",
        require_cpu_only=True,
    )
    assert profile.options()["num_gpu"] == 0


# --------------------------------------------------------------------------- #
# ledger
# --------------------------------------------------------------------------- #


def test_the_ledger_records_an_admission_durably(tmp_path: Path) -> None:
    led = ledger_mod.Ledger(tmp_path / "l.db")
    decision = inference_placement.decision_from_placement_plan(
        _plan(), decision_id="d" * 32, session_id="s", identity_id="i",
        occurred_at_utc=NOW,
    )
    led.append_admission(decision)
    assert led.get_admission(decision.decision_id) is not None


def test_content_is_addressed_not_inlined(tmp_path: Path) -> None:
    led = ledger_mod.Ledger(tmp_path / "l.db")
    digest = led.store_content_bytes(b"the bytes")
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert led.get_content(digest) == b"the bytes"


def test_an_append_only_table_rejects_an_update(tmp_path: Path) -> None:
    """The ledger is append-only, and the DATABASE enforces it, not just code."""
    led = ledger_mod.Ledger(tmp_path / "l.db")
    decision = inference_placement.decision_from_placement_plan(
        _plan(), decision_id="d" * 32, session_id="s", identity_id="i",
        occurred_at_utc=NOW,
    )
    led.append_admission(decision)
    with pytest.raises(Exception):
        led.connect().execute("UPDATE admission_record SET decision_id='tampered'")


def test_the_ledger_schema_holds_only_placement_tables(tmp_path: Path) -> None:
    """Checkpoint, memory-lane and authorization machinery stayed internal.

    Those tables were declared by the internal ledger but nothing in it ever read
    or wrote them, so removing them is a subtraction rather than a redesign.
    """
    led = ledger_mod.Ledger(tmp_path / "l.db")
    tables = {
        row[0] for row in led.connect().execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    tables.discard("sqlite_sequence")
    assert tables == {
        "content", "event", "global_event", "backend_observation",
        "hardware_profile_record", "model_profile_record", "admission_record",
        "resource_reservation", "resource_reservation_event", "topology_attestation",
        "backend_residency", "backend_residency_event", "identity_admission",
        "execution_attempt",
    }, tables


# --------------------------------------------------------------------------- #
# public-package properties
# --------------------------------------------------------------------------- #


def test_the_only_third_party_dependency_is_pydantic() -> None:
    """A placement library must not drag a UI toolkit in with it.

    Run in a subprocess so another package installed in the same environment
    cannot leak into the measurement.
    """
    result = subprocess.run(
        [sys.executable, "-c",
         "import json, sys, pgate_demo.placement;"
         "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    allowed = {"pydantic", "pydantic_core", "annotated_types", "typing_extensions"}
    foreign = [
        name for name in json.loads(result.stdout)
        if "." not in name
        and name not in allowed
        and name not in sys.stdlib_module_names
        and not name.startswith("_")
        and not name.startswith("wyrd")
        and not name.startswith("pgate_demo")
    ]
    assert foreign == [], foreign


def test_no_filesystem_paths_from_the_extraction_estate() -> None:
    """No source tree of the extraction environment may travel with the code."""
    drive_or_home = re.compile(r"[A-Za-z]:[\\/]|/Users/|/home/[a-z]|\\\\[A-Za-z0-9_.-]+\\")
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            if drive_or_home.search(line):
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{lineno}")
    assert offenders == [], offenders


def test_the_package_carries_no_estate_specific_module() -> None:
    """The 5,500-line controller, the TUI and the runtime service stayed behind.

    The seam is one method out of that controller; the rest of it -- memory
    lanes, checkpoint generations, authorization chains -- remains internal.
    """
    files = sorted(p.name for p in PACKAGE_DIR.rglob("*.py"))
    assert "controller.py" not in files, "the full controller stayed internal"
    for unwanted in ("app.py", "checkpoint_operator.py", "runtime_service.py"):
        assert unwanted not in files
    assert not [f for f in files if "memory_lane" in f or "checkpoint" in f]


def test_the_placement_seam_is_small() -> None:
    """One method, plus reconciliation. Not a controller wearing a hat."""
    import ast

    tree = ast.parse(
        (PACKAGE_DIR / "controller" / "placement.py").read_text(encoding="utf8")
    )
    runner = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PlacementRunner"
    )
    methods = [i.name for i in runner.body
               if isinstance(i, ast.FunctionDef)]
    assert methods == ["__init__", "set_backend", "place_and_load_model",
                       "_fail_and_reconcile"], methods


def test_no_intra_package_import_reaches_outside_the_package() -> None:
    """A relative import that climbs above pgate_demo.placement would break."""
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        depth = len(path.relative_to(PACKAGE_DIR).parts) - 1
        # from a module d directories below the package root, a legal relative
        # import climbs at most d+1 levels (…/backends/backends.py may use "..")
        for lineno, line in enumerate(path.read_text(encoding="utf8").splitlines(), 1):
            dots = re.match(r"\s*from (\.+)", line)
            if dots and len(dots.group(1)) > depth + 1:
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{lineno}: {line.strip()}")
    assert offenders == [], offenders


def test_version_is_exposed() -> None:
    from pgate_demo import PGATE_VERSION

    assert PGATE_VERSION == "0.3.0"


def test_the_public_entry_points_are_importable() -> None:
    from pgate_demo.placement.controller.placement import PlacementRunner
    from pgate_demo.placement.ledger.ledger import Ledger
    from pgate_demo.placement.policy.inference_placement import (
        decision_from_placement_plan,
        plan_inference_placement,
        verify_placement,
    )

    for entry in (plan_inference_placement, verify_placement,
                  decision_from_placement_plan, PlacementRunner, Ledger):
        assert callable(entry)


def test_admission_reasons_are_a_named_vocabulary() -> None:
    """Refusals must carry a stable reason, not a boolean."""
    reasons = {r.value for r in admission.AdmitReason}
    assert "ADMITTED" in reasons
    assert "HARDWARE_UNKNOWN" in reasons
    assert "EXCEEDS_RAM_BUDGET" in reasons


def test_placement_reasons_include_the_budget_and_hardware_refusals() -> None:
    reasons = {r.value for r in inference_placement.PlacementReason}
    assert "EXCEEDS_RESOURCE_BUDGET" in reasons
    assert "HARDWARE_UNKNOWN" in reasons
    assert "PLANNED" in reasons
