"""Lazy provider resolution for the upstream Placement Gate capabilities.

Two rules govern everything in this module.

1. **Named surfaces only.** Each capability names the exact upstream modules it
   needs. Nothing here globs, walks, or enumerates a source tree. Importing
   this module imports no upstream code at all.

2. **Resolution is a read, never a probe.** A capability that cannot be
   resolved is reported UNAVAILABLE. Veritas and Placement Gate never fall back
   to a local reimplementation, and never touch the filesystem to find out
   whether an import "might" work.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ENV_PREFIX = "PGATE_"

#: capability -> (env suffix, dotted module, required attributes)
REQUIREMENTS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "hardware_facts": (
        "HARDWARE_FACTS_PATH",
        "ollama_controller.hardware.facts",
        ("HardwareProfile", "MemoryFact", "DetectionStatus"),
    ),
    "hardware_observer": (
        "HARDWARE_OBSERVER_PATH",
        "ollama_controller.hardware.observer",
        ("RealHardwareObserver",),
    ),
    "hardware_memory": (
        "HARDWARE_MEMORY_PATH",
        "ollama_controller.hardware.windows_memory",
        ("WindowsMemoryProbe",),
    ),
    "hardware_nvidia": (
        "HARDWARE_NVIDIA_PATH",
        "ollama_controller.hardware.nvidia_query",
        ("NvidiaQueryAdapter",),
    ),
    "placement_planner": (
        "PLANNER_PATH",
        "ollama_controller.policy.inference_placement",
        (
            "PlacementPolicy",
            "PlacementPlanningPolicy",
            "PlacementReason",
            "ExecutionTopology",
            "PlacementVerificationState",
            "plan_inference_placement",
            "decision_from_placement_plan",
            "verify_placement",
        ),
    ),
    "model_profile": (
        "MODEL_PROFILE_PATH",
        "ollama_controller.policy.model_profile",
        ("ModelProfile", "ProfileSource"),
    ),
    "chat_adapter": (
        "CHAT_ADAPTER_PATH",
        "ollama_controller.backends.chat",
        ("OllamaChatAdapter", "OllamaChatProfile"),
    ),
    "backends_base": (
        "BACKENDS_BASE_PATH",
        "ollama_controller.backends.backends",
        ("ChatBackend", "ChatSendError"),
    ),
    "ledger": ("LEDGER_PATH", "ollama_controller.ledger.ledger", ("Ledger",)),
    "controller": (
        "CONTROLLER_PATH",
        "ollama_controller.controller.controller",
        ("Controller", "ControllerError"),
    ),
}

_CONFIG_NAME = "pgate.providers.json"


class CapabilityUnavailable(RuntimeError):
    def __init__(self, capability: str, detail: str = "") -> None:
        super().__init__(f"capability {capability!r} unavailable: {detail}")
        self.capability = capability
        self.detail = detail


@dataclass
class Resolution:
    capability: str
    module_name: str
    source: str
    detail: str = ""
    module: Any = None
    missing_attributes: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.module is not None and not self.missing_attributes

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "module": self.module_name,
            "source": self.source,
            "detail": self.detail,
            "resolved": self.ok,
            "missing_attributes": list(self.missing_attributes),
        }


@dataclass
class ProviderSet:
    resolutions: dict[str, Resolution] = field(default_factory=dict)

    def module(self, capability: str) -> Any:
        res = self.resolutions.get(capability)
        return res.module if res and res.ok else None

    def is_available(self, capability: str) -> bool:
        res = self.resolutions.get(capability)
        return bool(res and res.ok)

    def require(self, capability: str) -> Any:
        mod = self.module(capability)
        if mod is None:
            res = self.resolutions.get(capability)
            raise CapabilityUnavailable(capability, res.detail if res else "not resolved")
        return mod

    def missing(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, r in self.resolutions.items() if not r.ok))

    def as_dict(self) -> dict[str, Any]:
        return {n: r.as_dict() for n, r in sorted(self.resolutions.items())}


def config_path() -> Path:
    return Path(__file__).resolve().parent.parent / _CONFIG_NAME


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except (OSError, ValueError):
        return {}


def _load_from_file(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(f"_pgate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _seed_upstream_path() -> str:
    """Add the upstream package root to sys.path once, if the operator named one.

    ``PGATE_UPSTREAM_PATH`` is the single seam a user is expected to set: one
    directory containing the ``ollama_controller`` package. The per-capability
    variables remain available for unusual layouts, but they are per-module
    overrides rather than the normal route.
    """
    root = os.environ.get(ENV_PREFIX + "UPSTREAM_PATH", "").strip()
    if not root:
        return ""
    directory = str(Path(root))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return directory


def _resolve_one(capability: str, cfg: dict[str, Any]) -> Resolution:
    env_suffix, module_name, required = REQUIREMENTS[capability]
    env_value = os.environ.get(ENV_PREFIX + env_suffix, "").strip()

    attempts: list[tuple[str, str]] = []
    if env_value:
        attempts.append(("env:" + ENV_PREFIX + env_suffix, env_value))
    attempts.append(("env:" + ENV_PREFIX + "UPSTREAM_PATH", "seeded"))
    attempts.append(("installed", module_name))
    for entry in cfg.get("search_paths", []) or []:
        attempts.append(("config:search_paths", entry))

    first_error = ""
    for source, target in attempts:
        added = False
        try:
            if source == "env:" + ENV_PREFIX + "UPSTREAM_PATH":
                if not _seed_upstream_path():
                    continue
                module = importlib.import_module(module_name)
            elif source.startswith("env:"):
                candidate = Path(target)
                if candidate.is_dir():
                    if str(candidate) not in sys.path:
                        sys.path.insert(0, str(candidate))
                        added = True
                    module = importlib.import_module(module_name)
                else:
                    module = _load_from_file(candidate)
            elif source == "installed":
                module = importlib.import_module(module_name)
            else:
                directory = Path(target)
                if not directory.is_dir():
                    continue
                if str(directory) not in sys.path:
                    sys.path.insert(0, str(directory))
                    added = True
                module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            if not first_error:
                first_error = f"{source}: {type(exc).__name__}: {exc}"
            continue
        finally:
            if added:
                try:
                    sys.path.remove(str(Path(target)))
                except ValueError:
                    pass

        missing = tuple(a for a in required if not hasattr(module, a))
        return Resolution(
            capability=capability,
            module_name=module_name,
            source=source,
            detail="" if not missing else f"missing attributes: {', '.join(missing)}",
            module=module,
            missing_attributes=missing,
        )

    return Resolution(
        capability=capability,
        module_name=module_name,
        source="unresolved",
        detail=first_error or f"no provider for {module_name!r}",
    )


def resolve_providers(capabilities: tuple[str, ...] | None = None) -> ProviderSet:
    """Resolve the named capabilities. Imports nothing until a name is asked for."""
    cfg = load_config()
    names = capabilities if capabilities is not None else tuple(sorted(REQUIREMENTS))
    return ProviderSet(
        resolutions={name: _resolve_one(name, cfg) for name in names}
    )
