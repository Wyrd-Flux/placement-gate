"""Providers for Placement Gate's placement primitives.

The primitives live in :mod:`pgate_demo.placement`, folded into this repository, so
there is nothing to resolve across a package boundary and nothing to fail to
resolve. This module is a thin reporting seam over those files rather than a
plugin system.

It exists for one reason: the reporting vocabulary. Every command reports which
capability it used and whether it was available, and that shape is part of
Placement Gate's contract — a command whose capability is missing must say so and
exit non-zero rather than answer from a substitute.

Two rules this module keeps, because they were the point of the original:

- **Resolution is by name, never by scanning.** No ``os.walk``, no ``pkgutil``, no
  globbing. Each capability names one module inside this package.
- **A missing capability produces ``UNAVAILABLE``, never a plausible answer.**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["CapabilityUnavailable", "ProviderSet", "Provider", "resolve_providers"]


class CapabilityUnavailable(RuntimeError):
    """A required capability could not be supplied."""

    def __init__(self, capability: str, detail: str = "") -> None:
        super().__init__(f"capability {capability!r} unavailable: {detail}")
        self.capability = capability
        self.detail = detail


#: capability name -> (dotted module inside pgate_demo.placement, attributes)
REQUIREMENTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "hardware_facts": ("hardware.facts", ("HardwareProfile", "MemoryFact", "DetectionStatus")),
    "hardware_observer": ("hardware.observer", ("RealHardwareObserver",)),
    "hardware_memory": ("hardware.windows_memory", ("WindowsMemoryProbe",)),
    "hardware_nvidia": ("hardware.nvidia_query", ("NvidiaQueryAdapter",)),
    "placement_planner": (
        "policy.inference_placement",
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
    "model_profile": ("policy.model_profile", ("ModelProfile", "ProfileSource")),
    "chat_adapter": ("backends.chat", ("OllamaChatAdapter", "OllamaChatProfile")),
    "backends_base": ("backends.backends", ("ChatBackend", "ChatSendError")),
    "ledger": ("ledger.ledger", ("Ledger",)),
    "controller": (
        "controller.placement",
        ("PlacementRunner",),
    ),
}


@dataclass
class Provider:
    """One resolved capability."""

    capability: str
    module_name: str
    module: Any = None
    missing_attributes: tuple[str, ...] = ()
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.module is not None and not self.missing_attributes

    def as_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "module": self.module_name,
            "source": "dependency" if self.ok else "unresolved",
            "detail": self.detail,
            "resolved": self.ok,
            "missing_attributes": list(self.missing_attributes),
        }


@dataclass
class ProviderSet:
    """Every named capability, resolved once by import."""

    resolutions: dict[str, Provider] = field(default_factory=dict)

    def is_available(self, capability: str) -> bool:
        provider = self.resolutions.get(capability)
        return provider.ok if provider is not None else False

    def module(self, capability: str) -> Any:
        provider = self.resolutions.get(capability)
        return provider.module if provider is not None and provider.ok else None

    def require(self, capability: str) -> Any:
        provider = self.resolutions.get(capability)
        if provider is None:
            raise CapabilityUnavailable(
                capability, "not a declared capability of pgate_demo.placement"
            )
        if not provider.ok:
            raise CapabilityUnavailable(capability, provider.detail)
        return provider.module


def resolve_providers(capabilities: tuple[str, ...] | None = None) -> ProviderSet:
    """Resolve capabilities from the primitives folded into this package.

    Resolution is a named import of ``pgate_demo.placement.<module>``. Nothing is
    discovered, enumerated or searched.
    """
    import importlib

    resolutions: dict[str, Provider] = {}
    for name in (capabilities or tuple(REQUIREMENTS)):
        dotted, attributes = REQUIREMENTS[name]
        module_name = f"pgate_demo.placement.{dotted}"
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - reported, never substituted
            resolutions[name] = Provider(
                capability=name,
                module_name=module_name,
                detail=f"{type(exc).__name__}: {exc}",
            )
            continue
        missing = tuple(a for a in attributes if not hasattr(module, a))
        resolutions[name] = Provider(
            capability=name,
            module_name=module_name,
            module=module,
            missing_attributes=missing,
            detail=(
                f"missing attributes: {', '.join(missing)}"
                if missing
                else "pgate_demo.placement"
            ),
        )
    return ProviderSet(resolutions=resolutions)
