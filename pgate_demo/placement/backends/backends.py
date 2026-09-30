"""Backend profiles and adaptation interfaces.

Stage 1 defines the backend capability profiles (MANAGED_STRICT / ATTACH_RELAXED)
and the chat-backend interface. ONLY fake backends are used in Stage 1; the fake
server lives under tests/. No Stage 1 code discovers, defaults to, or contacts
the normal Ollama endpoint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

from ..ledger.models import BackendMode


class Capability(str, Enum):
    CPU_PLACEMENT_GUARANTEE = "CPU_PLACEMENT_GUARANTEE"
    MANAGED_UNLOADING = "MANAGED_UNLOADING"
    GLOBAL_EXCLUSION = "GLOBAL_EXCLUSION"
    FOREIGN_WORKLOAD_RECOGNITION = "FOREIGN_WORKLOAD_RECOGNITION"


@dataclass(frozen=True)
class BackendProfile:
    """Capability profile. Use profile.support(cap) rather than if mode checks."""

    mode: BackendMode
    _capabilities: frozenset[Capability] = field(default_factory=frozenset)

    def support(self, capability: Capability) -> bool:
        return capability in self._capabilities


MANAGED_STRICT_PROFILE = BackendProfile(
    mode=BackendMode.MANAGED_STRICT,
    _capabilities=frozenset(
        {
            Capability.CPU_PLACEMENT_GUARANTEE,
            Capability.MANAGED_UNLOADING,
            Capability.GLOBAL_EXCLUSION,
        }
    ),
)

ATTACH_RELAXED_PROFILE = BackendProfile(
    mode=BackendMode.ATTACH_RELAXED,
    _capabilities=frozenset({Capability.FOREIGN_WORKLOAD_RECOGNITION}),
)


class ChatBackend(ABC):
    """Abstract chat backend exposed to the controller runtime."""

    @abstractmethod
    def endpoint_id(self) -> str:
        """Stable identifier of this backend endpoint."""

    def send_chat(self, model: str, messages: list[dict], **kwargs) -> dict:
        """Send one typed chat request and return a validated response."""
        raise NotImplementedError


class ChatSendError(Exception):
    pass
