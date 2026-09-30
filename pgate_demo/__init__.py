"""Placement Gate: a thin adapter over the upstream Ollama_Controller placement path.

This package contains no placement policy. It resolves named upstream
capabilities lazily, builds upstream inputs, and reports upstream results.
"""

from .core import PGATE_VERSION, PlacementGateSession, Status

__all__ = ["PGATE_VERSION", "PlacementGateSession", "Status"]
