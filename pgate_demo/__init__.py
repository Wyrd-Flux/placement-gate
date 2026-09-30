"""Placement Gate: a CLI over wyrd-placement-core.

This package contains no placement policy. It resolves named capabilities of the
installed core lazily, builds the core's inputs, and reports the core's results
with its own reason strings.
"""

from .core import PGATE_VERSION, PlacementGateSession, Status

__all__ = ["PGATE_VERSION", "PlacementGateSession", "Status"]
