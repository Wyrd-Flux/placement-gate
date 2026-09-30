"""Stage 3A host memory observation via Windows APIs (stdlib ctypes only).

Read-only fact gathering: GlobalMemoryStatusEx (physical total/available).
Any API failure yields DetectionStatus.UNKNOWN — never fabricated numbers.

Positive GPU-ABSENCE evidence is a SEPARATE channel: an operator-supplied
enumeration fact (AbsenceEvidence) consumed by the observer. Probing failure
can never imply absence (v3 §6).
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

from .facts import MemoryFact, DetectionStatus


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class WindowsMemoryProbe:
    """Deterministic, stateless, read-only memory fact source."""

    def observe(self) -> MemoryFact:
        try:
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            if not ok:
                raise OSError("GlobalMemoryStatusEx returned 0")
            return MemoryFact(
                physical_total_bytes=int(stat.ullTotalPhys),
                observed_available_bytes=int(stat.ullAvailPhys),
                detection_status=DetectionStatus.OK,
            )
        except Exception:
            return MemoryFact(
                physical_total_bytes=0,
                observed_available_bytes=0,
                detection_status=DetectionStatus.UNKNOWN,
            )
