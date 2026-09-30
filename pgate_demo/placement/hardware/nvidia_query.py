"""G-3Q: fixed, bounded, read-only nvidia-smi query adapter.

Hard safety contract (v2 §C / v3 §6):
  - absolute operator-resolved executable path only (no PATH trust);
  - constant allowlisted argv tuples ONLY; caller/model content can never
    contribute command fragments;
  - shell=False;
  - bounded timeout, bounded stdout, bounded stderr;
  - malformed output -> UNKNOWN (never guessed);
  - execution failure / missing tool -> UNKNOWN unless INDEPENDENT positive
    evidence proves ABSENT (that evidence is a separate channel; this adapter
    never asserts absence);
  - tool/version provenance captured where safely obtainable.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Optional

from .facts import (
    AdapterKind, DetectionStatus, GpuAdapterFact, GpuUtilizationFact, MemoryFact,
)

MAX_STDOUT_BYTES = 64 * 1024
MAX_STDERR_BYTES = 8 * 1024
QUERY_TIMEOUT_SECONDS = 5.0

# Constant, allowlisted argv sets. NEVER assembled from caller data.
_ARGV_GPU_INFO = ("--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits")
_ARGV_COMPUTE_APPS = ("--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits")
_ARGV_UTILIZATION = (
    "--query-gpu=utilization.gpu,utilization.memory",
    "--format=csv,noheader,nounits",
)

_CSV_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


class NvidiaQueryAdapter:
    """Read-only GPU probe. Construct only with an absolute executable path."""

    def __init__(self, executable: str | Path):
        exe = Path(executable)
        if not exe.is_absolute():
            raise ValueError("nvidia-smi adapter requires an absolute executable path")
        self._exe = str(exe)

    # ------------------------------------------------------------------ #
    def probe(self) -> list[GpuAdapterFact]:
        """Return adapter facts; [] means NO CONCLUSION (unknown), not absent."""
        info = self._run(_ARGV_GPU_INFO)
        if info is None:
            return []
        try:
            return self._parse_gpu_info(info)
        except Exception:
            return []  # malformed -> UNKNOWN

    def foreign_compute_bytes(self) -> Optional[int]:
        out = self._run(_ARGV_COMPUTE_APPS)
        if out is None:
            return None
        try:
            total = 0
            for line in out.strip().splitlines():
                if not line.strip():
                    continue
                _pid, mem = line.split(",", 1)
                total += self._mib_to_bytes(self._num(mem))
            return total
        except Exception:
            return None

    def utilization(self) -> Optional[GpuUtilizationFact]:
        """Return bounded point-in-time utilization, or UNKNOWN as None."""
        out = self._run(_ARGV_UTILIZATION)
        if out is None:
            return None
        try:
            rows = [line for line in out.strip().splitlines() if line.strip()]
            if len(rows) != 1:
                return None  # multi-GPU aggregation is outside this stage
            gpu, memory = (part.strip() for part in rows[0].split(","))
            return GpuUtilizationFact(
                gpu_utilization_bps=int(round(self._num(gpu) * 100)),
                memory_utilization_bps=int(round(self._num(memory) * 100)),
                probe_source="nvidia-smi",
            )
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def _run(self, argv_tail: tuple[str, ...]) -> Optional[str]:
        argv = [self._exe, *argv_tail]  # constant prefix + constant tail only
        try:
            proc = subprocess.run(
                argv,
                shell=False,
                capture_output=True,
                timeout=QUERY_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0:
            return None
        out = proc.stdout[: MAX_STDOUT_BYTES + 1]
        if len(out) > MAX_STDOUT_BYTES:  # unbounded/broken tool -> unknown
            return None
        try:
            return out.decode("utf-8")
        except UnicodeDecodeError:
            return None

    @staticmethod
    def _num(text: str) -> float:
        m = _CSV_NUM_RE.match(text)
        if m is None:
            raise ValueError(f"malformed nvidia-smi numeric field: {text!r}")
        return float(m.group(1))

    @staticmethod
    def _mib_to_bytes(value_mib: float) -> int:
        # exact scaled arithmetic: MiB has finite binary representation
        return int(round(value_mib * 1024 * 1024))

    def _parse_gpu_info(self, text: str) -> list[GpuAdapterFact]:
        facts: list[GpuAdapterFact] = []
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty gpu info")
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 4:
                raise ValueError("malformed gpu info row")
            name, total, used, driver = parts
            total_b = self._mib_to_bytes(self._num(total))
            used_b = self._mib_to_bytes(self._num(used))
            facts.append(
                GpuAdapterFact(
                    kind=AdapterKind.DISCRETE,
                    vendor="NVIDIA",
                    name=name,
                    vram_total_bytes=total_b,
                    observed_free_vram_bytes=total_b - used_b,
                    foreign_usage_bytes=used_b,  # evidence only, pre-os/other
                    detection_status=DetectionStatus.OK,
                    probe_source="nvidia-smi",
                    probe_tool_version=driver,
                )
            )
        return facts
