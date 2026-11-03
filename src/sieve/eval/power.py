"""GPU power / VRAM sampling, energy integration and the ABAB cell scheduler.

* ``NvidiaSmiSampler`` polls ``nvidia-smi`` at 10 Hz (``-lms 100``). Without ``nvidia-smi``
  or without a GPU the state is explicitly ``no_gpu``: no numbers, never zeros.
* Energy is GPU-only (Kaggle exposes no RAPL), integrated by the trapezoid rule over the
  host monotonic timestamps of the samples, with an idle baseline subtracted.
* ``abab_order`` interleaves cells block-wise (A1 B1 A2 B2 ...; ``abba`` flips every other
  block) so slow thermal/co-tenancy drift hits both arms alike.
"""

from __future__ import annotations

import enum
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Protocol

QUERY = "index,power.draw,memory.used,clocks.sm,temperature.gpu,clocks_throttle_reasons.active"


class SamplerState(enum.Enum):
    MEASURED = "measured"
    NO_GPU = "no_gpu"
    SYNTHETIC = "synthetic"  # fake data from the mock pipeline; never a measurement


@dataclass(frozen=True)
class PowerSample:
    t: float  # host monotonic seconds
    gpu: int
    watts: float
    vram_mib: float
    sm_clock_mhz: float = 0.0
    temp_c: float = 0.0
    throttle_mask: int = 0


def parse_smi_line(line: str, t: float) -> PowerSample | None:
    """Parse one CSV line (no units); ``None`` for '[N/A]' or malformed lines."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 6:
        return None
    try:
        mask = int(parts[5], 16) if parts[5].lower().startswith("0x") else int(float(parts[5]))
        return PowerSample(
            t=t,
            gpu=int(parts[0]),
            watts=float(parts[1]),
            vram_mib=float(parts[2]),
            sm_clock_mhz=float(parts[3]),
            temp_c=float(parts[4]),
            throttle_mask=mask,
        )
    except ValueError:
        return None


def integrate_joules(samples: Sequence[PowerSample]) -> float:
    """Trapezoid integral of summed-over-GPU power versus time. Needs >= 2 timestamps."""
    by_t: dict[float, float] = {}
    for s in samples:
        by_t[s.t] = by_t.get(s.t, 0.0) + s.watts
    ts = sorted(by_t)
    if len(ts) < 2:
        return 0.0
    return sum(0.5 * (by_t[a] + by_t[b]) * (b - a) for a, b in pairwise(ts))


@dataclass(frozen=True)
class EnergyResult:
    """Energy of one measured window. ``joules_net`` is None unless state is measured/synthetic."""

    state: SamplerState
    cell: str
    block: int
    duration_s: float
    n_samples: int
    joules_gross: float | None
    joules_net: float | None  # idle-subtracted
    mean_watts_net: float | None
    peak_vram_mib: float | None
    idle_watts: float | None
    throttled: bool | None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "cell": self.cell,
            "block": self.block,
            "duration_s": self.duration_s,
            "n_samples": self.n_samples,
            "joules_gross": self.joules_gross,
            "joules_net": self.joules_net,
            "mean_watts_net": self.mean_watts_net,
            "peak_vram_mib": self.peak_vram_mib,
            "idle_watts": self.idle_watts,
            "throttled": self.throttled,
            "reason": self.reason,
        }


def summarise_window(
    samples: Sequence[PowerSample],
    *,
    cell: str,
    block: int,
    idle_watts: float,
    state: SamplerState = SamplerState.MEASURED,
) -> EnergyResult:
    if len(samples) < 2:
        return EnergyResult(
            state,
            cell,
            block,
            0.0,
            len(samples),
            None,
            None,
            None,
            None,
            idle_watts,
            None,
            "fewer than two samples",
        )
    ts = sorted({s.t for s in samples})
    dur = ts[-1] - ts[0]
    gross = integrate_joules(samples)
    n_gpu = len({s.gpu for s in samples})
    net = gross - idle_watts * n_gpu * dur if idle_watts is not None else gross
    return EnergyResult(
        state=state,
        cell=cell,
        block=block,
        duration_s=dur,
        n_samples=len(samples),
        joules_gross=gross,
        joules_net=max(net, 0.0),
        mean_watts_net=max(net, 0.0) / dur if dur > 0 else None,
        peak_vram_mib=max(s.vram_mib for s in samples),
        idle_watts=idle_watts,
        throttled=any(s.throttle_mask & 0xFC != 0 for s in samples),  # beyond idle/app-clock bits
    )


class Meter(Protocol):
    """Anything that can measure a window; real, absent-GPU or synthetic."""

    state: SamplerState

    def idle_baseline(self, seconds: float) -> float: ...

    def measure(self, cell: str, block: int) -> AbstractContextManager[None]: ...

    def results(self) -> list[EnergyResult]: ...


class NvidiaSmiSampler:
    """Background ``nvidia-smi -lms`` poller. Use through ``SmiMeter``."""

    def __init__(self, interval_ms: int = 100, nvidia_smi: str | None = None) -> None:
        self.interval_ms = interval_ms
        self._exe = nvidia_smi or shutil.which("nvidia-smi")
        self._proc: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[PowerSample] = []

    @property
    def available(self) -> bool:
        return self._exe is not None

    def start(self) -> bool:
        if self._exe is None:
            return False
        self._proc = subprocess.Popen(
            [
                self._exe,
                f"--query-gpu={QUERY}",
                "--format=csv,noheader,nounits",
                f"-lms={self.interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()
        return True

    def _pump(self) -> None:
        assert self._proc is not None
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            s = parse_smi_line(line, time.monotonic())
            if s is not None:
                with self._lock:
                    self._samples.append(s)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def window(self, t0: float, t1: float) -> list[PowerSample]:
        with self._lock:
            return [s for s in self._samples if t0 <= s.t <= t1]


@dataclass
class SmiMeter:
    """Real meter. If the GPU is absent every window reports state ``no_gpu`` and no numbers."""

    sampler: NvidiaSmiSampler = field(default_factory=NvidiaSmiSampler)
    idle_watts_value: float | None = None
    _results: list[EnergyResult] = field(default_factory=list)
    state: SamplerState = SamplerState.NO_GPU

    def start(self) -> SamplerState:
        self.state = SamplerState.MEASURED if self.sampler.start() else SamplerState.NO_GPU
        return self.state

    def stop(self) -> None:
        self.sampler.stop()

    def idle_baseline(self, seconds: float) -> float:
        if self.state is SamplerState.NO_GPU:
            return float("nan")
        t0 = time.monotonic()
        time.sleep(seconds)
        w = self.sampler.window(t0, time.monotonic())
        by_t: dict[float, float] = {}
        for s in w:
            by_t[s.t] = by_t.get(s.t, 0.0) + s.watts
        n_gpu = max(len({s.gpu for s in w}), 1)
        self.idle_watts_value = (sum(by_t.values()) / max(len(by_t), 1)) / n_gpu if by_t else 0.0
        return self.idle_watts_value

    @contextmanager
    def measure(self, cell: str, block: int) -> Iterator[None]:
        t0 = time.monotonic()
        try:
            yield
        finally:
            t1 = time.monotonic()
            if self.state is SamplerState.NO_GPU:
                self._results.append(
                    EnergyResult(
                        SamplerState.NO_GPU,
                        cell,
                        block,
                        t1 - t0,
                        0,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        "no GPU / nvidia-smi not available",
                    )
                )
            else:
                self._results.append(
                    summarise_window(
                        self.sampler.window(t0, t1),
                        cell=cell,
                        block=block,
                        idle_watts=self.idle_watts_value or 0.0,
                    )
                )

    def results(self) -> list[EnergyResult]:
        return list(self._results)


@dataclass
class SyntheticMeter:
    """Deterministic fake used by the mock pipeline. Every number is labelled synthetic."""

    clock: Callable[[], float]
    watts_by_cell: dict[str, float]
    idle_watts_value: float = 30.0
    state: SamplerState = SamplerState.SYNTHETIC
    _results: list[EnergyResult] = field(default_factory=list)

    def idle_baseline(self, seconds: float) -> float:
        return self.idle_watts_value

    @contextmanager
    def measure(self, cell: str, block: int) -> Iterator[None]:
        t0 = self.clock()
        try:
            yield
        finally:
            dur = self.clock() - t0
            watts = self.watts_by_cell.get(cell, 70.0)
            n = max(int(dur * 10), 2)
            samples = [
                PowerSample(
                    t=t0 + dur * i / (n - 1), gpu=0, watts=watts, vram_mib=4000.0 + 10 * i % 7
                )
                for i in range(n)
            ]
            self._results.append(
                summarise_window(
                    samples,
                    cell=cell,
                    block=block,
                    idle_watts=self.idle_watts_value,
                    state=SamplerState.SYNTHETIC,
                )
            )

    def results(self) -> list[EnergyResult]:
        return list(self._results)


def abab_order(
    cells: Sequence[str], n_blocks: int, *, counterbalance: bool = False
) -> list[tuple[str, int]]:
    """Block-interleaved ``[(A,0),(B,0),(A,1),(B,1),...]``; counterbalanced flips odd blocks."""
    out: list[tuple[str, int]] = []
    for b in range(n_blocks):
        order = list(cells)
        if counterbalance and b % 2 == 1:
            order.reverse()
        out.extend((c, b) for c in order)
    return out
