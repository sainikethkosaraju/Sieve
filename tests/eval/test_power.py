"""Power sampler, energy integration, ABAB scheduler, graceful no-GPU state."""

from __future__ import annotations

import shutil
import stat
import time
from pathlib import Path

import pytest

from sieve.eval.power import (
    NvidiaSmiSampler,
    PowerSample,
    SamplerState,
    SmiMeter,
    SyntheticMeter,
    abab_order,
    integrate_joules,
    parse_smi_line,
    summarise_window,
)


def test_parse_line_and_na() -> None:
    s = parse_smi_line("0, 71.52, 4310, 1590, 62, 0x0000000000000000", 1.0)
    assert s == PowerSample(1.0, 0, 71.52, 4310.0, 1590.0, 62.0, 0)
    assert parse_smi_line("0, [N/A], 4310, 1590, 62, 0x0", 1.0) is None
    assert parse_smi_line("garbage", 1.0) is None


def test_integrate_constant_power_is_watts_times_seconds() -> None:
    samples = [PowerSample(t * 0.1, 0, 80.0, 100.0) for t in range(101)]  # 10 s at 80 W
    assert integrate_joules(samples) == pytest.approx(800.0)
    two_gpus = samples + [PowerSample(t * 0.1, 1, 20.0, 100.0) for t in range(101)]
    assert integrate_joules(two_gpus) == pytest.approx(1000.0)
    assert integrate_joules(samples[:1]) == 0.0


def test_trapezoid_on_a_ramp() -> None:
    ramp = [PowerSample(float(t), 0, 10.0 * t, 0.0) for t in range(5)]  # 0..40 W over 4 s
    assert integrate_joules(ramp) == pytest.approx(80.0)


def test_idle_subtraction_and_peak_vram() -> None:
    samples = [PowerSample(t * 0.1, 0, 80.0, 1000.0 + t) for t in range(101)]
    r = summarise_window(samples, cell="c", block=0, idle_watts=30.0)
    assert r.joules_gross == pytest.approx(800.0)
    assert r.joules_net == pytest.approx(500.0)
    assert r.mean_watts_net == pytest.approx(50.0)
    assert r.peak_vram_mib == 1100.0 and r.state is SamplerState.MEASURED


def test_net_energy_never_negative() -> None:
    samples = [PowerSample(t * 0.1, 0, 20.0, 0.0) for t in range(11)]
    assert summarise_window(samples, cell="c", block=0, idle_watts=30.0).joules_net == 0.0


def test_too_few_samples_gives_no_numbers() -> None:
    r = summarise_window([PowerSample(0.0, 0, 50.0, 1.0)], cell="c", block=1, idle_watts=0.0)
    assert r.joules_net is None and r.reason


def test_abab_order() -> None:
    assert abab_order(["A", "B"], 3) == [("A", 0), ("B", 0), ("A", 1), ("B", 1), ("A", 2), ("B", 2)]
    assert abab_order(["A", "B"], 4, counterbalance=True) == [
        ("A", 0),
        ("B", 0),
        ("B", 1),
        ("A", 1),
        ("A", 2),
        ("B", 2),
        ("B", 3),
        ("A", 3),
    ]
    order = abab_order(["A", "B", "C"], 5)
    assert sorted(order) == sorted((c, b) for c in "ABC" for b in range(5))


def test_no_gpu_is_an_explicit_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shutil, "which", lambda _n: None)
    m = SmiMeter()
    assert m.start() is SamplerState.NO_GPU
    assert m.idle_baseline(0.0) != m.idle_baseline(0.0)  # NaN, not a made-up zero
    with m.measure("cell", 0):
        pass
    (r,) = m.results()
    assert r.state is SamplerState.NO_GPU
    assert r.joules_net is None and r.mean_watts_net is None and r.peak_vram_mib is None
    m.stop()


def test_sampler_reads_a_fake_nvidia_smi(tmp_path: Path) -> None:
    exe = tmp_path / "nvidia-smi"
    exe.write_text(
        "#!/bin/sh\nwhile true; do echo '0, 60.0, 2048, 1500, 55, 0x0'; sleep 0.05; done\n"
    )
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    m = SmiMeter(sampler=NvidiaSmiSampler(interval_ms=50, nvidia_smi=str(exe)))
    assert m.start() is SamplerState.MEASURED
    try:
        idle = m.idle_baseline(0.3)
        assert idle == pytest.approx(60.0)
        with m.measure("cell", 2):
            time.sleep(0.4)
    finally:
        m.stop()
    (r,) = m.results()
    assert r.state is SamplerState.MEASURED and r.n_samples >= 3
    assert r.joules_gross == pytest.approx(60.0 * r.duration_s, rel=0.05)
    assert r.joules_net == pytest.approx(0.0, abs=1.0)  # constant power == idle
    assert r.peak_vram_mib == 2048.0


def test_synthetic_meter_is_labelled_synthetic() -> None:
    t = [0.0]
    m = SyntheticMeter(lambda: t[0], {"a": 100.0}, idle_watts_value=30.0)
    with m.measure("a", 0):
        t[0] += 10.0
    (r,) = m.results()
    assert r.state is SamplerState.SYNTHETIC
    assert r.mean_watts_net == pytest.approx(70.0, rel=0.02)
