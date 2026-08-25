"""Power sampling: pure integration, host readers, and marginal-over-idle math.

No real threads and no sleeps — the meter's clock, sleep, and thread factory are
all injected, same as LiveTetrisAgent's thread_factory and PiPolicy's clock.
"""

import pytest

from tetris_agent import power
from tetris_agent.pricing import DEFAULT_KWH_PRICE, energy_usd

# --- integration -----------------------------------------------------------------------------


def test_integrate_wh_trapezoids_constant_draw():
    # 60 s at a flat 320 W = 320 W * (1/60) h = 5.333 Wh
    samples = [(0.0, 320.0), (30.0, 320.0), (60.0, 320.0)]
    assert round(power.integrate_wh(samples), 2) == 5.33


def test_integrate_wh_averages_across_a_ramp():
    # 0 -> 100 W over an hour averages 50 W, so 50 Wh.
    assert power.integrate_wh([(0.0, 0.0), (3600.0, 100.0)]) == pytest.approx(50.0)


def test_integrate_wh_needs_an_interval():
    assert power.integrate_wh([]) == 0.0
    assert power.integrate_wh([(0.0, 300.0)]) == 0.0


# --- macOS parser ----------------------------------------------------------------------------

COMBINED_SAMPLE = """
**** Processor usage ****
CPU Power: 3200 mW
GPU Power: 900 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 4100 mW
"""

DOMAINS_ONLY_SAMPLE = """
CPU Power: 3200 mW
GPU Power: 900 mW
ANE Power: 0 mW
"""


def test_parse_powermetrics_prefers_the_combined_line():
    assert power.parse_powermetrics(COMBINED_SAMPLE) == pytest.approx(4.1)


def test_parse_powermetrics_falls_back_to_summing_domains():
    # Older macOS omits the combined line; summing CPU+GPU+ANE gets the same watts.
    assert power.parse_powermetrics(DOMAINS_ONLY_SAMPLE) == pytest.approx(4.1)


def test_parse_powermetrics_returns_none_when_no_power_lines():
    assert power.parse_powermetrics("**** Processor usage ****\nsomething else\n") is None


# --- Linux readers ---------------------------------------------------------------------------


def test_read_hwmon_watts_sums_amdgpu_and_ignores_other_sensors(tmp_path):
    gpu = tmp_path / "hwmon0"
    gpu.mkdir()
    (gpu / "name").write_text("amdgpu\n")
    (gpu / "power1_input").write_text("8000000\n")  # 8 W in microwatts
    other = tmp_path / "hwmon1"
    other.mkdir()
    (other / "name").write_text("nvme\n")
    (other / "power1_input").write_text("999999999\n")
    assert power.read_hwmon_watts(tmp_path) == 8.0


def test_read_hwmon_watts_is_none_without_amdgpu(tmp_path):
    assert power.read_hwmon_watts(tmp_path) is None


def test_read_rapl_joules_converts_microjoules(tmp_path):
    d = tmp_path / "intel-rapl:0"
    d.mkdir()
    (d / "energy_uj").write_text("2500000\n")
    assert power.read_rapl_joules(tmp_path) == 2.5


def test_read_rapl_joules_is_none_when_absent(tmp_path):
    assert power.read_rapl_joules(tmp_path) is None


# --- meter -----------------------------------------------------------------------------------


def _unavailable(clock=None):
    return None, "no power backend for TestOS"


def test_unmeasured_run_reports_none_not_zero():
    """The bug being fixed is a misleading 0.0; n/a must not collapse back into it."""
    meter = power.EnergyMeter(detect=_unavailable)
    stats = meter.stats()
    assert meter.available is False
    assert stats["energy_wh"] is None
    assert stats["energy_usd"] is None
    assert stats["energy_source"] == "no power backend for TestOS"


def test_marginal_subtracts_the_idle_baseline():
    meter = power.EnergyMeter(source=lambda: 100.0)
    # An hour at 100 W with a 40 W idle floor: 100 Wh total, 60 Wh of it this run's.
    meter.samples = [(0.0, 100.0), (3600.0, 100.0)]
    meter.baseline_w = 40.0
    assert meter.total_wh() == pytest.approx(100.0)
    assert meter.marginal_wh() == pytest.approx(60.0)


def test_marginal_clamps_a_busier_baseline_to_zero():
    meter = power.EnergyMeter(source=lambda: 10.0)
    meter.samples = [(0.0, 10.0), (3600.0, 10.0)]
    meter.baseline_w = 50.0  # host was busier before the run than during it
    assert meter.marginal_wh() == 0.0


def test_stats_prices_marginal_watt_hours():
    meter = power.EnergyMeter(source=lambda: 100.0)
    meter.samples = [(0.0, 100.0), (3600.0, 100.0)]
    meter.baseline_w = 40.0
    stats = meter.stats()
    assert stats["energy_wh"] == pytest.approx(60.0)
    assert stats["energy_usd"] == pytest.approx(energy_usd(60.0, DEFAULT_KWH_PRICE))
    assert stats["energy_total_wh"] == pytest.approx(100.0)
    assert stats["energy_baseline_w"] == 40.0


def test_baseline_is_the_mean_of_idle_readings():
    # The clock is read once to set the deadline, then once per iteration before
    # each draw, so a 4 s window over a 1 s tick admits exactly three readings.
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])
    draws = iter([10.0, 20.0, 30.0])
    meter = power.EnergyMeter(
        source=lambda: next(draws),
        clock=lambda: next(ticks),
        sleep=lambda _s: None,
        interval_s=1.0,
        baseline_s=4.0,
    )
    assert meter.measure_baseline() == pytest.approx(20.0)


def test_sampling_loop_collects_until_stopped():
    """Drives the loop synchronously: the fake sleep stops it after three ticks."""
    ticks = iter([0.0, 2.0, 4.0, 6.0])
    meter = power.EnergyMeter(source=lambda: 50.0, clock=lambda: next(ticks), interval_s=2.0)
    calls = {"n": 0}

    def sleep(_s):
        calls["n"] += 1
        if calls["n"] >= 3:
            meter._stop.set()

    meter._sleep = sleep
    meter._loop()
    assert meter.samples == [(0.0, 50.0), (2.0, 50.0), (4.0, 50.0)]


def test_a_flaky_sensor_does_not_take_the_run_down():
    def boom():
        raise OSError("sensor went away mid-run")

    meter = power.EnergyMeter(source=boom)
    assert meter._read() is None


def test_scoped_sudoers_entry_counts_as_available():
    """A NOPASSWD line scoped to powermetrics alone must be detected.

    The obvious probe, `sudo -n true`, fails on exactly that host — it asks
    whether sudo works at all, not whether this one command is permitted.
    """
    seen = {}

    def runner(cmd, **kw):
        seen["cmd"] = cmd
        return type("P", (), {"returncode": 0})()

    assert power.can_sudo_powermetrics(runner=runner) is True
    assert seen["cmd"] == ["sudo", "-n", "-l", power.POWERMETRICS]


def test_no_sudoers_entry_is_unavailable():
    def runner(cmd, **kw):
        return type("P", (), {"returncode": 1})()

    assert power.can_sudo_powermetrics(runner=runner) is False
