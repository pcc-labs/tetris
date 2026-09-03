"""Host power sampling, so local arms stop reporting their cost as zero.

A `pi/*` arm bills nothing to an API, which is not the same as being free — it
burns watts on this machine instead. This module measures that draw and turns it
into watt-hours and dollars, so the benchmark's `cost_usd` column has something
honest to sit next to.

Three backends, picked by what the host actually offers:

- **macOS** `powermetrics`, which requires root. Probed with `sudo -n` so an
  unattended run degrades instead of blocking on a password prompt.
- **Linux** `nvidia-smi` plus amdgpu hwmon and Intel RAPL under `/sys` — the
  shape `../pokemon-kafka/scripts/power_sampler.py` uses, kept for the exe.dev
  VMs the README anticipates.
- **Neither**, in which case energy is reported as `None` and rendered `n/a`.
  Reporting a made-up number here would just replace one lie with another.

What we report is *marginal* draw: a short idle baseline is sampled before the
run and subtracted, so the figure answers "what did this run cost on top of
leaving the machine on" rather than "what does this laptop pull with Chrome
open". That is the number comparable to a cloud arm's API bill.
"""

import platform
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Iterable
from pathlib import Path

HWMON = Path("/sys/class/hwmon")
RAPL = Path("/sys/class/powercap")

# Sampling cadence during a run, and how long to watch an idle host first.
DEFAULT_INTERVAL_S = 2.0
DEFAULT_BASELINE_S = 3.0

WattSource = Callable[[], float | None]


# --- integration -----------------------------------------------------------------------------
# Pure, so it carries the tests without any host or subprocess involved.


def integrate_wh(samples: Iterable[tuple[float, float]]) -> float:
    """Trapezoid-integrate [(timestamp_s, watts)] into watt-hours.

    Fewer than two samples means no measured interval, which is 0.0 rather than
    an extrapolation from a single instant.
    """
    pts = sorted(samples)
    total = 0.0
    for (t0, w0), (t1, w1) in zip(pts, pts[1:]):
        total += (w0 + w1) / 2 * (t1 - t0) / 3600
    return total


# --- macOS ------------------------------------------------------------------------------------

# powermetrics reports milliwatts. Recent macOS emits a combined line; older and
# some configurations only emit the per-domain lines, so both shapes are handled
# rather than betting on one.
POWERMETRICS = "/usr/bin/powermetrics"

_COMBINED_RE = re.compile(r"^Combined Power \(.*\):\s*([\d.]+)\s*mW", re.MULTILINE)
_DOMAIN_RE = re.compile(r"^(CPU|GPU|ANE) Power:\s*([\d.]+)\s*mW", re.MULTILINE)


def parse_powermetrics(text: str) -> float | None:
    """Whole-package watts from one powermetrics sample, or None if absent.

    Prefers the combined line; falls back to summing the CPU/GPU/ANE domains.
    """
    combined = _COMBINED_RE.search(text)
    if combined:
        return float(combined.group(1)) / 1000.0
    domains = _DOMAIN_RE.findall(text)
    if domains:
        return sum(float(mw) for _, mw in domains) / 1000.0
    return None


def can_sudo_powermetrics(runner=None) -> bool:
    """True when powermetrics can run non-interactively.

    Probes `sudo -n -l powermetrics`, which asks whether *this one command* is
    permitted rather than whether sudo works at all. That matters: the sane way
    to grant this is a least-privilege sudoers line scoped to powermetrics, and
    such a host still fails a blanket `sudo -n true`. `-l` also only queries the
    policy, so the probe costs nothing and never actually samples.

    `-n` fails immediately instead of prompting, which is what keeps
    scripts/vm-demo.sh unattended on a host with no entry at all.
    """
    if platform.system() != "Darwin" or not Path(POWERMETRICS).exists():
        return False
    run = runner or subprocess.run
    try:
        probe = run(["sudo", "-n", "-l", POWERMETRICS], capture_output=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return probe.returncode == 0


def read_darwin_watts(sample_ms: int = 200) -> float | None:
    try:
        out = subprocess.run(
            ["sudo", "-n", POWERMETRICS, "--samplers", "cpu_power", "-i", str(sample_ms), "-n", "1"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_powermetrics(out)


# --- Linux ------------------------------------------------------------------------------------
# Ported from ../pokemon-kafka/scripts/power_sampler.py. The `root=` parameters
# exist so tests can point the readers at a tmp_path sysfs tree.


def read_gpu_watts() -> float | None:
    """Discrete NVIDIA GPU draw via nvidia-smi, or None when there is no card."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def read_hwmon_watts(root: Path | str | None = None) -> float | None:
    """Sum of amdgpu hwmon power1_input (microwatts), i.e. the integrated GPU."""
    total = None
    for h in sorted(Path(root or HWMON).glob("hwmon*")):
        try:
            if (h / "name").read_text().strip() != "amdgpu":
                continue
            uw = float((h / "power1_input").read_text().strip())
        except (OSError, ValueError):
            continue
        total = (total or 0.0) + uw / 1_000_000
    return total


def read_rapl_joules(root: Path | str | None = None) -> float | None:
    """CPU package energy counter in joules, or None when absent/root-only."""
    try:
        return float((Path(root or RAPL) / "intel-rapl:0" / "energy_uj").read_text().strip()) / 1_000_000
    except (OSError, ValueError):
        return None


class LinuxWattSource:
    """GPU + iGPU instantaneous draw, plus CPU package power from RAPL deltas.

    RAPL is a monotonic energy counter, so the first call has no delta to divide
    and contributes no CPU term; every later call does.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._last_j: float | None = None
        self._last_t: float | None = None

    def __call__(self) -> float | None:
        t = self._clock()
        parts = [p for p in (read_gpu_watts(), read_hwmon_watts()) if p is not None]
        j = read_rapl_joules()
        if j is not None and self._last_j is not None and t > self._last_t:
            parts.append((j - self._last_j) / (t - self._last_t))
        self._last_j, self._last_t = j, t
        return sum(parts) if parts else None


# --- backend selection ------------------------------------------------------------------------


def detect_source(clock: Callable[[], float] = time.monotonic) -> tuple[WattSource | None, str]:
    """Pick a backend for this host.

    Returns (reader, description). A None reader means energy is unmeasurable
    here, and the description says why so the benchmark row can explain its n/a.
    """
    system = platform.system()
    if system == "Darwin":
        if can_sudo_powermetrics():
            return read_darwin_watts, "powermetrics"
        if Path(POWERMETRICS).exists():
            return None, "powermetrics needs root — grant passwordless sudo to measure"
        return None, "no powermetrics on this host"
    if system == "Linux":
        source = LinuxWattSource(clock=clock)
        if source() is not None:
            return source, "nvidia-smi/hwmon/rapl"
        return None, "no readable nvidia-smi, amdgpu hwmon, or intel-rapl counter"
    return None, f"no power backend for {system}"


# --- metering ---------------------------------------------------------------------------------


class EnergyMeter:
    """Samples host draw across a run and reports marginal watt-hours.

    Used two ways: in-process around a host-side run, and behind the `sample`
    CLI for the VM demo, where the agent runs in a guest but inference (and so
    the power draw) happens out here on the host.

    `source`, `clock`, `sleep`, and `thread_factory` are injectable for the same
    reason they are on LiveTetrisAgent and PiPolicy — the tests run with a fake
    clock and no real threads.
    """

    def __init__(
        self,
        source: WattSource | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        thread_factory=threading.Thread,
        interval_s: float = DEFAULT_INTERVAL_S,
        baseline_s: float = DEFAULT_BASELINE_S,
        detect: Callable[..., tuple[WattSource | None, str]] = detect_source,
    ):
        if source is None:
            source, reason = detect(clock=clock)
        else:
            reason = "injected"
        self.source = source
        self.reason = reason
        self._clock = clock
        self._sleep = sleep
        self._thread_factory = thread_factory
        self._interval_s = interval_s
        self._baseline_s = baseline_s
        self.samples: list[tuple[float, float]] = []
        self.baseline_w: float | None = None
        self._stop = threading.Event()
        self._thread = None

    @property
    def available(self) -> bool:
        return self.source is not None

    def _read(self) -> float | None:
        try:
            return self.source() if self.source else None
        except Exception:  # a flaky sensor must not take the run down with it
            return None

    def measure_baseline(self) -> float | None:
        """Mean idle draw, sampled before the run starts."""
        if not self.available:
            return None
        readings, deadline = [], self._clock() + self._baseline_s
        while self._clock() < deadline:
            w = self._read()
            if w is not None:
                readings.append(w)
            self._sleep(self._interval_s)
        self.baseline_w = sum(readings) / len(readings) if readings else None
        return self.baseline_w

    def _loop(self) -> None:
        while not self._stop.is_set():
            w = self._read()
            if w is not None:
                self.samples.append((self._clock(), w))
            self._sleep(self._interval_s)

    def __enter__(self) -> "EnergyMeter":
        if self.available:
            self.measure_baseline()
            self._thread = self._thread_factory(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_s * 2 + 5)
        return None

    def total_wh(self) -> float | None:
        return integrate_wh(self.samples) if self.available and len(self.samples) >= 2 else None

    def marginal_wh(self) -> float | None:
        """Watt-hours above idle — the run's own cost.

        Clamped at zero: a host busier at baseline than during the run can push
        this negative, which is noise rather than energy given back.
        """
        total = self.total_wh()
        if total is None:
            return None
        if self.baseline_w is None:
            return round(total, 4)
        elapsed_h = (self.samples[-1][0] - self.samples[0][0]) / 3600
        return round(max(0.0, total - self.baseline_w * elapsed_h), 4)

    def stats(self) -> dict:
        """Energy keys for a run's policy stats. Always the same key set.

        `energy_wh` is None (not 0.0) when unmeasured — the whole point is to
        stop reporting an unmeasured local run as costing nothing.
        """
        from tetris_agent.pricing import DEFAULT_KWH_PRICE, energy_usd

        wh = self.marginal_wh()
        total = self.total_wh()
        # Draw, not just energy: the figures a write-up quotes ("mean 509 W, max
        # 608") and the ones that say whether a run sat near a power cap.
        mean_w = peak_w = None
        if total is not None:
            elapsed_h = (self.samples[-1][0] - self.samples[0][0]) / 3600
            mean_w = round(total / elapsed_h, 1) if elapsed_h > 0 else None
            peak_w = round(max(w for _, w in self.samples), 1)
        return {
            "energy_wh": wh,
            "energy_usd": None if wh is None else energy_usd(wh, DEFAULT_KWH_PRICE),
            "energy_source": self.reason,
            "energy_total_wh": None if total is None else round(total, 4),
            "energy_baseline_w": None if self.baseline_w is None else round(self.baseline_w, 1),
            "energy_mean_w": mean_w,
            "energy_peak_w": peak_w,
        }


# --- CLI --------------------------------------------------------------------------------------
# `sample` exists for scripts/vm-demo.sh: the agent runs inside the Lima guest,
# but inference happens on the host's Ollama, so the meter has to live out here.


def _sample(args) -> int:
    source, reason = detect_source()
    if source is None:
        print(f"[power] unavailable: {reason}")
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meter = EnergyMeter(source=source, interval_s=args.interval, baseline_s=args.baseline)
    print(f"[power] sampling via {reason} -> {out}")
    meter.measure_baseline()
    with open(out, "w") as f:
        f.write(f"# source={reason} baseline_w={meter.baseline_w}\n")
        f.write("ts,watts\n")
        try:
            while True:
                w = meter._read()
                if w is not None:
                    t = time.monotonic()
                    meter.samples.append((t, w))
                    f.write(f"{t},{w}\n")
                    f.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    print(f"[power] {len(meter.samples)} samples, marginal {meter.marginal_wh()} Wh")
    return 0


def _report(args) -> int:
    """Integrate a CSV written by `sample` — used after the sampler is killed."""
    baseline = None
    samples = []
    for line in Path(args.csv).read_text().splitlines():
        if line.startswith("#"):
            match = re.search(r"baseline_w=([\d.]+)", line)
            if match:
                baseline = float(match.group(1))
            continue
        parts = line.split(",")
        try:
            samples.append((float(parts[0]), float(parts[1])))
        except (ValueError, IndexError):
            continue
    meter = EnergyMeter(source=lambda: None)
    meter.samples, meter.baseline_w = samples, baseline
    wh = meter.marginal_wh()
    if wh is None:
        print("energy: n/a (no usable samples)")
        return 1
    from tetris_agent.pricing import energy_usd

    print(
        f"energy: {wh:.2f} Wh marginal over a {baseline:.0f} W idle floor "
        f"= ${energy_usd(wh, args.kwh_price):.4f} at ${args.kwh_price}/kWh "
        f"({len(samples)} samples, {meter.total_wh():.2f} Wh total draw)"
    )
    return 0


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="host power sampling for local arms")
    subs = parser.add_subparsers(dest="cmd", required=True)

    s = subs.add_parser("sample", help="sample host draw to a CSV until killed")
    s.add_argument("--out", required=True)
    s.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    s.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_S)
    s.set_defaults(fn=_sample)

    r = subs.add_parser("report", help="integrate a sampled CSV into Wh and dollars")
    r.add_argument("csv")
    r.add_argument("--kwh-price", type=float, default=None)
    r.set_defaults(fn=_report)

    c = subs.add_parser("check", help="report which power backend this host offers")
    c.set_defaults(fn=lambda a: (print(f"[power] {detect_source()[1]}"), 0)[1])

    args = parser.parse_args(argv)
    if getattr(args, "kwh_price", None) is None and args.cmd == "report":
        from tetris_agent.pricing import DEFAULT_KWH_PRICE

        args.kwh_price = DEFAULT_KWH_PRICE
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
