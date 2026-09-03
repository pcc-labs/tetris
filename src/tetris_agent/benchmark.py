"""Benchmark runner: model x harness x effort, against the heuristic control.

Every arm plays the same deterministic piece sequences (fixed timer_div seeds),
so differences in score are differences in play, not luck. Cost is a
first-class result column, and a hard budget cap aborts the matrix rather than
discovering the bill afterwards.
"""

import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from tetris_agent.fitness import race_score
from tetris_agent.pricing import DEFAULT_KWH_PRICE, energy_usd, is_pi, spec

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/benchmarks")
DEFAULT_SEEDS = (0x00,)
DEFAULT_MAX_PIECES = 30


@dataclass(frozen=True)
class Arm:
    policy: str  # "heuristic" | "model"
    model: str | None = None
    harness: str | None = None
    effort: str | None = None
    exemplars: bool = False  # human exemplars in the system prompt
    live: bool = False  # gravity keeps running while the model thinks
    # Bounded pause: the game freezes once per piece while the model thinks,
    # but a decision slower than this is discarded and the piece falls.
    deadline_s: float | None = None
    # The deadline controller may not step the effort tier: the row measures
    # the configured level, not what the ladder settled on.
    fixed_effort: bool = False

    @property
    def name(self) -> str:
        if self.policy != "model":
            return self.policy
        parts = [self.model, self.harness]
        if self.effort:
            parts.append(self.effort)
        suffix = "+ex" if self.exemplars else ""
        if self.live:
            suffix += "+live"
        elif self.deadline_s is not None:
            suffix += f"+p{self.deadline_s:g}"
        if self.fixed_effort:
            suffix += "+fixed"
        return "/".join(parts) + suffix


@dataclass
class ArmResult:
    arm: str
    seed: int
    fitness: dict = field(default_factory=dict)
    policy_stats: dict = field(default_factory=dict)
    error: str = ""

    @property
    def cost(self) -> float:
        return float(self.policy_stats.get("cost_usd", 0.0))


# Non-model reference arms, weakest first. `no-input` is the floor (nobody
# plays), `random` is chance within the legal action space, `heuristic` is the
# tuned-solver ceiling. A model arm is only interesting above `random`.
BASELINE_POLICIES = ("no-input", "random", "heuristic")


def expand_arms(
    models: list[str],
    harnesses: list[str],
    efforts: list[str],
    include_control: bool = True,
    exemplars: bool = False,
    live: bool = True,
    deadline_s: float | None = None,
    fixed_effort: bool = False,
    lookahead_control: bool = False,
) -> list[Arm]:
    """Cartesian product, minus combinations the API rejects.

    Haiku 4.5 does not accept output_config.effort, so it contributes one arm
    per harness instead of one per (harness, effort).

    Model arms default to live: the game does not pause while the model
    thinks. Baselines stay non-live — their decisions are instant, so pacing
    them to real time would change nothing but the wall clock.
    """
    arms = [Arm(policy=p) for p in reversed(BASELINE_POLICIES)] if include_control else []
    seen = set()
    for model, harness, effort in itertools.product(models, harnesses, efforts):
        eff = effort if spec(model).supports_effort else None
        arm = Arm(
            policy="model",
            model=model,
            harness=harness,
            effort=eff,
            exemplars=exemplars,
            live=live,
            deadline_s=None if live else deadline_s,
            fixed_effort=fixed_effort,
        )
        if arm.name not in seen:
            seen.add(arm.name)
            arms.append(arm)
    # Appended last, never inserted: a command run before this flag existed
    # must produce the same rows in the same order, plus this one at the end.
    if lookahead_control:
        arms.append(Arm(policy="lookahead"))
    return arms


def build_policy(arm: Arm, genome_params: dict | None = None, exemplar_block: str = ""):
    from tetris_agent.policy import Genome, HeuristicPolicy, LookaheadPolicy, NoInputPolicy, RandomPolicy

    genome = Genome.from_params(genome_params or {})
    if arm.policy == "heuristic":
        return HeuristicPolicy(genome)
    if arm.policy == "no-input":
        return NoInputPolicy()
    if arm.policy == "random":
        return RandomPolicy()
    if arm.policy == "lookahead":
        return LookaheadPolicy(genome)
    block = exemplar_block if arm.exemplars else ""
    if is_pi(arm.model):
        from tetris_agent.pi_policy import PiPolicy

        return PiPolicy(
            model=arm.model,
            harness=arm.harness,
            effort=arm.effort,
            exemplar_block=block,
            # Bounded pause: the deadline is the whole budget, kill at the mark.
            hard_deadline=arm.deadline_s is not None and not arm.live,
            genome=genome,
            fixed_effort=arm.fixed_effort,
        )
    from tetris_agent.model_policy import ModelPolicy

    return ModelPolicy(
        model=arm.model,
        harness=arm.harness,
        effort=arm.effort,
        exemplar_block=block,
        genome=genome,
        fixed_effort=arm.fixed_effort,
    )


def _arm_meta(arm: Arm, seed: int, max_pieces: int) -> dict:
    """The identity the viewer's banner renders: who is playing, under what rules."""
    if arm.live:
        mode = "live"
    elif arm.deadline_s is not None:
        mode = f"paused, {arm.deadline_s:g}s/decision"
    else:
        mode = "paused"
    return {
        "arm": arm.name,
        "model": arm.model or arm.policy,
        "harness": arm.harness,
        "effort": arm.effort,
        "mode": mode,
        "seed": seed,
        "max_pieces": max_pieces,
    }


def run_arm(
    arm: Arm,
    seed: int,
    rom_path,
    max_pieces: int,
    genome_params: dict | None = None,
    exemplar_block: str = "",
    level: int = 0,
    streamer=None,
    measure_power: bool = False,
) -> ArmResult:
    from tetris_agent.agent import TetrisAgent, _Tee
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.live import LiveEventSink
    from tetris_agent.live_agent import LiveTetrisAgent
    from tetris_agent.policy import Genome
    from tetris_agent.publisher import NoopPublisher

    policy = build_policy(arm, genome_params, exemplar_block)
    # Only local arms have an energy cost worth measuring — a cloud arm's draw is
    # someone else's datacenter, and its bill already shows up in cost_usd. A
    # fresh meter per run, so samples never carry across arms.
    meter = None
    if measure_power and arm.model and is_pi(arm.model):
        from tetris_agent.power import EnergyMeter

        meter = EnergyMeter()
    # Live arms need the wall-clock pacer. Paused arms run uncapped — unless a
    # viewer is watching, in which case real time is the point.
    paced = arm.live or streamer is not None
    emu = Emulator(rom_path, headless=True, speed=1) if paced else Emulator(rom_path)
    publisher = NoopPublisher() if streamer is None else _Tee(NoopPublisher(), LiveEventSink(streamer))
    frame_sinks = [] if streamer is None else [streamer.send_frame]
    agent = None
    try:
        agent_cls = LiveTetrisAgent if arm.live else TetrisAgent
        agent = agent_cls(
            emu,
            genome=Genome.from_params(genome_params or {}),
            collector=EventCollector(publisher),
            recorder=None,
            max_pieces=max_pieces,
            policy=policy,
            frame_sinks=frame_sinks,
            decision_deadline_s=arm.deadline_s,
            session_meta=_arm_meta(arm, seed, max_pieces),
            meter=meter,
        )
        if streamer is not None:
            emu.frame_hook = lambda: streamer.send_frame(agent.collector.turn, emu.screenshot())
        fitness = agent.run(timer_div=seed, level=level) if arm.live else agent.run(timer_div=seed)
    except Exception as exc:  # one bad arm must not kill the matrix
        logger.exception("arm %s seed %s failed", arm.name, seed)
        stats = {
            **policy.stats(),
            **getattr(agent, "live_stats", {}),
            **getattr(agent, "paused_stats", {}),
            **(meter.stats() if meter else {}),
        }
        return ArmResult(arm=arm.name, seed=seed, policy_stats=stats, error=repr(exc))
    finally:
        emu.stop()
    stats = {
        **policy.stats(),
        **getattr(agent, "live_stats", {}),
        **getattr(agent, "paused_stats", {}),
        **(meter.stats() if meter else {}),
    }
    return ArmResult(arm=arm.name, seed=seed, fitness=fitness, policy_stats=stats)


def estimate_cost(arms: list[Arm], seeds: list[int], max_pieces: int, tokens_per_decision: int = 2400) -> float:
    """Rough projection before spending anything: assumes a cached system
    prompt plus a fresh board per decision, and output roughly a tenth of input."""
    from tetris_agent.pricing import cost_usd

    total = 0.0
    for arm in arms:
        if arm.policy != "model":
            continue
        decisions = max_pieces * len(seeds)
        total += cost_usd(
            arm.model,
            input_tokens=tokens_per_decision * decisions,
            output_tokens=int(tokens_per_decision * 0.1) * decisions,
        )
    return round(total, 4)


def run_matrix(
    arms: list[Arm],
    seeds: list[int],
    rom_path,
    max_pieces: int = DEFAULT_MAX_PIECES,
    max_usd: float | None = None,
    runner=run_arm,
    on_result=None,
) -> list[ArmResult]:
    results: list[ArmResult] = []
    spent = 0.0
    for arm in arms:
        for seed in seeds:
            if max_usd is not None and spent >= max_usd:
                logger.warning("budget cap $%.2f reached; skipping the rest of the matrix", max_usd)
                return results
            result = runner(arm, seed, rom_path, max_pieces)
            spent += result.cost
            results.append(result)
            if on_result:
                on_result(result, spent)
    return results


def summarize(results: list[ArmResult]) -> list[dict]:
    """One row per arm, averaged across seeds, ranked by race score."""
    by_arm: dict[str, list[ArmResult]] = {}
    for r in results:
        by_arm.setdefault(r.arm, []).append(r)

    rows = []
    for arm, runs in by_arm.items():
        scored = [r for r in runs if r.fitness]
        n = max(len(scored), 1)
        latencies = [ms for r in runs for ms in r.policy_stats.get("decision_latencies_ms", [])]

        def pct_le(cap_ms: float) -> float:
            if not latencies:
                return 0.0
            return round(100 * sum(1 for ms in latencies if ms <= cap_ms) / len(latencies), 1)

        rows.append(
            {
                "arm": arm,
                "runs": len(runs),
                "errors": sum(1 for r in runs if r.error),
                "score": round(sum(r.fitness.get("score", 0) for r in scored) / n, 1),
                "lines": round(sum(r.fitness.get("lines", 0) for r in scored) / n, 2),
                "pieces": round(sum(r.fitness.get("pieces_placed", 0) for r in scored) / n, 1),
                "avg_holes": round(sum(r.fitness.get("avg_holes", 0) for r in scored) / n, 2),
                "topped_out": sum(1 for r in scored if r.fitness.get("topped_out")),
                "race_score": round(sum(race_score(r.fitness) for r in scored) / n, 1),
                "illegal": sum(r.policy_stats.get("illegal_count", 0) for r in runs),
                "late": sum(r.policy_stats.get("late", 0) for r in runs),
                "timeouts": sum(r.policy_stats.get("timeouts", 0) for r in runs),
                "pct_le_10s": pct_le(10_000),
                "pct_le_15s": pct_le(15_000),
                "latency_ms": round(sum(r.policy_stats.get("latency_ms_mean", 0) for r in runs) / max(len(runs), 1), 1),
                "tok_s": round(sum(r.policy_stats.get("tokens_per_second", 0) for r in runs) / max(len(runs), 1), 1),
                "cost_usd": round(sum(r.cost for r in runs), 4),
                **_energy_cells(runs),
            }
        )
    return sorted(rows, key=lambda r: r["race_score"], reverse=True)


def _energy_cells(runs: list[ArmResult]) -> dict:
    """Energy columns for one arm's rows, summed across seeds like cost_usd.

    An arm with nothing measured renders "n/a" rather than 0.0 — a cloud arm
    genuinely draws no local power, and an unmeasured local arm is unknown, but
    neither is "this run was free", which is the claim the zero used to make.
    """
    measured = [r.policy_stats.get("energy_wh") for r in runs]
    measured = [w for w in measured if w is not None]
    if not measured:
        return {"energy_wh": "n/a", "energy_usd": "n/a"}
    wh = sum(measured)
    return {"energy_wh": round(wh, 3), "energy_usd": round(energy_usd(wh), 4)}


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"
    cols = [
        "arm",
        "race_score",
        "score",
        "lines",
        "pieces",
        "avg_holes",
        "illegal",
        "late",
        "timeouts",
        "pct_le_10s",
        "pct_le_15s",
        "latency_ms",
        "tok_s",
        "cost_usd",
        "energy_wh",
        "energy_usd",
    ]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows)
    return f"{header}\n{sep}\n{body}"


def write_results(results: list[ArmResult], rows: list[dict], out_dir: Path = RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"benchmark-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "summary": rows,
                "runs": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    return path


def main(argv=None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="tetris-bench", description="Benchmark model x harness x effort")
    parser.add_argument("--models", nargs="+", default=["claude-opus-5"])
    parser.add_argument("--harnesses", nargs="+", default=["features"])
    parser.add_argument("--efforts", nargs="+", default=["medium"])
    parser.add_argument("--seeds", nargs="+", type=lambda s: int(s, 0), default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-pieces", type=int, default=DEFAULT_MAX_PIECES)
    parser.add_argument("--rom", default="rom/tetris.gb")
    parser.add_argument("--max-usd", type=float, default=5.0, help="abort the matrix once spend reaches this")
    parser.add_argument(
        "--no-power",
        action="store_true",
        help="skip host power sampling for local arms (their energy cost then reports n/a)",
    )
    parser.add_argument("--no-control", action="store_true", help="skip the heuristic control arm")
    parser.add_argument(
        "--lookahead-control",
        action="store_true",
        help="add the two-ply oracle as a ceiling arm (the arm that placement quality is graded against)",
    )
    parser.add_argument(
        "--paused",
        action="store_true",
        help="freeze the emulator during model calls (legacy mode, for A/B against historical rows)",
    )
    parser.add_argument(
        "--decision-deadline",
        type=float,
        default=None,
        metavar="SECONDS",
        help="bounded pause: freeze once per piece, but discard any decision slower than this "
        "and let the piece fall (implies --paused; arms are labeled +p<seconds>)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="stream every arm to the viewer's LIVE tab (uv run tetris-viewer) as it plays",
    )
    parser.add_argument("--viewer-url", default="ws://127.0.0.1:8000", help="viewer WebSocket base for --watch")
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=range(10),
        help="starting level/gravity for live arms (0 = slowest, ~15s per piece)",
    )
    parser.add_argument("--estimate", action="store_true", help="print a cost projection and exit")
    parser.add_argument("--skip-preflight", action="store_true", help="run pi arms without checking Ollama first")
    parser.add_argument(
        "--exemplars",
        nargs="?",
        const="runs",
        default=None,
        metavar="RUNS_DIR",
        help="inject verified human traces from RUNS_DIR (default runs/); model arms are labeled +ex",
    )
    parser.add_argument(
        "--fixed-effort",
        action="store_true",
        help="pin each arm to its configured effort (no deadline downshifts); arms are labeled +fixed",
    )
    args = parser.parse_args(argv)

    if args.decision_deadline is not None and not args.paused:
        # A bounded pause only means something when the game freezes to think.
        args.paused = True
        print(f"--decision-deadline {args.decision_deadline:g} implies --paused (bounded-pause mode)")

    if not args.skip_preflight and any(is_pi(m) for m in args.models):
        from tetris_agent.pi_policy import preflight

        problems = preflight(args.models)
        if problems:
            print("pi arms cannot run:")
            for problem in problems:
                print(f"  - {problem}")
            print("\n(--skip-preflight to try anyway)")
            return 1

    exemplar_block = ""
    if args.exemplars:
        from tetris_agent.traces import load_exemplar_block

        # Load before anything runs: a matrix that would silently drop its
        # exemplars mid-flight is worse than one that refuses to start.
        exemplar_block = load_exemplar_block(args.exemplars)

    arms = expand_arms(
        args.models,
        args.harnesses,
        args.efforts,
        include_control=not args.no_control,
        exemplars=bool(args.exemplars),
        live=not args.paused,
        deadline_s=args.decision_deadline,
        fixed_effort=args.fixed_effort,
        lookahead_control=args.lookahead_control,
    )
    projected = estimate_cost(arms, args.seeds, args.max_pieces)
    print(f"{len(arms)} arms x {len(args.seeds)} seed(s) x {args.max_pieces} pieces")
    print(f"arms: {', '.join(a.name for a in arms)}")
    print(f"projected cost: ~${projected:.2f} (cap ${args.max_usd:.2f})")
    live_runs = sum(1 for a in arms if a.live) * len(args.seeds)
    if live_runs:
        from tetris_agent.emulator import Emulator

        secs_per_piece = Emulator._GRAVITY_RELOADS[args.level] * 17 / 60
        minutes = live_runs * args.max_pieces * secs_per_piece / 60
        print(
            f"live arms run in real time: worst case ~{minutes:.0f} min wall-clock "
            f"(level {args.level}); --paused for the old fast mode"
        )
    if args.estimate:
        return 0

    def progress(result: ArmResult, spent: float) -> None:
        f = result.fitness
        status = result.error or f"score={f.get('score', 0)} pieces={f.get('pieces_placed', 0)}"
        print(f"  [{result.arm} seed={result.seed}] {status}  spent=${spent:.4f}")

    streamer = None
    if args.watch:
        from tetris_agent.live import LiveStreamer

        streamer = LiveStreamer(args.viewer_url)
        print(f"streaming arms to the viewer at {args.viewer_url} (LIVE tab)")

    def runner(arm, seed, rom_path, max_pieces):
        return run_arm(
            arm,
            seed,
            rom_path,
            max_pieces,
            exemplar_block=exemplar_block,
            level=args.level,
            streamer=streamer,
            measure_power=not args.no_power,
        )

    results = run_matrix(
        arms,
        args.seeds,
        args.rom,
        max_pieces=args.max_pieces,
        max_usd=args.max_usd,
        runner=runner,
        on_result=progress,
    )
    rows = summarize(results)
    print("\n" + render_table(rows))
    path = write_results(results, rows)
    print(f"\ntotal spend: ${sum(r.cost for r in results):.4f}")
    drawn = [r.policy_stats.get("energy_wh") for r in results]
    drawn = [w for w in drawn if w is not None]
    if drawn:
        print(
            f"local energy: {sum(drawn):.2f} Wh marginal over idle "
            f"= ${energy_usd(sum(drawn)):.4f} at ${DEFAULT_KWH_PRICE}/kWh"
        )
    elif not args.no_power:
        from tetris_agent.power import detect_source

        print(f"local energy: n/a ({detect_source()[1]})")
    print(f"results: {path}")
    return 0
