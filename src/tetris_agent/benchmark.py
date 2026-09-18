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
from tetris_agent.recorder import RunRecorder

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/benchmarks")
RUNS_DIR = Path("runs")
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
    # Where this arm's recording landed, when --record kept one. The viewer
    # replays a race from these, so a tab that never watched it live — or was
    # simply reloaded since — can still find the frames.
    run_id: str = ""

    @property
    def cost(self) -> float:
        return float(self.policy_stats.get("cost_usd", 0.0))


# Non-model reference arms, weakest first. `no-input` is the floor (nobody
# plays), `random` is chance within the legal action space, `heuristic` is the
# tuned-solver ceiling. A model arm is only interesting above `random`.
BASELINE_POLICIES = ("no-input", "random", "heuristic")

# Baselines may also be named directly in `--models`, which is how a race gives
# one of them a lane: `--models pi/gemma4 pi/gpt-oss:20b heuristic` is a far
# better matchup than the all-or-nothing `--no-control` switch allows.
NAMEABLE_BASELINES = BASELINE_POLICIES + ("lookahead",)


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

    A baseline named in `models` contributes one arm in the position it was
    written, not one per (harness, effort): the harness is what reasoning is
    done for a *model*, and means nothing to a solver.
    """
    arms = [Arm(policy=p) for p in reversed(BASELINE_POLICIES)] if include_control else []
    seen = {a.name for a in arms}
    for model in models:
        if model in NAMEABLE_BASELINES:
            arm = Arm(policy=model)
            if arm.name not in seen:
                seen.add(arm.name)
                arms.append(arm)
            continue
        for harness, effort in itertools.product(harnesses, efforts):
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


def _arm_meta(
    arm: Arm,
    seed: int,
    max_pieces: int,
    run_id: str | None = None,
    lane: int | None = None,
    host: str | None = None,
) -> dict:
    """The identity the viewer's banner renders: who is playing, under what rules.

    `run_id` is how the RACE tab finds this lane's frames afterwards: the viewer
    only ever learns about a run over the wire, so the id has to ride along with
    the identity rather than being discovered from the directory later.
    """
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
        "run_id": run_id,
        # Which lane of a race this is, and `None` for anything that is not a
        # race. The RACE grid belongs to races alone: a lone live session —
        # tetris-play, or a bare `--live` agent — is the LIVE tab's business,
        # and must not tear down a race the tab is showing.
        "lane": lane,
        # Where inference ran: `local`, `daytona`, or the remote hostname.
        # `None` for a solver, which does its own thinking and reaches no host —
        # so a mixed race shows at a glance which lanes needed a GPU at all.
        "host": host,
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
    grade_quality: bool = True,
    record_frames: bool = False,
    lane: int | None = None,
) -> ArmResult:
    from tetris_agent.agent import TetrisAgent, _Tee
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.live import LiveEventSink
    from tetris_agent.live_agent import LiveTetrisAgent
    from tetris_agent.policy import Genome
    from tetris_agent.publisher import NoopPublisher

    policy = build_policy(arm, genome_params, exemplar_block)
    # Only a pi/ arm reaches an Ollama; a solver and a cloud-API model do not.
    host = None
    if arm.model and is_pi(arm.model):
        from tetris_agent.pi_policy import inference_host

        host = inference_host()
    # Only local arms have an energy cost worth measuring — a cloud arm's draw is
    # someone else's datacenter, and its bill already shows up in cost_usd. A
    # fresh meter per run, so samples never carry across arms.
    meter = None
    if measure_power and arm.model and is_pi(arm.model):
        from tetris_agent.pi_policy import is_remote_ollama
        from tetris_agent.power import EnergyMeter

        # A remote Ollama (a Daytona GPU host) draws its watts there; metering
        # this box would attribute the emulator's draw to the model.
        if not is_remote_ollama():
            meter = EnergyMeter()
    # Placement grading, and the trace it writes. Events only by default:
    # frames are the expensive part of a recording and grading needs none of
    # them. `record_frames` is what --record buys — the pixels a replay needs.
    grader = recorder = None
    if grade_quality:
        import functools

        from tetris_agent.quality import DEFAULT_PLY, grade

        grader = functools.partial(grade, genome=Genome.from_params(genome_params or {}), ply=DEFAULT_PLY)
    if grade_quality or record_frames:
        recorder = RunRecorder(RUNS_DIR, label=arm.name)
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
            recorder=recorder,
            max_pieces=max_pieces,
            policy=policy,
            frame_sinks=frame_sinks,
            decision_deadline_s=arm.deadline_s,
            session_meta=_arm_meta(
                arm, seed, max_pieces, run_id=getattr(recorder, "run_id", None), lane=lane, host=host
            ),
            meter=meter,
            grader=grader,
            record_frames=record_frames,
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
        return ArmResult(
            arm=arm.name, seed=seed, policy_stats=stats, error=repr(exc), run_id=getattr(recorder, "run_id", "")
        )
    finally:
        emu.stop()
    stats = {
        **policy.stats(),
        **getattr(agent, "live_stats", {}),
        **getattr(agent, "paused_stats", {}),
        **(meter.stats() if meter else {}),
    }
    return ArmResult(
        arm=arm.name, seed=seed, fitness=fitness, policy_stats=stats, run_id=getattr(recorder, "run_id", "")
    )


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


def run_race(
    arms: list[Arm],
    seed: int,
    rom_path,
    max_pieces: int = DEFAULT_MAX_PIECES,
    lanes: int = 4,
    runner=None,
) -> list[ArmResult]:
    """Every arm at once, on one seed: a head-to-head race.

    The matrix runs arms one after another, which is the only way to measure
    latency honestly — but it means watching a five-arm matrix is five
    consecutive real-time games. A race trades that measurement honesty for a
    comparison you can see: identical pieces, identical clock, four screens.

    Lane index is the arm's viewer slot, so `runner` is called as
    `(arm, seed, rom_path, max_pieces, slot)`. Results come back in arm order,
    not completion order, so the summary table is deterministic.

    Threads, not processes: each lane drives its own PyBoy, policy, recorder
    and streamer, and touches no shared mutable state. The emulators are paced
    with sleeps and the model calls are subprocess or socket waits, so four
    lanes spend nearly all their time off the GIL.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if runner is None:
        runner = run_arm
    if not arms:
        return []
    if len(arms) > lanes:
        raise ValueError(f"{len(arms)} arms will not fit in {lanes} lanes: {', '.join(a.name for a in arms)}")

    results: list[ArmResult | None] = [None] * len(arms)
    with ThreadPoolExecutor(max_workers=len(arms)) as pool:
        futures = {pool.submit(runner, arm, seed, rom_path, max_pieces, i): i for i, arm in enumerate(arms)}
        for future in as_completed(futures):
            lane = futures[future]
            try:
                results[lane] = future.result()
            except Exception as exc:  # one bad lane must not kill the race
                logger.exception("lane %d (%s) failed", lane, arms[lane].name)
                results[lane] = ArmResult(arm=arms[lane].name, seed=seed, error=repr(exc))
    return [r for r in results if r is not None]


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
                **_quality_cells(runs),
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


def _quality_cells(runs: list[ArmResult]) -> dict:
    """Placement-quality columns for one arm's rows.

    Weighted by each seed's graded-decision count, so the figure is a mean over
    decisions rather than a mean of means. `n/a` rather than 0.0 when nothing was
    graded: quality off and every-decision-late are unknown, not perfect.
    """
    scored = [r for r in runs if r.fitness]
    total = sum(r.fitness.get("graded_decisions") or 0 for r in scored)
    if not total:
        return {"regret": "n/a", "top1": "n/a", "top3": "n/a", "graded": 0}

    def weighted(key: str) -> float:
        return sum((r.fitness.get(key) or 0) * (r.fitness.get("graded_decisions") or 0) for r in scored) / total

    return {
        "regret": round(weighted("mean_regret"), 4),
        "top1": round(weighted("top1_rate"), 3),
        "top3": round(weighted("top3_rate"), 3),
        "graded": total,
    }


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
        "regret",
        "top1",
        "top3",
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


def write_results(
    results: list[ArmResult],
    rows: list[dict],
    out_dir: Path = RESULTS_DIR,
    meta: dict | None = None,
) -> Path:
    """`meta` records how the matrix was run. A race's latency-derived columns
    are not comparable to serially-measured ones, and the file is otherwise
    indistinguishable from a serial one — so `{"race": True, "lanes": n}` rides
    along and the viewer's leaderboard says so."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"benchmark-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                **(meta or {}),
                "summary": rows,
                "runs": [asdict(r) for r in results],
            },
            indent=2,
        )
    )
    return path


def _apply_race_mode(args) -> int:
    """Validate --race and force the settings a shared clock makes mandatory.

    Lanes contend for cores and for one Ollama, so wall-clock latency in a race
    is partly the scheduler's doing. Three knobs read that inflated clock and
    would quietly change what is being measured; a race pins all three rather
    than letting a row describe the scheduler. Returns a nonzero exit code to
    abort, 0 to continue.
    """
    if args.paused or args.decision_deadline is not None:
        # Not refused, because on hardware where no model beats level-0 gravity
        # a live race is three boards filling with garbage — no demo at all.
        # The cost is that each lane freezes on its own model's clock, so the
        # lanes drift apart: same pieces, no longer the same moment.
        print(
            "--race with a bounded pause: each lane freezes while its own model thinks, so\n"
            "the lanes drift out of step — same pieces, not the same clock. Decisions slower\n"
            "than the deadline are still discarded, and under contention some of those\n"
            "discards are the scheduler's rather than the model's."
        )
    if len(args.seeds) != 1:
        print(f"--race is one piece sequence for everyone; got {len(args.seeds)} seeds. Pass a single --seeds value.")
        return 1
    if not args.no_control:
        args.no_control = True
        print("--race implies --no-control (name a baseline in --models to give it a lane)")
    if not args.fixed_effort:
        args.fixed_effort = True
        print(
            "--race implies --fixed-effort (the effort ladder steps down on observed latency,\n"
            "so contention would land an arm below the tier its row advertises)"
        )
    if not args.no_power:
        args.no_power = True
        print("--race implies --no-power (one host meter cannot attribute watts to one lane)")
    return 0


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
    parser.add_argument(
        "--race",
        action="store_true",
        help="run every arm at once on one seed — same pieces, same clock, one screen each "
        "in the viewer's RACE tab (pair with --watch)",
    )
    parser.add_argument(
        "--lanes",
        type=int,
        default=4,
        metavar="N",
        help="how many arms --race runs side by side (default 4)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="keep each arm's frames in runs/, so the viewer can replay it afterwards "
        "(off by default — frames are the expensive part of a recording)",
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
    parser.add_argument(
        "--no-quality",
        action="store_true",
        help="skip placement grading (no regret columns, no run traces)",
    )
    args = parser.parse_args(argv)

    if args.decision_deadline is not None and not args.paused:
        # A bounded pause only means something when the game freezes to think.
        args.paused = True
        print(f"--decision-deadline {args.decision_deadline:g} implies --paused (bounded-pause mode)")

    if args.race and (rc := _apply_race_mode(args)):
        return rc

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
    if args.race and len(arms) > args.lanes:
        print(f"--race has {args.lanes} lanes but this is {len(arms)} arms:")
        for arm in arms:
            print(f"  - {arm.name}")
        print("\nTrim --models / --harnesses / --efforts, or raise --lanes.")
        return 1

    projected = estimate_cost(arms, args.seeds, args.max_pieces)
    print(f"{len(arms)} arms x {len(args.seeds)} seed(s) x {args.max_pieces} pieces")
    print(f"arms: {', '.join(a.name for a in arms)}")
    # Say which box is answering, before anything runs. The same arm id means
    # something different on a laptop than on a rented H100, and "why is this
    # slow" is nearly always this line.
    pi_arms = [a for a in arms if a.model and is_pi(a.model)]
    inference = None
    if pi_arms:
        from tetris_agent.pi_policy import OLLAMA_URL, inference_host

        inference = inference_host()
        solvers = len(arms) - len(pi_arms)
        mix = f", plus {solvers} solver arm(s) reaching no host" if solvers else ""
        print(f"inference: {inference} ({OLLAMA_URL}) for {len(pi_arms)} pi/ arm(s){mix}")
    print(f"projected cost: ~${projected:.2f} (cap ${args.max_usd:.2f})")
    live_runs = sum(1 for a in arms if a.live) * len(args.seeds)
    if live_runs:
        from tetris_agent.emulator import Emulator

        secs_per_piece = Emulator._GRAVITY_RELOADS[args.level] * 17 / 60
        # A race overlaps its lanes, so the wall clock is one arm's game, not
        # the sum of them — that overlap is most of the point.
        concurrent = live_runs if not args.race else 1
        minutes = concurrent * args.max_pieces * secs_per_piece / 60
        print(
            f"live arms run in real time: worst case ~{minutes:.0f} min wall-clock "
            f"(level {args.level}); --paused for the old fast mode"
        )
    if args.estimate:
        return 0

    if args.race:
        print(
            "\nrace mode: lanes share cores and one Ollama, so latency_ms / late / timeouts /\n"
            "tok_s describe the contended clock. They are not comparable to serial rows;\n"
            "score, lines, pieces and holes are. The results file records race + lanes.\n"
        )
        if args.max_usd is not None and projected > args.max_usd:
            print(
                f"projected ${projected:.2f} is over the ${args.max_usd:.2f} cap. A race starts every arm\n"
                "at once, so there is no mid-matrix abort to fall back on — raise --max-usd or trim the race."
            )
            return 1

    def progress(result: ArmResult, spent: float) -> None:
        f = result.fitness
        status = result.error or f"score={f.get('score', 0)} pieces={f.get('pieces_placed', 0)}"
        print(f"  [{result.arm} seed={result.seed}] {status}  spent=${spent:.4f}")

    # One streamer per lane: a websockets.sync connection is not safe for
    # concurrent sends, and the slot is what tells the browser which screen a
    # frame belongs to. A serial matrix keeps its single slot-0 streamer.
    streamers = []
    if args.watch:
        from tetris_agent.live import LiveStreamer

        lanes = len(arms) if args.race else 1
        streamers = [LiveStreamer(args.viewer_url, slot=i) for i in range(lanes)]
        tab = "RACE" if args.race else "LIVE"
        print(f"streaming arms to the viewer at {args.viewer_url} ({tab} tab)")
        if args.race and not args.record:
            print("  (--record to keep the frames, so the RACE tab can replay this afterwards)")

    def runner(arm, seed, rom_path, max_pieces, slot=0):
        return run_arm(
            arm,
            seed,
            rom_path,
            max_pieces,
            exemplar_block=exemplar_block,
            level=args.level,
            streamer=streamers[slot] if streamers else None,
            measure_power=not args.no_power,
            grade_quality=not args.no_quality,
            record_frames=args.record,
            lane=slot if args.race else None,
        )

    try:
        if args.race:
            results = run_race(
                arms,
                args.seeds[0],
                args.rom,
                max_pieces=args.max_pieces,
                lanes=args.lanes,
                runner=runner,
            )
            for result in results:
                progress(result, sum(r.cost for r in results))
        else:
            results = run_matrix(
                arms,
                args.seeds,
                args.rom,
                max_pieces=args.max_pieces,
                max_usd=args.max_usd,
                runner=lambda a, s, r, m: runner(a, s, r, m),
                on_result=progress,
            )
    finally:
        for streamer in streamers:
            streamer.close()
    rows = summarize(results)
    print("\n" + render_table(rows))
    meta = {"race": True, "lanes": args.lanes} if args.race else {}
    if inference:
        meta["inference_host"] = inference
    path = write_results(results, rows, meta=meta or None)
    print(f"\ntotal spend: ${sum(r.cost for r in results):.4f}")
    drawn = [r.policy_stats.get("energy_wh") for r in results]
    drawn = [w for w in drawn if w is not None]
    if drawn:
        print(
            f"local energy: {sum(drawn):.2f} Wh marginal over idle "
            f"= ${energy_usd(sum(drawn)):.4f} at ${DEFAULT_KWH_PRICE}/kWh"
        )
    elif not args.no_power:
        from tetris_agent.pi_policy import OLLAMA_URL, is_remote_ollama
        from tetris_agent.power import detect_source

        why = f"inference ran on {OLLAMA_URL}, not this box" if is_remote_ollama() else detect_source()[1]
        print(f"local energy: n/a ({why})")
    print(f"results: {path}")
    return 0
