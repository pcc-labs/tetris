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
from tetris_agent.pricing import is_pi, spec

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

    @property
    def name(self) -> str:
        if self.policy == "heuristic":
            return "heuristic"
        parts = [self.model, self.harness]
        if self.effort:
            parts.append(self.effort)
        return "/".join(parts)


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


def expand_arms(models: list[str], harnesses: list[str], efforts: list[str], include_control: bool = True) -> list[Arm]:
    """Cartesian product, minus combinations the API rejects.

    Haiku 4.5 does not accept output_config.effort, so it contributes one arm
    per harness instead of one per (harness, effort).
    """
    arms = [Arm(policy="heuristic")] if include_control else []
    seen = set()
    for model, harness, effort in itertools.product(models, harnesses, efforts):
        eff = effort if spec(model).supports_effort else None
        arm = Arm(policy="model", model=model, harness=harness, effort=eff)
        if arm.name not in seen:
            seen.add(arm.name)
            arms.append(arm)
    return arms


def build_policy(arm: Arm, genome_params: dict | None = None):
    from tetris_agent.policy import Genome, HeuristicPolicy

    if arm.policy == "heuristic":
        return HeuristicPolicy(Genome.from_params(genome_params or {}))
    if is_pi(arm.model):
        from tetris_agent.pi_policy import PiPolicy

        return PiPolicy(model=arm.model, harness=arm.harness, effort=arm.effort)
    from tetris_agent.model_policy import ModelPolicy

    return ModelPolicy(model=arm.model, harness=arm.harness, effort=arm.effort)


def run_arm(arm: Arm, seed: int, rom_path, max_pieces: int, genome_params: dict | None = None) -> ArmResult:
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Genome
    from tetris_agent.publisher import NoopPublisher

    policy = build_policy(arm, genome_params)
    emu = Emulator(rom_path)
    try:
        agent = TetrisAgent(
            emu,
            genome=Genome.from_params(genome_params or {}),
            collector=EventCollector(NoopPublisher()),
            recorder=None,
            max_pieces=max_pieces,
            policy=policy,
        )
        fitness = agent.run(timer_div=seed)
    except Exception as exc:  # one bad arm must not kill the matrix
        logger.exception("arm %s seed %s failed", arm.name, seed)
        return ArmResult(arm=arm.name, seed=seed, policy_stats=policy.stats(), error=repr(exc))
    finally:
        emu.stop()
    return ArmResult(arm=arm.name, seed=seed, fitness=fitness, policy_stats=policy.stats())


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
                "latency_ms": round(
                    sum(r.policy_stats.get("latency_ms_mean", 0) for r in runs) / max(len(runs), 1), 1
                ),
                "cost_usd": round(sum(r.cost for r in runs), 4),
            }
        )
    return sorted(rows, key=lambda r: r["race_score"], reverse=True)


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "(no results)"
    cols = ["arm", "race_score", "score", "lines", "pieces", "avg_holes", "illegal", "latency_ms", "cost_usd"]
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
    parser.add_argument("--no-control", action="store_true", help="skip the heuristic control arm")
    parser.add_argument("--estimate", action="store_true", help="print a cost projection and exit")
    args = parser.parse_args(argv)

    arms = expand_arms(args.models, args.harnesses, args.efforts, include_control=not args.no_control)
    projected = estimate_cost(arms, args.seeds, args.max_pieces)
    print(f"{len(arms)} arms x {len(args.seeds)} seed(s) x {args.max_pieces} pieces")
    print(f"arms: {', '.join(a.name for a in arms)}")
    print(f"projected cost: ~${projected:.2f} (cap ${args.max_usd:.2f})")
    if args.estimate:
        return 0

    def progress(result: ArmResult, spent: float) -> None:
        f = result.fitness
        status = result.error or f"score={f.get('score', 0)} pieces={f.get('pieces_placed', 0)}"
        print(f"  [{result.arm} seed={result.seed}] {status}  spent=${spent:.4f}")

    results = run_matrix(
        arms, args.seeds, args.rom, max_pieces=args.max_pieces, max_usd=args.max_usd, on_result=progress
    )
    rows = summarize(results)
    print("\n" + render_table(rows))
    path = write_results(results, rows)
    print(f"\ntotal spend: ${sum(r.cost for r in results):.4f}")
    print(f"results: {path}")
    return 0
