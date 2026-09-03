"""Situation corpus: free boards from the local policies, and the class histogram over them.

No trace with per-decision boards exists on disk (data/benchmarks/ holds only
per-run aggregates), so the corpus is generated: HeuristicPolicy and
RandomPolicy cost nothing, and a recorded run replays into verified boards
through traces.mine_run. Heuristic play gives well-formed boards, random play
gives messy ones — together they bracket what a struggling model arm faces,
without being an LLM's distribution. This validates the classifier's coverage
and its priority conflicts, not its effect on play.

    uv run tetris-situations generate --seeds 0 1 2 3 --max-pieces 150
    uv run tetris-situations histogram            # exits 2 if one class dominates
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from tetris_agent.situation import Kind, classify
from tetris_agent.traces import mine_run

logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/situations")
POLICIES = ("heuristic", "random")
# Spec section 06: a class holding this share of boards means there is nothing to route.
DOMINANT = 0.9


def generate(
    rom_path,
    out_dir,
    policies: tuple[str, ...] = POLICIES,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    max_pieces: int = 150,
) -> list[Path]:
    """Record one paused, headless run per (policy, seed) under out_dir; return the run dirs."""
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Genome, HeuristicPolicy, RandomPolicy
    from tetris_agent.publisher import NoopPublisher
    from tetris_agent.recorder import RunRecorder

    out_dir = Path(out_dir)
    runs = []
    for name in policies:
        for seed in seeds:
            policy = HeuristicPolicy() if name == "heuristic" else RandomPolicy(seed=seed)
            recorder = RunRecorder(out_dir, label=f"{name}-seed{seed}")
            emu = Emulator(rom_path)
            try:
                agent = TetrisAgent(
                    emu,
                    genome=Genome(),
                    collector=EventCollector(NoopPublisher()),
                    recorder=recorder,
                    max_pieces=max_pieces,
                    policy=policy,
                )
                fitness = agent.run(timer_div=seed)
            finally:
                emu.stop()
            logger.info("%s seed %d: %d pieces", name, seed, fitness.get("pieces_placed", 0))
            runs.append(out_dir / recorder.run_id)
    return runs


def histogram(runs_dir, genome=None) -> dict[str, Counter]:
    """{policy: Counter{kind: n}} over every verified board replayed from runs_dir."""
    run_dirs = [p for p in sorted(Path(runs_dir).iterdir()) if p.is_dir()]
    counts: dict[str, Counter] = {}
    for name in POLICIES:
        c: Counter = Counter()
        for run_dir in run_dirs:
            for ex in mine_run(run_dir, policies=(name,)):
                c[classify(ex.board, ex.piece, ex.next_piece, genome).kind.value] += 1
        counts[name] = c
    return counts


def dominant(counts: dict[str, Counter], threshold: float = DOMINANT) -> str | None:
    """The kind holding at least `threshold` of all boards across policies, or None."""
    total: Counter = Counter()
    for c in counts.values():
        total.update(c)
    n = sum(total.values())
    if not n:
        return None
    kind, top = total.most_common(1)[0]
    return kind if top / n >= threshold else None


def render(counts: dict[str, Counter]) -> str:
    """Markdown table: kinds in priority order as rows, policies plus total and share as columns."""
    names = [p for p in POLICIES if p in counts] or list(counts)
    grand = sum(sum(counts[p].values()) for p in names)
    header = "| kind | " + " | ".join(names) + " | total | share |"
    sep = "|---|" + "---|" * (len(names) + 2)
    rows = []
    for kind in Kind:
        per = [counts[p].get(kind.value, 0) for p in names]
        total = sum(per)
        share = 100 * total / grand if grand else 0.0
        rows.append(f"| {kind.value} | " + " | ".join(str(x) for x in per) + f" | {total} | {share:.1f}% |")
    totals = [sum(counts[p].values()) for p in names]
    rows.append("| total | " + " | ".join(str(x) for x in totals) + f" | {grand} | {100.0 if grand else 0.0:.1f}% |")
    return "\n".join([header, sep, *rows])


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="tetris-situations", description="Board-situation corpus and histogram")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="record free heuristic/random runs to replay boards from")
    g.add_argument("--rom", default="rom/tetris.gb")
    g.add_argument("--out", default=str(CORPUS_DIR))
    g.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    g.add_argument("--seeds", nargs="+", type=lambda s: int(s, 0), default=[0, 1, 2, 3])
    g.add_argument("--max-pieces", type=int, default=150)
    h = sub.add_parser("histogram", help="classify every replayed board and tabulate by kind")
    h.add_argument("runs_dir", nargs="?", default=str(CORPUS_DIR))
    args = ap.parse_args(argv)

    if args.cmd == "generate":
        for run in generate(args.rom, args.out, tuple(args.policies), tuple(args.seeds), args.max_pieces):
            print(run)
        return 0

    counts = histogram(args.runs_dir)
    print(render(counts))
    out = Path(args.runs_dir) / "histogram.json"
    out.write_text(json.dumps({k: dict(v) for k, v in counts.items()}, indent=2))
    print(f"\nwritten: {out}")
    top = dominant(counts)
    if top:
        print(f"\nSTOP: {top} holds >= {DOMINANT:.0%} of boards — the table has nothing to route (spec section 06).")
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
