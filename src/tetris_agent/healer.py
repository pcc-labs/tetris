"""Between-run healing: fitness → rule → deterministic parameter race → genome.

Reads only the fitness JSON a run wrote (broker-free by design). A rule maps
bad fitness to the genome params it implicates; a seeded race of jittered
variants against the current genome decides — with a margin — whether to
adopt. Repeated failures escalate to the discovery queue for an LLM attempt.
check() never raises: healing must not break the run that triggered it.
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from tetris_agent import evolve, genome
from tetris_agent.evolve import decide, run_race, sample_variants

logger = logging.getLogger(__name__)

COOLDOWN_HOURS = 6
STATE_PATH = Path("data/healer_state.json")
QUEUE_PATH = Path("data/discovery_queue.json")


@dataclass(frozen=True)
class Rule:
    name: str
    params: tuple[str, ...]
    trigger: Callable[[dict], bool]


RULES = [
    Rule(
        "early-topout",
        ("w_lines", "w_agg_height", "w_holes", "w_bumpiness"),
        lambda f: bool(f.get("topped_out")) and f.get("pieces_placed", 0) < 40,
    ),
    Rule(
        "hole-thrash",
        ("w_holes", "w_bumpiness"),
        lambda f: f.get("avg_holes", 0) >= 4,
    ),
    Rule(
        "misexecution",
        ("ticks_per_press",),
        lambda f: f.get("max_misexec_streak", 0) >= 3 or f.get("misexec_count", 0) >= 10,
    ),
]


def evaluate_rules(fitness: dict) -> list[Rule]:
    return [rule for rule in RULES if rule.trigger(fitness)]


def default_seed(fitness_path: str | Path) -> int:
    return int(hashlib.sha256(str(fitness_path).encode()).hexdigest()[:8], 16)


def should_escalate(rule_name: str, state: dict) -> bool:
    """Pre-race check: the same rule firing after an accepted fix means tuning
    already had its shot — the capability itself is lacking."""
    return state.get("rules", {}).get(rule_name, {}).get("last_outcome") == "accepted"


def append_escalation(rule: Rule, fitness: dict, race_result: dict, path: str | Path = QUEUE_PATH) -> None:
    path = Path(path)
    queue = []
    if path.exists():
        try:
            queue = json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("discovery queue %s is corrupt; rewriting", path)
    queue.append(
        {
            "rule": rule.name,
            "fitness": fitness,
            "race": {"control": race_result["control"], "winner_score": race_result["winner"]["score"]},
            "queued_at": datetime.now(timezone.utc).isoformat(),
            "attempted": False,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(queue, indent=2))


def _load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("healer state %s is corrupt; resetting", path)
    return {"rules": {}, "races": []}


def _save_state(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


def _in_cooldown(rule_state: dict) -> bool:
    last = rule_state.get("last_race_at")
    if not last:
        return False
    return datetime.now(timezone.utc) - datetime.fromisoformat(last) < timedelta(hours=COOLDOWN_HOURS)


def check(
    fitness_path: str | Path,
    rom_path: str | Path = "rom/tetris.gb",
    state_path: str | Path = STATE_PATH,
    genome_path: str | Path = genome.GENOME_PATH,
    queue_path: str | Path = QUEUE_PATH,
    n_variants: int = 3,
    seeds_per: int = 2,
    max_pieces: int = evolve.DEFAULT_MAX_PIECES,
    race=run_race,
) -> dict:
    try:
        return _check(
            Path(fitness_path), rom_path, Path(state_path), genome_path, queue_path, n_variants, seeds_per,
            max_pieces, race,
        )
    except Exception:
        logger.exception("healer check failed")
        return {"status": "error"}


def _check(fitness_path, rom_path, state_path, genome_path, queue_path, n_variants, seeds_per, max_pieces, race):
    fitness = json.loads(fitness_path.read_text())
    fired = evaluate_rules(fitness)
    if not fired:
        return {"status": "no-rule"}
    rule = fired[0]

    state = _load_state(state_path)
    rule_state = state["rules"].setdefault(rule.name, {})
    if _in_cooldown(rule_state):
        return {"status": "cooldown", "rule": rule.name}

    escalate = should_escalate(rule.name, state)

    base = genome.load_params(genome_path)
    seed = default_seed(fitness_path)
    rng = random.Random(seed)
    variants = sample_variants(base, list(rule.params), n_variants, rng)
    seeds = [(seed + i * 0x11) & 0xFF for i in range(seeds_per)]

    def runner(params, timer_div, rom_path, max_pieces=max_pieces):
        return evolve.run_agent(params, timer_div=timer_div, rom_path=rom_path, max_pieces=max_pieces)

    result = race(base, variants, seeds, rom_path, runner=runner)
    accepted = decide(result["control"], result["winner"]["score"])

    if accepted:
        genome.append_genome(result["winner"]["params"], source=f"healer:{rule.name}", path=genome_path)
        rule_state["consecutive_rejections"] = 0
        rule_state["last_outcome"] = "accepted"
    else:
        rule_state["consecutive_rejections"] = rule_state.get("consecutive_rejections", 0) + 1
        rule_state["last_outcome"] = "rejected"
        if rule_state["consecutive_rejections"] >= 2:
            escalate = True
    rule_state["last_race_at"] = datetime.now(timezone.utc).isoformat()
    state["races"] = (
        state.get("races", [])
        + [{"rule": rule.name, "control": result["control"], "winner": result["winner"], "accepted": accepted}]
    )[-20:]
    _save_state(state, state_path)

    if escalate:
        append_escalation(rule, fitness, result, path=queue_path)
        logger.info("escalated %s to discovery queue", rule.name)

    status = "accepted" if accepted else "rejected"
    return {
        "status": status,
        "rule": rule.name,
        "control": result["control"],
        "winner_score": result["winner"]["score"],
        "escalated": escalate,
    }


def main(argv=None) -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="tetris-healer")
    sub = parser.add_subparsers(dest="command", required=True)
    check_parser = sub.add_parser("check", help="evaluate a fitness file and race if a rule fires")
    check_parser.add_argument("fitness_path")
    check_parser.add_argument("--rom", default="rom/tetris.gb")
    check_parser.add_argument("--variants", type=int, default=3)
    check_parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args(argv)

    report = check(args.fitness_path, rom_path=args.rom, n_variants=args.variants, seeds_per=args.seeds)
    print(json.dumps(report, indent=2))
    return 0
