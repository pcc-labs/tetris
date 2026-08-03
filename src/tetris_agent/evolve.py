"""Measured parameter races — the healer's decision engine.

Everything runs in-process (pokemon-kafka shelled out per candidate); races
can't recurse into self-healing because run_agent never touches the healer.
"""

import random

from tetris_agent.fitness import race_score

DEFAULT_MAX_PIECES = 150


def run_agent(params: dict, timer_div: int, rom_path, max_pieces: int = DEFAULT_MAX_PIECES) -> dict:
    """One deterministic, telemetry-free, non-recorded run; returns fitness."""
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Genome
    from tetris_agent.publisher import NoopPublisher

    emu = Emulator(rom_path)
    try:
        agent = TetrisAgent(
            emu,
            genome=Genome.from_params(params),
            collector=EventCollector(NoopPublisher()),
            recorder=None,
            max_pieces=max_pieces,
        )
        return agent.run(timer_div=timer_div)
    finally:
        emu.stop()


def sample_variants(base: dict, implicated: list[str], n: int, rng: random.Random) -> list[dict]:
    """Gaussian jitter on the implicated params only; ints stay positive ints."""
    variants = []
    for _ in range(n):
        variant = dict(base)
        for key in implicated:
            value = base[key]
            if isinstance(value, int):
                variant[key] = max(1, value + rng.choice([-1, 1]))
            else:
                sigma = max(0.05, 0.3 * abs(value))
                variant[key] = round(value + rng.gauss(0.0, sigma), 6)
        variants.append(variant)
    return variants


def run_race(control: dict, variants: list[dict], seeds: list[int], rom_path, runner=run_agent) -> dict:
    """Every candidate plays every seed; mean race_score decides."""

    def mean_score(params: dict) -> float:
        return sum(race_score(runner(params, timer_div=seed, rom_path=rom_path)) for seed in seeds) / len(seeds)

    control_score = mean_score(control)
    candidates = [{"params": params, "score": mean_score(params)} for params in variants]
    winner = max(candidates, key=lambda c: c["score"])
    return {"control": control_score, "candidates": candidates, "winner": winner}


def decide(control_score: float, winner_score: float, margin: float = 0.05) -> bool:
    """Adopt only on a clear win: 5% relative, or +10 absolute when control is tiny."""
    if control_score <= 0:
        return winner_score > control_score + 10.0
    return winner_score > control_score * (1.0 + margin)
