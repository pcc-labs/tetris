"""Placement policies: the heuristic control arm and the Policy contract.

A Policy is the benchmark's unit of comparison — the heuristic search here is
the control against which model-driven policies (model_policy.py) are scored.
"""

import random
from dataclasses import dataclass, fields
from typing import Protocol

import numpy as np

from tetris_agent.board import COLS, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width


@dataclass(frozen=True)
class Genome:
    # Evaluation weights (El-Tetris defaults); healed between runs.
    w_lines: float = 0.760666
    w_agg_height: float = -0.510066
    w_holes: float = -0.35663
    w_bumpiness: float = -0.184483
    # Controller timing; healed when the misexecution rule fires.
    ticks_per_press: int = 4
    # Situation thresholds (situation.py) for the routed harness; tuned like any other knob.
    topping_out_height: int = 12  # max column height that flips play into survival mode
    mound_rise: int = 3  # highest column minus lowest that counts as a mound
    well_depth: int = 4  # rows a one-column well must span to be Tetris-ready

    def to_params(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_params(cls, params: dict) -> "Genome":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in known})


@dataclass(frozen=True)
class Placement:
    rotation: int
    col: int
    score: float


def plan_placement(board: np.ndarray, piece: str, genome: Genome) -> Placement | None:
    """Best (rotation, col) by simulating every drop; None if nothing fits."""
    best: Placement | None = None
    best_key: tuple | None = None
    for rot in distinct_rotations(piece):
        shape = SHAPES[(piece, rot)]
        for col in range(10 - shape_width(shape) + 1):
            row = drop(board, shape, col)
            if row is None:
                continue
            new_board, lines = place(board, shape, col, row)
            f = features(new_board)
            score = (
                genome.w_lines * lines
                + genome.w_agg_height * f.agg_height
                + genome.w_holes * f.holes
                + genome.w_bumpiness * f.bumpiness
            )
            # Tie-break: lower landing (higher row index), then centermost column.
            key = (score, row, -abs(col - 4))
            if best_key is None or key > best_key:
                best_key = key
                best = Placement(rotation=rot, col=col, score=score)
    return best


class Policy(Protocol):
    """Chooses where each piece lands. The benchmark's unit of comparison."""

    name: str

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None: ...

    def stats(self) -> dict: ...


def legal_moves(board: np.ndarray, piece: str) -> list[tuple[int, int]]:
    """Every (rotation, col) the piece can be dropped into — the action space."""
    out = []
    for rot in distinct_rotations(piece):
        shape = SHAPES[(piece, rot)]
        for col in range(COLS - shape_width(shape) + 1):
            if drop(board, shape, col) is not None:
                out.append((rot, col))
    return out


class HeuristicPolicy:
    """Ceiling arm: weighted one-piece search. No model, no cost, no latency.

    Not a control — it is a tuned solver that evaluates every option exactly.
    It answers "how well can this be played", not "what does no skill look
    like". For that, see NoInputPolicy and RandomPolicy below.
    """

    def __init__(self, genome: Genome | None = None):
        self.genome = genome or Genome()
        self.name = "heuristic"

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        return plan_placement(board, piece, self.genome)

    def stats(self) -> dict:
        return {"policy": self.name, "decisions": 0, "cost_usd": 0.0}


class NoInputPolicy:
    """Floor arm: nobody plays. Every piece lands where it spawned.

    Game Boy Tetris spawns each piece at rotation 0 with its leftmost cell in
    column 3 — column 4 for O (measured against the ROM). Returning the spawn
    position means the controller presses nothing, so the piece drops straight
    down and the stack builds in the middle until it tops out. This is the
    score you get for letting the game run, and every other arm has to beat it
    before its decisions can be said to have done anything.
    """

    SPAWN_COL = {"O": 4}
    SPAWN_COL_DEFAULT = 3

    def __init__(self):
        self.name = "no-input"

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        return Placement(rotation=0, col=self.SPAWN_COL.get(piece, self.SPAWN_COL_DEFAULT), score=0.0)

    def stats(self) -> dict:
        return {"policy": self.name, "decisions": 0, "cost_usd": 0.0}


class RandomPolicy:
    """Chance arm: a uniformly random spot the piece actually fits.

    The same action space the model arms choose from — every pick is legal —
    but with no judgement about which one is good. This is the bar a model's
    "best guess at where the piece fits" has to clear to carry any information
    at all; beating the heuristic is a much later question.

    Seeded so a rerun on the same piece sequence reproduces exactly.
    """

    def __init__(self, seed: int = 0):
        self.name = "random"
        self._rng = random.Random(seed)

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        options = legal_moves(board, piece)
        if not options:
            return None
        rotation, col = self._rng.choice(options)
        return Placement(rotation=rotation, col=col, score=0.0)

    def stats(self) -> dict:
        return {"policy": self.name, "decisions": 0, "cost_usd": 0.0}
