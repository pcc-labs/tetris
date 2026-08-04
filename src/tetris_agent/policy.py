"""Placement policies: the heuristic control arm and the Policy contract.

A Policy is the benchmark's unit of comparison — the heuristic search here is
the control against which model-driven policies (model_policy.py) are scored.
"""

from dataclasses import dataclass, fields
from typing import Protocol

import numpy as np

from tetris_agent.board import drop, features, place
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


class HeuristicPolicy:
    """Control arm: weighted one-piece search. No model, no cost, no latency."""

    def __init__(self, genome: Genome | None = None):
        self.genome = genome or Genome()
        self.name = "heuristic"

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        return plan_placement(board, piece, self.genome)

    def stats(self) -> dict:
        return {"policy": self.name, "decisions": 0, "cost_usd": 0.0}
