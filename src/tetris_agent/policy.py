"""Placement search weighted by the genome the healer tunes."""

from dataclasses import dataclass, fields

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
