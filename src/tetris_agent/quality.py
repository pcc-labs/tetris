"""Grade a placement against a lookahead oracle.

The benchmark scores outcomes; this scores decisions. `rank_placements` ranks
every legal placement for the current piece by the same weighted evaluation the
heuristic control arm plays with, searched `ply` pieces deep. `grade` turns a
model's choice into a regret and a rank against that ranking.

The oracle is a strong heuristic, not ground truth: regret is disagreement with
a two-ply evaluator, so a model that outplays the oracle would show high regret.
The `lookahead` arm exists to put the oracle's own score in the same table.
"""

from dataclasses import dataclass

import numpy as np

from tetris_agent.board import COLS, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width
from tetris_agent.policy import Genome, Placement

DEFAULT_PLY = 2


def _value(board: np.ndarray, lines: int, genome: Genome) -> float:
    """The control arm's evaluation, so the oracle and the solver agree at ply=1."""
    f = features(board)
    return (
        genome.w_lines * lines
        + genome.w_agg_height * f.agg_height
        + genome.w_holes * f.holes
        + genome.w_bumpiness * f.bumpiness
    )


def _drops(board: np.ndarray, piece: str):
    """Every legal (rotation, col, landing row, shape) for `piece` on `board`."""
    for rot in distinct_rotations(piece):
        shape = SHAPES[(piece, rot)]
        for col in range(COLS - shape_width(shape) + 1):
            row = drop(board, shape, col)
            if row is not None:
                yield rot, col, row, shape


def rank_placements(
    board: np.ndarray,
    piece: str,
    next_piece: str | None = None,
    genome: Genome | None = None,
    ply: int = DEFAULT_PLY,
) -> list[tuple[tuple[int, int], float]]:
    """[((rotation, col), value)], best first.

    At ply=1 this is `policy.plan_placement`'s ranking, tie-break included. At
    ply=2 each placement is worth the best reply the known next piece has to it;
    a placement that leaves the next piece nowhere to go keeps its one-ply value
    rather than scoring as unboundedly bad.
    """
    genome = genome or Genome()
    scored = []
    for rot, col, row, shape in _drops(board, piece):
        after, lines = place(board, shape, col, row)
        value = _value(after, lines, genome)
        if ply > 1 and next_piece is not None:
            best_reply = None
            for _, col2, row2, shape2 in _drops(after, next_piece):
                board2, lines2 = place(after, shape2, col2, row2)
                reply = _value(board2, lines + lines2, genome)
                best_reply = reply if best_reply is None else max(best_reply, reply)
            if best_reply is not None:
                value = best_reply
        # Tie-break from policy.plan_placement: deeper landing, then centermost.
        scored.append(((rot, col), value, row, -abs(col - 4)))
    scored.sort(key=lambda e: (e[1], e[2], e[3]), reverse=True)
    return [(placement, value) for placement, value, _, _ in scored]


@dataclass(frozen=True)
class Grade:
    """One decision, scored against the oracle's ranking of the same board.

    `regret` is in the evaluation's arbitrary units, whose scale grows with
    board height, so `regret_norm` is the comparable one. Both are per-move and
    path-relative: the board graded is the one the model built, not the one the
    oracle would have built, so a ruined board played well scores near zero.
    Accumulated damage lives in the outcome metrics instead.
    """

    chosen: tuple[int, int]
    best: tuple[int, int]
    chosen_value: float
    best_value: float
    worst_value: float
    rank: int
    legal_count: int
    regret: float
    regret_norm: float
    ply: int
    genome: dict

    def to_dict(self) -> dict:
        return {
            "chosen": list(self.chosen),
            "best": list(self.best),
            "chosen_value": round(self.chosen_value, 6),
            "best_value": round(self.best_value, 6),
            "worst_value": round(self.worst_value, 6),
            "rank": self.rank,
            "legal_count": self.legal_count,
            "regret": round(self.regret, 6),
            "regret_norm": round(self.regret_norm, 6),
            "ply": self.ply,
            "genome": self.genome,
        }


def grade(
    board: np.ndarray,
    piece: str,
    next_piece: str | None,
    chosen: Placement,
    genome: Genome | None = None,
    ply: int = DEFAULT_PLY,
) -> Grade | None:
    """Score `chosen` against every alternative, or None if it cannot be graded.

    None means the choice was not in the legal set — a guard, not a path, since
    the policy validates against the same enumeration.
    """
    genome = genome or Genome()
    ranked = rank_placements(board, piece, next_piece, genome, ply)
    if not ranked:
        return None
    key = (chosen.rotation, chosen.col)
    index = next((i for i, (placement, _) in enumerate(ranked) if placement == key), None)
    if index is None:
        return None
    best_value, worst_value = ranked[0][1], ranked[-1][1]
    chosen_value = ranked[index][1]
    regret = best_value - chosen_value
    span = best_value - worst_value
    return Grade(
        chosen=key,
        best=ranked[0][0],
        chosen_value=chosen_value,
        best_value=best_value,
        worst_value=worst_value,
        rank=index + 1,
        legal_count=len(ranked),
        regret=regret,
        regret_norm=(regret / span) if span > 0 else 0.0,
        ply=ply,
        genome={
            "w_lines": genome.w_lines,
            "w_agg_height": genome.w_agg_height,
            "w_holes": genome.w_holes,
            "w_bumpiness": genome.w_bumpiness,
        },
    )
