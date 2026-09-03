"""Grade a placement against a lookahead oracle.

The benchmark scores outcomes; this scores decisions. `rank_placements` ranks
every legal placement for the current piece by the same weighted evaluation the
heuristic control arm plays with, searched `ply` pieces deep. `grade` turns a
model's choice into a regret and a rank against that ranking.

The oracle is a strong heuristic, not ground truth: regret is disagreement with
a two-ply evaluator, so a model that outplays the oracle would show high regret.
The `lookahead` arm exists to put the oracle's own score in the same table.
"""

import numpy as np

from tetris_agent.board import COLS, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width
from tetris_agent.policy import Genome

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
