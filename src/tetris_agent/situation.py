"""Board situations — the deterministic left-hand side of the routing table.

Neubauer's guidance is a small set of board states, each with a prescribed move.
`classify` names the state; `situation_block` renders the move for the `routed`
harness. Pure over the settled board, like board.py: no emulator, no model, no
I/O. Thresholds come from the Genome so evolve.py can tune them.

Priority is explicit and top-down — first match wins (see docs/superpowers/specs/
2026-09-02-situation-router-design.md, section 04). Real boards are mixtures, so
the order is a decision under test, not a derivation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from tetris_agent.board import COLS, ROWS, drop, features, place
from tetris_agent.pieces import SHAPES, distinct_rotations, shape_width
from tetris_agent.policy import Genome

HOLE_PRONE = ("O", "S", "Z")


class Kind(str, Enum):
    TOPPING_OUT = "TOPPING_OUT"
    TETRIS_READY = "TETRIS_READY"
    HOLE_RISK = "HOLE_RISK"
    MOUND = "MOUND"
    FLAT = "FLAT"


@dataclass(frozen=True)
class Situation:
    kind: Kind
    well_col: int | None = None  # TETRIS_READY only: the column the I goes in


# The consequence numbers the routed harness shows per placement. Names are
# LegalPlacement fields: `lines` on the placement itself, the rest on `.features`.
METRICS: dict[Kind, tuple[str, ...]] = {
    Kind.TOPPING_OUT: ("lines", "max_height"),
    Kind.TETRIS_READY: ("lines",),
    Kind.HOLE_RISK: ("holes",),
    Kind.MOUND: ("bumpiness",),
    Kind.FLAT: ("holes", "bumpiness"),
}


def column_heights(board: np.ndarray) -> np.ndarray:
    """Cells from the floor up to the topmost settled cell, per column (0 when empty)."""
    heights = np.zeros(COLS, dtype=int)
    for c in range(COLS):
        filled = np.flatnonzero(board[:, c])
        if filled.size:
            heights[c] = ROWS - filled[0]
    return heights


def tetris_well(board: np.ndarray, heights: np.ndarray, depth: int) -> int | None:
    """A column whose `depth` lowest empty rows are full in every other column, or None."""
    for w in range(COLS):
        floor = ROWS - 1 - int(heights[w])  # lowest empty row in this column
        top = floor - depth + 1
        if top < 0:
            continue
        rows = board[top : floor + 1]
        if np.delete(rows, w, axis=1).all() and not rows[:, w].any():
            return w
    return None


def min_new_holes(board: np.ndarray, piece: str) -> int | None:
    """Fewest holes any legal placement of `piece` opens; None if nothing fits."""
    base = features(board).holes
    best: int | None = None
    for rot in distinct_rotations(piece):
        shape = SHAPES[(piece, rot)]
        for col in range(COLS - shape_width(shape) + 1):
            row = drop(board, shape, col)
            if row is None:
                continue
            new_board, _ = place(board, shape, col, row)
            opened = max(0, features(new_board).holes - base)
            if best is None or opened < best:
                best = opened
            if best == 0:
                return 0
    return best


def classify(
    board: np.ndarray, piece: str, next_piece: str, genome: Genome | None = None
) -> Situation:
    """Name the board's situation. First match in priority order wins."""
    g = genome or Genome()
    heights = column_heights(board)
    if int(heights.max()) >= g.topping_out_height:
        return Situation(Kind.TOPPING_OUT)
    well = tetris_well(board, heights, g.well_depth)
    if well is not None:
        return Situation(Kind.TETRIS_READY, well_col=well)
    if piece in HOLE_PRONE and (min_new_holes(board, piece) or 0) >= 1:
        return Situation(Kind.HOLE_RISK)
    if int(heights.max() - heights.min()) >= g.mound_rise:
        return Situation(Kind.MOUND)
    return Situation(Kind.FLAT)


def situation_block(situation: Situation, board: np.ndarray, piece: str, next_piece: str) -> str:
    """The one technique this turn calls for, as the routed user prompt states it."""
    kind = situation.kind
    heights = column_heights(board)
    if kind is Kind.TOPPING_OUT:
        return (
            f"Situation: TOPPING_OUT — the stack is {int(heights.max())} rows high on an {ROWS}-row board. "
            "Survive: take a placement that clears lines if one exists; otherwise the one leaving the "
            "lowest max height. Do not set anything up."
        )
    if kind is Kind.TETRIS_READY:
        w = situation.well_col
        lead = f"Situation: TETRIS_READY — the well at column {w} is ready: the rows beside it are full. "
        if piece == "I":
            return lead + f"This piece is an I: drop it vertically (the width-1 rotation) in column {w}."
        if next_piece == "I":
            return lead + f"The next piece is an I: place this piece without covering column {w}."
        return lead + f"No I is in view yet: place this piece without covering column {w}, keeping the rest flat."
    if kind is Kind.HOLE_RISK:
        return (
            f"Situation: HOLE_RISK — this {piece} cannot land anywhere without opening a hole. "
            "Choose the placement that opens the fewest, as low on the board as possible."
        )
    if kind is Kind.MOUND:
        rise = int(heights.max() - heights.min())
        return (
            f"Situation: MOUND — the surface spans {rise} rows between its lowest and highest columns. "
            "Flatten: choose the placement that lowers bumpiness most without opening a hole."
        )
    return (
        "Situation: FLAT — the surface is even. Keep it that way: no column more than two above its "
        "neighbours, no gap deeper than two; prefer the least bumpiness and no new holes."
    )
