"""Pure board simulation over an 18x10 grid of booleans."""

from dataclasses import dataclass

import numpy as np

from tetris_agent.pieces import Cells, shape_width

BLANK = 47
ROWS, COLS = 18, 10


@dataclass(frozen=True)
class Features:
    holes: int
    agg_height: int
    bumpiness: int
    max_height: int


def settled_board(game_area: np.ndarray, falling_cells: Cells) -> np.ndarray:
    """Board of settled cells: non-blank tiles minus the falling piece."""
    board = np.asarray(game_area) != BLANK
    for r, c in falling_cells:
        if 0 <= r < ROWS and 0 <= c < COLS:
            board[r, c] = False
    return board


def _fits(board: np.ndarray, shape: Cells, col: int, row: int) -> bool:
    for r, c in shape:
        rr, cc = row + r, col + c
        if rr >= ROWS or board[rr, cc]:
            return False
    return True


def drop(board: np.ndarray, shape: Cells, col: int) -> int | None:
    """Landing top-row for a shape dropped in `col`, or None if it can't fit."""
    if col < 0 or col + shape_width(shape) > COLS:
        return None
    if not _fits(board, shape, col, 0):
        return None
    row = 0
    while _fits(board, shape, col, row + 1):
        row += 1
    return row


def place(board: np.ndarray, shape: Cells, col: int, row: int) -> tuple[np.ndarray, int]:
    """Lock the shape at (row, col); return (new board, lines cleared)."""
    new = board.copy()
    for r, c in shape:
        new[row + r, col + c] = True
    full = new.all(axis=1)
    lines = int(full.sum())
    if lines:
        kept = new[~full]
        new = np.vstack([np.zeros((lines, COLS), dtype=bool), kept])
    return new, lines


def features(board: np.ndarray) -> Features:
    heights = np.zeros(COLS, dtype=int)
    holes = 0
    for c in range(COLS):
        col = board[:, c]
        filled = np.flatnonzero(col)
        if filled.size:
            top = filled[0]
            heights[c] = ROWS - top
            holes += int((~col[top:]).sum())
    bumpiness = int(np.abs(np.diff(heights)).sum())
    return Features(
        holes=holes,
        agg_height=int(heights.sum()),
        bumpiness=bumpiness,
        max_height=int(heights.max()),
    )
