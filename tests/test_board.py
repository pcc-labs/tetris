import numpy as np

from tetris_agent.board import BLANK, drop, features, place, settled_board
from tetris_agent.pieces import SHAPES


def B(*rows: str) -> np.ndarray:
    """Build an 18x10 bool board from ascii rows (bottom-aligned). X = filled."""
    board = np.zeros((18, 10), dtype=bool)
    for i, row in enumerate(rows):
        board[18 - len(rows) + i] = [ch == "X" for ch in row]
    return board


def test_drop_o_on_empty_board_lands_on_floor():
    assert drop(np.zeros((18, 10), dtype=bool), SHAPES[("O", 0)], 0) == 16


def test_drop_stacks_on_existing_cells():
    board = B("X.........")
    assert drop(board, SHAPES[("O", 0)], 0) == 15


def test_drop_returns_none_when_column_out_of_range():
    assert drop(np.zeros((18, 10), dtype=bool), SHAPES[("I", 0)], 6) == 17
    assert drop(np.zeros((18, 10), dtype=bool), SHAPES[("I", 0)], 7) is None


def test_place_clears_full_row_and_shifts_down():
    board = B(
        "X.........",
        "XXXXXXXX..",
    )
    new_board, lines = place(board, SHAPES[("O", 0)], 8, 16)
    assert lines == 1
    # Bottom row cleared; the lone X from row 16 and O-remainder shift down.
    assert new_board[17].sum() == 3  # X + the O's top two cells
    assert new_board[16].sum() == 0


def test_features_counts_holes_and_bumpiness():
    board = B(
        "XX........",
        "X.........",
        "XX........",
    )
    f = features(board)
    assert f.max_height == 3
    assert f.agg_height == 6  # col0 height 3, col1 height 3 (with a gap inside)
    assert f.holes == 1  # col1 middle row is blank below its top
    assert f.bumpiness == 3  # |3-3| + |3-0| + zeros



def test_features_flat_board_zero_bumpiness():
    f = features(B("XXXXXXXXXX"))
    assert f.bumpiness == 0
    assert f.holes == 0


def test_settled_board_subtracts_falling_cells():
    area = np.full((18, 10), BLANK, dtype=int)
    area[0, 3] = 129
    area[17, 0] = 130
    board = settled_board(area, frozenset({(0, 3)}))
    assert not board[0, 3]
    assert board[17, 0]
