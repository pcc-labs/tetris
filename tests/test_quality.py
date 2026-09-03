"""The lookahead oracle and the grade it produces."""

import numpy as np

from tetris_agent import quality
from tetris_agent.policy import Genome, plan_placement

PIECES = ("I", "J", "L", "O", "S", "T", "Z")


def empty_board():
    return np.zeros((18, 10), dtype=bool)


def rough_board(seed=1, fill=0.55, top_row=12):
    """A half-full board with an uneven surface, so placements actually differ."""
    board = empty_board()
    rng = np.random.default_rng(seed)
    for r in range(top_row, 18):
        for c in range(10):
            board[r, c] = rng.random() < fill
    return board


def test_one_ply_reproduces_the_heuristic_solver():
    """ply=1 must be the control arm's own choice, tie-breaks included."""
    genome = Genome()
    for seed in (1, 2, 3):
        board = rough_board(seed)
        for piece in PIECES:
            ranked = quality.rank_placements(board, piece, "T", genome, ply=1)
            best = plan_placement(board, piece, genome)
            assert ranked, f"{piece} seed {seed}: no legal placements"
            assert ranked[0][0] == (best.rotation, best.col), f"{piece} seed {seed}"


def test_ranking_is_ordered_best_first():
    ranked = quality.rank_placements(rough_board(), "T", "I")
    values = [v for _, v in ranked]
    assert values == sorted(values, reverse=True)


def test_every_legal_placement_appears_exactly_once():
    from tetris_agent.policy import legal_moves

    board = rough_board()
    for piece in PIECES:
        ranked = quality.rank_placements(board, piece, "O")
        assert sorted(rc for rc, _ in ranked) == sorted(legal_moves(board, piece))


def test_two_ply_differs_from_one_ply_when_the_greedy_move_traps_the_next_piece():
    """A flat floor with one deep single-width well.

    One-ply happily fills the well with a vertical I. Two-ply, knowing another
    I is next, should value the alternatives differently — the point is only
    that looking ahead changes the ranking at all.
    """
    board = empty_board()
    board[14:18, :] = True
    board[14:18, 3] = False  # a 4-deep well in column 3
    one = quality.rank_placements(board, "I", "I", ply=1)
    two = quality.rank_placements(board, "I", "I", ply=2)
    assert [rc for rc, _ in one] != [rc for rc, _ in two]


def test_a_dead_next_piece_falls_back_to_the_one_ply_value():
    """No legal reply must not crash, and must not produce a non-finite value.

    Column 9 is left permanently empty so no row is ever fully True (placing
    a piece can't accidentally clear a line and free up space). Columns 0-2,
    5-8 are filled solid, blocking every column pair except (3, 4), which is
    open for exactly the two rows an O needs. That is the only legal
    placement for the current piece; once it lands there, every column but 9
    is solid, so the next O has nowhere to go.
    """
    board = empty_board()
    board[:, 0:3] = True
    board[2:18, 3:5] = True
    board[:, 5:9] = True
    ranked = quality.rank_placements(board, "O", "O", ply=2)
    assert ranked, "the current piece must still have a legal placement"
    for _, value in ranked:
        assert np.isfinite(value)
