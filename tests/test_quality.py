"""The lookahead oracle and the grade it produces."""

import numpy as np

from tetris_agent import quality
from tetris_agent.policy import Genome, Placement, plan_placement

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


def test_a_lethal_placement_ranks_last_when_live_options_exist():
    """The reviewer's minimal case: several legal placements for the current
    piece, of which only one leaves the next piece with nowhere to go. The
    lethal placement must rank LAST, not first. Asserting finiteness alone
    is not enough to catch the original bug — that test's board had exactly
    one legal placement, so there was nothing to compare a rank against; the
    inverted valuation (dead placement scoring highest) shipped anyway.

    Columns 0-2 and 7-8 are solid full height. Columns 3-6 are open only at
    rows 0-1 (solid below). Column 9 is left fully empty so no row is ever
    completely full (a line clear would change the board this test relies
    on). An O piece can only land at col 3, 4 or 5 (each lands at row 0,
    filling that column pair for rows 0-1). Landing at col 4 splits the rest
    of the opening into two isolated single-width gaps (col 3 and col 6),
    neither of which an O can occupy, so the next O has no legal reply.
    Landing at col 3 or col 5 instead leaves one two-wide gap the next O can
    fill.
    """
    board = empty_board()
    board[:, 0:3] = True
    board[:, 7:9] = True
    board[2:18, 3:7] = True

    ranked = quality.rank_placements(board, "O", "O", ply=2)
    assert len(ranked) > 1, "the test needs several placements to compare a rank against"
    placements = [rc for rc, _ in ranked]
    assert set(placements) == {(0, 3), (0, 4), (0, 5)}
    assert placements[-1] == (0, 4), "the lethal placement must rank last"
    for _, value in ranked:
        assert np.isfinite(value)


def placement_of(entry):
    (rotation, col), value = entry
    return Placement(rotation=rotation, col=col, score=value)


def test_grading_the_oracles_own_choice_is_zero_regret_rank_one():
    board = rough_board()
    ranked = quality.rank_placements(board, "T", "I")
    g = quality.grade(board, "T", "I", placement_of(ranked[0]))
    assert g.rank == 1
    assert g.regret == 0.0
    assert g.regret_norm == 0.0
    assert g.best == g.chosen
    assert g.legal_count == len(ranked)


def test_grading_the_worst_legal_choice_is_last_rank_and_full_regret():
    board = rough_board()
    ranked = quality.rank_placements(board, "T", "I")
    g = quality.grade(board, "T", "I", placement_of(ranked[-1]))
    assert g.rank == g.legal_count == len(ranked)
    assert g.regret_norm == 1.0
    assert g.regret > 0


def test_an_all_equal_board_is_zero_regret_not_a_division_by_zero():
    """An empty board with an O piece and bumpiness zeroed: every column is worth the same.

    With the default genome, an O piece on an empty board is not actually tied
    across every column: edge columns (0 and 8) have lower bumpiness than
    interior ones, so col=7 would tie for worst, not for best. Zeroing
    w_bumpiness removes that edge effect so every legal column is genuinely
    equal in value, which is what this test needs to exercise the
    best_value == worst_value guard.
    """
    genome = Genome(w_bumpiness=0)
    g = quality.grade(empty_board(), "O", "O", Placement(rotation=0, col=7, score=0.0), genome=genome, ply=1)
    assert g.regret_norm == 0.0
    assert g.best_value == g.worst_value


def test_a_placement_outside_the_legal_set_is_not_graded():
    """The board must have legal placements, so `grade` reaches the `index is
    None` branch this test is named for, instead of exiting earlier at the
    `if not ranked` guard (which an all-solid board hits vacuously)."""
    board = rough_board()
    ranked = quality.rank_placements(board, "T", "I")
    assert ranked, "the board must offer a legal placement for this test to mean anything"
    taken = {rc for rc, _ in ranked}
    # T at col=9 is out of bounds for a 3-wide shape on a 10-wide board, so it
    # can never be in the legal set regardless of the board's contents.
    illegal = Placement(rotation=0, col=9, score=0.0)
    assert (illegal.rotation, illegal.col) not in taken
    assert quality.grade(board, "T", "I", illegal) is None


def test_the_grade_records_the_weights_it_used():
    """The healer mutates the weights, so a label is only reproducible with them."""
    genome = Genome(w_holes=-0.9)
    board = rough_board()
    # Grade a placement taken from the ranking, so it is legal by construction.
    chosen = placement_of(quality.rank_placements(board, "T", "I", genome)[0])
    g = quality.grade(board, "T", "I", chosen, genome=genome)
    assert g.genome["w_holes"] == -0.9
    assert g.ply == 2


def test_to_dict_is_json_safe_and_carries_the_ranking():
    import json

    board = rough_board()
    ranked = quality.rank_placements(board, "T", "I")
    payload = quality.grade(board, "T", "I", placement_of(ranked[1])).to_dict()
    assert json.loads(json.dumps(payload))["rank"] == 2
    assert payload["best"] == list(ranked[0][0])
