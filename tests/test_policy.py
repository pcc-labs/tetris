import numpy as np

from tetris_agent.policy import Genome, plan_placement


def empty_board() -> np.ndarray:
    return np.zeros((18, 10), dtype=bool)


def test_i_piece_prefers_flat_placement_on_empty_board():
    placement = plan_placement(empty_board(), "I", Genome())
    assert placement is not None
    # Horizontal I (width 4) keeps agg_height at 4; vertical costs 1+2+3+4.
    from tetris_agent.pieces import SHAPES, shape_width

    assert shape_width(SHAPES[("I", placement.rotation)]) == 4


def test_planner_clears_a_line_when_available():
    board = empty_board()
    board[17, :9] = True  # bottom row missing only col 9
    board[16, :4] = True  # some noise so flat placements elsewhere are less attractive
    placement = plan_placement(board, "I", Genome())
    assert placement is not None
    from tetris_agent.pieces import SHAPES, shape_width

    # Vertical I in col 9 clears the bottom row.
    assert placement.col == 9
    assert shape_width(SHAPES[("I", placement.rotation)]) == 1


def test_genome_round_trip_and_unknown_keys_ignored():
    g = Genome(w_holes=-1.5)
    params = g.to_params()
    assert params["w_holes"] == -1.5
    assert params["ticks_per_press"] == 4
    g2 = Genome.from_params({**params, "bogus": 99})
    assert g2 == g


def test_plan_returns_none_when_nothing_fits():
    board = np.ones((18, 10), dtype=bool)
    board[0] = False
    board[0, 5] = True  # only scattered space in top row: nothing fits anywhere
    assert plan_placement(board, "O", Genome()) is None
