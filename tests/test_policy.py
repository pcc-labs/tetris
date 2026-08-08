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


def test_no_input_policy_returns_the_spawn_position_so_nothing_is_pressed():
    # Measured against the ROM: every piece spawns at rotation 0, leftmost
    # column 3, except O which spawns at 4. Returning that means the
    # controller has nothing to correct and the piece drops where it appeared.
    from tetris_agent.policy import NoInputPolicy

    p = NoInputPolicy()
    board = np.zeros((18, 10), dtype=bool)
    assert p.name == "no-input"
    for piece in ("I", "J", "L", "T", "S", "Z"):
        placement = p.plan(board, piece, "I", turn=1)
        assert (placement.rotation, placement.col) == (0, 3)
    assert p.plan(board, "O", "I", turn=1).col == 4


def test_no_input_policy_keeps_dropping_into_a_crowded_board():
    # The floor arm must not bail out when the stack is high — topping out is
    # the result we want to measure, not an error to avoid.
    from tetris_agent.policy import NoInputPolicy

    board = np.ones((18, 10), dtype=bool)
    assert NoInputPolicy().plan(board, "T", "I", turn=1) is not None


def test_random_policy_only_ever_picks_a_placement_that_fits():
    from tetris_agent.policy import RandomPolicy, legal_moves

    board = np.zeros((18, 10), dtype=bool)
    board[17, 0:4] = True
    board[16, 0:2] = True
    legal = set(legal_moves(board, "T"))
    p = RandomPolicy(seed=7)
    for _ in range(40):
        placement = p.plan(board, "T", "I", turn=1)
        assert (placement.rotation, placement.col) in legal


def test_random_policy_is_reproducible_and_actually_varies():
    from tetris_agent.policy import RandomPolicy

    board = np.zeros((18, 10), dtype=bool)

    def sequence(seed):
        policy = RandomPolicy(seed=seed)
        picks = [policy.plan(board, "T", "I", turn) for turn in range(12)]
        return [(p.rotation, p.col) for p in picks]

    assert sequence(3) == sequence(3)  # same seed, same run
    assert len(set(sequence(3))) > 1  # and it is not stuck on one move


def test_random_policy_returns_none_when_nothing_fits():
    from tetris_agent.policy import RandomPolicy

    assert RandomPolicy().plan(np.ones((18, 10), dtype=bool), "T", "I", turn=1) is None
