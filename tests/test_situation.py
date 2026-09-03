"""The deterministic left-hand side of the routing table: five situations, explicit priority."""

import numpy as np

from tetris_agent.policy import Genome
from tetris_agent.situation import (
    METRICS,
    Kind,
    Situation,
    classify,
    column_heights,
    min_new_holes,
    situation_block,
    tetris_well,
)


def board_from(*rows: str) -> np.ndarray:
    """An 18x10 board from its bottom rows, written top-first the way prompts render them."""
    b = np.zeros((18, 10), dtype=bool)
    for i, row in enumerate(reversed(rows)):
        assert len(row) == 10
        b[17 - i] = [ch == "#" for ch in row]
    return b


TETRIS_READY = board_from(*(["#########."] * 4))
SAW = board_from("#.#.#.#.#.")
MOUND = board_from(*(["###......."] * 3))


def test_column_heights_counts_from_the_floor():
    b = board_from("#.........", "#.#.......")
    assert column_heights(b).tolist() == [2, 0, 1, 0, 0, 0, 0, 0, 0, 0]


def test_empty_board_is_flat():
    assert classify(board_from(), "T", "O") == Situation(Kind.FLAT)


def test_tetris_ready_names_the_well_regardless_of_the_queue():
    assert classify(TETRIS_READY, "T", "O") == Situation(Kind.TETRIS_READY, well_col=9)
    assert tetris_well(TETRIS_READY, column_heights(TETRIS_READY), 4) == 9


def test_tetris_ready_beats_mound():
    # The same board has a height range of 4 (>= mound_rise); the well wins the priority.
    assert classify(TETRIS_READY, "T", "O").kind is Kind.TETRIS_READY


def test_three_deep_well_is_not_ready():
    assert classify(board_from(*(["#########."] * 3)), "I", "O").kind is Kind.MOUND


def test_topping_out_beats_tetris_ready():
    tall = board_from(*(["#########."] * 12))
    assert classify(tall, "I", "O") == Situation(Kind.TOPPING_OUT)


def test_hole_risk_when_no_placement_of_o_avoids_a_hole():
    assert min_new_holes(SAW, "O") == 1
    assert classify(SAW, "O", "T") == Situation(Kind.HOLE_RISK)


def test_hole_risk_is_only_for_o_s_z():
    assert classify(SAW, "T", "O") == Situation(Kind.FLAT)  # range 1 < mound_rise 3


def test_o_with_a_flat_pair_is_not_at_risk():
    assert min_new_holes(board_from("##........"), "O") == 0
    assert classify(board_from("##........"), "O", "T") == Situation(Kind.FLAT)


def test_mound_from_height_range():
    assert classify(MOUND, "T", "O") == Situation(Kind.MOUND)


def test_thresholds_come_from_the_genome():
    assert classify(MOUND, "T", "O", Genome(mound_rise=5)).kind is Kind.FLAT
    assert classify(MOUND, "T", "O", Genome(topping_out_height=3)).kind is Kind.TOPPING_OUT
    assert classify(board_from(*(["#########."] * 3)), "T", "O", Genome(well_depth=3)).kind is Kind.TETRIS_READY


def test_genome_carries_the_thresholds():
    params = Genome().to_params()
    assert (params["topping_out_height"], params["mound_rise"], params["well_depth"]) == (12, 3, 4)
    assert Genome.from_params({**params, "well_depth": 5}).well_depth == 5


def test_every_kind_has_metrics():
    for kind in Kind:
        assert METRICS[kind]
    assert METRICS[Kind.HOLE_RISK] == ("holes",)


def test_situation_block_states_the_technique_for_the_queue():
    ready = Situation(Kind.TETRIS_READY, well_col=9)
    assert "drop it vertically" in situation_block(ready, TETRIS_READY, "I", "O")
    assert "next piece is an I" in situation_block(ready, TETRIS_READY, "T", "I")
    assert "without covering column 9" in situation_block(ready, TETRIS_READY, "T", "O")
    assert "this O" in situation_block(Situation(Kind.HOLE_RISK), SAW, "O", "T")
    assert "spans 3 rows" in situation_block(Situation(Kind.MOUND), MOUND, "T", "O")
    assert "12 rows high" in situation_block(Situation(Kind.TOPPING_OUT), board_from(*(["#########."] * 12)), "T", "O")
    assert situation_block(Situation(Kind.FLAT), board_from(), "T", "O").startswith("Situation: FLAT")


def test_hole_risk_beats_mound_for_o_and_mound_wins_for_t():
    # A saw surface raised to range 3: O has no hole-free placement (HOLE_RISK), T is a MOUND.
    saw3 = board_from("#.#.#.#.#.", "#.#.#.#.#.", "#.#.#.#.#.")
    assert classify(saw3, "O", "T").kind is Kind.HOLE_RISK
    assert classify(saw3, "T", "O").kind is Kind.MOUND


def test_tetris_ready_beats_hole_risk():
    # A ready well over a saw: O would open a hole anywhere, but the well wins the priority.
    ready_over_saw = board_from("#.#.#.#.#.", *(["#########."] * 4))
    assert min_new_holes(ready_over_saw, "O") >= 1
    assert classify(ready_over_saw, "O", "T").kind is Kind.TETRIS_READY
