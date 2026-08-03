from tetris_agent.pieces import SHAPES, distinct_rotations, piece_name, shape_width


def test_every_shape_has_four_cells_and_is_normalized():
    for cells in SHAPES.values():
        assert len(cells) == 4
        assert min(r for r, _ in cells) == 0
        assert min(c for _, c in cells) == 0


def test_j_spawn_matches_probe_observation():
    assert SHAPES[("J", 0)] == frozenset({(0, 0), (0, 1), (0, 2), (1, 2)})


def test_distinct_rotations():
    assert distinct_rotations("O") == [0]
    assert len(distinct_rotations("I")) == 2
    assert len(distinct_rotations("S")) == 2
    assert len(distinct_rotations("Z")) == 2
    assert len(distinct_rotations("L")) == 4
    assert len(distinct_rotations("J")) == 4
    assert len(distinct_rotations("T")) == 4


def test_piece_name_masks_rotation_bits():
    assert piece_name(0x4) == "J"
    assert piece_name(0x6) == "J"
    assert piece_name(0x8) == "I"
    assert piece_name(0x0) == "L"


def test_shape_width():
    assert shape_width(SHAPES[("I", 0)]) == 4
    assert shape_width(SHAPES[("O", 0)]) == 2
