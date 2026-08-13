import pytest

from tetris_agent.pieces import PIECES, SHAPES, distinct_rotations
from tetris_agent.state import _cells_from_sprites, normalize_cells


def test_cells_from_sprites_maps_probe_data():
    # Captured by probe on 2026-08-03: J piece at rows 1-2, next-piece preview at x>=96.
    sprites = [(40, 8), (48, 8), (56, 8), (56, 16), (128, 112), (136, 112), (128, 120), (136, 120)]
    assert _cells_from_sprites(sprites) == frozenset({(1, 3), (1, 4), (1, 5), (2, 5)})


def test_normalize_cells():
    assert normalize_cells({(2, 3), (2, 4)}) == frozenset({(0, 0), (0, 1)})


@pytest.mark.rom
def test_read_state_on_fresh_game(rom_path):
    from tetris_agent.emulator import Emulator
    from tetris_agent.state import read_state

    emu = Emulator(rom_path)
    try:
        emu.start(timer_div=0x00)
        state = read_state(emu)
        assert state.next_piece in "LJIOZST"
        assert not state.game_over
        assert state.falling is not None
        assert normalize_cells(state.falling.cells) == SHAPES[(state.falling.name, state.falling.rotation)]
        # Board below the falling piece is empty at game start.
        assert state.board[6:].sum() == 0
    finally:
        emu.stop()


@pytest.mark.rom
def test_shapes_table_matches_rom_for_all_pieces_and_rotations(rom_path):
    """Ground truth: rotate naturally dealt pieces (set_tetromino's ROM patch is a no-op
    on World Rev 1) across deterministic seeds until every type has been validated."""
    from tetris_agent.emulator import CURRENT_PIECE_ADDR, Emulator
    from tetris_agent.state import read_state

    emu = Emulator(rom_path)
    validated: set[str] = set()
    try:
        emu.start(timer_div=0x00)
        for seed in (0x00, 0x40, 0x80, 0xC0, 0x10, 0x55, 0xAA, 0xF0):
            emu.game_wrapper.reset_game(timer_div=seed)
            for _piece_slot in range(4):  # a few pieces per seed, dropped from spawn
                state = read_state(emu)
                for _ in range(120):
                    if state.falling is not None or state.game_over:
                        break
                    emu.tick(2)
                    state = read_state(emu)
                if state.game_over or state.falling is None:
                    break
                name = state.falling.name
                for _rot_press in range(4):
                    state = read_state(emu)
                    if state.falling is None or state.falling.name != name:
                        break
                    rot = state.falling.rotation
                    observed = normalize_cells(state.falling.cells)
                    assert observed == SHAPES[(name, rot)], (
                        f"{name} rot {rot}: observed {sorted(observed)} != table {sorted(SHAPES[(name, rot)])}"
                    )
                    if len(distinct_rotations(name)) == 1:
                        break
                    emu.press("a", ticks_after=6)
                validated.add(name)
                # Hard-drop to the next piece.
                current = emu.read(CURRENT_PIECE_ADDR)
                for _ in range(80):
                    if emu.read(CURRENT_PIECE_ADDR) != current or len(emu.falling_sprite_cells()) != 4:
                        break
                    emu.press("down", ticks_after=2)
            if validated == set(PIECES):
                break
    finally:
        emu.stop()
    assert validated == set(PIECES), f"only validated {sorted(validated)}"


@pytest.mark.rom
def test_screenshot_returns_png(rom_path):
    from tetris_agent.emulator import Emulator

    emu = Emulator(rom_path)
    try:
        emu.start(timer_div=0x00)
        png = emu.screenshot()
    finally:
        emu.stop()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_bottom_row_is_the_lowest_occupied_cell():
    from tetris_agent.state import FallingPiece

    piece = FallingPiece(name="T", rotation=0, cells=frozenset({(4, 3), (4, 4), (4, 5), (5, 4)}))
    assert piece.bottom_row == 5
