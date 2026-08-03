"""Read a structured GameState from the emulator."""

from dataclasses import dataclass

import numpy as np

from tetris_agent.board import settled_board
from tetris_agent.pieces import Cells, piece_name

_PLAYFIELD_MAX_X = 96


def _cells_from_sprites(positions: list[tuple[int, int]]) -> Cells:
    """Map on-screen sprite pixel positions to board cells, excluding the preview."""
    return frozenset((y // 8, x // 8 - 2) for x, y in positions if x < _PLAYFIELD_MAX_X)


def normalize_cells(cells) -> Cells:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return frozenset((r - min_r, c - min_c) for r, c in cells)


@dataclass(frozen=True)
class FallingPiece:
    name: str
    rotation: int
    cells: Cells

    @property
    def col(self) -> int:
        return min(c for _, c in self.cells)


@dataclass(frozen=True)
class GameState:
    board: np.ndarray
    falling: FallingPiece | None
    next_piece: str
    score: int
    level: int
    lines: int
    game_over: bool


def read_state(emu) -> GameState:
    from tetris_agent.emulator import CURRENT_PIECE_ADDR, NEXT_PIECE_ADDR

    cells = emu.falling_sprite_cells()
    falling = None
    if len(cells) == 4:
        byte = emu.read(CURRENT_PIECE_ADDR)
        falling = FallingPiece(name=piece_name(byte), rotation=byte & 0b11, cells=cells)
    return GameState(
        board=settled_board(emu.game_area(), cells),
        falling=falling,
        next_piece=piece_name(emu.read(NEXT_PIECE_ADDR)),
        score=emu.score,
        level=emu.level,
        lines=emu.lines,
        game_over=emu.game_over,
    )
