"""Thin PyBoy wrapper — the only module that imports pyboy."""

from pathlib import Path

import numpy as np
from pyboy import PyBoy

CURRENT_PIECE_ADDR = 0xC203
NEXT_PIECE_ADDR = 0xC213
# The play field spans pixel x 16..95 (tile cols 2..11); sprites at x >= 96 are the preview.
_PLAYFIELD_MAX_X = 96


class Emulator:
    def __init__(self, rom_path: str | Path, headless: bool = True, speed: int = 0):
        """speed: emulation speed multiplier; 0 = unlimited, 1 = real time (for --live)."""
        self.pyboy = PyBoy(str(rom_path), window="null" if headless else "SDL2")
        self.pyboy.set_emulation_speed(speed)
        self.game_wrapper = self.pyboy.game_wrapper
        # Optional callback fired roughly every _FRAME_HOOK_TICKS ticks (live streaming).
        self.frame_hook = None
        self._ticks_since_frame = 0

    _FRAME_HOOK_TICKS = 12

    def start(self, timer_div: int | None = None) -> None:
        self.game_wrapper.start_game(timer_div=timer_div)

    def tick(self, n: int = 1) -> None:
        self.pyboy.tick(n, True)
        if self.frame_hook is not None:
            self._ticks_since_frame += n
            if self._ticks_since_frame >= self._FRAME_HOOK_TICKS:
                self._ticks_since_frame = 0
                self.frame_hook()

    def press(self, button: str, ticks_after: int = 4) -> None:
        self.pyboy.button(button)
        self.tick(ticks_after)

    def read(self, addr: int) -> int:
        return self.pyboy.memory[addr]

    def falling_sprite_cells(self) -> frozenset[tuple[int, int]]:
        positions = []
        for i in range(40):
            sprite = self.pyboy.get_sprite(i)
            if sprite.on_screen:
                positions.append((sprite.x, sprite.y))
        from tetris_agent.state import _cells_from_sprites

        return _cells_from_sprites(positions)

    def game_area(self) -> np.ndarray:
        return np.asarray(self.game_wrapper.game_area())

    def screenshot(self) -> bytes:
        """Current screen as PNG bytes (needs Pillow)."""
        import io

        buf = io.BytesIO()
        self.pyboy.screen.image.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()

    @property
    def score(self) -> int:
        return self.game_wrapper.score

    @property
    def level(self) -> int:
        return self.game_wrapper.level

    @property
    def lines(self) -> int:
        return self.game_wrapper.lines

    @property
    def game_over(self) -> bool:
        return self.game_wrapper.game_over()

    def save_state(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            self.pyboy.save_state(f)

    def load_state(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self.pyboy.load_state(f)

    def stop(self) -> None:
        self.pyboy.stop(save=False)
