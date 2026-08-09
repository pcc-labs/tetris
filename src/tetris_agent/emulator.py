"""Thin PyBoy wrapper — the only module that imports pyboy."""

import time
from pathlib import Path

import numpy as np
from pyboy import PyBoy

CURRENT_PIECE_ADDR = 0xC203
NEXT_PIECE_ADDR = 0xC213
# The play field spans pixel x 16..95 (tile cols 2..11); sprites at x >= 96 are the preview.
_PLAYFIELD_MAX_X = 96


class Emulator:
    def __init__(
        self,
        rom_path: str | Path,
        headless: bool = True,
        speed: int = 0,
        frame_hook_ticks: int | None = None,
    ):
        """speed: emulation speed multiplier; 0 = unlimited, 1 = real time (for --live).
        frame_hook_ticks: cadence of frame_hook; default suits recording (~5fps),
        browser play lowers it so streamed frames and input stay responsive."""
        self.pyboy = PyBoy(str(rom_path), window="null" if headless else "SDL2")
        self.pyboy.set_emulation_speed(speed)
        self.game_wrapper = self.pyboy.game_wrapper
        # Optional callback fired roughly every _FRAME_HOOK_TICKS ticks (live streaming).
        self.frame_hook = None
        self._ticks_since_frame = 0
        if frame_hook_ticks is not None:
            self._FRAME_HOOK_TICKS = frame_hook_ticks
        # pyboy's null window has a no-op frame_limiter, so headless + speed=1
        # runs uncapped (browser play came out ~2x real time). Pace ourselves.
        self._frame_s = (1.0 / 60.0) / speed if (headless and speed > 0) else 0.0
        self._next_frame_at = 0.0

    _FRAME_HOOK_TICKS = 12

    # Gravity reload value per level (frames between one-row falls, minus one),
    # read out of the running ROM: level 0 holds 52, level 9 holds 10.
    _LEVEL_ADDRS = (0xFFA9, 0xFFC2)
    _GRAVITY_RELOAD_ADDR = 0xFF9A
    _GRAVITY_COUNTDOWN_ADDR = 0xFF99
    _GRAVITY_RELOADS = (52, 48, 44, 40, 36, 32, 27, 21, 16, 10)

    def start(self, timer_div: int | None = None) -> None:
        self.game_wrapper.start_game(timer_div=timer_div)

    def force_level(self, level: int) -> None:
        """Pin the started game's level — and therefore its gravity.

        start_game presses blindly through the menus, inheriting whatever the
        level-select cursor happens to hold; live sessions came up at level 9,
        which is unplayable for a human over a stream. Poking the level and
        gravity-timer bytes (verified empirically: a level-9 game drops to the
        exact level-0 fall rate, persisting across pieces) makes the level
        deterministic no matter what the menus did. Piece RNG is untouched.
        """
        for addr in self._LEVEL_ADDRS:
            self.pyboy.memory[addr] = level
        reload = self._GRAVITY_RELOADS[level]
        self.pyboy.memory[self._GRAVITY_RELOAD_ADDR] = reload
        self.pyboy.memory[self._GRAVITY_COUNTDOWN_ADDR] = reload

    def tick(self, n: int = 1) -> bool:
        """Advance n frames. False once the window is closed (manual play)."""
        alive = self.pyboy.tick(n, True)
        if self._frame_s:
            now = time.monotonic()
            if not self._next_frame_at:
                self._next_frame_at = now
            self._next_frame_at += n * self._frame_s
            delay = self._next_frame_at - now
            if delay > 0:
                time.sleep(delay)
            else:
                self._next_frame_at = now  # fell behind; don't chase the deficit
        if self.frame_hook is not None:
            self._ticks_since_frame += n
            if self._ticks_since_frame >= self._FRAME_HOOK_TICKS:
                self._ticks_since_frame = 0
                self.frame_hook()
        return alive is not False

    def press(self, button: str, ticks_after: int = 4) -> None:
        self.pyboy.button(button)
        self.tick(ticks_after)

    def press_down(self, button: str) -> None:
        """Hold a button until release() — human input needs held keys (DAS,
        soft drop), which the tap-style press() can't express."""
        self.pyboy.button_press(button)

    def release(self, button: str) -> None:
        self.pyboy.button_release(button)

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
