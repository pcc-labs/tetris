"""Pure tests for the emulator wrapper's input passthroughs.

Constructed without PyBoy (object.__new__) so they run in the fast suite;
the ROM-marked tests in test_state.py cover the wrapper against the real
emulator.
"""


class FakePyBoy:
    def __init__(self):
        self.calls = []

    def button_press(self, name):
        self.calls.append(("press", name))

    def button_release(self, name):
        self.calls.append(("release", name))


def make_emulator():
    from tetris_agent.emulator import Emulator

    emu = object.__new__(Emulator)
    emu.pyboy = FakePyBoy()
    return emu


def test_press_down_and_release_pass_through_held_buttons():
    emu = make_emulator()
    emu.press_down("left")
    emu.release("left")
    assert emu.pyboy.calls == [("press", "left"), ("release", "left")]


def test_headless_realtime_tick_paces_itself(monkeypatch):
    # pyboy's null window has a no-op frame_limiter: speed=1 headless runs
    # uncapped, which made browser play unplayably fast. The wrapper must
    # hold 60fps wall-clock itself.
    from tetris_agent import emulator as emulator_mod
    from tetris_agent.emulator import Emulator

    sleeps = []
    clock = {"now": 100.0}
    monkeypatch.setattr(emulator_mod.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(emulator_mod.time, "sleep", lambda s: sleeps.append(s))

    emu = object.__new__(Emulator)
    emu.pyboy = type("P", (), {"tick": lambda self, n, render: True})()
    emu.frame_hook = None
    emu._ticks_since_frame = 0
    emu._frame_s = 1.0 / 60.0
    emu._next_frame_at = 0.0

    emu.tick(2)  # emulation is instant on the fake clock -> must sleep ~2 frames
    assert sleeps and abs(sleeps[0] - 2 / 60) < 1e-9


def test_unlimited_speed_never_sleeps(monkeypatch):
    from tetris_agent import emulator as emulator_mod
    from tetris_agent.emulator import Emulator

    sleeps = []
    monkeypatch.setattr(emulator_mod.time, "sleep", lambda s: sleeps.append(s))

    emu = object.__new__(Emulator)
    emu.pyboy = type("P", (), {"tick": lambda self, n, render: True})()
    emu.frame_hook = None
    emu._ticks_since_frame = 0
    emu._frame_s = 0.0
    emu._next_frame_at = 0.0

    emu.tick(4)
    assert sleeps == []


def test_force_level_pokes_level_and_gravity_reload():
    # start_game inherits whatever the level-select cursor held (live sessions
    # landed on level 9); force_level makes the started game's level — and its
    # gravity — deterministic. Addresses verified empirically against the ROM:
    # FFA9/FFC2 level value, FF9A gravity reload, FF99 current countdown.
    from tetris_agent.emulator import Emulator

    mem = {}

    class MemPyBoy:
        memory = mem

    emu = object.__new__(Emulator)
    emu.pyboy = MemPyBoy()
    emu.force_level(0)
    assert mem == {0xFFA9: 0, 0xFFC2: 0, 0xFF9A: 52, 0xFF99: 52}
    emu.force_level(9)
    assert mem[0xFF9A] == 10 and mem[0xFFA9] == 9
