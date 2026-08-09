"""Browser play: keystrokes arrive over the viewer's input channel, the game
runs headless at real speed, and the run is recorded exactly like tetris-manual.
"""

import queue

import pytest
from test_manual import FakeEmulator, install, state

from tetris_agent.browser_play import apply_inputs, play


class HeldFakeEmulator(FakeEmulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.held = []
        self.forced_level = None

    def force_level(self, level):
        self.forced_level = level

    def press_down(self, button):
        self.held.append(("press", button))

    def release(self, button):
        self.held.append(("release", button))


def test_apply_inputs_drains_queue_in_order():
    emu = HeldFakeEmulator([state(False)])
    q = queue.Queue()
    q.put({"type": "input", "button": "left", "action": "press"})
    q.put({"type": "input", "button": "left", "action": "release"})
    q.put({"type": "input", "button": "a", "action": "press"})
    apply_inputs(emu, q)
    assert emu.held == [("press", "left"), ("release", "left"), ("press", "a")]


def test_apply_inputs_ignores_unknown_buttons_and_shapes():
    emu = HeldFakeEmulator([state(False)])
    q = queue.Queue()
    q.put({"type": "input", "button": "escape", "action": "press"})  # not a GB button
    q.put({"type": "frame", "turn": 1})  # not an input message
    q.put({"button": "left"})  # no action
    apply_inputs(emu, q)
    assert emu.held == []


def test_play_fails_fast_when_viewer_unreachable(monkeypatch):
    def refuse(url, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("tetris_agent.browser_play._connect", refuse)
    with pytest.raises(RuntimeError, match="viewer"):
        play("rom.gb", viewer_url="http://127.0.0.1:9")


class FakeInputConn:
    """A websocket that never sends anything and closes when asked."""

    def __init__(self):
        self.closed = False

    def recv(self, timeout=None):
        raise TimeoutError

    def close(self):
        self.closed = True

    def __iter__(self):
        return iter(())


def test_play_records_a_human_run(monkeypatch, tmp_path):
    # Two pieces fall and lock on the scripted timeline; the browser sends no
    # input. The run must still be recorded as policy "human" — identical
    # artefacts to tetris-manual.
    timeline = []
    for i in range(2):
        timeline += [state(True, filled=i * 4), state(False, filled=(i + 1) * 4)]
    emu = HeldFakeEmulator(timeline)
    install(monkeypatch, emu)
    conn = FakeInputConn()
    monkeypatch.setattr("tetris_agent.browser_play._connect", lambda url, **k: conn)
    # The wait phase would consume the scripted timeline; player readiness is
    # covered by the wait_for_start tests.
    monkeypatch.setattr("tetris_agent.browser_play.wait_for_start", lambda *a, **k: True)

    published = []

    class Capture:
        def publish(self, event):
            published.append(event)

    from tetris_agent.events import EventCollector

    fitness = play(
        "rom.gb",
        viewer_url="http://127.0.0.1:8000",
        max_pieces=2,
        timer_div=5,
        collector=EventCollector(Capture()),
        streamer=None,
    )

    assert fitness["pieces_placed"] == 2
    assert fitness["policy"]["policy"] == "human"
    assert emu.started_with == 5
    kinds = [e["event_type"] for e in published]
    assert kinds.count("piece_spawn") == 2
    assert kinds.count("placement_decision") == 2
    assert kinds.count("piece_locked") == 2
    start = next(e for e in published if e["event_type"] == "session")
    assert start["data"]["policy"] == "human"
    assert conn.closed


def test_play_main_wires_browser_play(monkeypatch):
    from tetris_agent import cli

    seen = {}

    def fake_play(rom_path, viewer_url, max_pieces, timer_div, level, collector, recorder):
        seen.update(
            rom=rom_path, url=viewer_url, pieces=max_pieces, seed=timer_div, level=level, recorder=recorder
        )
        return {"score": 0, "lines": 0, "level": 0, "pieces_placed": 0, "avg_holes": 0.0,
                "max_stack_height": 0, "misexec_count": 0, "max_misexec_streak": 0,
                "topped_out": False}

    monkeypatch.setattr("tetris_agent.browser_play.play", fake_play)
    rc = cli.play_main(["--seed", "3", "--max-pieces", "7", "--no-save", "--no-record"])
    assert rc == 0
    assert seen["seed"] == 3
    assert seen["pieces"] == 7
    assert seen["rom"] == "rom/tetris.gb"
    assert seen["recorder"] is None


def test_agent_cli_exemplars_flag_reaches_the_policy(monkeypatch, tmp_path):
    from test_traces import O_AT_0, write_run

    from tetris_agent import cli

    write_run(tmp_path, "20260808-000009-cafe00", "human", [("O", "I", 0, 0, 0, O_AT_0)])
    cfg = cli.build_config(
        ["--policy", "model", "--model", "pi/gemma3", "--exemplars", str(tmp_path)]
    )
    p = cli.build_policy(cfg)
    assert "How a strong human placed pieces" in p.system_prompt


def test_frame_hook_announces_play_session_continuously():
    # Arming in the browser must never depend on catching a one-shot launch
    # event: a tab opened after tetris-play starts has to learn a session is
    # live from the stream itself.
    from tetris_agent.browser_play import _ANNOUNCE_EVERY_HOOKS, make_hook

    class FakeStreamer:
        def __init__(self):
            self.messages = []

        def send(self, message):
            self.messages.append(message)

        def send_frame(self, turn, png):
            self.messages.append({"type": "frame"})

    class ScreenshotEmulator(HeldFakeEmulator):
        def screenshot(self):
            return b"png"

    emu = ScreenshotEmulator([state(False)])
    streamer = FakeStreamer()

    class Turnless:
        turn = 0

    hook = make_hook(emu, queue.Queue(), streamer, None, Turnless())
    for _ in range(_ANNOUNCE_EVERY_HOOKS + 1):
        hook()
    announcements = [m for m in streamer.messages if m.get("type") == "play_session"]
    assert len(announcements) >= 1


START_PRESS = {"type": "input", "button": "start", "action": "press"}


def test_wait_for_start_returns_true_on_start_press_without_applying_it():
    from tetris_agent.browser_play import wait_for_start

    emu = HeldFakeEmulator([state(False)], alive_for=50)
    q = queue.Queue()
    q.put({"type": "input", "button": "left", "action": "press"})  # premature; ignored
    q.put(START_PRESS)
    assert wait_for_start(emu, q, None, None) is True
    # Watched, never applied: a manual start on the menu would begin an
    # unpinned game and break the seed's determinism.
    assert emu.held == []


def test_wait_for_start_returns_false_when_the_window_dies():
    from tetris_agent.browser_play import wait_for_start

    emu = HeldFakeEmulator([state(False)], alive_for=3)
    assert wait_for_start(emu, queue.Queue(), None, None) is False


HARD_DROP = {"type": "input", "button": "hard_drop", "action": "press"}


def test_hard_drop_press_holds_down():
    emu = HeldFakeEmulator([state(False)])
    q = queue.Queue()
    q.put(HARD_DROP)
    dropping = apply_inputs(emu, q)
    assert dropping is True
    assert emu.held == [("press", "down")]


def test_hook_releases_the_synthesized_drop_when_the_piece_locks():
    from tetris_agent.browser_play import make_hook

    class SpriteEmulator(HeldFakeEmulator):
        falling = frozenset()

        def falling_sprite_cells(self):
            return self.falling

    emu = SpriteEmulator([state(False)])
    emu.falling = frozenset({(1, 4)})
    q = queue.Queue()
    q.put(HARD_DROP)
    hook = make_hook(emu, q, None, None, None)
    hook()  # applies the hard drop; piece still falling -> down stays held
    assert emu.held == [("press", "down")]
    emu.falling = frozenset()  # piece locked
    hook()
    assert emu.held == [("press", "down"), ("release", "down")]


def test_wait_for_start_accepts_spacebar():
    from tetris_agent.browser_play import wait_for_start

    emu = HeldFakeEmulator([state(False)], alive_for=50)
    q = queue.Queue()
    q.put(HARD_DROP)  # spacebar at the title screen means "begin"
    assert wait_for_start(emu, q, None, None) is True
    assert emu.held == []
