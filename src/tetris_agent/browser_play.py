"""Play in the browser: frames stream out through the viewer's live hub and
keystrokes stream back over /ws/input.

The observation and recording are `manual.py`'s — same events, same
`policy: "human"` traces in runs/<id>/. Only the surface differs: the emulator
is headless, and the browser is the controller. That last part is why an
unreachable viewer is a hard error here, where agent --live degrades silently:
without the viewer there is no way to play at all.
"""

import json
import logging
import queue
import threading

logger = logging.getLogger(__name__)

_BUTTONS = frozenset({"left", "right", "up", "down", "a", "b", "start", "select"})
# frame_hook cadence in ticks: 2 ≈ 30fps streamed frames, and — because input
# is applied in the same hook — a keystroke-to-emulator latency of ~2 frames.
_STREAM_EVERY_TICKS = 2
# Decimate recording back to the ~5fps every other arm's replay uses.
_RECORD_EVERY_HOOKS = 6
# Re-announce the play session every ~2s of hooks: a browser tab opened after
# launch must be able to learn a session is live from the stream itself, never
# from having caught a one-shot start event.
_ANNOUNCE_EVERY_HOOKS = 60
# How long to idle at the title screen before giving up (~5 min of ticks).
_WAIT_START_TICKS = 60 * 60 * 5 // 2

_AUTO = object()

CONTROLS = """\
  space        start / drop to bottom  up arrow or a   rotate
  arrow keys   move / soft drop        s               rotate (other way)
  ctrl-c here stops early and still records the run"""


def _connect(url: str, **kwargs):
    from websockets.sync.client import connect

    return connect(url, **kwargs)


def _input_url(viewer_url: str) -> str:
    base = viewer_url.rstrip("/")
    base = base.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
    return base + "/ws/input"


def apply_inputs(emu, q: queue.Queue, dropping: bool = False) -> bool:
    """Drain pending browser keystrokes onto the emulator's buttons.

    The allowlist matters: these strings come off a websocket, and only real
    Game Boy buttons may reach pyboy. `hard_drop` is the one synthetic input:
    the Game Boy has no hard drop, so spacebar holds soft-drop until the piece
    locks — the returned flag tracks that hold so the hook can release it.
    """
    while True:
        try:
            msg = q.get_nowait()
        except queue.Empty:
            return dropping
        if not isinstance(msg, dict) or msg.get("type") != "input":
            continue
        button, action = msg.get("button"), msg.get("action")
        if button == "hard_drop":
            if action == "press" and not dropping:
                logger.info("input: hard_drop")
                emu.press_down("down")
                dropping = True
            continue
        if button not in _BUTTONS:
            continue
        if action == "press":
            logger.info("input: press %s", button)
            emu.press_down(button)
        elif action == "release":
            emu.release(button)


def make_hook(emu, q: queue.Queue, streamer, recorder, collector):
    """Per-tick work: apply queued keystrokes, resolve a synthesized hard
    drop, stream/record frames, and keep announcing that a play session is
    live for late-joining browser tabs."""
    hooks = 0
    dropping = False

    def hook():
        nonlocal hooks, dropping
        dropping = apply_inputs(emu, q, dropping)
        if dropping and not emu.falling_sprite_cells():
            emu.release("down")  # the hard-dropped piece locked
            dropping = False
        if streamer is None and recorder is None:
            return
        png = emu.screenshot()
        hooks += 1
        if streamer is not None:
            streamer.send_frame(collector.turn, png)
            if hooks % _ANNOUNCE_EVERY_HOOKS == 1:
                streamer.send({"type": "play_session"})
        if recorder is not None and hooks % _RECORD_EVERY_HOOKS == 0:
            recorder.record_frame(collector.turn, png)

    return hook


def wait_for_start(emu, q: queue.Queue, streamer, collector, limit: int = _WAIT_START_TICKS) -> bool:
    """Idle at the title screen until the player presses space (or start).

    The game must not begin while nobody is looking — under pure gravity
    seed 0 tops out in ~45s, so starting at launch records a junk human run
    whenever the tab isn't ready yet. Inputs are watched here, never applied:
    pressing start manually on the menu would begin an unpinned game and
    break the seed's determinism; start_game(timer_div) does it right.
    """
    hooks = 0
    for _ in range(limit):
        while True:
            try:
                msg = q.get_nowait()
            except queue.Empty:
                break
            if (
                isinstance(msg, dict)
                and msg.get("type") == "input"
                and msg.get("action") == "press"
                and msg.get("button") in ("start", "hard_drop")
            ):
                return True
        if not emu.tick(2):
            return False
        if streamer is not None:
            hooks += 1
            streamer.send_frame(0, emu.screenshot())
            if hooks % _ANNOUNCE_EVERY_HOOKS == 1:
                streamer.send({"type": "play_session"})
    return False


def _read_into(conn, q: queue.Queue) -> None:
    try:
        for raw in conn:
            try:
                q.put(json.loads(raw))
            except json.JSONDecodeError:
                continue
    except Exception:
        pass  # connection went away; the play loop ends on its own terms


def play(
    rom_path,
    viewer_url: str = "http://127.0.0.1:8000",
    max_pieces: int = 50,
    timer_div: int = 0,
    level: int = 0,
    collector=None,
    recorder=None,
    streamer=_AUTO,
) -> dict:
    """Browser-driven human play; returns the same fitness dict a policy arm does."""
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.fitness import FitnessTracker
    from tetris_agent.manual import observe_loop
    from tetris_agent.publisher import NoopPublisher

    try:
        conn = _connect(_input_url(viewer_url), open_timeout=3)
    except Exception as exc:
        raise RuntimeError(
            f"viewer not reachable at {viewer_url} — start `uv run tetris-viewer` first "
            f"and open it in a browser ({exc})"
        ) from exc

    collector = collector or EventCollector(NoopPublisher())
    if streamer is _AUTO:
        from tetris_agent.live import LiveStreamer

        streamer = LiveStreamer(viewer_url)
    if streamer is not None:
        from tetris_agent.agent import _Tee
        from tetris_agent.live import LiveEventSink

        collector.publisher = _Tee(collector.publisher, LiveEventSink(streamer))

    q: queue.Queue = queue.Queue()
    threading.Thread(target=_read_into, args=(conn, q), daemon=True).start()

    emu = Emulator(rom_path, headless=True, speed=1, frame_hook_ticks=_STREAM_EVERY_TICKS)

    print(f"\nBrowser play — seed {timer_div}, up to {max_pieces} pieces.")
    print(f"Waiting at the title screen: press SPACE in the LIVE tab at {viewer_url} to begin.")
    print(f"\n{CONTROLS}\n")
    # Nothing is recorded and no session exists until the player is actually
    # there — the wait phase only streams the title screen.
    try:
        ready = wait_for_start(emu, q, streamer, collector)
    except KeyboardInterrupt:
        ready = False
    if not ready:
        emu.stop()
        try:
            conn.close()
        except Exception:
            pass
        if streamer is not None:
            streamer.close()
        raise RuntimeError("nobody pressed start in the browser — no run recorded")

    emu.frame_hook = make_hook(emu, q, streamer, recorder, collector)
    tracker = FitnessTracker()
    placed = 0
    closed = False
    collector.session(
        "start",
        {"policy": "human", "timer_div": timer_div, "max_pieces": max_pieces, "surface": "browser"},
    )
    try:
        emu.start(timer_div=timer_div)
        emu.force_level(level)
        placed, closed = observe_loop(emu, collector, tracker, max_pieces)
    finally:
        fitness = tracker.compute(score=emu.score, lines=emu.lines, level=emu.level)
        fitness["policy"] = {"policy": "human", "decisions": placed, "cost_usd": 0.0}
        collector.game_over(fitness)
        collector.session("end", {"fitness": fitness})
        emu.stop()
        try:
            conn.close()
        except Exception:
            pass
        if streamer is not None:
            streamer.close()

    if closed:
        print("\n  connection closed.")
    if recorder is not None:
        out = recorder.finalize(fitness=fitness, params={})
        print(f"\n  run recorded to {out}")
    return fitness
