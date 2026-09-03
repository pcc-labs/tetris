"""The live loop: gravity keeps running while the policy thinks.

TetrisAgent freezes the emulator for as long as policy.plan takes, so a model
gets unlimited thinking time per piece. This subclass removes the pause: the
decision runs on a worker thread while this thread keeps ticking the emulator
in real time, making thinking time cost gravity distance exactly as it does
for a human. A decision that arrives after its piece already locked is
discarded as late, and a slow model degrades honestly toward the no-input
floor.

PyBoy is not thread-safe, so the emulator is only ever touched from the
thread running `run`; the worker sees a copy of the board and plain strings,
never the emulator handle. At most one decision is in flight at a time.
"""

import logging
import queue
import threading
import time

from tetris_agent.agent import TetrisAgent
from tetris_agent.board import ROWS, features
from tetris_agent.controller import Controller
from tetris_agent.fitness import FitnessTracker
from tetris_agent.policy import Placement, legal_moves
from tetris_agent.state import read_state

logger = logging.getLogger(__name__)

_MAX_SPAWN_WAIT = 240
# The decision must land before the piece does, and the controller still needs
# real frames to rotate/shift/drop it afterwards.
_EXEC_HEADROOM_S = 2.0


class _Pending:
    """One in-flight decision, tagged with the turn/piece it was asked for."""

    def __init__(self, turn: int, piece: str, board=None, next_piece: str | None = None):
        self.turn = turn
        self.piece = piece
        self.board = board
        self.next_piece = next_piece
        self.submitted_at = time.monotonic()
        self._q: queue.Queue = queue.Queue(maxsize=1)

    def put(self, outcome: tuple) -> None:
        self._q.put(outcome)

    def poll(self) -> tuple | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def latency_ms(self) -> float:
        return (time.monotonic() - self.submitted_at) * 1000.0


class LiveTetrisAgent(TetrisAgent):
    def __init__(self, *args, thread_factory=threading.Thread, **kwargs):
        super().__init__(*args, **kwargs)
        self.thread_factory = thread_factory
        self.live_stats = {"late": 0, "worker_errors": 0, "in_flight_at_end": False}
        self._last_decision = None

    def _play(self, timer_div: int | None = None, level: int | None = 0) -> dict:
        self._level = level
        self.emu.start(timer_div=timer_div)
        if level is not None:
            # Model arms otherwise inherit whatever level the blind menu
            # presses left; pinning it makes the gravity budget deterministic.
            self.emu.force_level(level)
        self.collector.session(
            "start",
            {
                "genome": self.genome.to_params(),
                "timer_div": timer_div,
                "policy": self.policy.name,
                "mode": "live",
                "level": level,
            },
        )
        tracker = FitnessTracker()
        controller = Controller(self.emu, ticks_per_press=self.genome.ticks_per_press)

        pending: _Pending | None = None
        placed = 0
        while placed < self.max_pieces:
            state, closed = self._await_spawn()
            if closed:
                break
            if state is None:
                self.collector.stuck(streak=0, detail="no piece spawned")
                break
            if state.game_over:
                tracker.on_game_over()
                break

            self.collector.spawn(state.falling.name, state.next_piece)
            self._capture_frame()
            lines_before = self.emu.lines
            if pending is None:
                pending = self._submit(state)

            pending, outcome, exec_result = self._ride_piece(pending, controller)

            if outcome == "executed":
                post = read_state(self.emu)
                f = features(post.board)
                tracker.on_lock(f, exec_result.misexec)
                self.collector.locked(self.emu.lines - lines_before, f, exec_result.misexec, score=self.emu.score)
                decided, self._last_decision = self._last_decision, None
                if self.grader is not None and decided is not None and not getattr(self.policy, "last_fallback", False):
                    board, piece, next_piece, placement = decided
                    g = self.grader(board, piece, next_piece, placement)
                    if g is not None:
                        tracker.on_grade(g)
                        self.collector.graded(g)
                self._capture_frame()
                placed += 1
                if exec_result.replanned:
                    self.collector.stuck(streak=exec_result.misexec, detail="placement drifted from plan")
                if post.game_over:
                    tracker.on_game_over()
                    break
                if not exec_result.locked:
                    self.collector.stuck(streak=exec_result.misexec, detail="piece never locked")
                    break
            elif outcome == "locked":
                post = self._lock_without_decision(tracker, lines_before)
                placed += 1
                if post.game_over:
                    tracker.on_game_over()
                    break
            elif outcome == "game_over":
                tracker.on_game_over()
                break
            elif outcome == "stop":
                self.collector.stuck(streak=0, detail="no placement fits (top-out imminent)")
                tracker.on_game_over()
                break
            else:  # "closed"
                break

        if pending is not None:
            self.live_stats["in_flight_at_end"] = True
        fitness = tracker.compute(score=self.emu.score, lines=self.emu.lines, level=self.emu.level)
        fitness["policy"] = {**self.policy.stats(), **self.live_stats, **self._energy_stats()}
        self.collector.game_over(fitness)
        self.collector.session("end", {"fitness": fitness})
        if self.recorder is not None:
            out = self.recorder.finalize(fitness=fitness, params=self.genome.to_params())
            logger.info("run recorded to %s", out)
        return fitness

    def _ride_piece(self, pending: _Pending, controller: Controller):
        """Tick until the piece resolves: executed, locked on its own, or the
        run ends. Returns (pending, outcome, exec_result) — pending survives
        a piece that locks while its decision is still in flight."""
        while True:
            result = pending.poll() if pending is not None else None
            if result is not None:
                latency = pending.latency_ms()
                stale = pending.turn != self.collector.turn
                s = read_state(self.emu)
                if stale:
                    # A decision for a piece that already locked. Discard it
                    # and ask again for the piece falling right now.
                    self._discard_late(result, pending, latency)
                    if s.game_over:
                        return None, "game_over", None
                    if s.falling is None:
                        return None, "locked", None
                    pending = self._submit(s)
                    continue
                if s.game_over:
                    return None, "game_over", None
                if s.falling is None or s.falling.name != pending.piece:
                    # Its own piece locked in the gap since the last poll:
                    # too late, and Controller.execute must not see it.
                    self._discard_late(result, pending, latency)
                    return None, "locked", None
                placement, reason = self._resolve(result, s)
                if placement is None:
                    return None, "stop", None
                self.collector.decision(
                    placement,
                    reason=reason,
                    latency_ms=latency,
                    tokens=getattr(self.policy, "last_output_tokens", None),
                )
                self._capture_frame()
                self._last_decision = (pending.board, pending.piece, pending.next_piece, placement)
                return None, "executed", controller.execute(placement)
            if not self.emu.tick(2):
                return pending, "closed", None
            s = read_state(self.emu)
            if s.game_over:
                return pending, "game_over", None
            if s.falling is None:
                return self._reap_on_lock(pending), "locked", None

    def _deadline_s(self, state) -> float:
        """Seconds until this piece reaches the floor, minus execution headroom.

        The policy budgets its thinking against this: the game does not pause.
        """
        from tetris_agent.emulator import Emulator

        level = self._level if self._level is not None else getattr(self.emu, "level", 0)
        frames_per_row = Emulator._GRAVITY_RELOADS[min(level, 9)] + 1
        rows_to_fall = (ROWS - 1) - state.falling.bottom_row
        return max(1.0, rows_to_fall * frames_per_row / 60 - _EXEC_HEADROOM_S)

    def _submit(self, state) -> _Pending:
        self.policy.deadline_s = self._deadline_s(state)
        board = state.board.copy()
        piece, next_piece, turn = state.falling.name, state.next_piece, self.collector.turn
        pending = _Pending(turn=turn, piece=piece, board=board, next_piece=next_piece)

        def work():
            try:
                pending.put(("ok", self.policy.plan(board, piece, next_piece, turn)))
            except Exception as exc:  # noqa: BLE001 — a dead worker must surface as a result
                pending.put(("error", exc))

        self.thread_factory(target=work, daemon=True, name="decision").start()
        return pending

    def _await_spawn(self):
        """Tick until a piece is falling; (state, closed) distinguishes a
        closed window from a spawn that never came."""
        for _ in range(_MAX_SPAWN_WAIT):
            state = read_state(self.emu)
            if state.falling is not None or state.game_over:
                return state, False
            if not self.emu.tick(2):
                return state, True
        return None, False

    def _lock_without_decision(self, tracker: FitnessTracker, lines_before: int):
        """Gravity placed the piece before any decision arrived."""
        settled = read_state(self.emu)
        f = features(settled.board)
        tracker.on_lock(f, misexec=0)
        self.collector.locked(self.emu.lines - lines_before, f, misexec=0, score=self.emu.score)
        self._capture_frame()
        return settled

    def _reap_on_lock(self, pending: _Pending) -> _Pending | None:
        """One last poll as the piece locks: a decision that landed on the
        locking tick is late, not still in flight."""
        result = pending.poll()
        if result is None:
            return pending
        self._discard_late(result, pending, pending.latency_ms())
        return None

    def _discard_late(self, result: tuple, pending: _Pending, latency: float) -> None:
        self.live_stats["late"] += 1
        kind, value = result
        if kind == "ok" and value is not None:
            self.collector.decision(
                value,
                reason=getattr(self.policy, "last_reason", ""),
                latency_ms=latency,
                late=True,
                turn=pending.turn,
            )

    def _resolve(self, result: tuple, state) -> tuple[Placement | None, str]:
        """Worker outcome → (placement, reason); (None, _) means top-out."""
        kind, value = result
        if kind == "ok":
            return value, getattr(self.policy, "last_reason", "")
        self.live_stats["worker_errors"] += 1
        logger.warning("decision worker failed: %s", value)
        moves = legal_moves(state.board, state.falling.name)
        if not moves:
            return None, ""
        return Placement(rotation=moves[0][0], col=moves[0][1], score=0.0), "worker error fallback"
