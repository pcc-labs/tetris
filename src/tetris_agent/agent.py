"""The main loop: spawn → read → plan → execute → emit, then fitness."""

import logging
import time

from tetris_agent.board import features
from tetris_agent.controller import Controller
from tetris_agent.events import EventCollector
from tetris_agent.fitness import FitnessTracker
from tetris_agent.policy import Genome, HeuristicPolicy, Policy
from tetris_agent.recorder import RunRecorder
from tetris_agent.state import read_state

logger = logging.getLogger(__name__)

_MAX_SPAWN_WAIT = 240


class _Tee:
    def __init__(self, *sinks):
        self.sinks = sinks

    def publish(self, event: dict) -> None:
        for sink in self.sinks:
            sink.publish(event)


class _RecorderSink:
    def __init__(self, recorder: RunRecorder):
        self.recorder = recorder

    def publish(self, event: dict) -> None:
        self.recorder.record_event(event)


class TetrisAgent:
    def __init__(
        self,
        emu,
        genome: Genome,
        collector: EventCollector,
        recorder: RunRecorder | None = None,
        max_pieces: int = 300,
        frame_sinks: list | None = None,
        policy: Policy | None = None,
        decision_deadline_s: float | None = None,
        session_meta: dict | None = None,
        clock=time.monotonic,
        meter=None,
        grader=None,
        record_frames: bool = True,
    ):
        self.emu = emu
        self.genome = genome
        self.policy = policy or HeuristicPolicy(genome)
        self.collector = collector
        self.recorder = recorder
        self.max_pieces = max_pieces
        # None means energy is not measured for this run. Left off by default so
        # constructing an agent never probes the host for power sensors; the CLI
        # and the benchmark switch it on for local arms, where the figure matters.
        self.meter = meter
        self.frame_sinks = list(frame_sinks or [])
        # Bounded pause: the game still freezes while the policy thinks (once
        # per piece — plan() is called exactly once), but a decision slower
        # than this many wall-clock seconds is discarded and the piece falls
        # untouched, mirroring the live loop's `late` semantics.
        self.decision_deadline_s = decision_deadline_s
        self.session_meta = dict(session_meta or {})
        self._clock = clock
        self.grader = grader
        self.paused_stats: dict = {"timeouts": 0, "decision_latencies_ms": []}
        if recorder is not None:
            collector.publisher = _Tee(collector.publisher, _RecorderSink(recorder))
            if record_frames:
                self.frame_sinks.append(recorder.record_frame)

    def _capture_frame(self) -> None:
        if not self.frame_sinks:
            return
        png = self.emu.screenshot()
        for sink in self.frame_sinks:
            sink(self.collector.turn, png)

    def run(self, *args, **kwargs) -> dict:
        """Play a run, sampling host power around it when a meter is attached.

        The meter is entered here rather than inside the loop so it is always
        stopped, including when the loop raises — `benchmark.run_arm` catches
        broadly, and a leaked sampler thread would keep spawning powermetrics
        subprocesses for the life of the process.
        """
        if self.meter is None:
            return self._play(*args, **kwargs)
        with self.meter:
            return self._play(*args, **kwargs)

    def _energy_stats(self) -> dict:
        """Energy keys for fitness, read while the meter is still sampling."""
        return self.meter.stats() if self.meter is not None else {}

    def _play(self, timer_div: int | None = None) -> dict:
        self.emu.start(timer_div=timer_div)
        self.collector.session(
            "start",
            {
                "genome": self.genome.to_params(),
                "timer_div": timer_div,
                "policy": self.policy.name,
                **self.session_meta,
            },
        )
        tracker = FitnessTracker()
        controller = Controller(self.emu, ticks_per_press=self.genome.ticks_per_press)

        placed = 0
        while placed < self.max_pieces:
            state = self._await_falling()
            if state is None:
                self.collector.stuck(streak=0, detail="no piece spawned")
                break
            if state.game_over:
                tracker.on_game_over()
                break
            if hasattr(self.policy, "deadline_s"):
                self.policy.deadline_s = self.decision_deadline_s
            # Spawn first: the viewer renders the think-freeze between spawn
            # and decision, and live mode already orders events this way. The
            # frame shows watchers the board being decided on while it's frozen.
            self.collector.spawn(state.falling.name, state.next_piece)
            self._capture_frame()
            plan_started = self._clock()
            placement = self.policy.plan(state.board, state.falling.name, state.next_piece, self.collector.turn)
            latency_ms = (self._clock() - plan_started) * 1000
            if placement is None:
                self.collector.stuck(streak=0, detail="no placement fits (top-out imminent)")
                tracker.on_game_over()
                break
            self.paused_stats["decision_latencies_ms"].append(round(latency_ms, 1))
            timed_out = self.decision_deadline_s is not None and latency_ms > self.decision_deadline_s * 1000
            self.collector.decision(
                placement,
                reason=getattr(self.policy, "last_reason", ""),
                latency_ms=latency_ms,
                late=timed_out,
                tokens=getattr(self.policy, "last_output_tokens", None),
            )
            self._capture_frame()
            if timed_out:
                # The answer missed its one pause; discard it and let gravity
                # place the piece, so a slow model degrades toward no-input.
                self.paused_stats["timeouts"] += 1
                result = controller.run_out()
            else:
                result = controller.execute(placement)
            post = read_state(self.emu)
            f = features(post.board)
            tracker.on_lock(f, result.misexec)
            self.collector.locked(result.lines_delta, f, result.misexec, score=self.emu.score)
            # After the lock, so grading costs the gap between pieces and never
            # the piece's own fall. Late decisions and fallbacks are not choices.
            if self.grader is not None and not timed_out and not getattr(self.policy, "last_fallback", False):
                g = self.grader(state.board, state.falling.name, state.next_piece, placement)
                if g is not None:
                    tracker.on_grade(g)
                    self.collector.graded(g, board=state.board)
            self._capture_frame()
            placed += 1
            if result.replanned:
                self.collector.stuck(streak=result.misexec, detail="placement drifted from plan")
            if post.game_over:
                tracker.on_game_over()
                break
            if not result.locked:
                self.collector.stuck(streak=result.misexec, detail="piece never locked")
                break

        fitness = tracker.compute(score=self.emu.score, lines=self.emu.lines, level=self.emu.level)
        fitness["policy"] = {**self.policy.stats(), **self.paused_stats, **self._energy_stats()}
        self.collector.game_over(fitness)
        self.collector.session("end", {"fitness": fitness})
        if self.recorder is not None:
            out = self.recorder.finalize(fitness=fitness, params=self.genome.to_params())
            logger.info("run recorded to %s", out)
        return fitness

    def _await_falling(self):
        for _ in range(_MAX_SPAWN_WAIT):
            state = read_state(self.emu)
            if state.falling is not None or state.game_over:
                return state
            self.emu.tick(2)
        return None
