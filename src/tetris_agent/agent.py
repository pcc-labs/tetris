"""The main loop: spawn → read → plan → execute → emit, then fitness."""

import logging

from tetris_agent.board import features
from tetris_agent.controller import Controller
from tetris_agent.events import EventCollector
from tetris_agent.fitness import FitnessTracker
from tetris_agent.policy import Genome, plan_placement
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
    ):
        self.emu = emu
        self.genome = genome
        self.collector = collector
        self.recorder = recorder
        self.max_pieces = max_pieces
        if recorder is not None:
            collector.publisher = _Tee(collector.publisher, _RecorderSink(recorder))

    def run(self, timer_div: int | None = None) -> dict:
        self.emu.start(timer_div=timer_div)
        self.collector.session("start", {"genome": self.genome.to_params(), "timer_div": timer_div})
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
            placement = plan_placement(state.board, state.falling.name, self.genome)
            if placement is None:
                self.collector.stuck(streak=0, detail="no placement fits (top-out imminent)")
                tracker.on_game_over()
                break
            self.collector.spawn(state.falling.name, state.next_piece)
            self.collector.decision(placement)
            result = controller.execute(placement)
            post = read_state(self.emu)
            f = features(post.board)
            tracker.on_lock(f, result.misexec)
            self.collector.locked(result.lines_delta, f, result.misexec)
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
