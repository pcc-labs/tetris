"""Command-line entrypoints."""

import argparse
import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path

from tetris_agent.pricing import EFFORTS, is_pi
from tetris_agent.prompts import HARNESSES

DEFAULT_FITNESS_PATH = "data/last_fitness.json"


@dataclass
class Config:
    rom: str
    max_pieces: int
    timer_div: int | None
    output_json: str | None
    telemetry_dir: str
    no_telemetry: bool
    runs_dir: str
    no_record: bool
    label: str
    no_self_heal: bool
    window: str
    live: bool
    level: int
    viewer_url: str
    policy: str
    model: str
    harness: str
    effort: str
    exemplars: str | None
    no_power: bool


def build_config(argv=None) -> Config:
    parser = argparse.ArgumentParser(prog="tetris-agent", description="Self-healing Game Boy Tetris agent")
    parser.add_argument("--rom", default="rom/tetris.gb")
    parser.add_argument("--max-pieces", type=int, default=300)
    parser.add_argument("--timer-div", type=int, default=None, help="fix the piece RNG (0-255) for determinism")
    parser.add_argument("--output-json", default=None, help=f"fitness output path (default {DEFAULT_FITNESS_PATH})")
    parser.add_argument("--telemetry-dir", default="data/telemetry")
    parser.add_argument("--no-telemetry", action="store_true")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--label", default="")
    parser.add_argument("--no-self-heal", action="store_true")
    parser.add_argument("--window", default="null", choices=["null", "SDL2"])
    parser.add_argument(
        "--live",
        action="store_true",
        help="play in real time — gravity keeps running while the model thinks; streams to the viewer",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        choices=range(10),
        help="starting level/gravity for --live; ignored otherwise",
    )
    parser.add_argument("--viewer-url", default="ws://127.0.0.1:8000", help="viewer WebSocket base for --live")
    parser.add_argument("--policy", default="heuristic", choices=["heuristic", "model"])
    parser.add_argument("--model", default="claude-opus-5", help="model id when --policy model")
    parser.add_argument("--harness", default="features", choices=list(HARNESSES))
    parser.add_argument("--effort", default="medium", choices=list(EFFORTS))
    parser.add_argument(
        "--exemplars",
        nargs="?",
        const="runs",
        default=None,
        metavar="RUNS_DIR",
        help="inject verified human traces from RUNS_DIR (default runs/) into the model's system prompt",
    )
    parser.add_argument(
        "--no-power",
        action="store_true",
        help="skip host power sampling for local arms (their energy cost then reports n/a)",
    )
    return Config(**vars(parser.parse_args(argv)))


def build_policy(cfg: Config):
    from tetris_agent.genome import load_params
    from tetris_agent.policy import Genome, HeuristicPolicy

    genome = Genome.from_params(load_params())
    if cfg.policy == "heuristic":
        return HeuristicPolicy(genome)
    from tetris_agent.pricing import is_pi

    block = ""
    if cfg.exemplars:
        from tetris_agent.traces import load_exemplar_block

        block = load_exemplar_block(cfg.exemplars)
    if is_pi(cfg.model):
        from tetris_agent.pi_policy import PiPolicy

        return PiPolicy(model=cfg.model, harness=cfg.harness, effort=cfg.effort, exemplar_block=block, genome=genome)
    from tetris_agent.model_policy import ModelPolicy

    return ModelPolicy(model=cfg.model, harness=cfg.harness, effort=cfg.effort, exemplar_block=block, genome=genome)


def _agent_class(live: bool):
    """Live runs use the no-pause loop; everything else keeps the frozen one."""
    if live:
        from tetris_agent.live_agent import LiveTetrisAgent

        return LiveTetrisAgent
    from tetris_agent.agent import TetrisAgent

    return TetrisAgent


def agent_main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    cfg = build_config(argv)

    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.genome import load_params
    from tetris_agent.policy import Genome
    from tetris_agent.publisher import JSONLPublisher, NoopPublisher
    from tetris_agent.recorder import RunRecorder

    genome = Genome.from_params(load_params())
    session_id = secrets.token_hex(4)
    publisher = NoopPublisher() if cfg.no_telemetry else JSONLPublisher(cfg.telemetry_dir, session_id)
    recorder = None if cfg.no_record else RunRecorder(cfg.runs_dir, label=cfg.label)

    streamer = None
    frame_sinks = []
    if cfg.live:
        from tetris_agent.agent import _Tee
        from tetris_agent.live import LiveEventSink, LiveStreamer

        streamer = LiveStreamer(cfg.viewer_url)
        publisher = _Tee(publisher, LiveEventSink(streamer))
        frame_sinks.append(streamer.send_frame)

    # Local arms bill no API, so their cost is watts. Cloud arms draw their power
    # in someone else's datacenter and already report it as cost_usd.
    meter = None
    if not cfg.no_power and cfg.policy == "model" and is_pi(cfg.model):
        from tetris_agent.power import EnergyMeter

        meter = EnergyMeter()

    emu = Emulator(cfg.rom, headless=cfg.window == "null", speed=1 if cfg.live else 0)
    try:
        agent = _agent_class(cfg.live)(
            emu, genome, EventCollector(publisher), recorder=recorder, max_pieces=cfg.max_pieces,
            frame_sinks=frame_sinks, policy=build_policy(cfg), meter=meter,
        )
        if streamer is not None:
            emu.frame_hook = lambda: streamer.send_frame(agent.collector.turn, emu.screenshot())
        if cfg.live:
            fitness = agent.run(timer_div=cfg.timer_div, level=cfg.level)
        else:
            fitness = agent.run(timer_div=cfg.timer_div)
    finally:
        emu.stop()
        if streamer is not None:
            streamer.close()

    print(json.dumps(fitness, indent=2))
    fitness_path = Path(cfg.output_json or DEFAULT_FITNESS_PATH)
    fitness_path.parent.mkdir(parents=True, exist_ok=True)
    fitness_path.write_text(json.dumps(fitness))

    if not cfg.no_self_heal:
        try:
            from tetris_agent.healer import check

            report = check(fitness_path, rom_path=cfg.rom)
            print(json.dumps({"healer": report}, indent=2))
        except ImportError:
            # Healer module lands in a later task; the chain activates itself then.
            pass
    return 0


def healer_main(argv=None) -> int:
    from tetris_agent.healer import main as _main

    return _main(argv)


def discovery_main(argv=None) -> int:
    from tetris_agent.discovery import main as _main

    return _main(argv)


def manual_main(argv=None) -> int:
    """Human baseline: play by hand, recorded as a benchmark arm."""
    import argparse

    from tetris_agent.benchmark import ArmResult, render_table, summarize, write_results
    from tetris_agent.manual import play

    parser = argparse.ArgumentParser(prog="tetris-manual", description="Play by hand to set a human baseline")
    parser.add_argument("--rom", default="rom/tetris.gb")
    parser.add_argument("--max-pieces", type=int, default=50, help="match the model arms you're comparing against")
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0, help="timer_div; same seed = same pieces")
    parser.add_argument("--label", default="human", help="arm name in the results table")
    parser.add_argument("--no-save", action="store_true", help="print the result without writing it")
    parser.add_argument("--runs-dir", default="runs", help="where to record the per-piece replay")
    parser.add_argument("--no-record", action="store_true", help="skip the runs/ replay recording")
    args = parser.parse_args(argv)

    from tetris_agent.agent import _RecorderSink
    from tetris_agent.events import EventCollector
    from tetris_agent.publisher import NoopPublisher
    from tetris_agent.recorder import RunRecorder

    recorder = None if args.no_record else RunRecorder(args.runs_dir, label=args.label)
    collector = EventCollector(NoopPublisher())
    if recorder is not None:
        collector.publisher = _RecorderSink(recorder)

    fitness = play(
        args.rom,
        max_pieces=args.max_pieces,
        timer_div=args.seed,
        collector=collector,
        recorder=recorder,
    )
    result = ArmResult(
        arm=args.label,
        seed=args.seed,
        fitness=fitness,
        policy_stats={"policy": args.label, "decisions": 0, "cost_usd": 0.0},
    )
    rows = summarize([result])
    print("\n" + render_table(rows))
    if not args.no_save:
        print(f"\nresults: {write_results([result], rows)}")
    return 0


def play_main(argv=None) -> int:
    """Human play in the browser: streams into the viewer, records a human arm."""
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    import argparse

    from tetris_agent import browser_play
    from tetris_agent.benchmark import ArmResult, render_table, summarize, write_results

    parser = argparse.ArgumentParser(
        prog="tetris-play", description="Play in the browser; traces recorded like any arm"
    )
    parser.add_argument("--rom", default="rom/tetris.gb")
    parser.add_argument("--max-pieces", type=int, default=50, help="match the model arms you're comparing against")
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0, help="timer_div; same seed = same pieces")
    parser.add_argument("--label", default="human", help="arm name in the results table")
    parser.add_argument("--viewer-url", default="http://127.0.0.1:8000")
    parser.add_argument("--level", type=int, default=0, choices=range(10), help="starting level (gravity)")
    parser.add_argument("--no-save", action="store_true", help="print the result without writing it")
    parser.add_argument("--runs-dir", default="runs", help="where to record the per-piece replay")
    parser.add_argument("--no-record", action="store_true", help="skip the runs/ replay recording")
    args = parser.parse_args(argv)

    from tetris_agent.agent import _RecorderSink
    from tetris_agent.events import EventCollector
    from tetris_agent.publisher import NoopPublisher
    from tetris_agent.recorder import RunRecorder

    recorder = None if args.no_record else RunRecorder(args.runs_dir, label=args.label)
    collector = EventCollector(NoopPublisher())
    if recorder is not None:
        collector.publisher = _RecorderSink(recorder)

    fitness = browser_play.play(
        args.rom,
        viewer_url=args.viewer_url,
        max_pieces=args.max_pieces,
        timer_div=args.seed,
        level=args.level,
        collector=collector,
        recorder=recorder,
    )
    result = ArmResult(
        arm=args.label,
        seed=args.seed,
        fitness=fitness,
        policy_stats={"policy": args.label, "decisions": 0, "cost_usd": 0.0},
    )
    rows = summarize([result])
    print("\n" + render_table(rows))
    if not args.no_save:
        print(f"\nresults: {write_results([result], rows)}")
    return 0
