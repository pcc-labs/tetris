"""Command-line entrypoints."""

import argparse
import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path

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
    return Config(**vars(parser.parse_args(argv)))


def agent_main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    cfg = build_config(argv)

    from tetris_agent.agent import TetrisAgent
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

    emu = Emulator(cfg.rom, headless=cfg.window == "null")
    try:
        agent = TetrisAgent(emu, genome, EventCollector(publisher), recorder=recorder, max_pieces=cfg.max_pieces)
        fitness = agent.run(timer_div=cfg.timer_div)
    finally:
        emu.stop()

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
