import json

import pytest

from tetris_agent.cli import build_config


def test_build_config_defaults():
    cfg = build_config([])
    assert cfg.rom == "rom/tetris.gb"
    assert cfg.max_pieces == 300
    assert cfg.timer_div is None
    assert cfg.output_json is None
    assert cfg.telemetry_dir == "data/telemetry"
    assert cfg.runs_dir == "runs"
    assert not cfg.no_telemetry and not cfg.no_record and not cfg.no_self_heal


def test_build_config_flags():
    cfg = build_config(
        ["--rom", "x.gb", "--max-pieces", "50", "--timer-div", "0", "--no-self-heal", "--no-record", "--label", "demo"]
    )
    assert cfg.rom == "x.gb"
    assert cfg.max_pieces == 50
    assert cfg.timer_div == 0
    assert cfg.no_self_heal and cfg.no_record
    assert cfg.label == "demo"


@pytest.mark.rom
def test_agent_places_pieces_and_records(rom_path, tmp_path):
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.emulator import Emulator
    from tetris_agent.events import EventCollector
    from tetris_agent.policy import Genome
    from tetris_agent.publisher import NoopPublisher
    from tetris_agent.recorder import RunRecorder

    emu = Emulator(rom_path)
    recorder = RunRecorder(tmp_path, label="test")
    agent = TetrisAgent(
        emu,
        genome=Genome(),
        collector=EventCollector(NoopPublisher()),
        recorder=recorder,
        max_pieces=5,
    )
    try:
        fitness = agent.run(timer_div=0x00)
    finally:
        emu.stop()
    assert fitness["pieces_placed"] == 5
    assert fitness["topped_out"] is False
    run_dir = tmp_path / recorder.run_id
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["fitness"] == fitness
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    spawns = [e for e in events if e["event_type"] == "piece_spawn"]
    assert len(spawns) == 5


def test_build_config_live_flag():
    cfg = build_config(["--live"])
    assert cfg.live is True
    assert build_config([]).live is False


def test_build_config_policy_defaults_and_overrides():
    cfg = build_config([])
    assert cfg.policy == "heuristic"
    assert cfg.harness == "features"
    cfg = build_config(["--policy", "model", "--model", "claude-sonnet-5", "--harness", "board", "--effort", "low"])
    assert (cfg.policy, cfg.model, cfg.harness, cfg.effort) == ("model", "claude-sonnet-5", "board", "low")


def test_build_policy_returns_heuristic_without_touching_the_api():
    from tetris_agent.cli import build_policy

    policy = build_policy(build_config([]))
    assert policy.name == "heuristic"
    assert policy.stats()["cost_usd"] == 0.0


def test_build_config_parses_level_for_live_gravity():
    assert build_config(["--live", "--level", "3"]).level == 3
    assert build_config([]).level == 0


def test_agent_class_routes_live_to_the_no_pause_loop():
    from tetris_agent.agent import TetrisAgent
    from tetris_agent.cli import _agent_class
    from tetris_agent.live_agent import LiveTetrisAgent

    assert _agent_class(True) is LiveTetrisAgent
    assert _agent_class(False) is TetrisAgent
