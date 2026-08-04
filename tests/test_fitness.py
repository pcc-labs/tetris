import json

from tetris_agent.board import Features
from tetris_agent.fitness import FitnessTracker, race_score
from tetris_agent.recorder import RunRecorder


def F(holes=0, max_height=2):
    return Features(holes=holes, agg_height=4, bumpiness=1, max_height=max_height)


def test_fitness_averages_holes_and_tracks_streaks():
    t = FitnessTracker()
    t.on_lock(F(holes=2), misexec=1)
    t.on_lock(F(holes=4), misexec=1)
    t.on_lock(F(holes=0), misexec=0)  # clean lock resets the streak
    t.on_lock(F(holes=2), misexec=3)
    fitness = t.compute(score=1200, lines=3, level=1)
    assert fitness["pieces_placed"] == 4
    assert fitness["avg_holes"] == 2.0
    assert fitness["misexec_count"] == 5
    assert fitness["max_misexec_streak"] == 2
    assert fitness["topped_out"] is False
    assert fitness["score"] == 1200 and fitness["lines"] == 3 and fitness["level"] == 1


def test_fitness_records_top_out_and_max_stack():
    t = FitnessTracker()
    t.on_lock(F(max_height=11), misexec=0)
    t.on_game_over()
    fitness = t.compute(score=0, lines=0, level=0)
    assert fitness["topped_out"] is True
    assert fitness["max_stack_height"] == 11


def test_race_score():
    assert race_score({"score": 100, "pieces_placed": 20}) == 200.0


def test_recorder_writes_run_directory(tmp_path):
    rec = RunRecorder(tmp_path, label="test-run")
    rec.record_event({"event_type": "piece_spawn", "turn": 1})
    rec.record_event({"event_type": "piece_locked", "turn": 1})
    out = rec.finalize(fitness={"score": 10}, params={"w_holes": -0.3})
    assert out.parent == tmp_path
    events = (out / "events.jsonl").read_text().splitlines()
    assert len(events) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["fitness"]["score"] == 10
    assert summary["params"]["w_holes"] == -0.3
    assert summary["run_id"] == out.name
    meta = json.loads((out / "meta.json").read_text())
    assert meta["label"] == "test-run"


def test_recorder_writes_frames_immediately(tmp_path):
    rec = RunRecorder(tmp_path, label="frames")
    rec.record_frame(1, b"\x89PNG-one")
    rec.record_frame(2, b"\x89PNG-two")
    frames = sorted((tmp_path / rec.run_id / "frames").iterdir())
    assert [f.name for f in frames] == ["0001-t1.png", "0002-t2.png"]
    assert frames[0].read_bytes() == b"\x89PNG-one"
