import json
from datetime import datetime, timedelta, timezone

from tetris_agent import discovery
from tetris_agent.discovery import COOLDOWN_HOURS, build_bundle, build_prompt, pick_entry, run, run_gates

ENTRY = {
    "rule": "hole-thrash",
    "fitness": {"avg_holes": 6.0, "score": 120, "pieces_placed": 50},
    "race": {"control": 100.0, "winner_score": 90.0},
    "queued_at": "2026-08-01T00:00:00+00:00",
    "attempted": False,
}


def test_pick_entry_skips_attempted_and_honors_cooldown():
    queue = [{**ENTRY, "attempted": True}, {**ENTRY, "queued_at": "2026-08-02T00:00:00+00:00"}]
    assert pick_entry(queue, {})["queued_at"] == "2026-08-02T00:00:00+00:00"
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert pick_entry(queue, {"last_attempt_at": recent}) is None
    old = (datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS + 1)).isoformat()
    assert pick_entry(queue, {"last_attempt_at": old}) is not None
    assert pick_entry([{**ENTRY, "attempted": True}], {}) is None


def test_prompt_contains_rule_fitness_and_files():
    bundle = build_bundle(ENTRY, {"races": [{"rule": "hole-thrash", "control": 100.0, "accepted": False}]})
    prompt = build_prompt(bundle)
    assert "hole-thrash" in prompt
    assert "6.0" in prompt
    assert "src/tetris_agent/policy.py" in prompt
    assert "data/genome.json" in prompt  # the do-not-touch instruction


def test_gates_short_circuit_on_pytest_failure(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)

        class R:
            returncode = 1
            stdout = ""
            stderr = "boom"

        return R()

    monkeypatch.setattr(discovery.subprocess, "run", fake_run)
    result = run_gates(tmp_path, "rom/tetris.gb", ENTRY)
    assert result["passed"] is False
    assert result["reason"] == "pytest"
    assert len(calls) == 1


def test_run_success_path_opens_pr(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps([ENTRY]))
    state_path = tmp_path / "state.json"
    events = []

    monkeypatch.setattr(discovery, "worktree_add", lambda root, branch: tmp_path / "wt")
    monkeypatch.setattr(discovery, "propose", lambda wt, prompt: events.append("propose") or True)
    monkeypatch.setattr(discovery, "run_gates", lambda wt, rom, entry: {"passed": True})
    monkeypatch.setattr(discovery, "push_and_pr", lambda wt, branch, entry: events.append("pr") or "http://pr/1")
    monkeypatch.setattr(discovery, "cleanup", lambda wt, branch: events.append("cleanup"))

    report = run(queue_path=queue_path, state_path=state_path, healer_state_path=tmp_path / "h.json")
    assert report["status"] == "pr"
    assert report["url"] == "http://pr/1"
    assert events == ["propose", "pr"]
    assert json.loads(queue_path.read_text())[0]["attempted"] is True
    assert "last_attempt_at" in json.loads(state_path.read_text())


def test_run_failed_gates_cleans_up(monkeypatch, tmp_path):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps([ENTRY]))
    events = []

    monkeypatch.setattr(discovery, "worktree_add", lambda root, branch: tmp_path / "wt")
    monkeypatch.setattr(discovery, "propose", lambda wt, prompt: True)
    monkeypatch.setattr(discovery, "run_gates", lambda wt, rom, entry: {"passed": False, "reason": "fitness"})
    monkeypatch.setattr(discovery, "push_and_pr", lambda wt, branch, entry: events.append("pr"))
    monkeypatch.setattr(discovery, "cleanup", lambda wt, branch: events.append("cleanup"))

    report = run(queue_path=queue_path, state_path=tmp_path / "state.json", healer_state_path=tmp_path / "h.json")
    assert report["status"] == "failed"
    assert report["reason"] == "fitness"
    assert events == ["cleanup"]
    assert json.loads(queue_path.read_text())[0]["attempted"] is True


def test_run_idle_when_queue_empty(tmp_path):
    report = run(
        queue_path=tmp_path / "missing.json", state_path=tmp_path / "state.json", healer_state_path=tmp_path / "h.json"
    )
    assert report["status"] == "idle"
