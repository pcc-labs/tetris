import json
from datetime import datetime, timedelta, timezone

import pytest

from tetris_agent.healer import (
    COOLDOWN_HOURS,
    RULES,
    check,
    default_seed,
    evaluate_rules,
    should_escalate,
)
from tetris_agent.policy import Genome

GOOD = {
    "score": 500,
    "lines": 5,
    "pieces_placed": 150,
    "avg_holes": 1.0,
    "misexec_count": 0,
    "max_misexec_streak": 0,
    "topped_out": False,
}


def fitness_file(tmp_path, **overrides):
    path = tmp_path / "fitness.json"
    path.write_text(json.dumps({**GOOD, **overrides}))
    return path


def paths(tmp_path):
    return dict(
        state_path=tmp_path / "healer_state.json",
        genome_path=tmp_path / "genome.json",
        queue_path=tmp_path / "queue.json",
    )


def accepting_race(control, variants, seeds, rom_path, runner=None):
    return {
        "control": 100.0,
        "candidates": [{"params": variants[0], "score": 200.0}],
        "winner": {"params": variants[0], "score": 200.0},
    }


def rejecting_race(control, variants, seeds, rom_path, runner=None):
    return {
        "control": 100.0,
        "candidates": [{"params": variants[0], "score": 90.0}],
        "winner": {"params": variants[0], "score": 90.0},
    }


def test_rule_matching_precedence_and_no_rule():
    assert evaluate_rules(GOOD) == []
    fired = evaluate_rules({**GOOD, "topped_out": True, "pieces_placed": 20, "avg_holes": 9})
    assert [r.name for r in fired] == ["early-topout", "hole-thrash"]
    assert [r.name for r in evaluate_rules({**GOOD, "max_misexec_streak": 3})] == ["misexecution"]


def test_no_rule_returns_early(tmp_path):
    report = check(fitness_file(tmp_path), race=accepting_race, **paths(tmp_path))
    assert report["status"] == "no-rule"


def test_accepted_race_writes_genome_and_state(tmp_path):
    p = paths(tmp_path)
    report = check(fitness_file(tmp_path, avg_holes=6.0), race=accepting_race, **p)
    assert report["status"] == "accepted"
    assert report["rule"] == "hole-thrash"
    stored = json.loads(p["genome_path"].read_text())
    assert stored["history"][0]["source"] == "healer:hole-thrash"
    state = json.loads(p["state_path"].read_text())
    assert state["rules"]["hole-thrash"]["last_outcome"] == "accepted"


def test_rejected_race_leaves_genome_untouched(tmp_path):
    p = paths(tmp_path)
    report = check(fitness_file(tmp_path, avg_holes=6.0), race=rejecting_race, **p)
    assert report["status"] == "rejected"
    assert not p["genome_path"].exists()


def test_cooldown_blocks_second_race(tmp_path):
    p = paths(tmp_path)
    check(fitness_file(tmp_path, avg_holes=6.0), race=accepting_race, **p)
    report = check(fitness_file(tmp_path, avg_holes=6.0), race=accepting_race, **p)
    assert report["status"] == "cooldown"


def test_cooldown_expires(tmp_path):
    p = paths(tmp_path)
    check(fitness_file(tmp_path, avg_holes=6.0), race=rejecting_race, **p)
    state = json.loads(p["state_path"].read_text())
    old = datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS + 1)
    state["rules"]["hole-thrash"]["last_race_at"] = old.isoformat()
    p["state_path"].write_text(json.dumps(state))
    report = check(fitness_file(tmp_path, avg_holes=6.0), race=rejecting_race, **p)
    assert report["status"] == "rejected"


def test_two_rejections_escalate(tmp_path):
    p = paths(tmp_path)
    for _ in range(2):
        check(fitness_file(tmp_path, avg_holes=6.0), race=rejecting_race, **p)
        state = json.loads(p["state_path"].read_text())
        old = datetime.now(timezone.utc) - timedelta(hours=COOLDOWN_HOURS + 1)
        state["rules"]["hole-thrash"]["last_race_at"] = old.isoformat()
        p["state_path"].write_text(json.dumps(state))
    queue = json.loads(p["queue_path"].read_text())
    assert len(queue) == 1
    assert queue[0]["rule"] == "hole-thrash"
    assert queue[0]["fitness"]["avg_holes"] == 6.0


def test_refire_after_accept_escalates():
    state = {"rules": {"hole-thrash": {"last_outcome": "accepted", "consecutive_rejections": 0}}}
    assert should_escalate("hole-thrash", state) is True
    assert should_escalate("early-topout", state) is False


def test_corrupt_fitness_returns_error(tmp_path):
    path = tmp_path / "fitness.json"
    path.write_text("{broken")
    report = check(path, race=accepting_race, **paths(tmp_path))
    assert report["status"] == "error"


def test_default_seed_deterministic():
    assert default_seed("a/b.json") == default_seed("a/b.json")
    assert default_seed("a/b.json") != default_seed("a/c.json")


def test_rules_reference_real_genome_params():
    genome_keys = set(Genome().to_params())
    for rule in RULES:
        assert set(rule.params) <= genome_keys


@pytest.mark.rom
def test_end_to_end_heal_with_sabotaged_genome(rom_path, tmp_path):
    """A genome that loves holes plays badly; the healer races it and reports."""
    from tetris_agent.evolve import run_agent
    from tetris_agent.genome import append_genome

    p = paths(tmp_path)
    sabotaged = {**Genome().to_params(), "w_holes": 1.0, "w_bumpiness": 0.5}
    append_genome(sabotaged, source="test:sabotage", path=p["genome_path"])

    fitness = run_agent(sabotaged, timer_div=0x00, rom_path=rom_path, max_pieces=20)
    assert fitness["avg_holes"] >= 4, f"sabotage was not bad enough: {fitness}"
    fpath = tmp_path / "fitness.json"
    fpath.write_text(json.dumps(fitness))

    report = check(fpath, rom_path=rom_path, n_variants=2, seeds_per=1, max_pieces=20, **p)
    assert report["status"] in ("accepted", "rejected")
    # A genome this bad can top out before 40 pieces, in which case early-topout outranks hole-thrash.
    assert report["rule"] in ("early-topout", "hole-thrash")
    state = json.loads(p["state_path"].read_text())
    assert len(state["races"]) == 1
