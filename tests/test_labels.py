"""Reviewer labels: written per run, read by the exemplar miner."""

import json

import numpy as np
from test_traces import O_AT_0, THEN_O_AT_2, write_run

from tetris_agent import jev
from tetris_agent.labels import clear_label, load_labels, set_label
from tetris_agent.traces import Exemplar, decisions, mine, mine_run, select_exemplars

THEN_O_AT_4 = {"holes": 0, "agg_height": 12, "bumpiness": 2, "max_height": 2}
THREE_O = [("O", "O", 0, 0, 0, O_AT_0), ("O", "O", 0, 2, 0, THEN_O_AT_2), ("O", "I", 0, 4, 0, THEN_O_AT_4)]


def test_set_and_clear_label_round_trip(tmp_path):
    label = set_label(tmp_path, 3, "promote", note="nice tuck", jev={"verdict": {"choice": "promote"}})
    assert label["verdict"] == "promote" and label["source"] == "human"
    assert load_labels(tmp_path) == {3: label}
    on_disk = json.loads((tmp_path / "labels.json").read_text())
    assert list(on_disk) == ["3"]  # turns keyed as strings, sorted
    assert clear_label(tmp_path, 3) is True
    assert clear_label(tmp_path, 3) is False
    assert load_labels(tmp_path) == {}


def test_unknown_verdicts_are_ignored_on_read_and_refused_on_write(tmp_path):
    (tmp_path / "labels.json").write_text('{"1": {"verdict": "maybe"}, "2": "promote", "3": {"verdict": "exclude"}}')
    assert list(load_labels(tmp_path)) == [3]
    try:
        set_label(tmp_path, 1, "maybe")
    except ValueError:
        pass
    else:
        raise AssertionError("bad verdict accepted")


def test_excluded_turns_leave_the_pool_without_breaking_the_replay(tmp_path):
    run = write_run(tmp_path, "20260808-000001-aaaaaa", "human", THREE_O)
    set_label(run, 2, "exclude")
    exemplars = mine_run(run)
    assert [e.col for e in exemplars] == [0, 4]
    # Turn 3 still saw turn 2's O on the board: the excluded turn was replayed, only not emitted.
    assert exemplars[1].board[16:, 0:4].all()


def test_promoted_turns_are_mined_from_a_model_run(tmp_path):
    run = write_run(tmp_path, "20260808-000002-bbbbbb", "heuristic", THREE_O)
    assert mine(tmp_path) == []
    set_label(run, 3, "promote")
    exemplars = mine(tmp_path)
    assert [(e.col, e.promoted) for e in exemplars] == [(4, True)]


def test_select_exemplars_takes_promoted_first_then_fills_round_robin():
    def ex(piece, col, promoted=False, lines=0):
        return Exemplar(
            board=np.zeros((18, 10), dtype=bool),
            piece=piece,
            next_piece="O",
            rotation=0,
            col=col,
            lines_delta=lines,
            holes=0,
            promoted=promoted,
        )

    pool = [ex("O", 0), ex("O", 2, lines=1), ex("I", 0), ex("T", 0, promoted=True), ex("T", 2, promoted=True)]
    picked = select_exemplars(pool, k=3)
    assert [(e.piece, e.col) for e in picked] == [("T", 0), ("T", 2), ("O", 2)]


def test_decisions_grade_a_human_run_and_carry_labels(tmp_path):
    run = write_run(tmp_path, "20260808-000003-cccccc", "human", THREE_O)
    set_label(run, 1, "exclude")
    out = decisions(run)
    assert [d["turn"] for d in out] == [1, 2, 3]
    first = out[0]
    assert first["piece"] == "O" and first["board"] == ["." * 10] * 18
    assert first["placed_cells"] == [[16, 0], [16, 1], [17, 0], [17, 1]]
    assert set(first["grade"]) >= {"regret", "regret_norm", "rank", "legal_count", "best"}
    assert "genome" not in first["grade"]
    assert len(first["best_cells"]) == 4
    assert first["label"]["verdict"] == "exclude" and out[1]["label"] is None


def test_decisions_prefer_the_recorded_grade(tmp_path):
    run = write_run(tmp_path, "20260808-000004-dddddd", "human", THREE_O[:1])
    recorded = {
        "schema": "tetris.game.v1",
        "event_type": "placement_graded",
        "turn": 1,
        "occurred_at": "2026-08-08T00:00:00+00:00",
        "data": {
            "chosen": [0, 0],
            "best": [0, 4],
            "regret": 42.0,
            "regret_norm": 0.5,
            "rank": 9,
            "legal_count": 9,
            "board": ["." * 10] * 18,
            "genome": {},
        },
    }
    with (run / "events.jsonl").open("a") as f:
        f.write(json.dumps(recorded) + "\n")
    [d] = decisions(run)
    assert d["grade"]["regret"] == 42.0 and d["grade"]["rank"] == 9
    assert "board" not in d["grade"]


def test_jev_state_carries_the_board_the_choice_and_the_grade(tmp_path):
    run = write_run(tmp_path, "20260808-000005-eeeeee", "human", THREE_O[:1])
    [d] = decisions(run)
    state = jev.decision_state(d)
    assert state["placement"] == {"rotation": 0, "col": 0}
    assert state["board_before"] == d["board"]
    assert state["oracle"]["rank"].endswith("legal placements")
    assert set(jev.QUESTIONS) == {q["id"] for q in jev.PUBLIC_QUESTIONS}
    assert all("instructions" not in q for q in jev.PUBLIC_QUESTIONS)


def test_jev_setup_state_never_leaks_the_key(monkeypatch):
    monkeypatch.setenv(jev.API_KEY_VAR, "  ")
    assert jev.setup_state()["configured"] is False
    monkeypatch.setenv(jev.API_KEY_VAR, "sk-secret")
    setup = jev.setup_state()
    assert setup["configured"] is True
    assert "sk-secret" not in json.dumps(setup)
