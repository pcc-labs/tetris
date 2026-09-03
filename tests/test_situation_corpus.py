"""Situation corpus: free boards from the local policies, histogrammed by kind."""

import json
from collections import Counter

import pytest

from tetris_agent.situation_corpus import dominant, histogram, render


def _envelope(event_type, turn, data):
    return {
        "schema": "tetris.game.v1",
        "event_type": event_type,
        "turn": turn,
        "occurred_at": "2026-09-02T00:00:00+00:00",
        "data": data,
    }


def write_run(runs_dir, run_id, policy, pieces):
    """pieces: list of (piece, next_piece, rotation, col, lines_delta, feature_dict)."""
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True)
    events = [_envelope("session", 0, {"phase": "start", "policy": policy, "timer_div": 0})]
    for turn, (piece, nxt, rot, col, lines, feats) in enumerate(pieces, start=1):
        events.append(_envelope("piece_spawn", turn, {"piece": piece, "next_piece": nxt}))
        events.append(_envelope("placement_decision", turn, {"rotation": rot, "col": col, "score": 0.0}))
        events.append(_envelope("piece_locked", turn, {"lines_delta": lines, "misexec": 0, "score": 0, **feats}))
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return run_dir


# Hand-computed, as in test_traces.py: one O at col 0 -> heights [2,2,0,...]; a second at col 2.
O_AT_0 = {"holes": 0, "agg_height": 4, "bumpiness": 2, "max_height": 2}
THEN_O_AT_2 = {"holes": 0, "agg_height": 8, "bumpiness": 2, "max_height": 2}


def test_histogram_counts_reconstructed_boards_per_policy(tmp_path):
    write_run(
        tmp_path,
        "20260902-000001-aaaaaa",
        "heuristic",
        [("O", "O", 0, 0, 0, O_AT_0), ("O", "I", 0, 2, 0, THEN_O_AT_2)],
    )
    write_run(tmp_path, "20260902-000002-bbbbbb", "random", [("O", "O", 0, 0, 0, O_AT_0)])
    (tmp_path / "20260902-000003-cccccc").mkdir()  # no events.jsonl: ignored
    counts = histogram(tmp_path)
    assert counts["heuristic"] == Counter({"FLAT": 2})
    assert counts["random"] == Counter({"FLAT": 1})


def test_dominant_flags_a_class_at_or_above_the_threshold():
    assert dominant({"heuristic": Counter({"FLAT": 9}), "random": Counter({"MOUND": 1})}) == "FLAT"
    assert dominant({"heuristic": Counter({"FLAT": 5}), "random": Counter({"MOUND": 5})}) is None
    assert dominant({}) is None


def test_render_lists_every_kind_in_priority_order_with_totals():
    table = render({"heuristic": Counter({"FLAT": 3, "MOUND": 1}), "random": Counter({"FLAT": 1})})
    lines = table.splitlines()
    assert lines[0] == "| kind | heuristic | random | total | share |"
    kinds = [line.split("|")[1].strip() for line in lines[2:]]
    assert kinds == ["TOPPING_OUT", "TETRIS_READY", "HOLE_RISK", "MOUND", "FLAT", "total"]
    assert "| FLAT | 3 | 1 | 4 | 80.0% |" in table
    assert "| total | 4 | 1 | 5 | 100.0% |" in table


def test_histogram_prefers_captured_boards_over_reconstruction(tmp_path):
    # Events that would fail reconstruction at turn 2 (bogus checksum), but a boards.jsonl
    # carrying all three decisions: every captured board is counted, nothing is truncated.
    corrupted = dict(THEN_O_AT_2, holes=3)
    run = write_run(
        tmp_path,
        "20260903-000001-aaaaaa",
        "heuristic",
        [("O", "O", 0, 0, 0, O_AT_0), ("O", "I", 0, 2, 0, corrupted), ("O", "T", 0, 4, 0, THEN_O_AT_2)],
    )
    empty = ["." * 10] * 18
    tall = ["." * 10] * 6 + ["#########."] * 12
    rows = [
        {"policy": "heuristic", "turn": 1, "piece": "O", "next_piece": "O", "board": empty},
        {"policy": "heuristic", "turn": 2, "piece": "O", "next_piece": "I", "board": empty},
        {"policy": "heuristic", "turn": 3, "piece": "T", "next_piece": "O", "board": tall},
    ]
    (run / "boards.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    counts = histogram(tmp_path)
    assert counts["heuristic"] == Counter({"FLAT": 2, "TOPPING_OUT": 1})
    assert counts["random"] == Counter()


def test_histogram_falls_back_to_reconstruction_without_captured_boards(tmp_path):
    write_run(tmp_path, "20260903-000002-bbbbbb", "random", [("O", "O", 0, 0, 0, O_AT_0)])
    assert histogram(tmp_path)["random"] == Counter({"FLAT": 1})


@pytest.mark.rom
def test_generate_records_replayable_runs(tmp_path, rom_path):
    from tetris_agent.situation_corpus import generate

    runs = generate(rom_path, tmp_path, policies=("heuristic",), seeds=(0,), max_pieces=5)
    assert len(runs) == 1 and (runs[0] / "events.jsonl").is_file()
    captured = [json.loads(line) for line in (runs[0] / "boards.jsonl").read_text().splitlines() if line.strip()]
    assert len(captured) == 5  # one board per decision, none lost to reconstruction
    assert captured[0]["policy"] == "heuristic" and len(captured[0]["board"]) == 18
    counts = histogram(tmp_path)
    assert sum(counts["heuristic"].values()) == 5


def test_cli_histogram_on_empty_corpus_exits_1(tmp_path, capsys):
    from tetris_agent.situation_corpus import main

    assert main(["histogram", str(tmp_path)]) == 1
    assert "no boards classified" in capsys.readouterr().out
