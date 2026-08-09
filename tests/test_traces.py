"""Trace mining: human runs -> verified (board, decision) exemplars.

The recorded per-piece features act as a checksum on the reconstruction: a
turn whose recomputed features disagree with what the emulator observed must
not become an exemplar, and neither may anything after it in that run.
"""

import json

import numpy as np

from tetris_agent.traces import Exemplar, mine, mine_run, select_exemplars


def _envelope(event_type, turn, data):
    return {
        "schema": "tetris.game.v1",
        "event_type": event_type,
        "turn": turn,
        "occurred_at": "2026-08-08T00:00:00+00:00",
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
        events.append(
            _envelope("piece_locked", turn, {"lines_delta": lines, "misexec": 0, "score": 0, **feats})
        )
    (run_dir / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return run_dir


# Hand-computed: an O piece is a 2x2 block. One at col 0 on an empty board
# gives heights [2,2,0,...]: agg 4, bumpiness |2-2|+|2-0| = 2, max 2, no holes.
O_AT_0 = {"holes": 0, "agg_height": 4, "bumpiness": 2, "max_height": 2}
# A second O at col 2: heights [2,2,2,2,0,...]: agg 8, bumpiness 2, max 2.
THEN_O_AT_2 = {"holes": 0, "agg_height": 8, "bumpiness": 2, "max_height": 2}


def test_mines_human_run_and_reconstructs_boards(tmp_path):
    run = write_run(
        tmp_path,
        "20260808-000001-aaaaaa",
        "human",
        [("O", "O", 0, 0, 0, O_AT_0), ("O", "I", 0, 2, 0, THEN_O_AT_2)],
    )
    exemplars = mine_run(run)
    assert len(exemplars) == 2
    first, second = exemplars
    assert (first.piece, first.next_piece, first.rotation, first.col) == ("O", "O", 0, 0)
    assert not first.board.any()  # first decision was taken on an empty board
    # The second decision saw the first O already settled at col 0.
    assert second.board[16:, 0:2].all()
    assert second.board.sum() == 4


def test_line_clear_turns_reconstruct(tmp_path):
    # Five O pieces across cols 0,2,4,6,8 fill rows 16-17 -> 2 lines clear,
    # leaving an empty board. Features after the clearing lock are all zero.
    pieces = []
    for i, col in enumerate((0, 2, 4, 6)):
        feats = {"holes": 0, "agg_height": 4 * (i + 1), "bumpiness": 2, "max_height": 2}
        pieces.append(("O", "O", 0, col, 0, feats))
    pieces.append(("O", "T", 0, 8, 2, {"holes": 0, "agg_height": 0, "bumpiness": 0, "max_height": 0}))
    run = write_run(tmp_path, "20260808-000002-bbbbbb", "human", pieces)

    exemplars = mine_run(run)
    assert len(exemplars) == 5
    assert exemplars[-1].lines_delta == 2
    assert exemplars[-1].board[16:, 0:8].all()  # board *before* the clearing piece


def test_checksum_mismatch_drops_that_turn_and_the_rest(tmp_path):
    corrupted = dict(THEN_O_AT_2, holes=3)  # emulator saw something we can't reproduce
    run = write_run(
        tmp_path,
        "20260808-000003-cccccc",
        "human",
        [("O", "O", 0, 0, 0, O_AT_0), ("O", "I", 0, 2, 0, corrupted), ("O", "I", 0, 4, 0, THEN_O_AT_2)],
    )
    exemplars = mine_run(run)
    assert len(exemplars) == 1  # verified prefix only


def test_non_human_runs_are_ignored(tmp_path):
    write_run(tmp_path, "20260808-000004-dddddd", "heuristic", [("O", "O", 0, 0, 0, O_AT_0)])
    assert mine(tmp_path) == []


def test_mine_scans_only_human_runs(tmp_path):
    write_run(tmp_path, "20260808-000005-eeeeee", "human", [("O", "O", 0, 0, 0, O_AT_0)])
    write_run(tmp_path, "20260808-000006-ffffff", "heuristic", [("O", "O", 0, 0, 0, O_AT_0)])
    (tmp_path / "20260808-000007-000000").mkdir()  # no events.jsonl at all
    exemplars = mine(tmp_path)
    assert len(exemplars) == 1
    assert exemplars[0].piece == "O"


def test_select_exemplars_spreads_across_piece_types():
    def ex(piece, col):
        return Exemplar(
            board=np.zeros((18, 10), dtype=bool),
            piece=piece,
            next_piece="O",
            rotation=0,
            col=col,
            lines_delta=0,
            holes=0,
        )

    pool = [ex("O", 0), ex("O", 2), ex("O", 4), ex("I", 0), ex("T", 0)]
    picked = select_exemplars(pool, k=3)
    assert [e.piece for e in picked] == ["O", "I", "T"]
    picked_all = select_exemplars(pool, k=5)
    assert len(picked_all) == 5


def test_select_exemplars_prefers_line_clears_within_a_piece_type():
    def ex(piece, lines):
        return Exemplar(
            board=np.zeros((18, 10), dtype=bool),
            piece=piece,
            next_piece="O",
            rotation=0,
            col=0,
            lines_delta=lines,
            holes=0,
        )

    pool = [ex("O", 0), ex("O", 1), ex("I", 0)]
    picked = select_exemplars(pool, k=2)
    assert [(e.piece, e.lines_delta) for e in picked] == [("O", 1), ("I", 0)]


def test_load_exemplar_block_fails_loudly_without_human_traces(tmp_path):
    import pytest

    from tetris_agent.traces import load_exemplar_block

    with pytest.raises(RuntimeError, match="tetris-play"):
        load_exemplar_block(tmp_path)


def test_load_exemplar_block_builds_from_verified_traces(tmp_path):
    from tetris_agent.traces import load_exemplar_block

    write_run(tmp_path, "20260808-000008-abcdef", "human", [("O", "I", 0, 0, 0, O_AT_0)])
    block = load_exemplar_block(tmp_path)
    assert "How a strong human placed pieces" in block
    assert "rotation=0 col=0" in block
