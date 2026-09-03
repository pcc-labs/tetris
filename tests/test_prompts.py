import numpy as np
import pytest

from tetris_agent.pricing import cost_usd, is_pi, spec
from tetris_agent.prompts import (
    HARNESSES,
    PLACEMENT_SCHEMA,
    build_user_prompt,
    legal_placements,
    render_board,
)


def empty_board():
    return np.zeros((18, 10), dtype=bool)


def test_render_board_marks_settled_cells():
    board = empty_board()
    board[17, 0] = True
    board[17, 9] = True
    lines = render_board(board).splitlines()
    assert len(lines) == 18
    assert lines[0] == "." * 10
    assert lines[17] == "#........#"


def test_legal_placements_cover_every_column_for_o():
    placements = legal_placements(empty_board(), "O")
    assert {p.col for p in placements} == set(range(9))  # width 2 -> cols 0..8
    assert all(p.landing_row == 16 for p in placements)


def test_legal_placements_report_line_clears():
    board = empty_board()
    board[17, :9] = True
    placements = legal_placements(board, "I")
    clearing = [p for p in placements if p.lines == 1]
    assert clearing and all(p.col == 9 for p in clearing)


@pytest.mark.parametrize("harness", HARNESSES)
def test_every_harness_includes_board_and_pieces(harness):
    board = empty_board()
    prompt = build_user_prompt(harness, board, "J", "O", legal_placements(board, "J"), turn=7)
    assert "Current piece: J" in prompt
    assert "Next piece: O" in prompt
    assert "." * 10 in prompt
    assert "Piece 7" in prompt


def test_scaffolding_ladder_is_strictly_increasing():
    board = empty_board()
    placements = legal_placements(board, "T")
    prompts = {h: build_user_prompt(h, board, "T", "I", placements, 1) for h in HARNESSES}
    assert "Legal placements" not in prompts["board"]
    assert "Legal placements" in prompts["legal"]
    assert "clears" not in prompts["legal"]
    assert "clears" in prompts["features"]
    assert len(prompts["features"]) > len(prompts["legal"]) > len(prompts["board"])


def test_unknown_harness_rejected():
    with pytest.raises(ValueError):
        build_user_prompt("telepathy", empty_board(), "T", "I", [], 1)


def test_placement_schema_is_structured_output_safe():
    # Numeric range constraints aren't supported; enums are.
    assert PLACEMENT_SCHEMA["additionalProperties"] is False
    assert PLACEMENT_SCHEMA["properties"]["col"]["enum"] == list(range(10))
    assert set(PLACEMENT_SCHEMA["required"]) == {"rotation", "col", "reason"}


def test_pricing_reflects_model_tiers_and_effort_support():
    assert spec("claude-opus-5").input_per_mtok < spec("claude-fable-5").input_per_mtok
    assert spec("claude-haiku-4-5").supports_effort is False
    assert spec("claude-opus-5").supports_effort is True


def test_cost_accounts_for_cache_discount():
    full = cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0)
    cached = cost_usd("claude-opus-5", input_tokens=0, output_tokens=0, cache_read_tokens=1_000_000)
    assert full == 5.0
    assert cached == pytest.approx(0.5)


def test_pi_models_are_free_and_gate_thinking_by_capability():
    assert is_pi("pi/gpt-oss:20b") and not is_pi("claude-opus-5")
    assert spec("pi/gpt-oss:20b").supports_effort is True
    assert spec("pi/gemma3").supports_effort is False
    assert cost_usd("pi/gpt-oss:20b", input_tokens=1_000_000, output_tokens=100_000) == 0.0


def test_unlisted_pi_model_falls_back_to_free_effortless_spec():
    s = spec("pi/some-model-nobody-registered")
    assert (s.input_per_mtok, s.output_per_mtok, s.supports_effort) == (0.0, 0.0, False)
    with pytest.raises(KeyError):
        spec("bogus-model")


def test_exemplar_block_renders_each_decision():
    from tetris_agent.prompts import build_exemplar_block
    from tetris_agent.traces import Exemplar

    board = empty_board()
    board[16:, 0:2] = True
    block = build_exemplar_block(
        [Exemplar(board=board, piece="I", next_piece="T", rotation=1, col=9, lines_delta=1, holes=0)]
    )
    assert "human" in block.lower()
    assert "Piece: I" in block and "Next: T" in block
    assert "rotation=1 col=9" in block
    assert "cleared 1 line" in block
    assert "##........" in block  # the board the human saw


def test_exemplar_block_empty_pool_is_empty_string():
    from tetris_agent.prompts import build_exemplar_block

    assert build_exemplar_block([]) == ""


def test_deadline_line_appended_only_when_given():
    import numpy as np

    from tetris_agent.prompts import build_user_prompt, legal_placements

    board = np.zeros((18, 10), dtype=bool)
    legal = legal_placements(board, "O")
    plain = build_user_prompt("features", board, "O", "I", legal, 1)
    assert "before this piece locks" not in plain
    timed = build_user_prompt("features", board, "O", "I", legal, 1, deadline_s=7.4)
    assert timed.startswith(plain)
    assert "about 7 seconds before this piece locks" in timed


def test_routed_prompt_names_the_situation_and_scopes_the_numbers():
    board = empty_board()
    legal = legal_placements(board, "T")
    prompt = build_user_prompt("routed", board, "T", "O", legal, 1)
    assert "Situation: FLAT" in prompt
    assert "-> holes 0, bumpiness" in prompt  # FLAT's metrics, nothing else
    assert "aggregate height" not in prompt
    assert "Choose one of them." in prompt


def test_routed_prompt_accepts_a_precomputed_situation():
    from tetris_agent.situation import Kind, Situation

    board = empty_board()
    legal = legal_placements(board, "O")
    prompt = build_user_prompt("routed", board, "O", "I", legal, 1, situation=Situation(Kind.HOLE_RISK))
    assert "Situation: HOLE_RISK" in prompt
    listing = prompt.split("Legal placements")[1]
    assert "-> holes" in listing and "bumpiness" not in listing


def test_routed_sits_between_legal_and_features_in_length():
    board = empty_board()
    legal = legal_placements(board, "T")
    sizes = {h: len(build_user_prompt(h, board, "T", "I", legal, 1)) for h in ("legal", "routed", "features")}
    assert sizes["legal"] < sizes["routed"] < sizes["features"]


def test_system_prompt_for_only_swaps_for_routed():
    from tetris_agent.prompts import ROUTED_SYSTEM_PROMPT, SYSTEM_PROMPT, system_prompt_for

    assert system_prompt_for("features") is SYSTEM_PROMPT
    assert system_prompt_for("chat") is SYSTEM_PROMPT
    assert system_prompt_for("routed") is ROUTED_SYSTEM_PROMPT
    assert "# What good play looks like" in SYSTEM_PROMPT
    assert "# What good play looks like" not in ROUTED_SYSTEM_PROMPT
    assert "# How to play" in ROUTED_SYSTEM_PROMPT
    assert ROUTED_SYSTEM_PROMPT.startswith("You are playing Game Boy Tetris")
    assert ROUTED_SYSTEM_PROMPT.endswith("one short sentence of reasoning.")
