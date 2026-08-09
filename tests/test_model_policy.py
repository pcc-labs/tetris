import json

import numpy as np

from tetris_agent.model_policy import ModelPolicy


class FakeUsage:
    def __init__(self, i=100, o=20, cr=0, cw=0):
        self.input_tokens = i
        self.output_tokens = o
        self.cache_read_input_tokens = cr
        self.cache_creation_input_tokens = cw


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeResponse:
    def __init__(self, payload, stop_reason="end_turn", usage=None):
        self.content = [FakeBlock(json.dumps(payload))] if payload is not None else []
        self.stop_reason = stop_reason
        self.usage = usage or FakeUsage()


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def empty_board():
    return np.zeros((18, 10), dtype=bool)


def policy(responses, **kwargs):
    return ModelPolicy(model="claude-opus-5", client=FakeClient(responses), **kwargs)


def test_valid_answer_becomes_a_placement_and_records_usage():
    p = policy([FakeResponse({"rotation": 0, "col": 3, "reason": "flat spot"})])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert (placement.rotation, placement.col) == (0, 3)
    assert p.last_reason == "flat spot"
    stats = p.stats()
    assert stats["decisions"] == 1
    assert stats["input_tokens"] == 100
    assert stats["cost_usd"] > 0
    assert stats["illegal_count"] == 0


def test_illegal_answer_retries_with_the_legal_set_then_succeeds():
    p = policy(
        [
            FakeResponse({"rotation": 0, "col": 9, "reason": "off the edge"}),  # O at col 9 doesn't fit
            FakeResponse({"rotation": 0, "col": 4, "reason": "recovered"}),
        ]
    )
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement.col == 4
    assert p.usage["retry_count"] == 1
    assert p.usage["illegal_count"] == 1
    retry_prompt = p.client.messages.calls[1]["messages"][-1]["content"]
    assert "not a legal placement" in retry_prompt
    assert "rot=0, col=0" in retry_prompt


def test_two_illegal_answers_fall_back_to_a_legal_placement():
    bad = {"rotation": 0, "col": 9, "reason": "nope"}
    p = policy([FakeResponse(bad), FakeResponse(bad)])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement is not None  # run continues
    assert p.usage["illegal_count"] == 3  # two rejections plus the fallback


def test_refusal_is_counted_and_does_not_retry():
    # Retrying an identical prompt after a refusal just spends money, so the
    # decision aborts to the fallback instead.
    p = policy([FakeResponse(None, stop_reason="refusal")])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement is not None
    assert p.usage["refusals"] == 1
    assert len(p.client.messages.calls) == 1


def test_unparseable_answer_is_counted_not_raised():
    p = policy([FakeResponse(None)])
    assert p.plan(empty_board(), "O", "I", turn=1) is not None
    assert p.usage["parse_failures"] == 1


def test_system_prompt_is_cached_and_effort_is_sent():
    p = policy([FakeResponse({"rotation": 0, "col": 0, "reason": "x"})], effort="xhigh")
    p.plan(empty_board(), "O", "I", turn=1)
    call = p.client.messages.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert call["output_config"]["effort"] == "xhigh"
    assert call["output_config"]["format"]["type"] == "json_schema"


def test_haiku_arms_omit_effort_entirely():
    p = ModelPolicy(
        model="claude-haiku-4-5",
        client=FakeClient([FakeResponse({"rotation": 0, "col": 0, "reason": "x"})]),
        effort="high",
    )
    assert p.effort is None
    p.plan(empty_board(), "O", "I", turn=1)
    assert "effort" not in p.client.messages.calls[0]["output_config"]
    assert p.name == "claude-haiku-4-5/features"


def test_chat_harness_accumulates_history_others_do_not():
    answer = {"rotation": 0, "col": 0, "reason": "x"}
    chat = policy([FakeResponse(answer), FakeResponse(answer)], harness="chat")
    chat.plan(empty_board(), "O", "I", 1)
    chat.plan(empty_board(), "O", "I", 2)
    assert len(chat.client.messages.calls[1]["messages"]) == 3  # prior pair + new prompt

    stateless = policy([FakeResponse(answer), FakeResponse(answer)], harness="board")
    stateless.plan(empty_board(), "O", "I", 1)
    stateless.plan(empty_board(), "O", "I", 2)
    assert len(stateless.client.messages.calls[1]["messages"]) == 1


def test_plan_returns_none_when_no_placement_fits():
    board = np.ones((18, 10), dtype=bool)
    p = policy([])
    assert p.plan(board, "O", "I", turn=1) is None


def test_exemplar_block_joins_the_cached_system_prompt():
    p = policy(
        [FakeResponse({"rotation": 0, "col": 3, "reason": "x"})],
        exemplar_block="# How a strong human placed pieces\nEXEMPLARS",
    )
    p.plan(empty_board(), "O", "I", turn=1)
    system = p.client.messages.calls[0]["system"]
    assert "EXEMPLARS" in system[0]["text"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}  # still cached


def test_without_exemplars_system_prompt_is_unchanged():
    from tetris_agent.prompts import SYSTEM_PROMPT

    p = policy([FakeResponse({"rotation": 0, "col": 3, "reason": "x"})])
    p.plan(empty_board(), "O", "I", turn=1)
    assert p.client.messages.calls[0]["system"][0]["text"] == SYSTEM_PROMPT
