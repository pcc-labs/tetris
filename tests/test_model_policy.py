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


class FakeThinkingDetails:
    def __init__(self, thinking_tokens):
        self.thinking_tokens = thinking_tokens


class FakeClock:
    """Scripted monotonic clock; repeats the last time once exhausted."""

    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


def test_stats_report_thinking_tokens_and_tokens_per_second():
    usage = FakeUsage(o=40)
    usage.output_tokens_details = FakeThinkingDetails(30)
    p = policy(
        [FakeResponse({"rotation": 0, "col": 3, "reason": "ok"}, usage=usage)],
        clock=FakeClock([0.0, 0.0, 2.0]),  # decide start, call start, call end
    )
    p.plan(empty_board(), "O", "I", turn=1)
    stats = p.stats()
    assert stats["thinking_tokens"] == 30
    assert stats["tokens_per_second"] == 20.0  # 40 output tokens over 2s of generation
    assert stats["tokens_per_decision"] == 40.0
    assert p.last_output_tokens == 40


def test_api_errors_inflate_latency_mean_but_not_tokens_per_second():
    import anthropic
    import httpx

    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.test"))

    class RaisingMessages(FakeMessages):
        def create(self, **kwargs):
            r = self.responses.pop(0)
            if isinstance(r, Exception):
                raise r
            self.calls.append(kwargs)
            return r

    client = FakeClient([])
    client.messages = RaisingMessages([err, FakeResponse({"rotation": 0, "col": 3, "reason": "ok"})])
    p = ModelPolicy(
        model="claude-opus-5",
        client=client,
        clock=FakeClock([0.0, 0.0, 10.0, 10.0, 10.0, 12.0]),
    )
    p.plan(empty_board(), "O", "I", turn=1)  # errors: falls back, 10s of latency
    p.plan(empty_board(), "O", "I", turn=2)  # succeeds: 2s of generation, 20 tokens
    stats = p.stats()
    assert stats["api_errors"] == 1
    assert stats["latency_ms_mean"] == 12000.0  # error latency still counted
    assert stats["tokens_per_second"] == 10.0  # but tok/s uses only the clean 2s


def test_deadline_reaches_the_user_prompt_but_not_the_cached_system_prompt():
    p = policy([FakeResponse({"rotation": 0, "col": 3, "reason": "ok"})])
    p.deadline_s = 6.0
    p.plan(empty_board(), "O", "I", turn=1)
    call = p.client.messages.calls[0]
    assert "about 6 seconds before this piece locks" in call["messages"][-1]["content"]
    assert "before this piece locks" not in call["system"][0]["text"]


def ok(rotation=0, col=3):
    return FakeResponse({"rotation": rotation, "col": col, "reason": "ok"})


def test_effort_steps_down_under_deadline_pressure_and_recovers():
    # Configured medium -> ladder [low, medium]. A 5s call against a 4s
    # deadline steps down; a fast call under a roomy deadline steps back up.
    p = policy([ok(), ok(), ok()], clock=FakeClock([0, 0, 5, 5, 5, 6, 6, 6, 7.5]))
    p.deadline_s = 4.0
    p.plan(empty_board(), "O", "I", turn=1)  # no EMA yet: configured medium, takes 5s
    p.plan(empty_board(), "O", "I", turn=2)  # EMA 5 > 0.9*4 -> low, takes 1s
    p.deadline_s = 10.0
    p.plan(empty_board(), "O", "I", turn=3)  # EMA 3 < 0.45*10 -> back to medium

    efforts = [c["output_config"].get("effort") for c in p.client.messages.calls]
    assert efforts == ["medium", "low", "medium"]
    assert p.stats()["downshifts"] == 1
    assert p.stats()["effort"] == "medium"  # arm identity keeps the configured tier


def test_no_deadline_means_the_configured_effort_always():
    p = policy([ok(), ok()], clock=FakeClock([0, 0, 30, 30, 30, 60]))
    p.plan(empty_board(), "O", "I", turn=1)  # slow, but no clock pressure
    p.plan(empty_board(), "O", "I", turn=2)
    efforts = [c["output_config"].get("effort") for c in p.client.messages.calls]
    assert efforts == ["medium", "medium"]
    assert p.stats()["downshifts"] == 0


def test_illegal_retry_is_skipped_when_the_deadline_is_tight():
    # The illegal answer alone took 3s of a 4s deadline; a second full call
    # cannot land in time, so the policy falls back instead of retrying.
    p = policy([FakeResponse({"rotation": 0, "col": 9, "reason": "off the edge"})], clock=FakeClock([0.0, 0.0, 3.0]))
    p.deadline_s = 4.0
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement is not None  # deterministic fallback
    assert len(p.client.messages.calls) == 1
    assert p.stats()["retry_count"] == 0


def test_routed_harness_swaps_the_system_prompt_and_names_the_situation():
    from tetris_agent.prompts import ROUTED_SYSTEM_PROMPT

    p = policy([FakeResponse({"rotation": 0, "col": 3, "reason": "flat"})], harness="routed")
    p.plan(empty_board(), "T", "O", turn=1)
    call = p.client.messages.calls[0]
    assert call["system"][0]["text"] == ROUTED_SYSTEM_PROMPT
    assert "Situation: FLAT" in call["messages"][-1]["content"]
    assert p.last_situation.kind.value == "FLAT"


def test_routed_thresholds_come_from_the_policy_genome():
    from tetris_agent.policy import Genome

    p = policy(
        [FakeResponse({"rotation": 0, "col": 3, "reason": "x"})],
        harness="routed",
        genome=Genome(topping_out_height=1),
    )
    board = empty_board()
    board[17, 0] = True
    p.plan(board, "T", "O", turn=1)
    assert "Situation: TOPPING_OUT" in p.client.messages.calls[0]["messages"][-1]["content"]


def test_non_routed_harness_has_no_situation():
    p = policy([FakeResponse({"rotation": 0, "col": 3, "reason": "x"})], harness="features")
    p.plan(empty_board(), "T", "O", turn=1)
    assert p.last_situation is None
    assert "Situation:" not in p.client.messages.calls[0]["messages"][-1]["content"]
