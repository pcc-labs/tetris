import json
import subprocess

import numpy as np
import pytest

from tetris_agent.pi_policy import PiPolicy


def jsonl_events(text, usage=None):
    """A minimal pi --mode json transcript, shaped like a real v0.80.10 run."""
    usage = usage or {"input": 121, "output": 19, "cacheRead": 0, "cacheWrite": 0}
    events = [
        {"type": "session", "version": 3, "id": "test"},
        {"type": "agent_start"},
        {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "prompt"}]}},
        {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}], "usage": usage},
        },
        {"type": "agent_end"},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


class FakeRunner:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.timeouts = []

    def __call__(self, cmd, timeout_s):
        self.calls.append(cmd)
        self.timeouts.append(timeout_s)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def completed(stdout, returncode=0, stderr=""):
    return subprocess.CompletedProcess(args=["pi"], returncode=returncode, stdout=stdout, stderr=stderr)


def empty_board():
    return np.zeros((18, 10), dtype=bool)


def policy(results, model="pi/gpt-oss:20b", **kwargs):
    return PiPolicy(model=model, runner=FakeRunner(results), **kwargs)


def prompt_of(cmd):
    return cmd[-1]


def test_valid_answer_becomes_a_placement_and_records_usage():
    p = policy([completed(jsonl_events('{"rotation": 0, "col": 3, "reason": "flat spot"}'))])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert (placement.rotation, placement.col) == (0, 3)
    assert p.last_reason == "flat spot"
    stats = p.stats()
    assert stats["decisions"] == 1
    assert stats["input_tokens"] == 121
    assert stats["output_tokens"] == 19
    assert stats["cost_usd"] == 0.0
    assert p.name == "pi/gpt-oss:20b/features/medium"


def test_fenced_or_prose_wrapped_json_still_parses():
    text = 'Sure! Here is my move:\n```json\n{"rotation": 0, "col": 3, "reason": "ok"}\n```\nGood luck!'
    p = policy([completed(jsonl_events(text))])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert (placement.rotation, placement.col) == (0, 3)
    assert p.usage["parse_failures"] == 0


def test_illegal_answer_retries_with_transcript_context():
    p = policy(
        [
            completed(jsonl_events('{"rotation": 0, "col": 9, "reason": "off the edge"}')),
            completed(jsonl_events('{"rotation": 0, "col": 4, "reason": "recovered"}')),
        ]
    )
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement.col == 4
    assert p.usage["retry_count"] == 1
    retry_prompt = prompt_of(p.runner.calls[1])
    assert "not a legal placement" in retry_prompt
    assert "[assistant]" in retry_prompt and "[user]" in retry_prompt


def test_nonzero_exit_counts_api_error_and_falls_back():
    p = policy([completed("", returncode=1, stderr="boom")])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement is not None
    assert p.usage["api_errors"] == 1
    assert p.usage["illegal_count"] == 1  # the fallback is counted


def test_timeout_counts_api_error():
    p = policy([subprocess.TimeoutExpired(cmd="pi", timeout=1)])
    assert p.plan(empty_board(), "O", "I", turn=1) is not None
    assert p.usage["api_errors"] == 1


def test_junk_stdout_counts_parse_failure():
    p = policy([completed("not json at all\n{broken\n")])
    assert p.plan(empty_board(), "O", "I", turn=1) is not None
    assert p.usage["parse_failures"] == 1


def test_command_assembly_gates_thinking_by_model_capability():
    p = policy([completed(jsonl_events('{"rotation": 0, "col": 0, "reason": "x"}'))], effort="high")
    p.plan(empty_board(), "O", "I", turn=1)
    cmd = p.runner.calls[0]
    assert cmd[cmd.index("--model") + 1] == "gpt-oss:20b"
    assert cmd[cmd.index("--thinking") + 1] == "high"
    for flag in ("--no-session", "--no-tools", "--no-context-files", "--no-extensions", "--offline"):
        assert flag in cmd
    assert str(cmd[cmd.index("-e") + 1]).endswith("tapes-proxy.ts")
    system = cmd[cmd.index("--system-prompt") + 1]
    assert "ONLY a single JSON object" in system

    ok = jsonl_events('{"rotation": 0, "col": 0, "reason": "x"}')
    effortless = policy([completed(ok)], model="pi/gemma3", effort="high")
    assert effortless.effort is None
    assert effortless.name == "pi/gemma3/features"
    effortless.plan(empty_board(), "O", "I", turn=1)
    assert "--thinking" not in effortless.runner.calls[0]


def test_chat_harness_renders_history_into_the_prompt():
    answer = '{"rotation": 0, "col": 0, "reason": "x"}'
    p = policy([completed(jsonl_events(answer)), completed(jsonl_events(answer))], harness="chat")
    p.plan(empty_board(), "O", "I", 1)
    p.plan(empty_board(), "O", "I", 2)
    second_prompt = prompt_of(p.runner.calls[1])
    assert "[assistant]" in second_prompt  # prior turn carried along

    stateless = policy([completed(jsonl_events(answer)), completed(jsonl_events(answer))], harness="board")
    stateless.plan(empty_board(), "O", "I", 1)
    stateless.plan(empty_board(), "O", "I", 2)
    assert "[assistant]" not in prompt_of(stateless.runner.calls[1])


def test_thinking_blocks_are_ignored_when_extracting_text():
    events = [
        {"type": "session"},
        {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm let me think"},
                    {"type": "text", "text": '{"rotation": 0, "col": 5, "reason": "clear"}'},
                ],
                "usage": {"input": 10, "output": 5, "cacheRead": 0, "cacheWrite": 0},
            },
        },
    ]
    stdout = "\n".join(json.dumps(e) for e in events)
    p = policy([completed(stdout)])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement.col == 5


def test_unknown_pi_model_is_accepted_effort_free():
    p = policy([completed(jsonl_events('{"rotation": 0, "col": 0, "reason": "x"}'))], model="pi/brand-new:7b")
    assert p.effort is None
    assert p.stats()["cost_usd"] == 0.0


def test_plan_returns_none_when_no_placement_fits():
    board = np.ones((18, 10), dtype=bool)
    p = policy([])
    assert p.plan(board, "O", "I", turn=1) is None
    assert p.runner.calls == []


def test_default_runner_closes_stdin():
    # pi --mode json holds the session open while stdin is open; the runner
    # must hand it a closed stdin or every decision hangs until timeout.
    from tetris_agent.pi_policy import _run_pi

    captured = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    import tetris_agent.pi_policy as mod

    original = mod.subprocess.run
    mod.subprocess.run = fake_run
    try:
        _run_pi(["pi"], timeout_s=5)
    finally:
        mod.subprocess.run = original
    assert captured["stdin"] == subprocess.DEVNULL
    assert captured["timeout"] == 5
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_pi_policy_rejects_non_pi_model():
    with pytest.raises(ValueError):
        PiPolicy(model="claude-opus-5", runner=FakeRunner([]))


def provider_error_events(message="Connection error.", attempts=3):
    """pi's real shape when the provider is unreachable: exit 0, empty content,
    stopReason=error, repeated once per internal auto-retry."""
    events = [{"type": "session", "version": 3, "id": "test"}]
    for _ in range(attempts):
        events += [
            {"type": "agent_start"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "usage": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "stopReason": "error",
                    "errorMessage": message,
                },
            },
        ]
    events.append({"type": "auto_retry_end", "success": False, "finalError": message})
    return "\n".join(json.dumps(e) for e in events) + "\n"


def test_unreachable_provider_is_an_api_error_not_the_model_failing():
    # pi exits 0 when Ollama is down; without reading stopReason this would be
    # charged to the model as a parse failure and drag the arm's score down.
    p = policy([completed(provider_error_events()), completed(provider_error_events())])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert placement is not None  # deterministic fallback keeps the run alive
    assert p.usage["api_errors"] == 1
    assert p.usage["parse_failures"] == 0
    assert p.usage["decisions"] == 0
    # _decide gives up on the first None, so the second result is never consumed.
    assert len(p.runner.calls) == 1


def test_provider_error_followed_by_a_successful_retry_counts_as_a_decision():
    # pi retries internally; the last assistant message is the verdict.
    stdout = provider_error_events(attempts=2) + jsonl_events('{"rotation": 0, "col": 3, "reason": "recovered"}')
    p = policy([completed(stdout)])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert (placement.rotation, placement.col) == (0, 3)
    assert p.usage["api_errors"] == 0
    assert p.usage["decisions"] == 1
    assert p.usage["input_tokens"] == 121  # usage taken from the winning message


def test_brace_inside_a_string_value_does_not_break_extraction():
    text = '{"rotation": 0, "col": 3, "reason": "fills the } notch"}'
    p = policy([completed(jsonl_events(text))])
    placement = p.plan(empty_board(), "O", "I", turn=1)
    assert (placement.rotation, placement.col) == (0, 3)
    assert p.last_reason == "fills the } notch"
    assert p.usage["parse_failures"] == 0


def test_preflight_is_a_noop_without_pi_arms():
    from tetris_agent.pi_policy import preflight

    assert preflight(["claude-opus-5", "claude-haiku-4-5"]) == []


def test_preflight_reports_an_unreachable_ollama_once():
    from tetris_agent.pi_policy import preflight

    problems = preflight(["pi/qwen3:8b", "pi/gemma3"], base_url="http://127.0.0.1:9")
    assert len(problems) == 1
    assert "unreachable" in problems[0]


def test_preflight_names_the_models_that_are_not_pulled(monkeypatch):
    import tetris_agent.pi_policy as mod

    monkeypatch.setattr(mod, "_ollama_models", lambda *a, **k: {"gemma3:latest": 3_300_000_000})
    problems = mod.preflight(["pi/qwen3:8b", "pi/gemma3"])
    # gemma3 resolves through the implicit :latest tag; qwen3:8b genuinely isn't there.
    assert len(problems) == 1
    assert "qwen3:8b" in problems[0] and "ollama pull" in problems[0]


def test_preflight_rejects_a_model_too_big_for_the_box(monkeypatch):
    import tetris_agent.pi_policy as mod

    # The walkman case: a 4 GB sandbox and a model that needs 5.2 GB resident.
    monkeypatch.setattr(mod, "_ollama_models", lambda *a, **k: {"qwen3:8b": 5_200_000_000})
    monkeypatch.setattr(mod, "_total_ram_bytes", lambda: 4_000_000_000)
    problems = mod.preflight(["pi/qwen3:8b"])
    assert len(problems) == 1
    assert "OOM-killed" in problems[0] and "4.0 GB RAM" in problems[0]


def test_preflight_passes_when_the_model_fits(monkeypatch):
    import tetris_agent.pi_policy as mod

    monkeypatch.setattr(mod, "_ollama_models", lambda *a, **k: {"qwen3:8b": 5_200_000_000})
    monkeypatch.setattr(mod, "_total_ram_bytes", lambda: 64_000_000_000)
    assert mod.preflight(["pi/qwen3:8b"]) == []


def test_preflight_cannot_judge_size_without_a_ram_reading(monkeypatch):
    import tetris_agent.pi_policy as mod

    monkeypatch.setattr(mod, "_ollama_models", lambda *a, **k: {"qwen3:8b": 5_200_000_000})
    monkeypatch.setattr(mod, "_total_ram_bytes", lambda: None)
    assert mod.preflight(["pi/qwen3:8b"]) == []  # unknown RAM must not block the run


def test_exemplar_block_reaches_the_pi_system_prompt():
    runner = FakeRunner([completed(jsonl_events('{"rotation": 0, "col": 3, "reason": "x"}'))])
    p = PiPolicy(model="pi/gemma3", runner=runner, exemplar_block="EXEMPLARS")
    p.plan(np.zeros((18, 10), dtype=bool), "O", "I", turn=1)
    cmd = runner.calls[0]
    system = cmd[cmd.index("--system-prompt") + 1]
    assert "EXEMPLARS" in system
    assert "Return ONLY" in system or "JSON" in system  # pi's format contract still appended


class FakeClock:
    """Scripted monotonic clock; repeats the last time once exhausted."""

    def __init__(self, times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0) if len(self.times) > 1 else self.times[0]


def test_reasoning_tokens_and_tokens_per_second_from_pi_usage():
    usage = {"input": 121, "output": 30, "cacheRead": 0, "cacheWrite": 0, "reasoning": 12}
    p = policy(
        [completed(jsonl_events('{"rotation": 0, "col": 3, "reason": "ok"}', usage=usage))],
        clock=FakeClock([0.0, 0.0, 3.0]),  # decide start, call start, call end
    )
    p.plan(empty_board(), "O", "I", turn=1)
    stats = p.stats()
    assert stats["thinking_tokens"] == 12
    assert stats["tokens_per_second"] == 10.0  # 30 output tokens over 3s
    assert stats["tokens_per_decision"] == 30.0
    assert p.last_output_tokens == 30


def ok_events():
    return completed(jsonl_events('{"rotation": 0, "col": 3, "reason": "ok"}'))


def test_pi_effort_ladder_bottoms_out_at_thinking_off():
    p = policy(
        [ok_events(), ok_events(), ok_events()],
        clock=FakeClock([0, 0, 5, 5, 5, 10, 10, 10, 15]),
    )
    p.deadline_s = 1.0
    p.plan(empty_board(), "O", "I", turn=1)  # configured medium
    p.plan(empty_board(), "O", "I", turn=2)  # EMA 5 > 0.9 -> low
    p.plan(empty_board(), "O", "I", turn=3)  # still over -> off

    runner = p.runner
    tiers = [cmd[cmd.index("--thinking") + 1] for cmd in runner.calls]
    assert tiers == ["medium", "low", "off"]
    assert p.stats()["downshifts"] == 2


def test_effort_off_pins_thinking_off_for_every_call():
    """`--efforts off` is a real arm: thinking disabled from the first piece, no ladder to climb."""
    p = policy([ok_events(), ok_events()], effort="off", clock=FakeClock([0, 0, 0.1, 0.1, 0.1, 0.2]))
    p.deadline_s = 1.0
    p.plan(empty_board(), "O", "I", turn=1)
    p.plan(empty_board(), "O", "I", turn=2)  # fast EMA would step *up* on a ladder; there is none
    tiers = [cmd[cmd.index("--thinking") + 1] for cmd in p.runner.calls]
    assert tiers == ["off", "off"]
    assert p.name == "pi/gpt-oss:20b/features/off"


def test_fixed_effort_never_downshifts_under_deadline_pressure():
    """Benchmarking reasoning levels needs each arm to *stay* at its level, clock or no clock."""
    p = policy(
        [ok_events(), ok_events(), ok_events()],
        effort="low",
        fixed_effort=True,
        clock=FakeClock([0, 0, 5, 5, 5, 10, 10, 10, 15]),
    )
    p.deadline_s = 1.0  # every call is 5x over the clock; the ladder would step to off
    for turn in (1, 2, 3):
        p.plan(empty_board(), "O", "I", turn=turn)
    tiers = [cmd[cmd.index("--thinking") + 1] for cmd in p.runner.calls]
    assert tiers == ["low", "low", "low"]
    assert p.stats()["downshifts"] == 0


def test_pi_timeout_is_capped_by_the_deadline():
    p = policy([ok_events(), ok_events()])
    p.plan(empty_board(), "O", "I", turn=1)  # no deadline: configured timeout
    p.deadline_s = 5.0
    p.plan(empty_board(), "O", "I", turn=2)  # capped at max(15, 2*deadline)
    assert p.runner.timeouts == [180.0, 15.0]


def test_a_timed_out_call_counts_as_latency_so_the_controller_downshifts():
    # A model that deliberates past the clock never completes a call, so the
    # EMA would never learn it is slow. The timeout itself is the evidence.
    timeout = subprocess.TimeoutExpired(cmd="pi", timeout=15)
    p = policy(
        [timeout, timeout, ok_events()],
        clock=FakeClock([0, 0, 15, 15, 15, 30, 30, 30, 31]),
    )
    p.deadline_s = 7.0
    p.plan(empty_board(), "O", "I", turn=1)  # configured medium, times out
    p.plan(empty_board(), "O", "I", turn=2)  # EMA 15 > 0.9*7 -> low, times out
    p.plan(empty_board(), "O", "I", turn=3)  # still over -> off, answers in 1s

    tiers = [cmd[cmd.index("--thinking") + 1] for cmd in p.runner.calls]
    assert tiers == ["medium", "low", "off"]
    assert p.stats()["downshifts"] == 2
    assert p.usage["api_errors"] == 2
    assert p.usage["decisions"] == 1


def test_hard_deadline_kills_the_call_at_exactly_the_deadline():
    # Bounded-pause mode: the game is frozen, so the deadline IS the budget —
    # no 2x gravity slack like the live path.
    p = policy([ok_events(), ok_events()], hard_deadline=True)
    p.plan(empty_board(), "O", "I", turn=1)  # no deadline set: configured timeout
    p.deadline_s = 15.0
    p.plan(empty_board(), "O", "I", turn=2)
    assert p.runner.timeouts == [180.0, 15.0]


def test_routed_system_prompt_and_situation_reach_pi():
    import subprocess

    import numpy as np

    from tetris_agent.pi_policy import PiPolicy
    from tetris_agent.prompts import ROUTED_SYSTEM_PROMPT

    seen = {}

    def runner(cmd, timeout):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    p = PiPolicy(model="pi/gemma3", runner=runner, harness="routed")
    p.plan(np.zeros((18, 10), dtype=bool), "T", "O", turn=1)
    cmd = seen["cmd"]
    assert cmd[cmd.index("--system-prompt") + 1].startswith(ROUTED_SYSTEM_PROMPT)
    assert "Situation: FLAT" in cmd[-1]


def test_last_fallback_marks_a_placement_the_model_did_not_choose():
    """A fallback is not a decision, so the grader must skip it.

    Drives one policy instance through a failed decision followed by a
    successful one, so the assertion proves last_fallback RESETS on the next
    call rather than merely differing between two independently constructed
    instances (which would prove nothing about resetting at all).
    """
    p = policy([completed("", returncode=1), ok_events()])

    p.plan(empty_board(), "O", "I", turn=1)
    assert p.last_fallback is True
    assert p.stats()["illegal_count"] == 1

    p.plan(empty_board(), "O", "I", turn=2)
    assert p.last_fallback is False
