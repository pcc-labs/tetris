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

    def __call__(self, cmd, timeout_s):
        self.calls.append(cmd)
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
