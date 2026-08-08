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
