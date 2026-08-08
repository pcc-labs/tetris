"""A placement policy that shells out to the `pi` coding agent per decision.

The open-weight arms of the benchmark: `pi/<ollama-model>` ids run one
`pi -p --mode json` subprocess per piece against a local Ollama model. pi is
part of the harness under measurement — its overhead is part of the delta
against the Anthropic-API arms, not noise to subtract.

pi has no structured-output mode, so the placement JSON is prompt-instructed
and parsed leniently. Retry turns and chat-harness history are flattened into
a single prompt per call (no pi session files).
"""

import json
import logging
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from tetris_agent.model_policy import LLMPlacementPolicy
from tetris_agent.pricing import PI_PREFIX
from tetris_agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 180.0
OLLAMA_URL = os.environ.get("TETRIS_OLLAMA_URL", "http://127.0.0.1:11434")
# A model this large relative to total RAM will thrash or get OOM-killed rather
# than run. Measured the hard way: an 8B model needing ~10 GB resident took a
# machine into continuous swap and had its benchmark killed twice, silently,
# after ten minutes of producing nothing.
_WONT_FIT = 0.60
_TIGHT = 0.25
# Ships inside the package: reroutes pi's ollama provider through the tapes
# capture proxy when TETRIS_TAPES_OLLAMA_URL is set, no-ops otherwise.
TAPES_EXTENSION = Path(__file__).parent / "pi_extension" / "tapes-proxy.ts"

PI_JSON_INSTRUCTIONS = (
    "\n\n# Output format\n"
    "Reply with ONLY a single JSON object, no code fences, no prose before or after:\n"
    '{"rotation": <int 0-3>, "col": <int 0-9>, "reason": "<one short sentence>"}'
)

# Open-weight models ignore format rules that only live in the system prompt;
# the tail of the user turn is what they actually obey. The Anthropic arms get
# hard schema enforcement from output_config, so this is the pi arms' (weaker)
# equivalent, not an extra hint.
PI_PROMPT_SUFFIX = (
    '\n\nAnswer with ONLY the JSON object, e.g. {"rotation": 0, "col": 4, "reason": "flat"} — nothing else.'
)


def _total_ram_bytes() -> int | None:
    """Physical RAM, or None where the platform won't say."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def _ollama_models(base_url: str = OLLAMA_URL, timeout_s: float = 3.0) -> dict[str, int] | None:
    """Installed model name -> size in bytes, or None if Ollama isn't reachable."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout_s) as response:
            payload = json.load(response)
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None
    return {m["name"]: int(m.get("size") or 0) for m in payload.get("models", []) if m.get("name")}


def _match(name: str, installed: dict[str, int]) -> int | None:
    """Ollama reports `gemma3:latest`; an arm may just say `gemma3`."""
    for candidate in (name, f"{name}:latest"):
        if candidate in installed:
            return installed[candidate]
    return None


def preflight(model_ids, base_url: str = OLLAMA_URL) -> list[str]:
    """Reasons the pi arms in `model_ids` would fail — empty list means go.

    Worth doing because every one of these fails *slowly*: an unreachable
    Ollama or an unpulled model burns the per-decision timeout on all 50
    pieces, and a model too big for the box gets OOM-killed with no error
    anywhere. Seconds here saves tens of minutes of silence.
    """
    wanted = sorted({m.removeprefix(PI_PREFIX) for m in model_ids if m.startswith(PI_PREFIX)})
    if not wanted:
        return []

    installed = _ollama_models(base_url)
    if installed is None:
        return [f"ollama unreachable at {base_url} — pi arms need it running (`ollama serve`)"]

    problems = []
    total_ram = _total_ram_bytes()
    for name in wanted:
        size = _match(name, installed)
        if size is None:
            have = ", ".join(sorted(installed)) or "nothing"
            problems.append(f"{name!r} is not pulled — run `ollama pull {name}` (installed: {have})")
            continue
        if total_ram and size > total_ram * _WONT_FIT:
            problems.append(
                f"{name!r} needs ~{size / 1e9:.1f} GB but this box has {total_ram / 1e9:.1f} GB RAM — "
                "it will thrash or be OOM-killed mid-run; use a larger machine or a smaller model"
            )
        elif total_ram and size > total_ram * _TIGHT:
            logger.warning(
                "%s needs ~%.1f GB of this box's %.1f GB — close other work or expect swapping",
                name,
                size / 1e9,
                total_ram / 1e9,
            )
    return problems


def _run_pi(cmd: list[str], timeout_s: float) -> subprocess.CompletedProcess:
    # stdin must be closed: pi --mode json keeps the session open (reading
    # follow-up input) as long as stdin is, and the subprocess never exits.
    return subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _final_assistant_output(stdout: str) -> tuple[str | None, dict, str | None]:
    """Text, usage, and error of the last assistant message in a pi --mode json stream.

    Every assumption about pi's JSONL event shape lives here (pinned against
    v0.80.10): assistant turns arrive as
    {"type": "message_end", "message": {"role": "assistant",
     "content": [{"type": "text", "text": ...}, ...],
     "usage": {"input": .., "output": .., "cacheRead": .., "cacheWrite": ..}}}
    Unparseable lines are skipped so stray non-JSON output can't kill a run.

    A provider failure (Ollama down, model not pulled, proxy misconfigured) does
    not make pi exit nonzero — it reports `"stopReason": "error"` with an
    `errorMessage` and an empty content list, so that has to be read out of the
    stream or it masquerades as the model failing to produce JSON. pi retries
    internally, so the *last* assistant message is the verdict: a retry that
    eventually succeeded ends in a normal message, one that gave up ends in an
    error.
    """
    message: dict = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "message_end":
            continue
        candidate = event.get("message") or {}
        if candidate.get("role") == "assistant":
            message = candidate

    failed = message.get("stopReason") == "error"
    error = (message.get("errorMessage") or "unknown error") if failed else None
    blocks = [b.get("text", "") for b in message.get("content", []) if b.get("type") == "text"]
    return ("\n".join(blocks) if blocks else None, message.get("usage") or {}, error)


def _extract_json(text: str) -> dict | None:
    """First JSON object in the text, fences and prose tolerated.

    Scans with the real decoder rather than counting braces, so a brace inside
    a string value (`"reason": "fills the } notch"`) doesn't truncate the object.
    """
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            return decoder.raw_decode(text, start)[0]
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


class PiPolicy(LLMPlacementPolicy):
    def __init__(
        self,
        model: str,
        harness: str = "features",
        effort: str | None = "medium",
        runner: Callable[[list[str], float], subprocess.CompletedProcess] | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        pi_bin: str = "pi",
        extension: Path = TAPES_EXTENSION,
    ):
        if not model.startswith(PI_PREFIX):
            raise ValueError(f"PiPolicy needs a {PI_PREFIX}* model id, got {model!r}")
        super().__init__(model, harness, effort)
        self.ollama_model = model.removeprefix(PI_PREFIX)
        self.runner = runner or _run_pi
        self.timeout_s = timeout_s
        self.pi_bin = pi_bin
        self.extension = extension

    def _build_cmd(self, prompt: str) -> list[str]:
        cmd = [
            self.pi_bin,
            "-p",
            "--mode",
            "json",
            "--offline",
            "--provider",
            "ollama",
            "--model",
            self.ollama_model,
            "--no-session",
            "--no-tools",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
            "--no-extensions",
            "--system-prompt",
            SYSTEM_PROMPT + PI_JSON_INSTRUCTIONS,
            "-e",
            str(self.extension),
        ]
        if self.effort:
            cmd += ["--thinking", self.effort]
        cmd.append(prompt)
        return cmd

    def _render_transcript(self, messages: list[dict]) -> str:
        if len(messages) == 1:
            return messages[0]["content"]
        turns = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        return turns + "\n\nReply to the last [user] turn."

    def _call(self, messages: list[dict]) -> dict | None:
        cmd = self._build_cmd(self._render_transcript(messages) + PI_PROMPT_SUFFIX)
        started = time.monotonic()
        try:
            proc = self.runner(cmd, self.timeout_s)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("pi decision call failed: %s", exc)
            self.usage["api_errors"] += 1
            self.usage["latency_ms_total"] += (time.monotonic() - started) * 1000
            return None
        self.usage["latency_ms_total"] += (time.monotonic() - started) * 1000
        if proc.returncode != 0:
            logger.warning("pi exited %d: %s", proc.returncode, (proc.stderr or "")[-500:])
            self.usage["api_errors"] += 1
            return None

        text, usage, error = _final_assistant_output(proc.stdout)
        self.usage["input_tokens"] += usage.get("input", 0) or 0
        self.usage["output_tokens"] += usage.get("output", 0) or 0
        self.usage["cache_read_tokens"] += usage.get("cacheRead", 0) or 0
        self.usage["cache_write_tokens"] += usage.get("cacheWrite", 0) or 0
        if error is not None:
            # pi still exits 0 here; without this the arm's `parse_failures` and
            # `illegal` would blame the model for a dead provider.
            logger.warning("pi provider error for %s: %s", self.model, error)
            self.usage["api_errors"] += 1
            return None
        self.usage["decisions"] += 1

        answer = _extract_json(text) if text else None
        if answer is None:
            self.usage["parse_failures"] += 1
            return None
        try:
            return {
                "rotation": int(answer["rotation"]),
                "col": int(answer["col"]),
                "reason": str(answer.get("reason", "")),
            }
        except (KeyError, TypeError, ValueError):
            self.usage["parse_failures"] += 1
            return None
