"""A Claude-driven placement policy — one decision call per piece.

This is the benchmark's subject. The model sees the board through a harness
(how much scaffolding it gets) and answers with a structured placement; the
policy validates that answer against the legal set, retries once on an illegal
one, and accounts for every token, dollar, and millisecond it spent.
"""

import json
import logging
import time

import anthropic
import numpy as np

from tetris_agent.policy import Placement
from tetris_agent.pricing import cost_usd, spec
from tetris_agent.prompts import (
    PLACEMENT_SCHEMA,
    SYSTEM_PROMPT,
    build_user_prompt,
    legal_placements,
)

logger = logging.getLogger(__name__)

MAX_TOKENS = 8192
CHAT_HISTORY_TURNS = 12  # kept pairs in the `chat` harness before trimming


class ModelPolicy:
    def __init__(
        self,
        model: str,
        harness: str = "features",
        effort: str | None = "medium",
        client=None,
        max_tokens: int = MAX_TOKENS,
    ):
        self.model = model
        self.harness = harness
        self.spec = spec(model)
        # Haiku 4.5 rejects output_config.effort; its arms are effort-free by construction.
        self.effort = effort if self.spec.supports_effort else None
        self.client = client or anthropic.Anthropic()
        self.max_tokens = max_tokens
        self.name = f"{model}/{harness}" + (f"/{self.effort}" if self.effort else "")
        self._history: list[dict] = []
        self.usage = {
            "decisions": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "illegal_count": 0,
            "retry_count": 0,
            "parse_failures": 0,
            "refusals": 0,
            "api_errors": 0,
            "latency_ms_total": 0.0,
        }
        self.last_reason = ""

    # ---- the Policy contract -------------------------------------------------

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        legal = legal_placements(board, piece)
        if not legal:
            return None

        prompt = build_user_prompt(self.harness, board, piece, next_piece, legal, turn)
        choice = self._decide(prompt, legal)
        if choice is None:
            # Every attempt failed; keep the run alive with a deterministic
            # fallback so the arm is still comparable, and count it as illegal.
            self.usage["illegal_count"] += 1
            choice = legal[0]
        return Placement(rotation=choice.rotation, col=choice.col, score=0.0)

    def stats(self) -> dict:
        u = self.usage
        decisions = max(u["decisions"], 1)
        return {
            "policy": self.name,
            "model": self.model,
            "harness": self.harness,
            "effort": self.effort,
            **{k: v for k, v in u.items() if k != "latency_ms_total"},
            "latency_ms_mean": round(u["latency_ms_total"] / decisions, 1),
            "cost_usd": cost_usd(
                self.model,
                u["input_tokens"],
                u["output_tokens"],
                u["cache_read_tokens"],
                u["cache_write_tokens"],
            ),
        }

    # ---- internals -----------------------------------------------------------

    def _decide(self, prompt: str, legal: list):
        """Ask, validate, retry once with the failure explained, then give up."""
        messages = self._history + [{"role": "user", "content": prompt}]
        for attempt in range(2):
            answer = self._call(messages)
            if answer is None:
                return None
            match = next(
                (p for p in legal if p.rotation == answer["rotation"] and p.col == answer["col"]),
                None,
            )
            if match is not None:
                self.last_reason = answer.get("reason", "")
                self._remember(prompt, answer)
                return match
            self.usage["illegal_count"] += 1
            if attempt == 0:
                self.usage["retry_count"] += 1
                options = ", ".join(f"(rot={p.rotation}, col={p.col})" for p in legal)
                messages = messages + [
                    {"role": "assistant", "content": json.dumps(answer)},
                    {
                        "role": "user",
                        "content": (
                            f"rotation={answer['rotation']} col={answer['col']} is not a legal placement "
                            f"for this piece — it does not fit. Choose one of: {options}"
                        ),
                    },
                ]
        return None

    def _call(self, messages: list[dict]) -> dict | None:
        output_config: dict = {"format": {"type": "json_schema", "schema": PLACEMENT_SCHEMA}}
        if self.effort:
            output_config["effort"] = self.effort

        started = time.monotonic()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                output_config=output_config,
                messages=messages,
            )
        except anthropic.APIError:
            logger.exception("decision call failed")
            self.usage["api_errors"] += 1
            self.usage["latency_ms_total"] += (time.monotonic() - started) * 1000
            return None
        self.usage["latency_ms_total"] += (time.monotonic() - started) * 1000
        self.usage["decisions"] += 1
        self._record_usage(response.usage)

        if response.stop_reason == "refusal":
            self.usage["refusals"] += 1
            return None
        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            self.usage["parse_failures"] += 1
            return None
        try:
            answer = json.loads(text)
            return {"rotation": int(answer["rotation"]), "col": int(answer["col"]), "reason": answer.get("reason", "")}
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            self.usage["parse_failures"] += 1
            return None

    def _record_usage(self, usage) -> None:
        self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        self.usage["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def _remember(self, prompt: str, answer: dict) -> None:
        if self.harness != "chat":
            return
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": json.dumps(answer)})
        if len(self._history) > CHAT_HISTORY_TURNS * 2:
            self._history = self._history[-CHAT_HISTORY_TURNS * 2 :]
