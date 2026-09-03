"""A Claude-driven placement policy — one decision call per piece.

This is the benchmark's subject. The model sees the board through a harness
(how much scaffolding it gets) and answers with a structured placement; the
policy validates that answer against the legal set, retries once on an illegal
one, and accounts for every token, dollar, and millisecond it spent.

`LLMPlacementPolicy` holds everything provider-agnostic (validate, retry,
chat history, accounting); a subclass supplies `_call`, which turns a message
list into a `{"rotation", "col", "reason"}` dict or None on failure.
"""

import json
import logging
import time

import anthropic
import numpy as np

from tetris_agent.policy import Genome, Placement
from tetris_agent.pricing import EFFORTS, cost_usd, spec
from tetris_agent.prompts import (
    PLACEMENT_SCHEMA,
    build_user_prompt,
    legal_placements,
    system_prompt_for,
)
from tetris_agent.situation import Situation, classify

logger = logging.getLogger(__name__)

MAX_TOKENS = 8192
CHAT_HISTORY_TURNS = 12  # kept pairs in the `chat` harness before trimming


class LLMPlacementPolicy:
    def __init__(
        self,
        model: str,
        harness: str = "features",
        effort: str | None = "medium",
        max_tokens: int = MAX_TOKENS,
        exemplar_block: str = "",
        clock=time.monotonic,
        genome: Genome | None = None,
        fixed_effort: bool = False,
    ):
        self.model = model
        self.harness = harness
        self.spec = spec(model)
        # Haiku 4.5 rejects output_config.effort; its arms are effort-free by construction.
        self.effort = effort if self.spec.supports_effort else None
        # Pinned: the deadline controller may not step the tier. For a matrix
        # whose axis *is* the reasoning level, a row that quietly slid to a
        # lower tier under the clock would be measuring the controller instead.
        self.fixed_effort = fixed_effort
        self.max_tokens = max_tokens
        self._clock = clock
        # Situation thresholds for the routed harness — the same genome the heuristic plays with.
        self.genome = genome or Genome()
        # Exemplars are static across the run, so they belong in the system
        # prompt — cached — never in the per-piece user turn.
        self.system_prompt = system_prompt_for(harness) + (f"\n\n{exemplar_block}" if exemplar_block else "")
        self.name = f"{model}/{harness}" + (f"/{self.effort}" if self.effort else "")
        self._history: list[dict] = []
        self.last_situation: Situation | None = None
        self.usage = {
            "decisions": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "thinking_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "illegal_count": 0,
            "retry_count": 0,
            "parse_failures": 0,
            "refusals": 0,
            "api_errors": 0,
            "downshifts": 0,
            "latency_ms_total": 0.0,
            # Successful calls only: latency_ms_total also counts errors, so
            # tokens-per-second is derived from this pair, never from it.
            "gen_ms_total": 0.0,
        }
        self.last_reason = ""
        self.last_output_tokens: int | None = None
        # Live mode sets this before each decision; None means no clock.
        self.deadline_s: float | None = None
        self._decide_started: float | None = None
        self._ema_latency_s: float | None = None
        self._tier_index: int | None = None

    # ---- the Policy contract -------------------------------------------------

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        legal = legal_placements(board, piece)
        if not legal:
            return None

        situation = classify(board, piece, next_piece, self.genome) if self.harness == "routed" else None
        self.last_situation = situation
        prompt = build_user_prompt(
            self.harness, board, piece, next_piece, legal, turn, deadline_s=self.deadline_s, situation=situation
        )
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
        gen_s = u["gen_ms_total"] / 1000
        return {
            "policy": self.name,
            "model": self.model,
            "harness": self.harness,
            "effort": self.effort,
            **{k: v for k, v in u.items() if k not in ("latency_ms_total", "gen_ms_total")},
            "latency_ms_mean": round(u["latency_ms_total"] / decisions, 1),
            "tokens_per_second": round(u["output_tokens"] / gen_s, 1) if gen_s > 0 else 0.0,
            "tokens_per_decision": round(u["output_tokens"] / decisions, 1),
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
        self._decide_started = self._clock()
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
                if self._out_of_time():
                    return None
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
        """Return {"rotation", "col", "reason"} or None; account usage/latency/errors."""
        raise NotImplementedError

    def _effort_ladder(self) -> list[str] | None:
        """Tiers the deadline controller may pick from, floor first, configured last."""
        if not self.effort:
            return None
        if self.fixed_effort:
            return [self.effort]
        return list(EFFORTS[: EFFORTS.index(self.effort) + 1])

    def _choose_effort(self) -> str | None:
        """The effort for this call: the configured tier, stepped down one level
        at a time while the observed latency (EMA) can't beat the clock, and
        back up when there's room. Arm names keep the configured tier."""
        ladder = self._effort_ladder()
        if ladder is None:
            return self.effort
        if self._tier_index is None:
            self._tier_index = len(ladder) - 1
        if self.deadline_s is not None and self._ema_latency_s is not None:
            if self._ema_latency_s > 0.9 * self.deadline_s and self._tier_index > 0:
                self._tier_index -= 1
                self.usage["downshifts"] += 1
            elif self._ema_latency_s < 0.45 * self.deadline_s and self._tier_index < len(ladder) - 1:
                self._tier_index += 1
        return ladder[self._tier_index]

    def _out_of_time(self) -> bool:
        """A retry means a whole second call; skip it when the clock says no."""
        if self.deadline_s is None or self._decide_started is None:
            return False
        return (self._clock() - self._decide_started) > 0.5 * self.deadline_s

    def _remember(self, prompt: str, answer: dict) -> None:
        if self.harness != "chat":
            return
        self._history.append({"role": "user", "content": prompt})
        self._history.append({"role": "assistant", "content": json.dumps(answer)})
        if len(self._history) > CHAT_HISTORY_TURNS * 2:
            self._history = self._history[-CHAT_HISTORY_TURNS * 2 :]


class ModelPolicy(LLMPlacementPolicy):
    def __init__(
        self,
        model: str,
        harness: str = "features",
        effort: str | None = "medium",
        client=None,
        max_tokens: int = MAX_TOKENS,
        exemplar_block: str = "",
        clock=time.monotonic,
        genome: Genome | None = None,
    ):
        super().__init__(model, harness, effort, max_tokens, exemplar_block, clock=clock, genome=genome)
        # X-Tapes-Agent-Name tags captured turns when the traffic exits
        # through a tapes proxy (scripts/tapes-up.sh, scripts/vm-demo.sh);
        # api.anthropic.com ignores it when no proxy is in the path. Session
        # grouping is conversation-chained on the capture side, so only the
        # `chat` harness — whose history carries across pieces — lands a whole
        # run as one session; the stateless harnesses land one per piece.
        self.client = client or anthropic.Anthropic(default_headers={"X-Tapes-Agent-Name": "tetris-agent"})

    def _call(self, messages: list[dict]) -> dict | None:
        output_config: dict = {"format": {"type": "json_schema", "schema": PLACEMENT_SCHEMA}}
        effort = self._choose_effort()
        if effort:
            output_config["effort"] = effort

        started = self._clock()
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=[{"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}],
                output_config=output_config,
                messages=messages,
            )
        except anthropic.APIError:
            logger.exception("decision call failed")
            self.usage["api_errors"] += 1
            self.usage["latency_ms_total"] += (self._clock() - started) * 1000
            return None
        gen_s = self._clock() - started
        self.usage["latency_ms_total"] += gen_s * 1000
        self.usage["gen_ms_total"] += gen_s * 1000
        self.usage["decisions"] += 1
        self._ema_latency_s = gen_s if self._ema_latency_s is None else 0.5 * gen_s + 0.5 * self._ema_latency_s
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
        out = getattr(usage, "output_tokens", 0) or 0
        self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        self.usage["output_tokens"] += out
        # A subset of output_tokens — never added into cost on top of it.
        self.usage["thinking_tokens"] += (
            getattr(getattr(usage, "output_tokens_details", None), "thinking_tokens", 0) or 0
        )
        self.usage["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
        self.usage["cache_write_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.last_output_tokens = out
