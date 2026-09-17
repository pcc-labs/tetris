"""A Jev-driven placement policy — one typed `choice` per piece.

The LLM arms generate a placement and the policy parses and validates it.
Jev (TypeSafe's System One model, see jev.py) cannot generate at all: it is
handed the legal placements as the options of a `choice` question and returns
one of them with a probability for every option. So this arm has no parse
failures and no illegal answers by construction, one round trip of a few
hundred milliseconds per piece, and a bill of $0.042 per million input
tokens with output free. What it may lack is the ability to plan — it is a
calibrated judgment model, not a search — which is exactly what the benchmark
row measures.

The harness axis still means something: `board` shows Jev the board and the
option keys only; every other harness (the default, `features`) also gives
each option its consequences — lines cleared, holes, heights — the way the
`features` prompt does for an LLM.
"""

import logging
import time

import numpy as np

from tetris_agent import jev
from tetris_agent.policy import Genome, Placement
from tetris_agent.pricing import cost_usd
from tetris_agent.prompts import LegalPlacement, legal_placements, render_board

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are placing the current piece in Game Boy Tetris (10 columns, 18 rows, row 0 at the top). "
    "Every option is a legal resting place for the piece: `rotation` is the piece's rotation state and "
    "`col` the leftmost column it occupies once dropped straight down. Choose the placement that keeps "
    "the stack low and flat, creates no covered empty cells (holes), and clears lines when it can, "
    "bearing in mind the next piece still has to fit afterwards."
)


def option_key(p: LegalPlacement) -> str:
    return f"r{p.rotation}c{p.col}"


def describe(p: LegalPlacement) -> str:
    f = p.features
    return (
        f"rotation {p.rotation}, column {p.col}: clears {p.lines} line(s), leaves {f.holes} hole(s), "
        f"aggregate height {f.agg_height}, bumpiness {f.bumpiness}, max height {f.max_height}"
    )


class JevPolicy:
    def __init__(
        self,
        model: str = jev.DEFAULT_MODEL,
        harness: str = "features",
        exemplar_block: str = "",
        clock=time.monotonic,
        genome: Genome | None = None,
        ask=jev.ask,
    ):
        self.model = model
        self.harness = harness
        self.effort = None  # Jev has no reasoning dial
        self.genome = genome or Genome()
        # Jev takes no system prompt, so the (static) exemplar block rides in
        # the state of every call instead. Cheap at Jev's price, and it keeps
        # the +ex label honest.
        self.exemplar_block = exemplar_block
        self._clock = clock
        self._ask = ask
        self.name = f"{model}/{harness}"
        self.usage = {
            "decisions": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "illegal_count": 0,  # always 0: the answer is constrained to the legal set
            "api_errors": 0,
            "latency_ms_total": 0.0,
        }
        self.last_reason = ""
        self.last_fallback = False
        self.last_output_tokens: int | None = None
        self.last_probabilities: dict[str, float] = {}
        self.deadline_s: float | None = None  # set by live mode; a Jev call is far inside any deadline

    # ---- the Policy contract -------------------------------------------------

    def plan(self, board: np.ndarray, piece: str, next_piece: str, turn: int) -> Placement | None:
        legal = legal_placements(board, piece)
        if not legal:
            return None
        by_key = {option_key(p): p for p in legal}
        state = self._state(board, piece, next_piece, turn, legal)
        question = {"placement": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": self._criteria(legal)}}

        started = self._clock()
        try:
            result = self._ask(state, question)
        except Exception as err:
            self.usage["api_errors"] += 1
            self.usage["latency_ms_total"] += (self._clock() - started) * 1000
            logger.warning("jev: %s", err)
            self.last_fallback = True
            self.last_reason = ""
            self.last_probabilities = {}
            choice = legal[0]
            return Placement(rotation=choice.rotation, col=choice.col, score=0.0)
        self.usage["latency_ms_total"] += (self._clock() - started) * 1000
        self.usage["decisions"] += 1
        usage = result.get("usage") or {}
        self.usage["input_tokens"] += int(usage.get("input_tokens", 0))
        self.usage["output_tokens"] += int(usage.get("output_tokens", 0))
        self.last_output_tokens = int(usage.get("output_tokens", 0))

        answer = (result.get("answers") or {}).get("placement") or {}
        probabilities = answer.get("probabilities") or {}
        choice = by_key.get(answer.get("choice"))
        if choice is None:
            # Cannot happen when the API honours its contract; counted so a
            # violation shows in the table rather than passing as a decision.
            self.usage["illegal_count"] += 1
            self.last_fallback = True
            choice = legal[0]
        else:
            self.last_fallback = False
        self.last_probabilities = probabilities
        self.last_reason = self._reason(answer.get("choice"), probabilities)
        confidence = float(probabilities.get(option_key(choice), 0))
        return Placement(rotation=choice.rotation, col=choice.col, score=confidence)

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
            "tokens_per_second": 0.0,  # no generation
            "tokens_per_decision": 0.0,
            "cost_usd": cost_usd(self.model, u["input_tokens"], u["output_tokens"]),
        }

    # ---- internals -----------------------------------------------------------

    def _criteria(self, legal: list[LegalPlacement]) -> dict[str, str | None]:
        if self.harness == "board":
            return {option_key(p): None for p in legal}
        return {option_key(p): describe(p) for p in legal}

    def _state(self, board, piece: str, next_piece: str, turn: int, legal: list[LegalPlacement]) -> dict:
        state = {
            "turn": turn,
            "piece": piece,
            "next_piece": next_piece,
            "board": render_board(board).split("\n"),
            "board_legend": "'.' empty, '#' settled, row 0 at the top",
            "legal_placements": [option_key(p) for p in legal],
        }
        if self.exemplar_block:
            state["how_a_strong_human_played"] = self.exemplar_block
        return state

    @staticmethod
    def _reason(chosen: str | None, probabilities: dict[str, float]) -> str:
        if not chosen:
            return ""
        ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        runner_up = next(((k, p) for k, p in ranked if k != chosen), None)
        text = f"{chosen} p={probabilities.get(chosen, 0):.2f}"
        if runner_up:
            text += f", next {runner_up[0]} p={runner_up[1]:.2f}"
        return text
