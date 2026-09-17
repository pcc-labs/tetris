"""The Jev arm: a typed choice over the legal set, so nothing to parse and nothing illegal."""

import numpy as np

from tetris_agent.benchmark import Arm, build_policy
from tetris_agent.jev_policy import JevPolicy, option_key
from tetris_agent.pricing import cost_usd, is_jev, spec
from tetris_agent.prompts import legal_placements


class FakeJev:
    def __init__(self, pick=None, fail=False):
        self.calls = []
        self.pick = pick
        self.fail = fail

    def __call__(self, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("TypeSafe 529: overloaded")
        options = list(questions["placement"]["criteria"])
        chosen = self.pick or options[-1]
        probabilities = {k: (0.7 if k == chosen else 0.3 / max(len(options) - 1, 1)) for k in options}
        return {
            "model": "jev-test",
            "answers": {
                "placement": {"type": "choice", "choice": chosen, "probabilities": probabilities, "confidence": 0.7}
            },
            "usage": {"input_tokens": 800, "output_tokens": 0},
        }


def test_plan_returns_the_option_jev_chose_and_accounts_for_it():
    board = np.zeros((18, 10), dtype=bool)
    fake = FakeJev(pick="r0c7")
    policy = JevPolicy(ask=fake, clock=iter([0.0, 0.25]).__next__)
    placement = policy.plan(board, "O", "I", turn=1)
    assert (placement.rotation, placement.col) == (0, 7)
    assert placement.score == 0.7
    assert policy.last_fallback is False
    assert policy.last_reason.startswith("r0c7 p=0.70, next ")
    stats = policy.stats()
    assert stats["decisions"] == 1 and stats["illegal_count"] == 0
    assert stats["latency_ms_mean"] == 250.0
    assert stats["cost_usd"] == cost_usd("jev-latest", 800, 0)
    assert stats["effort"] is None


def test_features_harness_describes_every_legal_option_and_board_harness_does_not():
    board = np.zeros((18, 10), dtype=bool)
    legal = legal_placements(board, "T")
    fake = FakeJev()
    JevPolicy(harness="features", ask=fake).plan(board, "T", "O", turn=3)
    state, questions = fake.calls[0]
    criteria = questions["placement"]["criteria"]
    assert set(criteria) == {option_key(p) for p in legal}
    assert all("clears" in text for text in criteria.values())
    assert state["piece"] == "T" and state["next_piece"] == "O" and len(state["board"]) == 18
    assert state["legal_placements"] == list(criteria)

    bare = FakeJev()
    JevPolicy(harness="board", ask=bare).plan(board, "T", "O", turn=3)
    assert all(v is None for v in bare.calls[0][1]["placement"]["criteria"].values())


def test_exemplars_ride_in_the_state():
    fake = FakeJev()
    JevPolicy(ask=fake, exemplar_block="# How a strong human placed pieces").plan(
        np.zeros((18, 10), dtype=bool), "I", "I", turn=1
    )
    assert fake.calls[0][0]["how_a_strong_human_played"].startswith("# How a strong human")


def test_api_failure_falls_back_and_is_not_a_choice():
    fake = FakeJev(fail=True)
    policy = JevPolicy(ask=fake)
    placement = policy.plan(np.zeros((18, 10), dtype=bool), "O", "I", turn=1)
    assert placement is not None
    assert policy.last_fallback is True  # the grader must skip it
    assert policy.stats()["api_errors"] == 1 and policy.stats()["decisions"] == 0


def test_benchmark_routes_jev_models_to_the_jev_policy():
    assert is_jev("jev-latest") and not is_jev("claude-opus-5")
    assert spec("jev-latest").supports_effort is False
    policy = build_policy(Arm(policy="model", model="jev-latest", harness="features"))
    assert isinstance(policy, JevPolicy)
    assert policy.name == "jev-latest/features"
    assert Arm(policy="model", model="jev-latest", harness="features", exemplars=True).name == "jev-latest/features+ex"
