"""Cost accounting: local pi arms are free, Ollama cloud arms are not."""

import pytest

from tetris_agent.pricing import cost_usd, spec


def test_local_pi_arm_costs_nothing():
    assert cost_usd("pi/gpt-oss:20b", 1_000_000, 1_000_000) == 0.0


def test_ollama_cloud_tag_is_priced_at_its_published_rate():
    # ollama.com/library/kimi-k3:cloud: $3.00 in / $15.00 out per million.
    assert cost_usd("pi/kimi-k3:cloud", 1_000_000, 1_000_000) == pytest.approx(18.0)
    # ollama.com/library/gpt-oss:120b-cloud: $0.15 in / $0.60 out per million.
    assert cost_usd("pi/gpt-oss:120b-cloud", 1_000_000, 0) == pytest.approx(0.15)


def test_cloud_tags_accept_the_thinking_flag():
    # Ollama's cloud tags take reasoning_effort (incl. "none"); the effort lands in the arm id,
    # so the effort-free rows recorded on 09-03 stay distinguishable from these.
    assert spec("pi/gpt-oss:120b-cloud").supports_effort is True
    assert spec("pi/kimi-k3:cloud").supports_effort is True


def test_unlisted_cloud_tag_is_refused_rather_than_priced_free():
    """A cloud tag we have no rate for must not silently report $0 — that is the bug."""
    with pytest.raises(KeyError, match="ollama.com/library/brand-new:cloud"):
        spec("pi/brand-new:cloud")
    with pytest.raises(KeyError):
        spec("pi/brand-new:7b-cloud")


def test_unlisted_local_pi_model_is_still_free():
    assert cost_usd("pi/brand-new:7b", 1_000_000, 1_000_000) == 0.0
