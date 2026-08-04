"""Model capabilities and cost accounting for benchmark arms."""

from dataclasses import dataclass

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    supports_effort: bool
    # Minimum cacheable prefix; a shorter system prompt silently won't cache.
    cache_min_tokens: int


MODELS: dict[str, ModelSpec] = {
    "claude-fable-5": ModelSpec("claude-fable-5", 10.0, 50.0, True, 512),
    "claude-opus-5": ModelSpec("claude-opus-5", 5.0, 25.0, True, 512),
    # Sonnet 5 sticker is $3/$15; introductory pricing runs through 2026-08-31.
    "claude-sonnet-5": ModelSpec("claude-sonnet-5", 2.0, 10.0, True, 1024),
    # Haiku 4.5 rejects output_config.effort — its arms run effort-free.
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", 1.0, 5.0, False, 4096),
}

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def spec(model_id: str) -> ModelSpec:
    if model_id not in MODELS:
        raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODELS)}")
    return MODELS[model_id]


def cost_usd(
    model_id: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float:
    s = spec(model_id)
    per_input = s.input_per_mtok / 1_000_000
    return round(
        input_tokens * per_input
        + output_tokens * (s.output_per_mtok / 1_000_000)
        + cache_read_tokens * per_input * CACHE_READ_MULTIPLIER
        + cache_write_tokens * per_input * CACHE_WRITE_MULTIPLIER,
        6,
    )
