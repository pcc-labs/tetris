"""Model capabilities and cost accounting for benchmark arms."""

from dataclasses import dataclass

CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25  # 5-minute TTL

# What a local arm's watt-hours are worth. `../pokemon-kafka` drifted between
# three values for this (launcher 0.39, writeups 0.30, bench_report default 0.0,
# which silently zeroed the column) — one constant here, non-zero, and always
# echoed alongside the figure so the rate is never invisible.
DEFAULT_KWH_PRICE = 0.39


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    input_per_mtok: float
    output_per_mtok: float
    supports_effort: bool
    # Minimum cacheable prefix; a shorter system prompt silently won't cache.
    cache_min_tokens: int


# `pi/<ollama-model>` arms run through the pi agent against local Ollama: free,
# and supports_effort mirrors whether the model accepts pi's --thinking flag.
PI_PREFIX = "pi/"

MODELS: dict[str, ModelSpec] = {
    "claude-fable-5": ModelSpec("claude-fable-5", 10.0, 50.0, True, 512),
    "claude-opus-5": ModelSpec("claude-opus-5", 5.0, 25.0, True, 512),
    # Sonnet 5 sticker is $3/$15; introductory pricing runs through 2026-08-31.
    "claude-sonnet-5": ModelSpec("claude-sonnet-5", 2.0, 10.0, True, 1024),
    # Haiku 4.5 rejects output_config.effort — its arms run effort-free.
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", 1.0, 5.0, False, 4096),
    "pi/gpt-oss:20b": ModelSpec("pi/gpt-oss:20b", 0.0, 0.0, True, 0),
    "pi/glm-4.7-flash": ModelSpec("pi/glm-4.7-flash", 0.0, 0.0, True, 0),
    "pi/qwen3.6:27b": ModelSpec("pi/qwen3.6:27b", 0.0, 0.0, True, 0),
    # num_ctx-capped variants: stock glm/qwen auto-size to 202k/262k and the
    # runner dies loading them on the iGPU. Same blobs, PARAMETER num_ctx 32768.
    "pi/glm-4.7-flash-32k": ModelSpec("pi/glm-4.7-flash-32k", 0.0, 0.0, True, 0),
    "pi/qwen3.6-27b-32k": ModelSpec("pi/qwen3.6-27b-32k", 0.0, 0.0, True, 0),
    # Nemotron 3.5 Lightning 30B-A3B MoE: ~67 tok/s steady-state decode on the
    # iGPU (same bracket as glm-4.7-flash). Native context is 1M, so only the
    # -32k variant is registered. Leave draft_num_predict at its default.
    "pi/nemotron-3.5-lightning-32k": ModelSpec("pi/nemotron-3.5-lightning-32k", 0.0, 0.0, True, 0),
    "pi/qwen3:8b": ModelSpec("pi/qwen3:8b", 0.0, 0.0, False, 0),
    "pi/gemma3": ModelSpec("pi/gemma3", 0.0, 0.0, False, 0),
}

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def is_pi(model_id: str) -> bool:
    return model_id.startswith(PI_PREFIX)


def spec(model_id: str) -> ModelSpec:
    if model_id in MODELS:
        return MODELS[model_id]
    if is_pi(model_id):
        # Unlisted local model: free, and effort-free so we never send
        # --thinking to a model we don't know accepts it.
        return ModelSpec(model_id, 0.0, 0.0, False, 0)
    raise KeyError(f"unknown model {model_id!r}; known: {sorted(MODELS)}")


def energy_usd(wh: float, kwh_price: float = DEFAULT_KWH_PRICE) -> float:
    """Dollar cost of watt-hours drawn.

    Deliberately separate from `cost_usd` rather than folded into it: `cost_usd`
    is token-only by signature and is called from `benchmark.estimate_cost`
    before any run exists, where there is no measured draw to fold in. Keeping
    them apart also keeps `--max-usd` an API-spend cap, so a local arm never
    consumes a dollar budget it is not billing.
    """
    return round(wh / 1000 * kwh_price, 6)


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
