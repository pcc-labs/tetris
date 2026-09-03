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
    # Ollama cloud tags proxy through the local daemon but bill the account's
    # credits at the "Cost /1M tokens" rate on each tag's ollama.com page
    # (read 2026-09-03). They take pi's --thinking — Ollama's OpenAI endpoint maps
    # it to reasoning_effort, "off" to "none" — but only once the tag is listed in
    # ~/.pi/agent/models.json with reasoning: true and thinkingLevelMap.off = "none";
    # an unlisted tag silently runs at Ollama's default thinking whatever the flag
    # says. The effort-free rows on record (arm ids with no effort segment) were
    # exactly that. Ollama's cached-input rate is per model and not the uniform
    # 10 % below; pi has reported zero cacheRead for Ollama so far.
    "pi/gpt-oss:20b-cloud": ModelSpec("pi/gpt-oss:20b-cloud", 0.07, 0.30, True, 0),
    "pi/gpt-oss:120b-cloud": ModelSpec("pi/gpt-oss:120b-cloud", 0.15, 0.60, True, 0),
    "pi/gemma4:cloud": ModelSpec("pi/gemma4:cloud", 0.14, 0.40, True, 0),
    "pi/gemma4:31b-cloud": ModelSpec("pi/gemma4:31b-cloud", 0.14, 0.40, True, 0),
    "pi/nemotron-3-super:cloud": ModelSpec("pi/nemotron-3-super:cloud", 0.015, 0.60, True, 0),
    "pi/kimi-k3:cloud": ModelSpec("pi/kimi-k3:cloud", 3.00, 15.00, True, 0),
    "pi/deepseek-v4-flash:0731-cloud": ModelSpec("pi/deepseek-v4-flash:0731-cloud", 0.44, 1.32, True, 0),
    "pi/deepseek-v4-pro:cloud": ModelSpec("pi/deepseek-v4-pro:cloud", 1.32, 3.96, True, 0),
    "pi/glm-5.3:cloud": ModelSpec("pi/glm-5.3:cloud", 1.40, 4.40, True, 0),
    "pi/glm-5.3-flash:cloud": ModelSpec("pi/glm-5.3-flash:cloud", 0.15, 0.50, True, 0),
    "pi/glm-5.2:cloud": ModelSpec("pi/glm-5.2:cloud", 1.40, 4.40, True, 0),
    # glm-5.1's page reads $0.20 / $1.00 / $3.20; cached below input is the
    # convention everywhere else on the site, so input is taken as $1.00.
    "pi/glm-5.1:cloud": ModelSpec("pi/glm-5.1:cloud", 1.00, 3.20, True, 0),
    "pi/kimi-k2.7-code:cloud": ModelSpec("pi/kimi-k2.7-code:cloud", 0.95, 4.00, True, 0),
    "pi/kimi-k2.6:cloud": ModelSpec("pi/kimi-k2.6:cloud", 0.95, 4.00, True, 0),
    "pi/minimax-m3:cloud": ModelSpec("pi/minimax-m3:cloud", 0.60, 2.40, True, 0),
    "pi/minimax-m2.7:cloud": ModelSpec("pi/minimax-m2.7:cloud", 0.30, 1.20, True, 0),
    "pi/qwen3.5:cloud": ModelSpec("pi/qwen3.5:cloud", 0.60, 3.60, True, 0),
    "pi/nemotron-3-ultra:cloud": ModelSpec("pi/nemotron-3-ultra:cloud", 0.10, 3.00, True, 0),
}


def is_ollama_cloud(model_id: str) -> bool:
    """`pi/<name>:cloud` or `pi/<name>:<size>-cloud` — served by Ollama's cloud, not this box."""
    tag = model_id.removeprefix(PI_PREFIX)
    return is_pi(model_id) and (tag.endswith(":cloud") or tag.endswith("-cloud"))


EFFORTS = ("low", "medium", "high", "xhigh", "max")


def is_pi(model_id: str) -> bool:
    return model_id.startswith(PI_PREFIX)


def spec(model_id: str) -> ModelSpec:
    if model_id in MODELS:
        return MODELS[model_id]
    if is_ollama_cloud(model_id):
        # A cloud tag bills credits; defaulting it to free would report a bill
        # of $0 for a run that cost money. Unknown and free are different claims.
        tag = model_id.removeprefix(PI_PREFIX)
        raise KeyError(
            f"no rate on file for {model_id!r} — copy the 'Cost /1M tokens' figures from "
            f"https://ollama.com/library/{tag} into pricing.MODELS"
        )
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
