# Benchmark Phase: Model × Harness × Effort

**Goal:** Turn the repo from "heuristic that plays Tetris" into "benchmark harness that measures how well a *model* plays Tetris, under a given harness, at a given reasoning effort." The heuristic becomes the control arm.

**Why this is the right substrate:** the pieces already in place are exactly what a benchmark needs — `timer_div` gives identical piece sequences across arms (fair comparison), `fitness.py` already produces comparable metrics, `evolve.run_race` is already a "run N configs across M seeds" runner, and `runs/` + the viewer replay any arm.

## The three axes

| Axis | Values | Where it lives |
|---|---|---|
| **Model** | `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`, `claude-fable-5` | `model=` |
| **Effort** | `low`, `medium`, `high`, `xhigh`, `max` | `output_config.effort` — **Haiku 4.5 rejects this param**, so its arms run effort-free |
| **Harness** | `board`, `legal`, `features`, `chat` | how the board is presented + what scaffolding the model gets |

Harness ladder, from least to most scaffolded:
1. **`board`** — ASCII board + current piece + next piece. The model must derive legality and consequences itself.
2. **`legal`** — board plus the enumerated list of legal `(rotation, col)` placements. Removes legality reasoning; keeps evaluation.
3. **`features`** — legal placements each annotated with the resulting `lines/holes/agg_height/bumpiness`. The model gets the heuristic's *inputs* but must weigh them itself.
4. **`chat`** — like `board` but keeps conversation history across pieces, so the model can pursue a plan (and pays growing context).

Plus the **`heuristic`** control arm: no LLM, free, instant — the score every model arm is measured against.

## Architecture

- `policy.py` grows a `Policy` protocol (`plan(state) -> Placement | None`); the existing weighted search becomes `HeuristicPolicy`.
- `prompts.py` — board rendering and the four harness prompt builders. Pure and unit-testable.
- `model_policy.py` — `ModelPolicy`: one Anthropic call per piece, structured output (`output_config.format` json_schema), validation against legal placements, one retry on an illegal answer, then nearest-legal fallback. Accumulates usage.
- `pricing.py` — per-model $/MTok, cache-read/write multipliers, `supports_effort`.
- `benchmark.py` — matrix expansion, per-arm runs across seeds, results JSONL + a ranked table.
- `agent.py` takes a `policy` rather than a genome (the genome now only configures the heuristic).

## Cost discipline (this is the risk)

One API call per piece. A 4×4×3×2-seed matrix at 50 pieces is ~4,800 calls — real money. Mitigations, all mandatory:

- Static system prompt is cached (`cache_control: ephemeral`); the volatile board goes last. Verify with `usage.cache_read_input_tokens` — note the minimum cacheable prefix is 512 tokens on Opus 5 but **4096 on Haiku 4.5**, so Haiku arms likely won't cache at all.
- Every run reports `cost_usd`, tokens, and per-decision latency alongside score.
- `tetris-bench` takes `--max-usd` and aborts the matrix when projected spend crosses it.
- `--estimate` prints projected cost from `count_tokens` without calling the model.
- Ship with a **small** default (`--max-pieces 30`, 1 seed) and let the user scale up deliberately.

## New metrics (on top of fitness)

`illegal_count` (model proposed a placement that doesn't fit), `retry_count`, `parse_failures`, `input/output/cache_read/cache_write tokens`, `cost_usd`, `latency_ms_mean`, `decisions`.

## Deferred

Leaderboard UI in the viewer; healer tuning the *harness* rather than the genome (a natural follow-on — the genome concept generalizes to prompt/effort/retry parameters).
