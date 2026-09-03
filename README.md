# tetris-agent

<img width="934" height="572" alt="Screenshot 2026-08-03 at 5 54 25 PM" src="https://github.com/user-attachments/assets/da0b47fb-5cc3-4c3d-b18b-4134dbb87d4b" />

A benchmark harness for open-weight models playing Game Boy Tetris, with a
self-healing heuristic as the control arm. The piece sequence is deterministic
(`--timer-div` pins the RNG), so every arm plays the same game and score
differences are differences in play.

The standing goal: **a local model that plays as well as the cloud one.**

## Setup

```bash
uv sync
# Put the ROM at rom/tetris.gb (gitignored).

# Open-weight arms need Ollama and the pi coding agent
ollama pull gemma4
npm install -g @earendil-works/pi-coding-agent@0.80.10
```

Ollama is expected at `http://127.0.0.1:11434`; set `TETRIS_OLLAMA_URL` to
point elsewhere. Any model in `ollama list` works as a `pi/<model>` arm.

## Run it

```bash
# Watch a model play in the browser
uv run tetris-viewer                                        # http://127.0.0.1:8000
uv run tetris-agent --policy model --model pi/gemma4 --harness routed --live

# Benchmark a matrix: one row per model × harness
uv run tetris-bench --models pi/gemma4 pi/gpt-oss:20b \
                    --harnesses features routed \
                    --seeds 1 --max-pieces 30 --decision-deadline 15 --watch

# Play it yourself (keyboard in the viewer's LIVE tab)
uv run tetris-play --seed 0 --max-pieces 50
```

Results land in `data/benchmarks/` (gitignored) and print as a ranked table.
Results worth keeping go in [`benchmarks/`](benchmarks/) as a dated write-up.

### Harnesses

The harness is how much reasoning is done *for* the model before it decides:

- **`board`** — ASCII board, current piece, next piece.
- **`legal`** — plus the enumerated legal `(rotation, col)` placements.
- **`features`** — plus holes, height, bumpiness, and line clears per placement.
- **`chat`** — like `board`, but conversation history carries across pieces.
- **`routed`** — `legal`, plus the board's *situation* (`TOPPING_OUT`,
  `TETRIS_READY`, `HOLE_RISK`, `MOUND`, or `FLAT`, in `situation.py`) with the
  one technique it calls for and only the numbers that technique needs. The
  situations and techniques come from Jonas Neubauer's advice in Polygon's
  [Tetris tips from a seven-time world champion](https://www.polygon.com/guides/2019/2/22/18225349/tetris-strategy-tips-how-to-jonas-neubauer/).
  `uv run tetris-situations generate` then `histogram` shows the class split
  over a free corpus.

### The clock

Model arms play **live** by default: gravity keeps pulling while the model
thinks (level 0 ≈ 15 s from spawn to floor), and a decision that arrives after
its piece locked is counted as `late`. `--decision-deadline 15` is the
screening mode: the game freezes per piece, but a decision slower than the
deadline is discarded and counted in `timeouts`. `--paused` removes the clock
entirely. `--watch` streams any matrix to the viewer.

Every arm reports tokens per decision, seconds per decision, and `tok_s`, so a
flatlined arm is attributable: slow hardware or an overspent thinking budget.
(`tok_s` is end-to-end — output tokens over the pi subprocess's wall clock — so
it reads low for terse arms and is comparable across pi arms, not with a
provider's decode rate.)

### Placement quality

Every decision is graded against a two-ply oracle: the same weighted
evaluation the control arm plays with, searched one piece deeper. A row
reports `regret` (0 is the oracle's own choice, 1 is the worst legal
placement), `top1` and `top3`.

The oracle is a strong heuristic, not ground truth. Regret is disagreement with
a two-ply evaluator, so a model that outplays the oracle shows high regret; the
`--lookahead-control` arm puts the oracle's own score in the same table so the
comparison is visible. Regret is also per-move and path-relative: each decision
is graded against the board the model built, so a ruined board played well
scores near zero. Accumulated damage shows up in lines, holes and pieces
survived instead.

Grades are written to `runs/<id>/events.jsonl` as `placement_graded` events,
alongside the spawn, decision and lock events the exemplar miner already reads.
`--no-quality` turns all of it off.

Grading costs 4–33 ms per decision, negligible against a live decision's
multi-second budget. The one-ply `heuristic` control arm — 490 on the live
screen (random 140, no-input 55) — comes back with a normalized regret of
0.15 against the two-ply oracle: the gap the extra ply of search buys.

### Reasoning level

Thinking-capable models take `--efforts off|low|medium|high`, and the level is
the benchmark's main axis: on this task the level decides who plays at all.
By default an adaptive controller steps effort down when observed latency
can't beat the clock; `--fixed-effort` pins each arm to its configured level
(arms are labeled `+fixed`) so a matrix over `off low medium` measures the
level, not the controller. `off` is Ollama's `reasoning_effort: "none"`;
gpt-oss has no off and reasons at its default there.

The flag only reaches an Ollama model that pi knows about. Each tag must be
listed in `~/.pi/agent/models.json` under the `ollama` provider with
`"reasoning": true` and `"thinkingLevelMap": {"off": "none"}`; for an
unlisted tag pi silently drops `--thinking` and the model runs at Ollama's
default, whatever the arm id says. Every cloud row before 2026-09-03 was
recorded that way.

**Four cloud tags ignore the flag altogether**, as served by Ollama cloud on
2026-09-03: `glm-5.3`, `glm-5.3-flash`, `minimax-m3`, and `nemotron-3-ultra`.
With `reasoning_effort: "none"` on the OpenAI endpoint — and with
`think: false` on the raw `/api/generate` — glm-5.3 and glm-5.3-flash still
write their reasoning into the visible response (650–1,000 tokens, 60–160 s
per decision) and minimax-m3 still fills a hidden thinking block; nemotron-3-ultra
does not answer a one-word prompt inside two minutes either way. They flatline
at every level and are excluded from the effort matrices for that reason, not
for lack of capability. The earlier glm-4.7-flash (local) and glm-5.1 / 5.2
(cloud) do obey it.

### Local cost

Ollama cloud tags (`pi/<tag>:cloud`, `pi/<tag>-cloud`) proxy through the local
daemon but bill the account's credits at the "Cost /1M tokens" rate on each
tag's ollama.com page. `pricing.MODELS` carries those rates, `cost_usd` uses
them, and `--max-usd` caps them; a cloud tag with no rate on file is refused
rather than priced at $0. pi reports no usage for a call it kills at the
deadline, so a flatlined cloud arm's cost is a floor.

`cost_usd` is $0 for a local arm, which is not the same as free. Local runs
sample host power and report `energy_wh` / `energy_usd`, marginal over an idle
baseline. `uv run python -m tetris_agent.power check` shows which backend the
host offers; an unmeasured run reports `n/a`, never `0.0`.

`tetris-bench` preflights every `pi/` arm (Ollama reachable, model pulled,
model fits the box) and refuses in seconds rather than OOM-ing in silence.
Run one local model at a time.

### Learn from your own games

`tetris-play` records every piece you place to `runs/<id>/` as the same
spawn → decision → locked events a model arm emits. `--exemplars` mines those
runs, reconstructs each board you faced, verifies the reconstruction against
the recorded features, and injects the best decisions into the system prompt.
Exemplar arms are labeled `+ex` in results.

```bash
uv run tetris-bench --models pi/gemma4 --harnesses routed --exemplars
```

## Self-healing

The heuristic control arm is tuned by the system. Its playing strength is a
**genome** of evaluation weights, adjusted through three loops:

```
                 ┌────────────────────────────────────────────────┐
                 │                                                ▼
  tetris-agent ──┤ fitness.json ──► healer: rule table ──► parameter race
       ▲         │                      │                       │
       │         │                      │ tuning exhausted      │ winner beats
  data/genome.json ◄────────────────────┼───────────────────────┘ control by 5%
       ▲         │                      ▼
       │         │            data/discovery_queue.json
       │         │                      │
       │         │                      ▼
       └─────────┘        tetris-discovery: coding-agent worktree
                          → tests + lint + fitness gates → PR
```

1. **In-run recovery** (`controller.py`): every rotate/shift press is verified
   against emulator memory and sprites. Unverified presses count as
   misexecutions, and a piece that drifts from its plan surfaces as a stuck
   event.
2. **Between-run healing** (`healer.py`): each run writes a fitness JSON. A
   rule table maps bad fitness to the genome parameters it implicates
   (`early-topout`, `hole-thrash`, `misexecution`), then a deterministic race
   of jittered variants against the current genome decides, with a 5% margin,
   whether to adopt a change into `data/genome.json`. 6h cooldown per rule.
3. **Capability growth** (`discovery.py`): a rule that re-fires after an
   accepted fix, or two consecutive rejected races, escalates to the discovery
   queue. `tetris-discovery run` hands the evidence bundle to a coding-agent
   session in an isolated git worktree; the proposal only becomes a PR if it
   passes pytest, ruff, and a measured fitness race against the baseline.

```bash
uv run tetris-agent --max-pieces 200          # play; self-heals at session end
uv run tetris-healer check data/last_fitness.json
uv run tetris-discovery run --dry-run         # print the escalation prompt
```

`--no-self-heal`, `--no-record`, and `--no-telemetry` turn off the healing
chain, `runs/` recording, and JSONL telemetry. Healer races derive their seeds
from a hash of the fitness file path, so every race is reproducible.

## Benchmarks

Race is `score + 5 × pieces survived`. The baselines on the live screen are
heuristic 490, random 140, no-input 55; race 50–55 means no decision ever
arrived before the piece locked.

### Reasoning level × model, live gravity (2026-09-03)

Thirteen Ollama cloud models at `off`, `low` and `medium`, effort pinned, seed
1, 30 pieces, `features`. Eleven of them play at exactly one level — `off` —
and top out at any thinking at all; kimi-k3 and gpt-oss:120b also play at
`low`, the only two served fast enough to afford a few hundred tokens inside
the ~15 s fall. The cutoff is a latency: under ~3 s per decision every arm
placed all 30 pieces, over ~9 s every arm topped out.

| arm | race | lines | tok/decision | s/decision | tok/s | $/M in / out | run cost |
|---|---|---|---|---|---|---|---|
| nemotron-3-super / off | **730** | 10 | 28 | 2.4 | 11.7 | $0.015 / $0.60 | $0.001 |
| kimi-k3 / low | 590 | 10 | 268 | 4.5 | 60.2 | $3.00 / $15.00 | $0.226 |
| gpt-oss:120b / low | 570 | 10 | 107 | 1.4 | 76.6 | $0.15 / $0.60 | $0.006 |
| gpt-oss:20b / low | 530 | 9 | 134 | 3.0 | 45.3 | $0.07 / $0.30 | $0.003 |
| deepseek-v4-pro / off | 530 | 9 | 28 | 1.4 | 20.4 | $1.32 / $3.96 | $0.067 |
| gemma4 31B / off | 530 | 9 | 30 | 1.3 | 22.6 | $0.14 / $0.40 | $0.007 |
| deepseek-v4-flash / off | 510 | 9 | 31 | 1.2 | 25.3 | $0.44 / $1.32 | $0.022 |
| same models at `low` / `medium` (except kimi-k3, gpt-oss) | 50–80 | 0 | 460–2,140 | 13–72 | — | — | ≤ $0.03 |

The full 40-arm table with every model's curve is in
[`benchmarks/2026-09-03-live-reasoning-matrix.md`](benchmarks/2026-09-03-live-reasoning-matrix.md).
The whole matrix cost $0.98 in credits; kimi-k3 was over half of it and did
not buy the top score. gpt-oss:20b at `low` — 530, 3 s per decision — is the
same 21B that runs on the iGPU, which makes it the local row to chase.

### The local roster on the same matrix (2026-09-03)

Same screen, the models that fit the iGPU, power sampled, one resident at a
time. **gemma4:26b (26B-A4B, 19 GB) with thinking off scores 530** — nine
lines, zero holes, 3 s per decision, ~50 W — which is the cloud 530 cluster
and the original reference. Local parity on this screen, three days after the
best local row was 220. The curve shape is the cloud's: six of seven models
play only at `off`, gpt-oss:20b only at `low`.

| arm | race | lines | avg holes | tok/decision | s/decision | tok/s | mean W | Wh |
|---|---|---|---|---|---|---|---|---|
| gemma4:26b / off | **530** | 9 | 0.00 | 33 | 3.0 | 10.7 | 51 | ~1.5* |
| gemma4 e4b / off | 330 | 4 | 10.20 | 34 | 2.2 | 15.8 | 46 | 0.88 |
| laguna-xs / off | 310 | 4 | 5.77 | 26 | 2.4 | 10.8 | 51 | 1.23 |
| glm-4.7-flash / off | 230 | 2 | 6.50 | 30 | 2.5 | 12.0 | 53 | 0.88 |
| gpt-oss:20b / low | 225 | 2 | 0.48 | 107 | 3.6 | 29.6 | 63 | 1.40 |
| nemotron-3.5-lightning / off | 215 | 2 | 14.89 | 33 | 2.8 | 11.8 | 56 | 0.81 |
| any model at `low`/`medium` (gpt-oss at default/`medium`); gemma4-31b at `off` | 55 | 0 | ~13 | 0–1,500 | 18–72 | — | 74–88 | 1.1–1.7 |

Full table with peak watts and idle baselines in
[`benchmarks/2026-09-03-local-reasoning-matrix.md`](benchmarks/2026-09-03-local-reasoning-matrix.md).
(*The 26b `off` baseline read high, so its marginal Wh is an artefact; its
mean draw is the figure.) gpt-oss:20b at `low` is 225 locally against 530 for
its cloud twin at the same latency — the first question a per-placement
quality measure needs to answer.

### Harness × model, bounded pause (2026-09-03, superseded screen)

Earlier the same day, at Ollama's default thinking and with the game frozen
per piece (`--decision-deadline 15`):

| model | legal | features | routed |
|---|---|---|---|
| gpt-oss:120b-cloud (baseline) | 50 | **530** | 190 |
| gpt-oss:20b-cloud | 55 | **170** | 80 |
| gpt-oss:20b (local, 13 GB) | 65 | 170 | **175** |
| gemma4 (local, 9.6 GB) | — | 50 | **220** |
| qwen3.6-27b-32k, gemma4-31b-32k, laguna-xs-32k (local) | — | 50 | 50–70 |

Two findings so far. `routed` cuts output tokens per decision for every model,
but it cost the cloud model most of its play: the `MOUND` technique asks the
model to avoid holes without showing a holes number, which `routed-v2` will
fix. For the smallest local model the shorter prompt is the difference between
never answering inside 15 s and being the local leader; the dense 27–31B
models flatline either way. The next lever is the prompt gemma4 sees, not a
larger model.

The same arm can move ~100 race between sessions on identical settings, so
treat ±100 at one seed as noise. Write-ups, one dated file per run day:

- [`benchmarks/2026-09-03-local-reasoning-matrix.md`](benchmarks/2026-09-03-local-reasoning-matrix.md) — the local roster × {off, low, medium} on live gravity, with power; gemma4:26b `off` 530
- [`benchmarks/2026-09-03-live-reasoning-matrix.md`](benchmarks/2026-09-03-live-reasoning-matrix.md) — thirteen cloud models × {off, low, medium} on live gravity, with rates and tok/s
- [`benchmarks/2026-09-03-cloud-thinking.md`](benchmarks/2026-09-03-cloud-thinking.md) — the effort curve on gpt-oss:120b and the roster with thinking off, bounded pause
- [`benchmarks/2026-09-03-cloud-roster.md`](benchmarks/2026-09-03-cloud-roster.md) — every cloud tag at Ollama's default thinking, bounded pause
- [`benchmarks/2026-09-03-situation-router.md`](benchmarks/2026-09-03-situation-router.md) — `routed` vs `legal` vs `features` per model
- [`benchmarks/2026-09-03-local-roster-routed.md`](benchmarks/2026-09-03-local-roster-routed.md) — the local roster under `routed`
- [`benchmarks/2026-09-03-situation-histogram.md`](benchmarks/2026-09-03-situation-histogram.md) — class split of the situation classifier
- [`benchmarks/2026-08-22-p15-deadline-screen.md`](benchmarks/2026-08-22-p15-deadline-screen.md) — every local model under a 15 s deadline

The `heuristic` arm (the one-piece search in `policy.py`) runs as a control,
but it is an exhaustive solver, not a player. Read model arms against human
play and each other, not against it.

## VM demo

`scripts/vm-demo.sh` runs a pi arm inside a Lima VM: the agent plays in the
VM, frames stream to the viewer on the host, and every inference call exits the
VM through a tapes capture proxy to the host's Ollama. Default arm is
`pi/gemma3`. Needs Docker, `tapes`, `limactl`, and Ollama; the script header
has the topology. Profile:
[`benchmarks/2026-08-25-vm-demo-gemma3.md`](benchmarks/2026-08-25-vm-demo-gemma3.md).

## Session capture (tapes)

LLM traffic is captured locally by [tapes](https://docs.tapes.dev):

```bash
tapes local up            # once: Postgres in Docker
./scripts/tapes-up.sh     # proxies + API + derive worker; Ctrl-C tears down
export TETRIS_TAPES_OLLAMA_URL=http://127.0.0.1:8092/v1
```

The bundled pi extension (`src/tetris_agent/pi_extension/tapes-proxy.ts`)
reroutes pi's Ollama provider through the proxy when the variable is set. Read
back with `tapesctl sessions list --tapes-url http://localhost:8081`.

## Telemetry

Events (`tetris.game.v1`) are appended to
`data/telemetry/YYYY-MM-DD/events-<session>.jsonl`, broker-free. Recorded runs
land in `runs/<id>/` with `events.jsonl`, `summary.json`, and PNG frames at each
spawn/lock, so the viewer's REPLAY tab plays them anywhere without the emulator.

## Development

```bash
uv run pytest            # pure tests (fast, no emulator)
uv run pytest -m ""      # everything, including ROM-marked emulator tests
uv run ruff check .
```

`pieces.py` / `board.py` / `policy.py` are pure and unit-tested; `emulator.py`
is the only module importing pyboy; `state.py` reads the board from the
background tilemap minus falling-piece sprites; `agent.py` runs the loop;
`evolve.py` runs in-process races.
