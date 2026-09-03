# tetris-agent

<img width="934" height="572" alt="Screenshot 2026-08-03 at 5 54 25 PM" src="https://github.com/user-attachments/assets/da0b47fb-5cc3-4c3d-b18b-4134dbb87d4b" />


A benchmark harness for **model × harness × effort** on Game Boy Tetris, with a
self-healing heuristic as the control arm.

The question it answers: given a model, a way of presenting the game to it, and
a reasoning-effort setting, how well does it actually play — and what does that
cost? Tetris is a good testbed because the piece sequence is deterministic
(`timer_div` pins the RNG), so every arm plays the *same* game and differences
in score are differences in play rather than luck.

## The three axes

| Axis | Values |
|---|---|
| **Model** | `claude-opus-5`, `claude-sonnet-5`, `claude-haiku-4-5`, `claude-fable-5`, plus `pi/<ollama-model>` open-weight arms (see below) |
| **Effort** | `low`, `medium`, `high`, `xhigh`, `max` — Haiku 4.5 rejects the parameter, so its arms run effort-free |
| **Harness** | `board`, `legal`, `features`, `chat`, `routed` |

The harness is the interesting axis — it is how much of the reasoning is done
*for* the model before it decides:

- **`board`** — ASCII board, current piece, next piece. The model derives
  legality and consequences itself.
- **`legal`** — plus the enumerated legal `(rotation, col)` placements.
- **`features`** — plus the resulting holes, height, bumpiness, and line clears
  for each placement. The model gets the heuristic's inputs but must weigh them.
- **`chat`** — like `board`, but conversation history carries across pieces, so
  the model can pursue a plan (and pays for the growing context).
- **`routed`** — `legal`, plus the board's *situation* — `TOPPING_OUT`,
  `TETRIS_READY`, `HOLE_RISK`, `MOUND`, or `FLAT`, in that priority
  (`situation.py`) — with the one technique it calls for and only the
  consequence numbers that technique turns on. The general play guidance in the
  system prompt is replaced by "follow the named technique"; the thresholds are
  genome fields. Run `uv run tetris-situations generate` then `uv run tetris-situations histogram`
  to see the class split over a free corpus (`benchmarks/2026-09-03-situation-histogram.md`).

The reference that matters is **human play**: `tetris-play` records your own
games as traces (see below), and the `no-input` / `random` arms bound the floor
and chance. The **`heuristic`** arm (the weighted one-piece search in
`policy.py`) still runs as a control, but it is an exhaustive solver, not a
player — don't read model arms against it.

## The VM demo

`scripts/vm-demo.sh` runs an open-weight pi arm inside a Lima VM (standing in
for an exe.dev-style shell computer): the agent plays in the VM, its frames
stream to the GRAMBOY viewer on the host, and every inference call exits the
VM through a tapes capture proxy to the host's Ollama — no hosted API in the
loop, and the whole run lands as one queryable session (project `tetris-vm`,
`--harness chat` so the conversation chain holds it together). Default arm is
`pi/gemma3`: it decides in ~2-5s, inside live gravity's ~15s window, where
thinking models (`pi/qwen3:8b`) blow the per-piece deadline and lose to
gravity. Expect a scrappy game — that gap between fast-but-weak and
smart-but-late IS the benchmark's finding, and `--exemplars` (your recorded
human games in the system prompt) is the lever the repo offers for closing
it. The script's header comment has the topology; it needs Docker, `tapes`,
`limactl`, and Ollama.

### Shipping this to an exe.dev-style setup

The Lima layer is the only local stand-in; everything else maps onto real
shell computers directly. What changes per piece:

- **The VM** becomes an exe.dev VM. Step 3's provisioning (uv + repo + node +
  pi + `~/.pi/agent/models.json`) is a plain-English prompt to Shelley or a
  first-boot script — nothing in it is Lima-specific.
- **The capture proxy** can't stay on your laptop (a cloud VM can't reach
  `host.lima.internal`). It moves to one of two places, both already
  supported: as a sidecar on each VM (`tapes serve proxy` is a single binary;
  pi's provider `baseUrl` points at `localhost` instead of the host gateway),
  or on one shared gateway VM that every agent VM's provider config points
  at — the same boundary where exe already puts its LLM Gateway.
- **Inference** stays wherever the weights run fast. exe VMs are 2-vCPU
  CPU-only boxes, so the proxy's `--upstream` points at a GPU box you run, a
  hosted open-weight endpoint, or exe's own gateway — the capture shape is
  identical in all three.
- **Storage centralizes.** Every proxy takes `--postgres <dsn>`: point the
  whole fleet at one Postgres and every VM's runs land in the same sessions
  store, with a single `tapes serve api` + `derive-worker` next to the
  database (both are CPU-light — another exe VM is fine). Locally the demo
  already runs this exact topology in miniature: the proxy, API, and worker
  all share `tapes-postgres-1`; a fleet just moves the DSN off-box.
- **Downstream analytics** is the kafka-cassette's job, not another database:
  it drains derived span events from the central store to Kafka/Confluent
  Cloud, which is where dashboards, anomaly detection, and stream processing
  belong — the tapes Postgres stays the system of record.

## Play it yourself, then let the models learn from you

```bash
uv run tetris-viewer                     # the GRAMBOY viewer, http://127.0.0.1:8000
uv run tetris-play --seed 0 --max-pieces 50
```

`tetris-play` runs the emulator headless and streams it into the viewer's LIVE
tab; your keyboard drives it from the browser (`space` starts the session and
hard-drops in play, arrows move/soft-drop, `up` or `a`/`s` rotate). The game idles at
the title screen until you press space, so nothing is recorded before you're
actually playing. Every piece you place is recorded to `runs/<id>/` as
the same spawn → decision → locked events a model arm emits, tagged
`policy: "human"`. (`tetris-manual` still does the same in a native SDL window.)

Those traces then feed the model arms. `--exemplars` mines your recorded runs
— reconstructing each board you faced and *verifying* the reconstruction
against the recorded per-piece features, so a misread never becomes a training
pair — and injects the best decisions into the (cached) system prompt:

```bash
uv run tetris-agent --policy model --model claude-sonnet-5 --harness features --exemplars --live
uv run tetris-bench --models claude-sonnet-5 --harnesses features --exemplars
```

Exemplar arms are labeled `+ex` in results, so with-and-without stays an
honest comparison.

```bash
# Project the spend without calling anything
uv run tetris-bench --models claude-opus-5 claude-haiku-4-5 \
                    --harnesses board features --efforts medium \
                    --max-pieces 30 --estimate

# Run it, with a hard budget cap
uv run tetris-bench --models claude-opus-5 --harnesses features \
                    --efforts low high --max-usd 2.00

# Watch one model play in the browser
uv run tetris-agent --policy model --model claude-opus-5 --harness features --live
```

Cost is a first-class result column, not an afterthought: the static system
prompt is cached, every run reports tokens, dollars, and latency, `--estimate`
projects spend before you commit, and `--max-usd` aborts the matrix rather than
letting you find the bill afterwards. One API call per piece adds up fast.

Results land in `data/benchmarks/` and print as a ranked table. Model arms also
report `illegal` — placements the model proposed that don't fit — which is a
useful capability signal on its own, separate from score.

`data/` is gitignored, so those JSON files stay on the machine that made them.
Results worth keeping go in [`benchmarks/`](benchmarks/) as a dated markdown
writeup — append-only, one file per run day, every row traceable to a run id.

Model arms play **live** by default: the game does not pause while the model
thinks. The decision runs on a worker thread while gravity keeps pulling the
piece down in real time (level 0 ≈ 15s from spawn to floor), so latency is part
of the game — the same clock a human plays against. A decision that arrives
after its piece already locked is discarded and counted in the `late` column,
and each `placement_decision` event carries its per-piece `latency_ms` and
`tokens`. Live arms are labeled `+live` and run in real time (50 pieces ≈
12–15 minutes per arm); `--paused` restores the freeze-while-thinking mode for
A/B against historical rows, and `--level` raises the gravity.

Latency decomposes as prompt processing plus tokens generated over the model's
**tokens per second** — reported per arm in the `tok_s` column (with
`tokens_per_decision` and `thinking_tokens` in the results JSON), so a
flatlined arm is attributable: low `tok_s` means the hardware or API is slow,
high tokens-per-decision means the model overspent thinking. Live arms also
budget themselves to the clock: each decision is told how many seconds remain
before its piece locks, and an adaptive controller steps the thinking effort
down one tier at a time (`--thinking off` is the floor for pi arms) while the
observed latency can't beat the deadline, back up when there's room —
`downshifts` in the results JSON counts the interventions. The configured
effort still names the arm, so rows stay comparable. An arm whose floor —
prompt processing alone, zero thinking — exceeds the clock will still
flatline, and the numbers now say exactly why.

### Bounded pause: `--decision-deadline`

Between full-live and the uncapped legacy `--paused` sits the screening mode:

```bash
uv run tetris-bench --models pi/gpt-oss:20b --harnesses features \
                    --decision-deadline 15 --watch --max-pieces 30
```

The game still freezes exactly once per piece while the model thinks (so
execution is never racing gravity), but a decision slower than the deadline is
discarded — the subprocess is killed at the mark for `pi/` arms — and the piece
falls untouched, counted in the `timeouts` column. Arms are labeled
`+p<seconds>`, and the summary adds `pct_le_10s` / `pct_le_15s` (share of
decisions inside 10s/15s, from the per-decision latencies in the results JSON),
so one run answers "which models can decide a placement in 10–15 seconds?"
without also making them race the fall animation.

`--watch` streams any matrix — live, paused, or bounded — to the viewer's LIVE
tab (`uv run tetris-viewer`, then http://127.0.0.1:8000): the board freezes
while the model thinks, a **NOW PLAYING** banner names the model, harness,
effort, mode, and seed, and per-piece status shows THINKING… / placed in Ns /
TOO SLOW — dropped. Watched paused arms run in real time (that is the point);
unwatched paused arms stay uncapped. A tab opened mid-arm still gets the
banner — the viewer replays the last session start to late joiners.

## Open-weight arms (pi + Ollama)

Model ids prefixed `pi/` run through the [pi coding agent](https://github.com/earendil-works/pi-mono)
(tested against v0.80.10) driving a local Ollama model — one `pi -p --mode json`
subprocess per piece:

```bash
uv run tetris-bench --models claude-haiku-4-5 pi/gpt-oss:20b \
                    --harnesses features --efforts medium --max-pieces 10
```

Registered ids: `pi/gpt-oss:20b`, `pi/glm-4.7-flash`, `pi/qwen3.6:27b`,
`pi/nemotron-3.5-lightning-32k` (thinking-capable — effort maps onto pi's `--thinking` level verbatim) and
`pi/qwen3:8b`, `pi/gemma3` (effort-free, they collapse like Haiku). Any other
`pi/<model>` id is accepted as an unbilled, effort-free arm, so anything in
`ollama list` works. pi arms bill $0 to an API but are not free — they draw
watts here instead, which the `energy_wh` / `energy_usd` columns price (see
below) — and they cost time: a 20B-class decision can take tens of seconds, so
keep `--max-pieces` modest in pi-heavy matrices.

### What a local arm actually costs

`cost_usd` is an API bill, so a local arm reports `$0.0000` there and that used
to be the whole story — which read as "free" when it actually means "billed
somewhere else". Local runs now sample host power and report it:

```
uv run python -m tetris_agent.power check      # which backend this host offers
tetris-bench --models pi/gemma3 ...            # energy_wh / energy_usd columns
```

The figure is **marginal over idle**: a short baseline is sampled before the run
and subtracted, so it answers "what did this run cost on top of leaving the
machine on" rather than counting the browser you left open. Watt-hours are
priced at `DEFAULT_KWH_PRICE` in `pricing.py`, echoed in the summary line so the
rate is never invisible.

Backends, in the order they are tried: macOS `powermetrics` (needs root — grant
passwordless sudo for `/usr/bin/powermetrics`, or the column reads `n/a` with the
reason), Linux `nvidia-smi` + amdgpu hwmon + Intel RAPL for the exe.dev-style VMs
below. With neither, energy is `n/a` rather than `0.0` — an unmeasured run is
unknown, not free. `--no-power` opts out.

In `scripts/vm-demo.sh` the sampler runs on the **host**, not in the guest: the
agent plays inside the VM but inference happens on the host's Ollama, so that is
where the watts are.

### What these arms need to run

Open-weight arms shift cost from a bill to a machine. Budget for all three
before starting a matrix:

| | measured |
|---|---|
| Resident memory | `qwen3:8b` holds **~10 GB** while loaded |
| Wall clock | **~40 s per decision** on an idle machine — 50 pieces is over half an hour for one seed |
| Contention | sharing the machine with another benchmark roughly doubles that |

The memory figure is the one that bites. On a box already under pressure the
run does not fail loudly — it gets OOM-killed after ten minutes having printed
nothing, because progress is only reported once an arm finishes. A dedicated
sandbox is the right way to run these, but size it for the model: a 4 GB
instance cannot host an 8B model, whatever else it has going for it.

`tetris-bench` preflights any `pi/` arm before it starts — Ollama reachable,
model actually pulled, model small enough for the box — and refuses in seconds
rather than failing in silence. `--skip-preflight` overrides it.

The delta is deliberate: pi is part of the harness under measurement. It has no
structured-output mode, so the placement JSON is prompt-instructed and parsed
leniently; `parse_failures` and `illegal` in the results are capability signals
for the open-weight arms just like they are for the Claude ones.

## Self-healing

The heuristic control arm is itself tuned by the system. Its playing strength is
a **genome** of evaluation weights, adjusted through three healing loops:

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
       └─────────┘        tetris-discovery: claude worktree
                          → tests + lint + fitness gates → PR
```

1. **In-run recovery** (`controller.py`): every rotate/shift press is verified
   against emulator memory and sprites; unverified presses count as
   misexecutions, and a piece that drifts from its plan surfaces as a stuck
   event instead of silently diverging.
2. **Between-run healing** (`healer.py`): each run writes a fitness JSON. A
   rule table maps bad fitness to the genome parameters it implicates
   (`early-topout`, `hole-thrash`, `misexecution`), then a deterministic race
   of jittered variants against the current genome decides — with a 5% margin
   — whether to adopt a change into `data/genome.json`. 6h cooldown per rule.
3. **Capability growth** (`discovery.py`): a rule that re-fires after an
   accepted fix, or two consecutive rejected races, escalates to the discovery
   queue. `tetris-discovery run` hands the evidence bundle to a Claude Code
   session in an isolated git worktree; the proposal only becomes a PR if it
   passes pytest, ruff, and a measured fitness race against the baseline.

## Quickstart

```bash
uv sync
# Put the ROM at rom/tetris.gb (gitignored).

uv run tetris-agent --max-pieces 200          # play; self-heals at session end
uv run tetris-agent --timer-div 0             # deterministic piece sequence
uv run tetris-healer check data/last_fitness.json
uv run tetris-discovery run --dry-run         # print the escalation prompt
```

## Demo: the GRAMBOY viewer

A Game Boy in your browser — live view and session replay:

```bash
uv run tetris-viewer                          # http://127.0.0.1:8000
uv run tetris-agent --live --max-pieces 60    # no-pause play, streamed into the viewer
```

**LIVE** shows the running agent on the LCD with telemetry (score/lines/holes/
misexec) and the event bus. **REPLAY** lists recorded runs as cartridges —
click one to insert it, then use the transport (or START/A/B on the shell) to
play, scrub, and change speed. Every recorded run captures PNG frames at each
piece spawn/lock into `runs/<id>/frames/`, so demos replay anywhere without
the emulator. Streaming is best-effort: if the viewer isn't running, `--live`
just plays at real-time speed and drops frames silently.

`--no-self-heal`, `--no-record`, and `--no-telemetry` turn off the healing
chain, `runs/` recording, and JSONL telemetry respectively.

## Determinism

`--timer-div <0-255>` pins the game's piece RNG, so a seed replays the same
piece sequence every time. Healer races derive their seeds from a hash of the
fitness file path, which makes every race reproducible.

## Telemetry

Events (`tetris.game.v1`) are envelopes of
`{schema, event_type, turn, occurred_at, data}` appended to
`data/telemetry/YYYY-MM-DD/events-<session>.jsonl`. The agent is deliberately
broker-free: a bridge process (see pokemon-kafka's `docker/game-event-bridge`)
can tail these files into Kafka without the agent knowing. Recorded runs land
in `runs/<id>/` with `events.jsonl` and `summary.json` (the healer's input).

## Session capture (tapes)

Benchmark LLM traffic is captured locally by **[tapes](https://docs.tapes.dev)**,
keeping benchmark noise out of the Paper org. One script brings up the whole
capture stack — two single-provider proxies, the read API (:8081), and the
derive worker (the proxies sit on :8091/:8092 because pi's `mlx` provider
already owns :8080):

```bash
tapes local up            # once: Postgres (+ Ollama embeddings) in Docker
./scripts/tapes-up.sh     # proxies + API + derive worker; Ctrl-C tears down
```

Then, in the shell that runs the benchmark:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:8091          # Claude arms
export TETRIS_TAPES_OLLAMA_URL=http://127.0.0.1:8092/v1  # pi arms
```

The Anthropic SDK honors `ANTHROPIC_BASE_URL` natively; pi arms pick up
`TETRIS_TAPES_OLLAMA_URL` through the bundled extension
(`src/tetris_agent/pi_extension/tapes-proxy.ts`), which reroutes pi's ollama
provider through the capture proxy. Unset either var and that side simply runs
uncaptured. Read back what was captured with
`tapesctl sessions list --tapes-url http://localhost:8081`.

When sessions *should* land in the Paper org instead — real work, not
benchmark noise — run from a paper-gatewayed shell (`paper start`) without the
exports above. Game telemetry is separate either way: JSONL events under
`data/telemetry/` and recorded runs under `runs/`.

## How this differs from pokemon-kafka

- One installable package (`src/tetris_agent/`), not 28 flat scripts.
- Components call each other in-process; the only subprocesses are
  discovery → `claude`/`git`/`gh`.
- The genome is a real JSON file with history, not a regex-parsed markdown
  comment block.
- Races are natively deterministic via the emulator's `timer_div` piece RNG.
- Game state comes from PyBoy's built-in Tetris wrapper + sprites, not a
  hand-rolled memory map. (Caveat learned the hard way: the wrapper's
  `set_tetromino` ROM patch is a no-op on World Rev 1, and the rotation bits
  at 0xC203 count counter-clockwise.)

## Development

```bash
uv run pytest            # pure tests (fast, no emulator)
uv run pytest -m ""      # everything, including ROM-marked emulator tests
uv run ruff check .
```

Layout: `pieces.py` / `board.py` / `policy.py` are pure and fully unit-tested;
`emulator.py` is the only module importing pyboy; `state.py` reads the board
from the background tilemap minus falling-piece sprites; `agent.py` runs the
loop; `evolve.py` runs in-process races (they can't recurse into self-healing
by construction).
