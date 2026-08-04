# tetris-agent

A self-healing Game Boy Tetris agent — the pokemon-kafka paradigms rebuilt clean.

The agent plays Tetris (World, Rev 1) headless through PyBoy, and its playing
strength is a **genome** of evaluation weights that the system tunes itself
through three healing loops:

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
uv run tetris-agent --live --max-pieces 60    # streams into the viewer at 1x speed
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

## Session capture

Agent-session traces are captured by **paperd** (the Paper gateway), not by
anything in this repo. Gameplay itself makes no LLM calls — the only LLM
sessions here are discovery's `claude -p` proposals, and that subprocess
inherits the environment, so running from a paper-gatewayed shell routes its
traces through paperd automatically. Game telemetry is separate: JSONL events
under `data/telemetry/` and recorded runs under `runs/`.

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
