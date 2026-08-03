# Tetris Self-Healing Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A clean sibling repo to pokemon-kafka: an autonomous Game Boy Tetris agent whose evaluation weights are a genome tuned by a fitness-driven healer race, with LLM escalation for capability gaps.

**Architecture:** Three loops ported from pokemon-kafka — (1) in-run recovery: the controller verifies every button press against emulator memory/sprites and replans on drift; (2) between-run healing: the agent emits a fitness JSON, a rule table maps bad fitness to implicated genome parameters, and a deterministic parameter race (seeded `timer_div` piece sequences) decides whether to adopt a variant genome; (3) capability growth: repeated rule firings escalate to a Claude Code worktree that must pass test+lint+fitness gates before opening a PR. Unlike pokemon-kafka, everything is one installable package with in-process calls (no subprocess chains except discovery→claude), the genome lives in a real JSON file (not regex-parsed markdown), and the agent stays broker-free: telemetry is date-partitioned JSONL that a Kafka bridge can tail later.

**Tech Stack:** Python ≥3.11, uv, PyBoy (built-in `GameWrapperTetris`), numpy, pytest, ruff.

## Global Constraints

- Package name `tetris-agent`, import name `tetris_agent`, src layout (`src/tetris_agent/`), absolute imports only.
- `requires-python = ">=3.11"`. Runtime deps: `pyboy`, `numpy` only. Dev: `pytest`, `pytest-cov`, `ruff`.
- ROM at `rom/tetris.gb` (Tetris (World) (Rev 1)). ROM is gitignored; never commit it.
- Tests that boot PyBoy are marked `@pytest.mark.rom`; default `pytest` run excludes them (`-m "not rom"` via addopts); CI-style full run is `pytest -m ""`.
- Ruff: `E,F,I,W`, line-length 120. No coverage `fail_under` mandate.
- Event schema string is exactly `"tetris.game.v1"`. Envelope keys: `schema, event_type, turn, occurred_at, data`.
- Mutable state files live under `data/` (gitignored): `data/genome.json`, `data/healer_state.json`, `data/discovery_queue.json`, `data/telemetry/`. Runs under `runs/` (gitignored).
- Healer `check` never raises and the self-heal chain is opt-out (`--no-self-heal`); race children never self-heal (guaranteed structurally: races call `evolve.run_agent`, which never invokes the healer).
- Board is 18 rows × 10 cols. Tile 47 = blank. Falling-piece sprites have pixel x < 96; cell mapping `row = y // 8`, `col = x // 8 - 2`. Current piece addr `0xC203`, next `0xC213`; `& 0b11111100` → type (L=0, J=4, I=8, O=12, Z=16, S=20, T=24), `& 0b11` → rotation.
- Commit after every task with a conventional-commit message.

## Empirical facts (verified by probe on 2026-08-03, do not re-derive)

- `pyboy.game_wrapper` for this ROM is `GameWrapperTetris` with `start_game(timer_div=)`, `reset_game(timer_div=)`, `game_area()` (18×10 memoryview), `score`, `level`, `lines`, `next_tetromino()`, `game_over()`, `set_tetromino(shape)`.
- `game_area()` includes the falling piece; the settled board is `game_area != 47` minus the falling-piece sprite cells.
- `pyboy.button("left")` then `pyboy.tick(5, True)` reliably shifts the piece one column (verified: two lefts moved leftmost col 3→1... i.e. one col per press).
- `timer_div` fixes the piece RNG → identical piece sequences across runs.

## File structure

```
src/tetris_agent/
  __init__.py
  pieces.py       # tetromino silhouettes per rotation (pure)
  board.py        # settled-board ops: drop, place, clear, features (pure)
  policy.py       # Genome weights + placement search (pure)
  emulator.py     # PyBoy wrapper (only module importing pyboy)
  state.py        # GameState read from emulator
  controller.py   # execute a Placement with press verification
  events.py       # tetris.game.v1 envelope + builders + Collector
  publisher.py    # Publisher protocol, JSONLPublisher, NoopPublisher
  fitness.py      # FitnessTracker -> fitness dict
  recorder.py     # runs/<id>/ events.jsonl + summary.json
  agent.py        # TetrisAgent loop
  genome.py       # data/genome.json load/append
  evolve.py       # run_agent, sample_variants, run_race
  healer.py       # RULES, cooldown, check()
  discovery.py    # escalation queue -> claude worktree -> gates -> PR
  cli.py          # tetris-agent / tetris-healer / tetris-discovery entrypoints
tests/            # test_<module>.py mirrors; conftest.py has rom fixtures
rom/tetris.gb     # gitignored
```

---

### Task 1: Repo scaffold

**Files:** Create `pyproject.toml`, `.gitignore`, `README.md`, `AGENTS.md`, `src/tetris_agent/__init__.py`, `tests/conftest.py`, `tests/test_scaffold.py`.

**Produces:** installable `tetris_agent` package; `uv run pytest` green; `rom` marker configured; `ROM_PATH` fixture.

- [ ] Step 1: `git init`, write `.gitignore` (`.venv/`, `__pycache__/`, `data/`, `runs/`, `rom/*.gb`, `*.state`, `.coverage`, `.ruff_cache/`, `.pytest_cache/`, `dist/`)
- [ ] Step 2: Write `pyproject.toml`:

```toml
[project]
name = "tetris-agent"
version = "0.1.0"
description = "Self-healing Game Boy Tetris agent: fitness-driven genome tuning with LLM escalation"
requires-python = ">=3.11"
dependencies = ["pyboy>=2.4", "numpy>=1.26"]

[project.scripts]
tetris-agent = "tetris_agent.cli:agent_main"
tetris-healer = "tetris_agent.cli:healer_main"
tetris-discovery = "tetris_agent.cli:discovery_main"

[dependency-groups]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.6"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tetris_agent"]

[tool.pytest.ini_options]
addopts = "-m 'not rom'"
markers = ["rom: boots PyBoy with the real ROM (slow)"]

[tool.ruff]
line-length = 120
[tool.ruff.lint]
select = ["E", "F", "I", "W"]
```

- [ ] Step 3: Write `tests/conftest.py`:

```python
from pathlib import Path

import pytest

ROM_PATH = Path(__file__).resolve().parent.parent / "rom" / "tetris.gb"


@pytest.fixture(scope="session")
def rom_path() -> Path:
    if not ROM_PATH.exists():
        pytest.skip("rom/tetris.gb not present")
    return ROM_PATH
```

- [ ] Step 4: Write `tests/test_scaffold.py` asserting `import tetris_agent` works and `tetris_agent.__version__ == "0.1.0"`; `src/tetris_agent/__init__.py` sets `__version__ = "0.1.0"`.
- [ ] Step 5: `uv sync && uv run pytest && uv run ruff check .` → green. Write minimal `README.md` (what it is, `uv run tetris-agent --max-pieces 200`) and `AGENTS.md` (use uv; pure tests default, `pytest -m ""` for ROM tests). Commit `chore: scaffold tetris-agent package`.

### Task 2: pieces.py — silhouettes

**Interfaces produced:** `PIECES: dict[str, int]` (name→id: L=0,J=4,I=8,O=12,Z=16,S=20,T=24), `piece_name(byte) -> str`, `SHAPES: dict[tuple[str, int], frozenset[tuple[int, int]]]` mapping (name, rot 0-3) → normalized (row, col) cell offsets, `distinct_rotations(name) -> list[int]` (unique silhouettes only), `shape_width(cells) -> int`.

- [ ] Step 1: Failing tests in `tests/test_pieces.py`:

```python
from tetris_agent.pieces import SHAPES, distinct_rotations, piece_name, shape_width

def test_every_shape_has_four_cells_and_is_normalized():
    for cells in SHAPES.values():
        assert len(cells) == 4
        assert min(r for r, _ in cells) == 0 and min(c for _, c in cells) == 0

def test_j_spawn_matches_probe_observation():
    assert SHAPES[("J", 0)] == frozenset({(0, 0), (0, 1), (0, 2), (1, 2)})

def test_distinct_rotations():
    assert distinct_rotations("O") == [0]
    assert len(distinct_rotations("I")) == 2 and len(distinct_rotations("T")) == 4

def test_piece_name_masks_rotation_bits():
    assert piece_name(0x4) == "J" and piece_name(0x6) == "J" and piece_name(0x8) == "I"

def test_shape_width():
    assert shape_width(SHAPES[("I", 0)]) == 4
```

- [ ] Step 2: Implement: spawn shapes `L:{(0,0),(0,1),(0,2),(1,0)}, J:{(0,0),(0,1),(0,2),(1,2)}, I:{(0,0),(0,1),(0,2),(0,3)}, O:{(0,0),(0,1),(1,0),(1,1)}, Z:{(0,0),(0,1),(1,1),(1,2)}, S:{(0,1),(0,2),(1,0),(1,1)}, T:{(0,0),(0,1),(0,2),(1,1)}`; `_rotate_cw(cells)` = `(c, max_r - r)` then normalize; build `SHAPES[(name, k)]` by k applications; `distinct_rotations` returns first rotation index per unique silhouette in press order.
- [ ] Step 3: Tests pass; commit `feat: tetromino silhouette tables`.
- [ ] Step 4 (deferred validation): Task 5 adds a ROM test that uses `set_tetromino` + A presses to compare observed sprite silhouettes per rotation value against `SHAPES`; if press order is counter-clockwise, flip `_rotate_cw` there.

### Task 3: board.py — pure board ops

**Interfaces produced:** `BLANK = 47`; `settled_board(game_area, falling_cells) -> np.ndarray[bool] (18,10)`; `drop(board, shape, col) -> int | None` (landing top-row, None if column overflow/doesn't fit); `place(board, shape, col, row) -> tuple[np.ndarray, int]` (new board, lines cleared); `features(board) -> Features` dataclass with `holes, agg_height, bumpiness, max_height`.

- [ ] Step 1: Failing tests `tests/test_board.py` (build boards from ascii helper `B("....X.....", ...)`): drop an O into empty board lands at row 16; drop onto a one-cell column stacks above it; `place` clearing a full bottom row returns lines=1 and shifts rows down; `features` on a board with an overhang counts 1 hole; bumpiness of flat board is 0; `settled_board` subtracts falling cells.
- [ ] Step 2: Implement with numpy; heights per column = 18 - first occupied row; holes = blanks below column top; bumpiness = sum |h[i]-h[i+1]|.
- [ ] Step 3: Green; commit `feat: board simulation and features`.

### Task 4: policy.py — genome + placement search

**Interfaces produced:**

```python
@dataclass(frozen=True)
class Genome:
    w_lines: float = 0.760666
    w_agg_height: float = -0.510066
    w_holes: float = -0.35663
    w_bumpiness: float = -0.184483
    ticks_per_press: int = 4
    def to_params(self) -> dict[str, float]  # includes all fields
    @classmethod
    def from_params(cls, params: dict) -> "Genome"  # ignores unknown keys

@dataclass(frozen=True)
class Placement:
    rotation: int
    col: int
    score: float

def plan_placement(board: np.ndarray, piece: str, genome: Genome) -> Placement | None
```

`plan_placement` enumerates `distinct_rotations(piece)` × valid columns, simulates `drop`+`place`, scores `w_lines*lines + w_agg_height*f.agg_height + w_holes*f.holes + w_bumpiness*f.bumpiness`, returns argmax (ties → lowest landing, then centermost col). Returns None only if nothing fits (top-out imminent).

- [ ] Step 1: Failing tests: on an empty board an I piece prefers flat placement over vertical; with a nine-cell bottom row and an I piece, planner picks the placement that clears the line; genome round-trips through `to_params`/`from_params`; unknown param keys are ignored.
- [ ] Step 2: Implement; green; commit `feat: weighted placement policy with default El-Tetris genome`.

### Task 5: emulator.py + state.py

**Interfaces produced:**

```python
class Emulator:
    def __init__(self, rom_path: str | Path, headless: bool = True)
    def start(self, timer_div: int | None = None) -> None   # game_wrapper.start_game
    def tick(self, n: int = 1) -> None
    def press(self, button: str, ticks_after: int = 4) -> None  # pyboy.button + tick
    def read(self, addr: int) -> int
    def falling_sprite_cells(self) -> frozenset[tuple[int, int]]  # sprites x<96 mapped
    def game_area(self) -> np.ndarray
    @property score/level/lines/game_over
    def save_state(self, path) / load_state(self, path)
    def stop(self) -> None

@dataclass(frozen=True)
class FallingPiece:
    name: str; rotation: int; cells: frozenset[tuple[int, int]]
    @property def col(self) -> int  # min col of cells

@dataclass(frozen=True)
class GameState:
    board: np.ndarray; falling: FallingPiece | None; next_piece: str
    score: int; level: int; lines: int; game_over: bool

def read_state(emu: Emulator) -> GameState   # falling None if <4 sprite cells
CURRENT_PIECE_ADDR = 0xC203
```

- [ ] Step 1: Failing pure tests for the sprite→cell mapping math (factor as `_cells_from_sprites(list[tuple[x, y]])`) using probe data: `[(40,8),(48,8),(56,8),(56,16)]` → `{(1,3),(1,4),(1,5),(2,5)}`; preview sprites at x≥96 excluded.
- [ ] Step 2: `@pytest.mark.rom` tests: boot with `timer_div=0x00`, assert `read_state` returns a `FallingPiece` whose `cells` match one of `SHAPES[(name, rot)]` translated; assert `next_piece in "LJIOZST"`; assert board initially empty.
- [ ] Step 3: `@pytest.mark.rom` pieces-table validation: for each piece name, `set_tetromino(name)`, reset, for each of 4 A-presses record normalized sprite silhouette and compare against `SHAPES[(name, k)]`; correct `pieces.py` rotation direction/tables from failures until green.
- [ ] Step 4: Implement both modules; run `pytest -m rom` green; commit `feat: emulator wrapper and game state reader`.

### Task 6: controller.py — verified execution (in-run healing layer 1)

**Interfaces produced:**

```python
@dataclass
class ExecResult:
    locked: bool; misexec: int; lines_delta: int; replanned: bool

class Controller:
    def __init__(self, emu: Emulator, ticks_per_press: int = 4)
    def execute(self, target: Placement) -> ExecResult
```

Behavior: (a) rotate — press "a" up to 6 times until `emu.read(0xC203) & 0b11 == target.rotation`; unverified press → `misexec += 1`; (b) shift — compare `read_state().falling.col` to `target.col`, press left/right one at a time, verify col changed, re-read; give up a direction after 3 consecutive unverified presses (wall/wedge) and set `replanned=True`; (c) drop — press "down" repeatedly (ticks_after=2) until piece type byte changes, settled board gains cells, or `game_over`; record `lines_delta` from wrapper. If the piece locked mid-rotate/shift, return early with `locked=True`.

- [ ] Step 1: Pure tests with a `FakeEmulator` (scripted memory/sprite responses) covering: rotation converges in ≤ needed presses; unverified rotation counts misexec; shift stops at wall and sets replanned.
- [ ] Step 2: `@pytest.mark.rom` test: boot deterministic game, plan with default `Genome`, `execute` → `locked=True`, settled board gained exactly 4 cells, `misexec <= 2`.
- [ ] Step 3: Implement; both suites green; commit `feat: verified placement controller`.

### Task 7: events.py + publisher.py

**Interfaces produced:** `SCHEMA = "tetris.game.v1"`; builders returning envelopes: `build_session_event(phase, data)`, `build_spawn_event(turn, piece, next_piece)`, `build_decision_event(turn, placement, score)`, `build_locked_event(turn, lines_delta, features, misexec)`, `build_stuck_event(turn, streak, detail)`, `build_game_over_event(turn, fitness)`; `class EventCollector(publisher)` with `.spawn/.decision/.locked/.stuck/.game_over/.session` methods and `.turn` counter; `Publisher` Protocol `publish(event: dict) -> None`; `JSONLPublisher(root: Path, session_id: str)` writing `root/YYYY-MM-DD/events-<session_id>.jsonl` (mkdir parents, append, flush per line); `NoopPublisher`.

- [ ] Step 1: Failing tests: envelope has exactly the five global-constraint keys, `occurred_at` ISO-8601 UTC with `+00:00`; JSONLPublisher creates date dir and appends valid JSON lines; collector increments turn on spawn.
- [ ] Step 2: Implement; green; commit `feat: tetris.game.v1 events and JSONL publisher`.

### Task 8: fitness.py + recorder.py

**Interfaces produced:**

```python
class FitnessTracker:
    def on_lock(self, features: Features, misexec: int) -> None
    def on_game_over(self) -> None
    def compute(self, score: int, lines: int, level: int) -> dict
    # keys: score, lines, level, pieces_placed, avg_holes, max_stack_height,
    #       misexec_count, max_misexec_streak, topped_out

def race_score(fitness: dict) -> float  # score + 5 * pieces_placed

class RunRecorder:
    def __init__(self, runs_dir: Path, label: str = "")
    run_id: str  # YYYYMMDD-HHMMSS-<hex6>
    def record_event(self, event: dict) -> None       # buffers
    def finalize(self, fitness: dict, params: dict) -> Path  # writes runs/<id>/{events.jsonl,summary.json,meta.json}
```

`summary.json` = `{"run_id", "fitness", "params"}` — the healer's input format. misexec streak = consecutive locks with misexec>0.

- [ ] Step 1: Failing tests: avg_holes averaged over locks; streak resets on clean lock; `finalize` writes all three files with parseable JSON; `race_score` math.
- [ ] Step 2: Implement; green; commit `feat: fitness tracking and run recorder`.

### Task 9: agent.py + cli.py — the loop

**Interfaces produced:**

```python
class TetrisAgent:
    def __init__(self, emu, genome, collector, recorder=None, max_pieces=300)
    def run(self, timer_div: int | None = None) -> dict  # returns fitness

def agent_main(argv=None) -> int  # flags: --rom (default rom/tetris.gb), --max-pieces,
    # --timer-div (int, default None), --output-json PATH, --telemetry-dir (default data/telemetry),
    # --no-telemetry, --runs-dir (default runs), --no-record, --label, --no-self-heal, --window (default null)
```

`run()`: `emu.start(timer_div)`; loop: wait for falling piece (tick up to 120 with re-read; stuck event + break if none), `read_state`, `plan_placement` (None → stuck event, break), emit spawn+decision, `Controller.execute`, `tracker.on_lock` with post-lock features, emit locked; exit on `game_over` or `max_pieces`; emit game_over + session end; `finalize`; return fitness. `agent_main` wires everything, writes `--output-json`, then unless `--no-self-heal`: `from tetris_agent.healer import check; check(output_json_path)` inside try/except that logs and continues (Task 12 lands `check`; until then guard with try/ImportError — Task 12 removes the guard).

- [ ] Step 1: Failing pure test: `agent_main(["--rom", "x", "--no-telemetry", ...])` argument parsing factored as `build_config(argv)`; test config defaults.
- [ ] Step 2: `@pytest.mark.rom` integration: `TetrisAgent(max_pieces=5)` with `NoopPublisher` and tmp runs dir, `timer_div=0x00` → fitness `pieces_placed == 5`, `topped_out is False`, summary.json exists, events.jsonl has ≥ 5 spawn events.
- [ ] Step 3: Implement; `uv run tetris-agent --max-pieces 30 --timer-div 0 --output-json data/fit.json` completes and prints fitness; commit `feat: agent loop and CLI`.

### Task 10: genome.py — persistent genome store

**Interfaces produced:** `GENOME_PATH = Path("data/genome.json")`; `load_params(path=GENOME_PATH) -> dict` (defaults from `Genome().to_params()` merged under stored `current`); `append_genome(params: dict, source: str, path=GENOME_PATH) -> None` (sets `current`, appends `{"params", "source", "ts"}` to `history`, atomic write via tmp+rename).

- [ ] Step 1: Failing tests: load on missing file returns defaults; append then load round-trips; history grows; corrupt file → defaults (and warning), never raises.
- [ ] Step 2: Implement; green; commit `feat: JSON genome store`.

### Task 11: evolve.py — measured races

**Interfaces produced:**

```python
DEFAULT_MAX_PIECES = 150
def run_agent(params: dict, timer_div: int, rom_path, max_pieces=DEFAULT_MAX_PIECES) -> dict
    # in-process: Emulator + TetrisAgent + NoopPublisher, no recorder, no self-heal; returns fitness
def sample_variants(base: dict, implicated: list[str], n: int, rng: random.Random) -> list[dict]
    # gaussian jitter sigma=0.3*|value| (min 0.05) on implicated float keys; ints jitter ±1, floor 1
def run_race(control: dict, variants: list[dict], seeds: list[int], rom_path, runner=run_agent) -> dict
    # every candidate runs every seed; mean race_score per candidate
    # returns {"control": float, "candidates": [{"params", "score"}], "winner": {"params", "score"}}
def decide(control_score: float, winner_score: float, margin: float = 0.05) -> bool
```

- [ ] Step 1: Failing pure tests with a fake `runner` returning scripted fitness: race averages across seeds; winner selection; `decide` requires strictly > control*(1+margin) (and handles control ≤ 0 by requiring winner > control + 10); `sample_variants` only perturbs implicated keys and is deterministic given rng seed.
- [ ] Step 2: `@pytest.mark.rom` smoke: `run_agent(Genome().to_params(), timer_div=0, max_pieces=10)` returns dict with `pieces_placed == 10`.
- [ ] Step 3: Implement; green; commit `feat: in-process evolution races`.

### Task 12: healer.py — the healing loop

**Interfaces produced:**

```python
@dataclass(frozen=True)
class Rule:
    name: str; params: tuple[str, ...]; trigger: Callable[[dict], bool]

RULES = [
    Rule("early-topout", ("w_lines", "w_agg_height", "w_holes", "w_bumpiness"),
         lambda f: f.get("topped_out") and f.get("pieces_placed", 0) < 40),
    Rule("hole-thrash", ("w_holes", "w_bumpiness"), lambda f: f.get("avg_holes", 0) >= 4),
    Rule("misexecution", ("ticks_per_press",),
         lambda f: f.get("max_misexec_streak", 0) >= 3 or f.get("misexec_count", 0) >= 10),
]
COOLDOWN_HOURS = 6
STATE_PATH = Path("data/healer_state.json")
QUEUE_PATH = Path("data/discovery_queue.json")

def evaluate_rules(fitness: dict) -> list[Rule]
def default_seed(fitness_path: str) -> int          # int(sha256(path)[:8], 16)
def should_escalate(rule_name: str, accepted: bool, state: dict) -> bool
    # same rule fired again after last race for it was accepted, OR 2 consecutive rejections for it
def append_escalation(rule, fitness, state, path=QUEUE_PATH) -> None
def check(fitness_path: str | Path, rom_path="rom/tetris.gb", state_path=STATE_PATH,
          genome_path=None, n_variants=3, seeds_per=2, race=evolve.run_race) -> dict
    # never raises; returns report dict {"status": "no-rule"|"cooldown"|"accepted"|"rejected"|"error", ...}
```

`check` flow: read fitness JSON → `evaluate_rules` (first match wins) → cooldown vs `state["last_race_at"][rule.name]` → base = `genome.load_params()` → `rng = random.Random(default_seed(path))`, `variants = sample_variants(base, rule.params, n_variants, rng)`, `seeds = [(default_seed(path) + i * 0x11) & 0xFF for i in range(seeds_per)]` (these are `timer_div` values) → `race(...)` → `decide` → accepted: `genome.append_genome(winner, source=f"healer:{rule.name}")`; update state (race history capped 20, per-rule outcome, timestamps) → `should_escalate` → `append_escalation`. Whole body inside try/except returning `{"status": "error"}`.

- [ ] Step 1: Failing pure tests (fake `race`, tmp paths): rule matching precedence; cooldown blocks and expires; accepted race writes genome + state; rejection does not touch genome; two rejections escalate; refire-after-accept escalates; corrupt fitness file → `{"status": "error"}` without raising; `default_seed` deterministic.
- [ ] Step 2: Implement + `healer_main` CLI (`tetris-healer check <fitness.json> [--rom]` printing the report JSON, always exit 0); remove the ImportError guard in `agent.py`'s self-heal chain.
- [ ] Step 3: Green; `@pytest.mark.rom` end-to-end: run agent 20 pieces with a sabotaged genome (`w_holes=+1.0`) → fitness triggers `hole-thrash` → `check` with `n_variants=2, seeds_per=1` completes with status accepted/rejected and state file written. Commit `feat: fitness-driven healer with parameter races`.

### Task 13: discovery.py — LLM escalation

**Interfaces produced:**

```python
RULE_CODE_MAP = {
    "early-topout": ["src/tetris_agent/policy.py", "src/tetris_agent/board.py"],
    "hole-thrash": ["src/tetris_agent/policy.py", "src/tetris_agent/board.py"],
    "misexecution": ["src/tetris_agent/controller.py", "src/tetris_agent/emulator.py"],
}
COOLDOWN_HOURS = 24
def load_queue(path) -> list[dict]; def pick_entry(queue, state) -> dict | None  # oldest un-attempted
def build_bundle(entry, healer_state) -> dict   # escalation + recent races + implicated file list
def build_prompt(bundle) -> str                  # PROMPT_TEMPLATE: context, fitness, race history,
                                                 # files to consider, gates it must pass, "do not touch genome files"
def worktree_add(repo_root, branch) -> Path      # git worktree add .worktrees/<branch> -b <branch>
def propose(worktree, prompt) -> bool            # subprocess: claude -p <prompt> --permission-mode acceptEdits --max-turns 40
def run_gates(worktree, rom_path, entry) -> dict # uv run pytest -m "not rom" && uv run ruff check .
                                                 # && fitness race: candidate worktree run_agent vs baseline, decide(margin=0.05)
def push_and_pr(worktree, branch, entry) -> str  # git push -u origin, gh pr create --fill; returns url
def cleanup(worktree, branch) -> None
def run(queue_path=QUEUE_PATH, ...) -> dict      # orchestrates; one attempt per entry; 24h cooldown; never raises
```

The fitness gate runs the candidate's code via `uv --directory <worktree> run tetris-agent --max-pieces 150 --timer-div <seed> --no-self-heal --no-record --no-telemetry --output-json <tmp>` compared to the same command in the main checkout, two seeds, `decide` margin 0.05.

- [ ] Step 1: Failing tests with monkeypatched `subprocess.run`: pick_entry skips attempted entries and honors cooldown; prompt contains rule name, fitness numbers, and file list; gates short-circuit on pytest failure; successful gates call push_and_pr; failure calls cleanup; state records the attempt either way.
- [ ] Step 2: Implement + `discovery_main` CLI (`tetris-discovery run [--dry-run]`; dry-run prints the prompt and exits). Green; commit `feat: discovery escalation to claude worktree with gates`.

### Task 14: README + end-to-end verification

- [ ] Step 1: Rewrite `README.md`: the three loops (diagram of agent → fitness.json → healer race → genome.json → next run; escalation → discovery → PR), quickstart, determinism via `timer_div`, event schema, how this differs from pokemon-kafka (in-process, JSON genome, broker-free with JSONL bridge point). Document `data/` and `runs/` layout.
- [ ] Step 2: Full verification: `uv run pytest -m "" -v` (all pure + ROM tests), `uv run ruff check .`, then a real session: `uv run tetris-agent --max-pieces 150 --output-json data/last_fitness.json` and confirm the self-heal chain prints a healer report. Fix anything found.
- [ ] Step 3: Commit `docs: README with self-healing architecture`; final review pass with superpowers:requesting-code-review.

## Deferred (explicitly out of scope)

- Kafka/Flink docker stack: the JSONL sink shape is bridge-compatible with pokemon-kafka's `docker/game-event-bridge`; port it only when streaming detection is actually wanted.
- Viewer UI, multi-agent evolution (`run_10_agents` analog), frames capture.

## Self-review notes

- Spec coverage: three healing layers → Task 6 (in-run), Tasks 10-12 (tune), Task 13 (grow). Determinism → timer_div threaded through Tasks 9, 11, 12. Clean-architecture goals → global constraints (package, in-process, JSON genome).
- Type consistency check: `Placement` produced in Task 4, consumed Tasks 6/9; `Features` produced Task 3, consumed Tasks 4/8; fitness dict keys defined Task 8, consumed Tasks 11/12; `race_score` defined Task 8, used Task 11; `run_race` signature Task 11 matches healer's `race=` kwarg Task 12.
