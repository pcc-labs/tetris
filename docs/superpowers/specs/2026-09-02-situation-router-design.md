# Situation Router — design

Status: approved for implementation (2026-09-02). Owner: bdougie. Mechanism: deterministic.
Review copy: https://claude.ai/code/artifact/1e53ed94-9759-4afc-a63f-df764c200d2a

Classify the board into one of five situations, then hand the model only the technique
that situation calls for — instead of one general placement prompt for every piece.

## 01 — Why route at all

The premise comes from Jonas Neubauer's strategy guidance ([Polygon, 2019](https://www.polygon.com/guides/2019/2/22/18225349/tetris-strategy-tips-how-to-jonas-neubauer/)), which is not one
general placement heuristic but a small set of board states each with a prescribed move.
That structure is the useful part: it is a routing table, and the left-hand side is
computable from the board array.

The payoff is expected to be tokens before it is accuracy. Every arm currently receives the
full "What good play looks like" block on every decision. A situation-specific fragment is
materially shorter, which pushes directly on the tokens-per-decision figure standing between
the local arms and being usable. Neubauer's own top-level rule — decide fast, even
sub-optimally, and build around the result — is the same claim the live-play and
`--decision-deadline` modes already test.

## 02 — A constraint found while scoping: no trace data on disk

`traces.mine_run` reads `events.jsonl` from a run directory, and there is not one in the
repository. `data/benchmarks/` holds 21 files, but they carry per-run aggregates only
(score, lines, avg_holes, timeouts) with no per-decision boards. The classifier cannot be
validated by replaying existing traces; section 06 substitutes a corpus that costs nothing.

## 03 — Where it sits

A new pure module, `src/tetris_agent/situation.py`, exposing one entry point:

```
classify(board, piece, next_piece) -> Situation
```

Pure NumPy over the settled board. No emulator, no API key, no I/O — the same shape as
`board.py` and `prompts.py`, both already unit-tested without a running ROM.

It is consumed by a **new** harness value in `prompts.py`:

```
HARNESSES = ("board", "legal", "features", "chat", "routed")
```

This is the load-bearing decision. Routing enters as another arm — a measurable independent
variable beside model and effort — rather than as an edit to how existing arms are prompted.
Every published row stays comparable; a routed arm is compared against `features` on
identical seeds.

## 04 — The five situations, in priority order

Neubauer implies an ordering but never states one, and real boards are mixtures. The order
below is a decision to be tested, not an inference. Read top-down; first match wins.

| # | situation | when | technique |
|---|---|---|---|
| 1 | `TOPPING_OUT` | max height past threshold | Survive. Flatten; take any line clear on offer. |
| 2 | `TETRIS_READY` | nine columns filled four deep, one well open, I current or next | Put the I in the well. GB has no hold, so the window is current + preview only. |
| 3 | `HOLE_RISK` | piece is O, S or Z and no flat gap two wide exists | Placement that creates the fewest new holes. These pieces are where most garbage originates. |
| 4 | `MOUND` | bumpiness past threshold | Flatten toward the chosen side. Neubauer biases left on NES; measure both. |
| 5 | `FLAT` | default | Stay flat: nothing over two high, no hole over two deep. |

Bottom six rows of a `TETRIS_READY` board in the `.`/`#` rendering the arms already receive
(18×10 playfield, well at column 9):

```
..........
..........
#########.
#########.
#########.
#########.
```

## 05 — Thresholds belong to the genome

Topping-out height, mound bumpiness and well depth are `genome.py` fields, so `evolve.py`
tunes them with existing machinery. This also makes the priority order falsifiable: a
threshold driven to an extreme means that class is doing no work.

## 06 — Validation without traces

Generate the corpus. `HeuristicPolicy` and `RandomPolicy` are local and free — run them
headless with the recorder on to produce the `events.jsonl` that does not currently exist.
Heuristic play yields well-formed boards; random play yields messy ones; together they
bracket the range a struggling model arm lands in. **Neither is an LLM's board
distribution** — this validates coverage and priority conflicts, not effect on play.

First artifact: a class-distribution histogram, and a genuine stopping point. If one class
swallows ~90% of boards, the table has nothing to route and further work is not justified.

## 07 — Testing

Test-first, `tests/test_situation.py`: hand-built boards per class, plus one case per
priority conflict (Tetris-ready while mounded; topping out while Tetris-ready; hole-risk
while flat). `classify` is pure — no emulator, no fixtures beyond the arrays.

## 08 — Success criterion

`--harness routed` against `--harness features` on identical seeds:

| measure | expectation | source |
|---|---|---|
| tokens per decision | falls — the primary hypothesis | results JSON, existing field |
| score | does not regress | results JSON, existing field |
| class distribution | no class above ~90% | new histogram (06) |

Then a model sweep under `routed` to find which model performs best with routing on —
one resident Ollama model at a time (see the benchmark setup notes), cloud arms as available.

## 09 — Tabled

- **Semantic router** (embed/classify the board description, route on similarity): parked,
  not rejected. Handles mixed boards more gracefully, but adds a model call per piece to the
  latency budget routing exists to protect. Revisit only if the histogram shows classes are
  well spread *and* the fixed order is demonstrably mis-sorting real boards.
- **Routing on model identity**: `references/model_fit.json`-style measured characters rest
  on 1–2 sessions per model — too thin to route on without encoding noise.
