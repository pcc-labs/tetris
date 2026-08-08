---
name: manual-baseline
description: Use when the user wants to play Game Boy Tetris by hand to set or refresh the human baseline for the benchmark — opens the viewer in a browser, records every piece and where they landed it, and reports the run alongside the model arms.
---

# Manual baseline run

A human baseline is the reference that makes "can a model do this" mean
anything. The other non-model arms answer narrower questions: `no-input` is the
floor (score 0, tops out ~10 pieces), `random` is chance (score 0, ~21 pieces).
Neither says what a person scores on the same pieces.

Manual play emits the same events a model arm does, so the run lands in
`runs/<id>/` with one spawn → decision → locked triple per piece, and the
viewer replays it without knowing a human was driving.

## Before starting

Ask which seed and piece count if the user hasn't said. Defaults: **seed 0, 50
pieces** — that matches the model arms already recorded, and `--seed` fixes the
piece sequence, so the same seed means they played the identical game the models
did. A different seed is not comparable.

## Steps

**1. Start the viewer if it isn't already up.**

```bash
curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8000/api/runs
```

Anything other than `200` means start it in the background:

```bash
uv run tetris-viewer --port 8000
```

**2. Open the browser.**

```bash
open http://127.0.0.1:8000
```

**3. Hand over for play.** Run this in the **background** — it opens a real
Game Boy window and blocks until they top out, hit the piece limit, or quit, which
is far longer than a foreground command should wait:

```bash
uv run tetris-manual --seed 0 --max-pieces 50
```

Tell them the controls and then stop talking: arrow keys move and soft-drop,
`a` rotates, `return` starts. `ctrl-c` or closing the window stops early and
still records what they did. Do not narrate while they play.

**4. When it exits, summarise the run.** Find the newest run directory and read
its events:

```bash
ls -td runs/*/ | head -1
```

Build a per-piece table from `events.jsonl` — `piece_spawn` carries the block,
`placement_decision` carries the rotation and column they landed it at, and
`piece_locked` carries what that cost them:

```bash
uv run python -c "
import json, sys, pathlib
run = pathlib.Path(sys.argv[1])
events = [json.loads(l) for l in (run / 'events.jsonl').read_text().splitlines() if l.strip()]
piece = None
print(f\"{'#':>3}  {'block':<5} {'rot':>3} {'col':>3} {'lines':>5} {'holes':>5} {'score':>6}\")
n = 0
for e in events:
    d, t = e.get('data', {}), e.get('event_type')
    if t == 'piece_spawn': piece = d.get('piece')
    elif t == 'placement_decision': rot, col = d.get('rotation'), d.get('col')
    elif t == 'piece_locked':
        n += 1
        print(f\"{n:>3}  {piece:<5} {rot:>3} {col:>3} {d.get('lines_delta',0):>5} {d.get('holes',0):>5} {d.get('score',0):>6}\")
    elif t == 'game_over':
        f = d.get('fitness', {})
        print(f\"\nscore {f.get('score')}  lines {f.get('lines')}  pieces {f.get('pieces_placed')}  avg_holes {f.get('avg_holes')}  topped_out {f.get('topped_out')}\")
" "$(ls -td runs/*/ | head -1)"
```

**5. Place it on the ladder.** Report their score next to the arms already
measured on the same seeds, so the number means something:

| arm | score | lines | pieces survived |
|---|---|---|---|
| `sonnet/features` | 800 | 14.0 | 50 |
| `haiku/features` | 760 | 14.5 | 50 |
| `sonnet/board` | 300 | — | 48 |
| random legal | 0 | 0 | 21.0 |
| `haiku/board` | 0 | 0 | 13 |
| no input | 0 | 0 | 10.8 |

Call out `avg_holes` specifically — it is the metric that separated the models
from each other, and the one a human should win on most clearly.

## Caveats to state when reporting

- **The models had no clock.** Each got as long as it wanted per piece; the
  human plays against gravity. That asymmetry favours the models, so a human
  win is a decisive win.
- **One run is an anecdote.** Piece luck moved the same unchanged algorithm
  from 920 to 1040 across seeds. Offer a second run on `--seed 1` before
  treating the number as settled.
- Do not compare against the heuristic arm. It is an exhaustive solver, not a
  player, and it is excluded from this benchmark's analysis.
