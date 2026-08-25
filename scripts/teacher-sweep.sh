#!/usr/bin/env bash
# teacher-sweep.sh — accumulate frontier "teacher" runs for distillation.
#
# Drives tetris-agent (NOT tetris-bench: bench arms run recorder=None and
# leave no per-decision record) in its default paused mode — the emulator
# freezes while the model thinks, so there is no deadline and the traces are
# the model's best play, not its fastest. Each run lands in runs/<id>/ with
# events.jsonl, which traces.py can replay into verified (board, placement)
# exemplars. The final step mines every teacher run and writes the export
# Distil Labs consumes: data/teacher/<slug>/exemplars.jsonl.
#
# Determinism: --timer-div pins the piece RNG, so a seed names a piece
# sequence. Sweeping seeds is how the dataset gets board diversity; the same
# seeds later name the held-out eval games (train on some, hold out others).
#
# Usage:
#   scripts/teacher-sweep.sh                       # defaults below
#   MODEL=claude-opus-5 SEEDS="1 2 3" scripts/teacher-sweep.sh
#   scripts/teacher-sweep.sh -y                    # skip the cost prompt
#
# Knobs (env vars):
#   MODEL       teacher model id (default claude-opus-5; pi/<ollama-model> works
#               too for a local teacher — set TETRIS_TAPES_OLLAMA_URL to also
#               capture those calls in tapes)
#   HARNESS     board|legal|features|chat (default features)
#   EFFORT      low|medium|high|xhigh|max (default high — teacher runs have no
#               clock, so buy the thinking)
#   SEEDS       timer_div values, space-separated (default "1 2 3 4 5 6 7 8")
#   MAX_PIECES  pieces per run (default 60)
#   RUNS_DIR    where runs land (default runs/)
#
# Resume-safe: a seed with a completed run in the manifest is skipped, so
# rerunning the script only fills gaps.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-claude-opus-5}"
HARNESS="${HARNESS:-features}"
EFFORT="${EFFORT:-high}"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
MAX_PIECES="${MAX_PIECES:-60}"
RUNS_DIR="${RUNS_DIR:-runs}"

SLUG="$(echo "${MODEL}-${HARNESS}-${EFFORT}" | tr '/:' '--')"
OUT_DIR="data/teacher/${SLUG}"
MANIFEST="${OUT_DIR}/manifest.tsv"   # seed<TAB>run_id<TAB>pieces
mkdir -p "${OUT_DIR}"
touch "${MANIFEST}"

seed_count=$(wc -w <<<"${SEEDS}" | tr -d ' ')
echo "teacher sweep: ${MODEL} / ${HARNESS} / ${EFFORT}"
echo "  ${seed_count} seed(s) x ${MAX_PIECES} pieces, paused mode (no clock)"
echo "  runs -> ${RUNS_DIR}/   export -> ${OUT_DIR}/exemplars.jsonl"

# Rough spend estimate for hosted arms, using the bench's planning number
# (~2,400 tok/decision). Local pi/ teachers bill $0 (watts instead).
case "${MODEL}" in
  pi/*) echo "  cost: local arm — \$0 API, energy metered per run" ;;
  *)
    est=$(uv run python - "$MODEL" "$seed_count" "$MAX_PIECES" <<'PY'
import sys
from tetris_agent.pricing import spec
model, seeds, pieces = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
s = spec(model)
# ~2k input + ~400 output per decision, one decision per piece
d = seeds * pieces
print(f"{(d * 2000 * s.input_per_mtok + d * 400 * s.output_per_mtok) / 1e6:.2f}")
PY
    ) || est="?"
    echo "  cost: ~\$${est} projected"
    ;;
esac

if [[ "${1:-}" != "-y" ]]; then
  read -r -p "run it? [y/N] " ok
  [[ "${ok}" == "y" || "${ok}" == "Y" ]] || exit 0
fi

for seed in ${SEEDS}; do
  if awk -F'\t' -v s="${seed}" '$1 == s {found=1} END {exit !found}' "${MANIFEST}"; then
    echo "seed ${seed}: already recorded, skipping"
    continue
  fi

  echo "--- seed ${seed} ---"
  before=$(ls -1 "${RUNS_DIR}" 2>/dev/null | sort | tail -1 || true)
  uv run tetris-agent \
    --policy model --model "${MODEL}" \
    --harness "${HARNESS}" --effort "${EFFORT}" \
    --timer-div "${seed}" --max-pieces "${MAX_PIECES}" \
    --window null --no-self-heal \
    --runs-dir "${RUNS_DIR}" --label "teacher-${SLUG}-seed${seed}"
  after=$(ls -1 "${RUNS_DIR}" | sort | tail -1)
  if [[ -z "${after}" || "${after}" == "${before}" ]]; then
    echo "seed ${seed}: no run directory appeared — not recording in manifest" >&2
    continue
  fi
  printf '%s\t%s\t%s\n' "${seed}" "${after}" "${MAX_PIECES}" >> "${MANIFEST}"
done

# Mine every manifest run into the export. Verified replay only: a turn whose
# recomputed features miss the recorded checksum is dropped, with everything
# after it, so a bad reconstruction can never become training data.
uv run python - "${MODEL}" "${HARNESS}" "${EFFORT}" "${RUNS_DIR}" "${MANIFEST}" "${OUT_DIR}/exemplars.jsonl" <<'PY'
import json, sys
from pathlib import Path
from tetris_agent.traces import mine_run

model, harness, effort, runs_dir, manifest, out_path = sys.argv[1:7]
policy = f"{model}/{harness}" + (f"/{effort}" if effort else "")
rows, per_run = [], {}
for line in Path(manifest).read_text().splitlines():
    seed, run_id, _ = line.split("\t")
    exs = mine_run(Path(runs_dir) / run_id, policies=(policy,))
    per_run[f"seed {seed} ({run_id})"] = len(exs)
    for e in exs:
        rows.append({
            "seed": int(seed),
            "board": ["".join("#" if c else "." for c in row) for row in e.board],
            "piece": e.piece, "next_piece": e.next_piece,
            "rotation": e.rotation, "col": e.col,
            "lines_delta": e.lines_delta, "holes": e.holes,
        })
Path(out_path).write_text("".join(json.dumps(r) + "\n" for r in rows))
for k, v in per_run.items():
    print(f"  {k}: {v} exemplars")
print(f"export: {out_path} ({len(rows)} pairs, teacher {policy})")
PY
