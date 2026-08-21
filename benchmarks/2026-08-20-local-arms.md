# 2026-08-20 — first local open-weight arms (Strix Halo iGPU)

First `tetris-bench` rows for `pi/` arms on this box. Live mode (no pause), level-0
gravity (~15 s per piece), 30-piece cap, seed 1, `features` harness, `--thinking medium`.

## Hardware and harness constants

- **iGPU, not the 5090**: the TB5 eGPU is physically disconnected (`boltctl:
  disconnected`, `NVRM: No NVIDIA GPU found`). Everything runs on the Radeon 8060S
  via ROCm (`gfx1151`), ~61.4 GiB addressable (512M UMA carveout + GTT ≈ half of
  128 GB system RAM). Bandwidth ~256 GB/s, so *active* parameters dominate decode.
- **num_ctx must be capped.** Stock `glm-4.7-flash` and `qwen3.6:27b` auto-size to
  their native 202k/262k contexts and the runner dies loading the KV cache
  (`llama runner terminated exit status 2`). The `-32k` variants pin
  `PARAMETER num_ctx 32768` via `ollama create` — same blobs, no re-download.
  Tetris prompts are a single ASCII board (~2 k tokens); pokemon-kafka's 128k
  constant is reason-specific to long operator transcripts and does not transfer.
- **One resident model at a time.** Ollama's `keep_alive=5m` keeps recently used
  models on the GPU; three roster models ≈ 65 GB > the ~62 GiB GTT ceiling, and a
  saturated GPU times out *every* decision. Evict
  (`curl :11434/api/generate -d '{"model":M,"keep_alive":0}'`) and run one
  `--models` arm per `tetris-bench` invocation, `--no-control` after the first.
  An invalid triple-resident run was killed unpublished, per the harness-death rule.
- pi 0.73.1 (`@mariozechner/pi-coding-agent`), Ollama /api, ROM `Tetris (World)
  (Rev 1)` md5 `982ed5d2...`.

## Results (seed 1, 30-piece cap, level 0 live)

| arm | race | score | lines | pieces | avg holes | decisions | illegal | late | api err | downshifts | s/decision | tok/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heuristic | 490 | 340 | 8 | 30 | 0.13 | — | 0 | 0 | — | — | — | — |
| pi/gpt-oss:20b | 180 | 80 | 2 | 20 | 8.85 | 11 | 2 | 3 | 2 | 2 | 10.0 | 35.3 |
| pi/glm-4.7-flash-32k | 55 | 0 | 0 | 11 | 12.73 | 0 | 4 | 4 | 4 | 0 | 69.3 | 0.0 |
| pi/qwen3.6-27b-32k | 55 | 0 | 0 | 11 | 12.73 | 0 | 4 | 4 | 4 | 0 | 69.3 | 0.0 |
| random | 140 | 0 | 0 | 28 | 24.14 | — | 0 | 0 | — | — | — | — |
| no-input | 55 | 0 | 0 | 11 | — | — | 0 | 0 | — | — | — | — |

## Reading the glm and qwen rows

Both flatlined identically: race 55 is *exactly* the no-input floor — the game
played itself. Zero completed decisions each; 4 attempts, 4 timeouts, 0 output
tokens ever landed. Both load and answer a trivial prompt in ~5 s, so this is
not the num_ctx load failure (that is fixed). Follow-ups pinned down two
*different* failure modes:

- **glm-4.7-flash-32k: thinking verbosity.** Replaying the exact failed bench
  command with a 300 s clock finishes in **111 s with valid JSON — after 4,769
  output tokens of reasoning for one placement** (~43 tok/s decode, which is
  healthy). A full `--thinking low` arm flatlined identically (row above uses
  medium; low was benchmark-20260821-024913.json), so pi's thinking knob does
  not curb it. The model is fine; it deliberates ~30x longer than the clock.
- **qwen3.6-27b-32k: decode bandwidth.** The same replay blew even a 300 s
  ceiling on the *empty-board opening piece*. Direct measurement: **11.1 tok/s**
  decode (242 tok in 21.9 s) — the dense-27B wall on ~256 GB/s of iGPU
  bandwidth, exactly as predicted when the roster was chosen. Even terse
  thinking overruns 15 s at that rate.

**Roster implication:** on this box, live arms need *both* MoE-class decode
speed *and* terse reasoning. gpt-oss:20b is currently the only arm with both.
Candidates to try next should be small-active-parameter MoE models with short
CoT habits — or run deliberative models under `--paused` and label the rows
legacy-mode, per the README's A/B convention.

## Reading the gpt-oss row

It beats random — the floor arms exist exactly for this claim — but 9 of its 20
pieces locked before any answer arrived, and the adaptive controller downshifted
thinking twice trying to keep pace. Mean decision 10.0 s against a ~15 s clock:
viable, no headroom. 0 parse failures — the losses are speed, not JSON discipline.

Results JSON: `data/benchmarks/benchmark-20260821-024335.json` (controls + gpt-oss),
`benchmark-20260821-024523.json` (glm), `benchmark-20260821-024719.json` (qwen).
