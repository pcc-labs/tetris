#!/usr/bin/env bash
# The exe.dev pitch demo: an open-weight model plays Game Boy Tetris inside a
# VM, every inference call exits the VM through a tapes capture proxy to the
# host's Ollama, and you watch the board stream live in the GRAMBOY viewer.
# No hosted API anywhere in the loop — the whole stack is local open weights.
#
#   VM (lima "tetris", standing in for an exe.dev shell computer)
#     └─ tetris-agent --live --policy model --model pi/<ollama-model>
#          ├─ pi → ~/.pi/agent/models.json baseUrl → host:8092
#          │    └─ tapes capture proxy → host Ollama :11434
#          │         (session → project tetris-vm)
#          └─ frames → ws://host:8000    GRAMBOY viewer LIVE tab
#
# Afterwards the captured session is a tapes session like any other: derive a
# skill from it (talk-emoji's 📼, or POST /v1/cassettes/skills/generate).
#
# Usage:  scripts/vm-demo.sh [max_pieces]   (default 15; ~15s/piece at level 0)
set -euo pipefail

cd "$(dirname "$0")/.."
PIECES="${1:-15}"
MODEL="${TETRIS_VM_MODEL:-pi/gemma3}"
OLLAMA_MODEL="${MODEL#pi/}"

step() { printf '\n\033[1m%s\033[0m\n' "$*"; }

step "1/5 host services (viewer :8000, capture proxy :8092 → ollama, derive worker)"
nc -z localhost 5432 || { echo "postgres down — run 'tapes local up' or start tapes-postgres-1"; exit 1; }
curl -sf -m 2 http://localhost:8081/ping >/dev/null || { echo "tapes api down — tapes serve api"; exit 1; }
curl -sf -m 2 http://localhost:11434/api/tags >/dev/null || { echo "ollama down — ollama serve"; exit 1; }
curl -s http://localhost:11434/api/tags | grep -q "\"$OLLAMA_MODEL\"" \
  || { echo "$OLLAMA_MODEL not pulled — ollama pull $OLLAMA_MODEL"; exit 1; }
if ! curl -sf -m 2 http://localhost:8000/ >/dev/null; then
  (nohup uv run tetris-viewer --host 0.0.0.0 >/tmp/tetris-viewer.log 2>&1 &)
fi
if ! lsof -iTCP:8092 -sTCP:LISTEN -t >/dev/null; then
  (nohup tapes serve proxy --provider openai --upstream http://127.0.0.1:11434 \
    --listen 0.0.0.0:8092 --project tetris-vm >/tmp/tetris-tapes-proxy.log 2>&1 &)
fi
pgrep -f "tapes serve derive-worker" >/dev/null || \
  (nohup tapes serve derive-worker >/tmp/tetris-derive-worker.log 2>&1 &)
sleep 1
echo "viewer + proxy + derive worker up"

step "2/5 vm"
# 12GiB: inference runs on the HOST's Ollama, but pi_policy's preflight sizes
# the model against local RAM — the VM needs headroom to pass that guard.
if ! limactl list --format '{{.Name}} {{.Status}}' | grep -q '^tetris Running'; then
  limactl list --format '{{.Name}}' | grep -q '^tetris$' \
    && limactl start tetris --tty=false \
    || { limactl create --name tetris --memory 12 --tty=false template://ubuntu-lts; limactl start tetris --tty=false; }
fi
limactl shell tetris -- true && echo "vm running"

step "3/5 provision (uv + repo + node + pi, first run only)"
if ! limactl shell tetris -- bash -c 'test -x ~/.local/bin/uv && test -d ~/tetris'; then
  limactl shell tetris -- bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh' >/dev/null
  tar czf - --exclude .venv --exclude __pycache__ --exclude runs \
    --exclude screenshots --exclude data/telemetry -C .. tetris \
    | limactl shell tetris -- bash -c 'tar xzf - -C ~ 2>/dev/null || true'
  limactl shell tetris -- bash -c 'export PATH=$HOME/.local/bin:$PATH && cd ~/tetris && uv sync'
fi
if ! limactl shell tetris -- bash -c 'command -v pi >/dev/null'; then
  limactl shell tetris -- sudo bash -c \
    'curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs' >/dev/null 2>&1
  # Pinned: pi_policy reads pi's JSONL event shape as of v0.80.10.
  limactl shell tetris -- sudo npm install -g @earendil-works/pi-coding-agent@0.80.10 >/dev/null 2>&1
fi
# pi's ollama provider inside the VM: baseUrl is the capture proxy, so every
# inference call is captured by construction (no extension env needed).
limactl shell tetris -- bash -c 'mkdir -p ~/.pi/agent && cat > ~/.pi/agent/models.json << "EOF"
{
 "providers": {
  "ollama": {
   "api": "openai-completions",
   "apiKey": "ollama",
   "baseUrl": "http://host.lima.internal:8092/v1",
   "models": [
    {"id": "gemma3", "contextWindow": 131072},
    {"id": "qwen2.5-coder:7b", "contextWindow": 32768},
    {"id": "qwen3:8b", "contextWindow": 40960},
    {"id": "gpt-oss:20b", "contextWindow": 131072, "reasoning": true}
   ]
  }
 }
}
EOF'
echo "provisioned"

step "4/5 open the viewer"
echo "watch: http://127.0.0.1:8000  (LIVE tab)"

step "5/5 $MODEL plays $PIECES pieces in the vm"
limactl shell tetris -- bash -c "
  export PATH=\$HOME/.local/bin:\$PATH \
    TETRIS_OLLAMA_URL=http://host.lima.internal:11434
  cd ~/tetris && uv run tetris-agent --live --policy model --model $MODEL \
    --harness chat --effort low --max-pieces $PIECES --timer-div 0 --no-self-heal \
    --viewer-url ws://host.lima.internal:8000"
# --harness chat on purpose: its history carries across pieces, so the whole
# run captures as ONE tapes session (stateless harnesses land one per piece).
# TETRIS_OLLAMA_URL points preflight at the host's real Ollama for the
# model-installed check; inference itself goes through the proxy baseUrl.

step "captured session (project tetris-vm)"
sleep 25   # derive worker debounce
curl -s "http://localhost:8081/v1/sessions?project=tetris-vm&limit=5" | python3 -c "
import json,sys
for s in json.load(sys.stdin)['items']:
    r = s.get('rollup') or {}
    print(s['id'], 'turns:', r.get('turn_count'),
          'calls:', sum(m['calls'] for m in r.get('model_usage', [])))"
echo
echo "derive a skill from it: react 📼 in talk-emoji (localhost:7777), or:"
echo '  curl -X POST localhost:8081/v1/cassettes/skills/generate -H "content-type: application/json" -d '"'"'{"sessionIds":["<id>"]}'"'"''
