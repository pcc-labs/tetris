#!/usr/bin/env bash
# Local tapes capture stack for the benchmark: two single-provider proxies
# (Claude arms and pi/Ollama arms), the read API, and the derive worker.
# Everything lands in the local tapes store — nothing reaches the Paper org.
# Ctrl-C tears the whole stack down.
set -euo pipefail

# :8080 is intentionally avoided — pi's mlx provider already binds it.
ANTHROPIC_PORT="${TAPES_ANTHROPIC_PORT:-8091}"
OLLAMA_PORT="${TAPES_OLLAMA_PORT:-8092}"
API_PORT="${TAPES_API_PORT:-8081}"
PROJECT="${TAPES_PROJECT:-tetris}"

if ! command -v tapes >/dev/null 2>&1; then
  echo "tapes is not installed: curl -fsSL https://download.tapes.dev/install | bash" >&2
  exit 1
fi

# Storage is the `tapes local up` Postgres container (see ~/.tapes/config.toml).
if ! nc -z localhost 5432 >/dev/null 2>&1; then
  echo "postgres (localhost:5432) is not reachable — run 'tapes local up' first (needs Docker)" >&2
  exit 1
fi

for port in "$ANTHROPIC_PORT" "$OLLAMA_PORT" "$API_PORT"; do
  if lsof -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "port $port is already in use — another tapes stack or service is running" >&2
    exit 1
  fi
done

trap 'kill 0' EXIT INT TERM

tapes serve api --listen "127.0.0.1:${API_PORT}" &
tapes serve derive-worker &
tapes serve proxy --provider anthropic --upstream https://api.anthropic.com \
  --listen "127.0.0.1:${ANTHROPIC_PORT}" --project "$PROJECT" &
tapes serve proxy --provider openai --upstream http://127.0.0.1:11434 \
  --listen "127.0.0.1:${OLLAMA_PORT}" --project "$PROJECT" &

sleep 1
cat <<EOF

tapes capture stack is up (project: ${PROJECT}). In your benchmark shell:

  export ANTHROPIC_BASE_URL=http://127.0.0.1:${ANTHROPIC_PORT}
  export TETRIS_TAPES_OLLAMA_URL=http://127.0.0.1:${OLLAMA_PORT}/v1

Claude arms route through :${ANTHROPIC_PORT} to api.anthropic.com; pi arms route
through :${OLLAMA_PORT} to the local Ollama (:11434). Read back captured
sessions on http://127.0.0.1:${API_PORT}. Ctrl-C stops everything.
EOF
wait
