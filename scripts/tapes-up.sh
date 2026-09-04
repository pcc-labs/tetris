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

# Storage is the `tapes local up` Postgres container. Every `tapes serve` below
# must be handed the DSN explicitly: it does not read TAPES_PG_DSN itself, and
# ~/.tapes/config.toml ships an empty [storage] section, so without --postgres
# each one exits immediately with "empty postgres dsn" — into a backgrounded
# shell where nobody sees it, leaving a stack that looks up and captures nothing.
PG_DSN="${TAPES_PG_DSN:-postgres://tapes:tapes@127.0.0.1:5432/tapes?sslmode=disable}"

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

tapes serve api --listen "127.0.0.1:${API_PORT}" --postgres "$PG_DSN" &
tapes serve derive-worker --postgres "$PG_DSN" &
tapes serve proxy --provider anthropic --upstream https://api.anthropic.com \
  --listen "127.0.0.1:${ANTHROPIC_PORT}" --project "$PROJECT" --postgres "$PG_DSN" &
tapes serve proxy --provider openai --upstream http://127.0.0.1:11434 \
  --listen "127.0.0.1:${OLLAMA_PORT}" --project "$PROJECT" --postgres "$PG_DSN" &

# A proxy that died on startup leaves the port closed. Say so here rather than
# letting a whole benchmark run land nowhere. Each proxy connects to Postgres
# before it binds, so give it up to 10 s rather than one probe at t=1 s -- a
# single early probe was killing healthy stacks on a loaded box.
wait_for_port() {  # port label provider
  local port=$1 label=$2 provider=$3 i
  for i in $(seq 1 20); do
    nc -z 127.0.0.1 "$port" >/dev/null 2>&1 && return 0
    sleep 0.5
  done
  echo "tapes ${label} proxy is not listening on ${port} after 10 s — it exited on startup." >&2
  echo "Run it in the foreground to see why:" >&2
  echo "  tapes serve proxy --provider ${provider} --listen 127.0.0.1:${port} --postgres '${PG_DSN}'" >&2
  return 1
}
wait_for_port "$ANTHROPIC_PORT" anthropic anthropic || exit 1
wait_for_port "$OLLAMA_PORT" ollama openai || exit 1
cat <<EOF

tapes capture stack is up (project: ${PROJECT}). In your benchmark shell:

  export ANTHROPIC_BASE_URL=http://127.0.0.1:${ANTHROPIC_PORT}
  export TETRIS_TAPES_OLLAMA_URL=http://127.0.0.1:${OLLAMA_PORT}/v1

Claude arms route through :${ANTHROPIC_PORT} to api.anthropic.com; pi arms route
through :${OLLAMA_PORT} to the local Ollama (:11434). Read back captured
sessions on http://127.0.0.1:${API_PORT}. Ctrl-C stops everything.
EOF
wait
