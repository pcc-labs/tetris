// Point pi's ollama provider somewhere other than this box's :11434.
//
//   TETRIS_TAPES_OLLAMA_URL  — the local tapes capture proxy (scripts/tapes-up.sh);
//                              an openai-compatible base, `/v1` included. Wins.
//   TETRIS_OLLAMA_URL        — a remote Ollama (scripts/daytona_host.py, or the
//                              Framework over Tailscale); the daemon's root URL,
//                              same value pi_policy's preflight reads. Loopback
//                              values are ignored: that is already pi's default.
//
// No-op otherwise, so pi arms run fine — local and uncaptured — with neither set.
// Only baseUrl is overridden, which keeps the model list from ~/.pi/agent/models.json.
export default function (pi: any) {
  const base = ollamaBase(process.env);
  if (!base) return;
  pi.registerProvider("ollama", { baseUrl: base });
}

export function ollamaBase(env: Record<string, string | undefined>): string | null {
  if (env.TETRIS_TAPES_OLLAMA_URL) return env.TETRIS_TAPES_OLLAMA_URL;
  const raw = env.TETRIS_OLLAMA_URL;
  if (!raw) return null;
  let host: string;
  try {
    host = new URL(raw).hostname;
  } catch {
    return null;
  }
  if (["127.0.0.1", "localhost", "::1", "[::1]", "0.0.0.0", ""].includes(host)) return null;
  return raw.replace(/\/+$/, "") + "/v1";
}
