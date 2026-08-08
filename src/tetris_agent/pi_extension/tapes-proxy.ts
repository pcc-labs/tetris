// Reroute pi's ollama provider through the local tapes capture proxy when
// TETRIS_TAPES_OLLAMA_URL is set (see scripts/tapes-up.sh). No-op otherwise,
// so pi arms run fine — uncaptured — without the tapes stack.
export default function (pi: any) {
  const base = process.env.TETRIS_TAPES_OLLAMA_URL;
  if (!base) return;
  pi.registerProvider("ollama", { baseUrl: base });
}
