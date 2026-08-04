/* GRAMBOY bench viewer: LIVE (ws) and REPLAY (cartridges over runs/). */

const $ = (id) => document.getElementById(id);
const canvas = $("screen");
const ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;

const state = {
  mode: "live",          // "live" | "replay"
  ws: null,
  run: null,             // loaded run detail
  frame: 0,
  playing: false,
  speed: 1,
  timer: null,
  hud: { score: 0, lines: 0, level: 0, piece: 0, holes: 0, misexec: 0 },
};

/* ── HUD ── */
function pad(n, w) { return String(n).padStart(w, "0"); }
function renderHud() {
  const h = state.hud;
  $("r-score").textContent = pad(h.score, 6);
  $("r-lines").textContent = pad(h.lines, 4);
  $("r-level").textContent = pad(h.level, 2);
  $("r-piece").textContent = pad(h.piece, 4);
  $("r-holes").textContent = pad(h.holes, 2);
  $("r-misexec").textContent = pad(h.misexec, 2);
}

function tick(msg, cls) {
  const li = document.createElement("li");
  if (cls) li.className = cls;
  const turn = document.createElement("span");
  turn.className = "t";
  turn.textContent = `t${msg.turn ?? 0}`;
  li.appendChild(turn);
  li.appendChild(document.createTextNode(msg.text));
  const ticker = $("ticker");
  ticker.prepend(li);
  while (ticker.children.length > 40) ticker.lastChild.remove();
}

function applyEvent(e) {
  const d = e.data || {};
  switch (e.event_type) {
    case "piece_spawn":
      state.hud.piece = e.turn;
      tick({ turn: e.turn, text: `spawn ${d.piece} (next ${d.next_piece})` });
      break;
    case "placement_decision":
      tick({ turn: e.turn, text: `plan rot=${d.rotation} col=${d.col}` });
      if (d.reason) tick({ turn: e.turn, text: `“${d.reason}”` }, "reason");
      break;
    case "piece_locked":
      state.hud.lines += d.lines_delta || 0;
      state.hud.holes = d.holes ?? state.hud.holes;
      state.hud.misexec += d.misexec || 0;
      if (d.score) state.hud.score = d.score;
      state.hud.level = Math.floor(state.hud.lines / 10);
      if (d.lines_delta) tick({ turn: e.turn, text: `LINE CLEAR ×${d.lines_delta}` }, "warn");
      break;
    case "stuck":
      tick({ turn: e.turn, text: `STUCK: ${d.detail}` }, "bad");
      break;
    case "game_over":
      tick({ turn: e.turn, text: `game over — score ${d.fitness?.score}` }, "bad");
      break;
    case "session":
      tick({ turn: 0, text: `session ${d.phase}${d.policy ? ` — ${d.policy}` : ""}` });
      if (d.fitness?.policy?.cost_usd) {
        tick({ turn: 0, text: `cost $${d.fitness.policy.cost_usd.toFixed(4)}` }, "warn");
      }
      break;
  }
  renderHud();
}

function resetHud() {
  state.hud = { score: 0, lines: 0, level: 0, piece: 0, holes: 0, misexec: 0 };
  $("ticker").replaceChildren();
  renderHud();
}

/* ── screen ── */
function drawDataUrl(url) {
  const img = new Image();
  img.onload = () => { ctx.drawImage(img, 0, 0, 160, 144); };
  img.src = url;
  $("lcd-notice").classList.add("hidden");
}

/* ── LIVE mode ── */
function connectLive() {
  if (state.ws) return;
  const ws = new WebSocket(`ws://${location.host}/ws/live`);
  state.ws = ws;
  ws.onopen = () => { $("ws-status").textContent = "ws: connected"; };
  ws.onclose = () => {
    $("ws-status").textContent = "ws: disconnected";
    $("power-led").classList.remove("on");
    $("feed-state").textContent = "IDLE";
    $("feed-state").classList.remove("live");
    state.ws = null;
    if (state.mode === "live") setTimeout(connectLive, 1500);
  };
  ws.onmessage = (evt) => {
    if (state.mode !== "live") return;
    const msg = JSON.parse(evt.data);
    $("power-led").classList.add("on");
    $("feed-state").textContent = "LIVE";
    $("feed-state").classList.add("live");
    if (msg.type === "frame") drawDataUrl(`data:image/png;base64,${msg.png}`);
    else if (msg.type === "event") applyEvent(msg.event);
  };
}

/* ── REPLAY mode ── */
async function loadShelf() {
  const runs = await (await fetch("/api/runs")).json();
  const shelf = $("carts");
  shelf.replaceChildren();
  for (const run of runs) {
    const cart = document.createElement("button");
    cart.className = "cart";
    cart.dataset.runId = run.run_id;
    const label = document.createElement("span");
    label.className = "cart-label";
    label.textContent = run.label || run.run_id;
    const meta = document.createElement("span");
    meta.className = "cart-meta";
    meta.innerHTML = `<span>${run.fitness.lines ?? 0} lines · <b>${run.fitness.score ?? 0}</b></span><span>${run.frame_count}f</span>`;
    cart.append(label, meta);
    cart.onclick = () => insertCart(run.run_id);
    shelf.appendChild(cart);
  }
  if (!runs.length) shelf.textContent = "no recorded runs yet — play one with recording on";
}

async function insertCart(runId) {
  document.querySelectorAll(".cart").forEach((c) => c.classList.toggle("inserted", c.dataset.runId === runId));
  state.run = await (await fetch(`/api/runs/${runId}`)).json();
  state.frame = 0;
  resetHud();
  $("t-scrub").max = Math.max(0, state.run.frames.length - 1);
  showFrame(0);
  setPlaying(true);
}

function turnOfFrame(name) {
  const m = name.match(/-t(\d+)\.png$/);
  return m ? parseInt(m[1], 10) : 0;
}

function showFrame(i) {
  if (!state.run || !state.run.frames.length) return;
  state.frame = Math.max(0, Math.min(i, state.run.frames.length - 1));
  drawDataUrl(state.run.frames[state.frame]);
  $("t-scrub").value = state.frame;
  $("t-frame").textContent = `${state.frame + 1}/${state.run.frames.length}`;
  // Rebuild HUD from events up to this frame's turn.
  const turn = turnOfFrame(state.run.frames[state.frame]);
  resetHud();
  for (const e of state.run.events) {
    if ((e.turn ?? 0) <= turn) applyEvent(e);
  }
}

function setPlaying(playing) {
  state.playing = playing;
  $("t-play").textContent = playing ? "❚❚" : "▶";
  clearInterval(state.timer);
  if (playing) {
    state.timer = setInterval(() => {
      if (state.frame >= state.run.frames.length - 1) { setPlaying(false); return; }
      showFrame(state.frame + 1);
    }, 300 / state.speed);
  }
}

function cycleSpeed() {
  state.speed = state.speed >= 4 ? 1 : state.speed * 2;
  $("t-speed").textContent = `${state.speed}×`;
  if (state.playing) setPlaying(true);
}

/* ── BENCH mode ── */
const BENCH_COLS = [
  ["arm", "ARM"], ["race_score", "RACE"], ["score", "SCORE"], ["lines", "LINES"],
  ["pieces", "PIECES"], ["avg_holes", "HOLES"], ["illegal", "ILLEGAL"],
  ["latency_ms", "MS/DEC"], ["cost_usd", "COST $"],
];

async function loadBench() {
  const runs = await (await fetch("/api/benchmarks")).json();
  const body = $("bench-body");
  if (!runs.length) {
    body.textContent = "no benchmark runs yet — try: uv run tetris-bench --estimate";
    $("bench-stamp").textContent = "model × harness × effort";
    return;
  }
  const latest = runs[0];
  $("bench-stamp").textContent = latest.recorded_at.slice(0, 19).replace("T", " ") + " UTC";
  const table = document.createElement("table");
  table.className = "bench-table";
  const head = table.insertRow();
  for (const [, label] of BENCH_COLS) {
    const th = document.createElement("th");
    th.textContent = label;
    head.appendChild(th);
  }
  latest.summary.forEach((row, i) => {
    const tr = table.insertRow();
    if (row.arm === "heuristic") tr.className = "control";
    else if (i === 0) tr.className = "winner";
    for (const [key] of BENCH_COLS) {
      const td = tr.insertCell();
      td.textContent = row[key];
    }
  });
  body.replaceChildren(table);
}

/* ── mode switching ── */
function setMode(mode) {
  state.mode = mode;
  $("btn-live").classList.toggle("active", mode === "live");
  $("btn-replay").classList.toggle("active", mode === "replay");
  $("btn-bench").classList.toggle("active", mode === "bench");
  $("shelf").classList.toggle("hidden", mode !== "replay");
  $("replay-deck").classList.toggle("hidden", mode !== "replay");
  $("bench-panel").classList.toggle("hidden", mode !== "bench");
  $("lcd-notice").classList.remove("hidden");
  resetHud();
  if (mode === "replay") { setPlaying(false); loadShelf(); }
  else if (mode === "bench") { setPlaying(false); loadBench(); }
  else connectLive();
}

const MODES = ["live", "replay", "bench"];
$("btn-live").onclick = () => setMode("live");
$("btn-replay").onclick = () => setMode("replay");
$("btn-bench").onclick = () => setMode("bench");
$("pad-select").onclick = () => setMode(MODES[(MODES.indexOf(state.mode) + 1) % MODES.length]);
$("pad-start").onclick = () => state.run && setPlaying(!state.playing);
$("t-play").onclick = () => state.run && setPlaying(!state.playing);
$("t-speed").onclick = cycleSpeed;
$("pad-a").onclick = cycleSpeed;
$("pad-b").onclick = () => { state.speed = 1; $("t-speed").textContent = "1×"; if (state.playing) setPlaying(true); };
$("t-scrub").oninput = (e) => { setPlaying(false); showFrame(parseInt(e.target.value, 10)); };

renderHud();
setMode("live");
