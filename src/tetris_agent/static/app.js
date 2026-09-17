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
  playMode: false,       // a tetris-play session is live; keyboard is the controller
  inputWs: null,
  armMax: 0,          // pieces in the arm now streaming (banner progress)
  hud: { score: 0, lines: 0, level: 0, piece: 0, holes: 0, misexec: 0 },
  deck: {              // LABEL mode: the decisions of one run, one at a time
    runId: null,
    decisions: [],
    index: 0,
    jev: null,         // { questions, groups, setup } from /api/jev/questions
    answers: null,     // Jev's answers for the current decision
    prevAnswers: null, // ...and for the previous one, so moved rows can flash
    asking: null,      // AbortController of the request in flight
    timer: null,
  },
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
      if (state.armMax) $("arm-progress").textContent = `piece ${e.turn}/${state.armMax}`;
      setArmStatus("THINKING…", "think");
      break;
    case "placement_decision": {
      tick({ turn: e.turn, text: `plan rot=${d.rotation} col=${d.col}` });
      if (d.reason) tick({ turn: e.turn, text: `“${d.reason}”` }, "reason");
      const secs = d.latency_ms != null ? ` in ${(d.latency_ms / 1000).toFixed(1)}s` : "";
      if (d.late) setArmStatus(`TOO SLOW${secs} — dropped`, "late");
      else setArmStatus(`placed${secs}`, "ok");
      break;
    }
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
      if (d.phase === "start") setArm(d.policy);
      if (d.fitness?.policy?.cost_usd) {
        tick({ turn: 0, text: `cost $${d.fitness.policy.cost_usd.toFixed(4)}` }, "warn");
      }
      if (d.phase === "start") showArmBanner(d);
      if (d.phase === "end" && d.fitness) {
        setArmStatus(`DONE — score ${d.fitness.score ?? 0}`, "ok");
      }
      // A browser-play session hands the keyboard to the viewer.
      if (d.phase === "start" && d.surface === "browser") setPlayMode(true);
      if (d.phase === "end") setPlayMode(false);
      break;
  }
  renderHud();
}

// The bezel caption is the only always-visible spot that names the player, and
// it used to hard-code "SELF-HEALING AGENT" — wrong for every model arm, and
// wrong for human play. The arm arrives on session start and stays pinned; the
// event-bus line scrolls away after a few pieces.
const IDLE_CAPTION = "DOT MATRIX · SELF-HEALING AGENT";

function setArm(policy) {
  $("bezel-caption").textContent = policy ? `DOT MATRIX · ${policy.toUpperCase()}` : IDLE_CAPTION;
}

function resetHud() {
  state.hud = { score: 0, lines: 0, level: 0, piece: 0, holes: 0, misexec: 0 };
  $("ticker").replaceChildren();
  renderHud();
}

/* ── arm banner: who is playing right now ── */
function setArmStatus(text, cls) {
  const el = $("arm-status");
  el.textContent = text;
  el.className = `arm-status${cls ? ` ${cls}` : ""}`;
}

function showArmBanner(d) {
  // Benchmark arms announce their identity on session start; human play and
  // bare agent sessions don't, and keep the plain telemetry layout.
  if (!d.model) return;
  resetHud();
  state.armMax = d.max_pieces || 0;
  $("arm-model").textContent = d.model;
  $("arm-sub").textContent = [
    d.harness && `harness ${d.harness}`,
    d.effort && `effort ${d.effort}`,
    d.mode,
    d.seed != null && `seed ${d.seed}`,
  ].filter(Boolean).join(" · ");
  $("arm-progress").textContent = state.armMax ? `piece 0/${state.armMax}` : "";
  setArmStatus("");
  $("arm-banner").classList.remove("hidden");
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
    setArm(null);
    setPlayMode(false);
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
    else if (msg.type === "play_session") setPlayMode(true);  // continuous announce; late tabs arm too
  };
}

/* ── browser play: keyboard → /ws/input ── */
// Arrows move/soft-drop, up-arrow or a/s rotate, space starts the session and
// hard-drops in play (the backend synthesizes the drop — the Game Boy has
// none). The Game Boy ignores d-pad up in play, so ↑ maps to the A button —
// the rotate key every modern Tetris player reaches for.
// Press/release pairs, so the Game Boy's own DAS applies to held arrows.
const KEYMAP = {
  ArrowLeft: "left", ArrowRight: "right", ArrowDown: "down",
  ArrowUp: "a", a: "a", A: "a", s: "b", S: "b",
  Enter: "start", " ": "hard_drop",
};

// Keys are ALWAYS captured and sent in live mode — arming must never depend
// on having caught a session-start broadcast (a tab opened after tetris-play
// launches would stay deaf forever). playMode only drives the indicator.
function connectInput() {
  if (state.inputWs) return;
  const ws = new WebSocket(`ws://${location.host}/ws/input`);
  ws.onclose = () => {
    state.inputWs = null;
    if (state.mode === "live") setTimeout(connectInput, 1500);
  };
  state.inputWs = ws;
}

function setPlayMode(on) {
  if (on === state.playMode) return;
  state.playMode = on;
  $("feed-state").textContent = on ? "PLAYING" : (state.ws ? "LIVE" : "IDLE");
  $("feed-state").classList.toggle("playing", on);
}

function sendInput(button, action) {
  const ws = state.inputWs;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "input", button, action }));
  }
}

window.addEventListener("keydown", (e) => {
  const button = KEYMAP[e.key];
  if (state.mode !== "live" || !button) return;
  e.preventDefault();
  if (!e.repeat) sendInput(button, "press");  // held keys: the GB does its own DAS
});

window.addEventListener("keyup", (e) => {
  const button = KEYMAP[e.key];
  if (state.mode !== "live" || !button) return;
  e.preventDefault();
  sendInput(button, "release");
});

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
    cart.onclick = () => (state.mode === "label" ? loadDeck(run.run_id) : insertCart(run.run_id));
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
  ["late", "LATE"], ["latency_ms", "MS/DEC"], ["tok_s", "TOK/S"], ["cost_usd", "COST $"],
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
      td.textContent = row[key] ?? ""; // older result files predate some columns
    }
  });
  body.replaceChildren(table);
}

/* ── LABEL mode ──
   One decision at a time, in the Typewriter's shape: pick a cartridge, step
   through its pieces, and for each one see the board it was taken on (on the
   LCD), the oracle's regret, Jev's typed judgments (re-asked on every step,
   abortable, last answers held while the next are on the wire), and the
   verdict the exemplar miner reads back. */

const ROWS = 18, COLS = 10, CELL = 8;   // the 10×18 well drawn 8px a cell, centred on the 160×144 LCD
const BOARD_X = (160 - COLS * CELL) / 2;
const JEV_DEBOUNCE_MS = 120;

function drawBoard(d) {
  // Game Boy greens: the well is the light shade, settled cells the darkest,
  // the placed piece one step lighter so it reads as "just landed", and the
  // oracle's pick (when it differs) an outline the eye can compare against.
  ctx.fillStyle = "#0f380f";
  ctx.fillRect(0, 0, 160, 144);
  ctx.fillStyle = "#9bbc0f";
  ctx.fillRect(BOARD_X, 0, COLS * CELL, ROWS * CELL);
  d.board.forEach((row, r) => {
    for (let c = 0; c < COLS; c++) {
      if (row[c] !== "#") continue;
      ctx.fillStyle = "#0f380f";
      ctx.fillRect(BOARD_X + c * CELL, r * CELL, CELL, CELL);
      ctx.fillStyle = "#306230";
      ctx.fillRect(BOARD_X + c * CELL + 1, r * CELL + 1, CELL - 2, CELL - 2);
    }
  });
  for (const [r, c] of d.placed_cells || []) {
    ctx.fillStyle = "#0f380f";
    ctx.fillRect(BOARD_X + c * CELL, r * CELL, CELL, CELL);
    ctx.fillStyle = "#8bac0f";
    ctx.fillRect(BOARD_X + c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4);
  }
  const placed = new Set((d.placed_cells || []).map(([r, c]) => `${r},${c}`));
  const bestDiffers = (d.best_cells || []).some(([r, c]) => !placed.has(`${r},${c}`));
  if (bestDiffers) {
    ctx.strokeStyle = "#0f380f";
    ctx.setLineDash([2, 2]);
    for (const [r, c] of d.best_cells) ctx.strokeRect(BOARD_X + c * CELL + 0.5, r * CELL + 0.5, CELL - 1, CELL - 1);
    ctx.setLineDash([]);
  }
  $("lcd-notice").classList.add("hidden");
}

function regretHeat(norm) {
  return norm >= 0.5 ? "hot" : norm >= 0.15 ? "warm" : "";
}

async function loadJevMeta() {
  if (state.deck.jev) return;
  try {
    state.deck.jev = await (await fetch("/api/jev/questions")).json();
  } catch {
    state.deck.jev = { questions: [], groups: [], setup: { configured: false } };
  }
}

async function loadDeck(runId) {
  document.querySelectorAll(".cart").forEach((c) => c.classList.toggle("inserted", c.dataset.runId === runId));
  await loadJevMeta();
  const deck = state.deck;
  deck.runId = runId;
  const body = await (await fetch(`/api/runs/${runId}/decisions`)).json();
  deck.decisions = body.decisions;
  deck.index = 0;
  deck.answers = deck.prevAnswers = null;
  // The replay verifies a prefix; a tuck or misread ends it. Say how much of
  // the run is on the deck so a 5-of-50 never reads as a 5-piece run.
  const verified = body.decisions.length === body.placed ? "" : ` · ${body.decisions.length} of ${body.placed} verified`;
  $("label-file").textContent = `runs/${runId}/labels.json${verified}`;
  $("label-empty").classList.toggle("hidden", true);
  $("label-deck").classList.remove("hidden");
  renderDeckList();
  showDecision(0);
}

function renderDeckList() {
  const list = $("deck-list");
  list.replaceChildren();
  state.deck.decisions.forEach((d, i) => {
    const li = document.createElement("li");
    li.dataset.index = i;
    const norm = d.grade?.regret_norm ?? 0;
    const verdict = d.label?.verdict || "";
    li.innerHTML =
      `<span class="t">t${d.turn}</span><span class="p">${d.piece}</span>` +
      `<span class="bar"><i class="${regretHeat(norm)}" style="width:${Math.round(norm * 100)}%"></i></span>` +
      `<span class="r">${d.grade ? d.grade.regret.toFixed(1) : "—"}</span>` +
      `<span class="v ${verdict}">${verdict === "promote" ? "★" : verdict === "exclude" ? "✕" : ""}</span>`;
    li.onclick = () => showDecision(i);
    list.appendChild(li);
  });
  renderDeckCounts();
}

function renderDeckCounts() {
  const ds = state.deck.decisions;
  const promoted = ds.filter((d) => d.label?.verdict === "promote").length;
  const excluded = ds.filter((d) => d.label?.verdict === "exclude").length;
  $("d-counts").textContent = `${promoted} promoted · ${excluded} excluded`;
}

function currentDecision() {
  return state.deck.decisions[state.deck.index] || null;
}

function showDecision(i) {
  const deck = state.deck;
  if (!deck.decisions.length) return;
  deck.index = Math.max(0, Math.min(i, deck.decisions.length - 1));
  const d = currentDecision();
  drawBoard(d);
  $("d-pos").textContent = `${deck.index + 1}/${deck.decisions.length}`;
  $("d-piece").textContent = `${d.piece} → ${d.next_piece}`;
  const g = d.grade;
  const regretOut = $("d-regret");
  regretOut.textContent = g ? g.regret.toFixed(1) : "—";
  regretOut.className = g ? regretHeat(g.regret_norm).replace(/^(.)/, "regret-$1") : "";
  $("d-rank").textContent = g ? `${g.rank}/${g.legal_count}` : "—";
  $("d-chosen").textContent = `r${d.rotation} c${d.col}`;
  $("d-best").textContent = g ? `r${g.best[0]} c${g.best[1]}` : "—";
  $("d-result").textContent = d.lines_delta ? `${d.lines_delta} LINE${d.lines_delta > 1 ? "S" : ""}` : `${d.holes} HOLES`;
  const bar = $("d-regret-bar");
  bar.style.width = `${Math.round((g?.regret_norm ?? 0) * 100)}%`;
  bar.className = `regret-fill ${g ? regretHeat(g.regret_norm) : ""}`;
  renderVerdict();
  document.querySelectorAll("#deck-list li").forEach((li) => li.classList.toggle("current", Number(li.dataset.index) === deck.index));
  document.querySelector("#deck-list li.current")?.scrollIntoView({ block: "nearest" });
  askJev();
}

function renderVerdict() {
  const d = currentDecision();
  const verdict = d?.label?.verdict || "";
  const chip = $("d-verdict");
  chip.textContent = verdict ? `${verdict.toUpperCase()}${d.label.source === "jev" ? " · via jev" : ""}` : "";
  chip.className = `verdict-chip ${verdict}`;
  $("v-promote").classList.toggle("on", verdict === "promote");
  $("v-exclude").classList.toggle("on", verdict === "exclude");
  const proposal = state.deck.answers?.verdict?.choice;
  $("v-accept").classList.toggle("hidden", !(proposal === "promote" || proposal === "exclude") || proposal === verdict);
}

/* Jev: one call per decision, every question at once. A step cancels the
   pending debounce and aborts the request in flight, so paging quickly never
   queues stale work, and the meters keep the last answers until new ones land. */
function setJevStatus(text, cls) {
  const el = $("jev-status");
  el.textContent = text;
  el.className = `jev-status${cls ? ` ${cls}` : ""}`;
}

function askJev() {
  const deck = state.deck;
  clearTimeout(deck.timer);
  deck.asking?.abort();
  const setup = deck.jev?.setup;
  if (setup && !setup.configured) {
    setJevStatus(`needs ${setup.keyVar}`, "needs-setup");
    const notice = $("jev-notice");
    notice.innerHTML = `No TypeSafe key: set <code>${setup.keyVar}</code> in the viewer's environment ` +
      `(<a href="${setup.consoleUrl}" target="_blank" rel="noreferrer">get one</a>) and restart it. ` +
      `Labels still work without Jev.`;
    notice.classList.remove("hidden");
    renderMeters();
    return;
  }
  $("jev-notice").classList.add("hidden");
  const d = currentDecision();
  if (!d) return;
  const ctrl = new AbortController();
  deck.asking = ctrl;
  deck.timer = setTimeout(async () => {
    setJevStatus("asking…", "asking");
    const started = performance.now();
    try {
      const res = await fetch(`/api/runs/${deck.runId}/decisions/${d.turn}/judge`, { method: "POST", signal: ctrl.signal });
      const body = await res.text();
      let data = null;
      try { data = JSON.parse(body); } catch { /* an HTML error page; fall through to HTTP status */ }
      if (ctrl.signal.aborted) return;
      if (!res.ok || !data) {
        setJevStatus(data?.detail?.error || data?.detail?.reason || `HTTP ${res.status}`, "error");
        return;
      }
      deck.prevAnswers = deck.answers;
      deck.answers = data.answers;
      setJevStatus(`${Object.keys(data.answers).length} questions · ${Math.round(performance.now() - started)} ms`, "ok");
      renderMeters();
      renderVerdict();
    } catch (err) {
      if (err?.name === "AbortError") return;
      setJevStatus(String(err), "error");
    }
  }, JEV_DEBOUNCE_MS);
}

/* Answer → one 0–1 number for the bar: P(yes) for noul, position on the
   scale for score, the winner's probability for choice. */
function fillOf(meta, a) {
  if (!a) return 0;
  if (a.type === "noul") return a.noul;
  if (a.type === "score") return (meta.criteria?.length > 1) ? a.score / (meta.criteria.length - 1) : 0;
  if (a.type === "choice") return a.probabilities?.[a.choice] ?? a.confidence ?? 0;
  return 0;
}

function headlineOf(meta, a) {
  if (!a) return { text: "—", detail: "" };
  const pct = (p) => `${Math.round(p * 100)}%`;
  if (a.type === "noul") return { text: a.noul >= 0.5 ? "Yes" : "No", detail: pct(a.noul) };
  if (a.type === "score") {
    const levels = meta.criteria || [];
    const i = Math.max(0, Math.min(levels.length - 1, Math.round(a.score)));
    return { text: levels[i] ?? a.score.toFixed(2), detail: a.score.toFixed(2) };
  }
  if (a.type === "choice") return { text: a.choice, detail: pct(a.probabilities?.[a.choice] ?? a.confidence ?? 0) };
  return { text: "—", detail: "" };
}

function moved(prev, next) {
  if (!prev || !next || prev.type !== next.type) return false;
  if (next.type === "noul") return (prev.noul >= 0.5) !== (next.noul >= 0.5) || Math.abs(prev.noul - next.noul) >= 0.15;
  if (next.type === "score") return Math.abs(prev.score - next.score) >= 0.75;
  if (next.type === "choice") return prev.choice !== next.choice;
  return false;
}

function renderMeters() {
  const deck = state.deck;
  const root = $("jev-meters");
  root.replaceChildren();
  if (!deck.jev) return;
  for (const group of deck.jev.groups) {
    const box = document.createElement("div");
    box.className = "jev-group";
    box.innerHTML = `<div class="jev-group-head"><b>${group.title.toUpperCase()}</b><span>${group.blurb}</span></div>`;
    for (const meta of deck.jev.questions.filter((q) => q.group === group.id)) {
      const a = deck.answers?.[meta.id];
      const { text, detail } = headlineOf(meta, a);
      const row = document.createElement("div");
      row.className = "jev-row";
      row.dataset.q = meta.id;
      if (a?.type === "noul" && a.noul < 0.5) row.classList.add("no");
      if (moved(deck.prevAnswers?.[meta.id], a)) row.classList.add("flash");
      row.innerHTML =
        `<span class="jev-label">${meta.label}</span>` +
        `<span class="jev-bar"><i class="jev-fill" style="width:${Math.round(fillOf(meta, a) * 100)}%"></i></span>` +
        `<span class="jev-answer" data-value="${text}">${text}${detail ? `<small>${detail}</small>` : ""}</span>`;
      box.appendChild(row);
    }
    root.appendChild(box);
  }
}

/* Verdicts: written to runs/<id>/labels.json, which the exemplar miner reads. */
async function setVerdict(verdict, source = "human") {
  const deck = state.deck;
  const d = currentDecision();
  if (!d) return;
  const url = `/api/runs/${deck.runId}/labels/${d.turn}`;
  let res;
  if (verdict) {
    res = await fetch(url, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ verdict, source, jev: deck.answers }),
    });
    if (res.ok) d.label = await res.json();
  } else {
    res = await fetch(url, { method: "DELETE" });
    if (res.ok || res.status === 404) d.label = null;
  }
  if (!res.ok && res.status !== 404) { setJevStatus(`label HTTP ${res.status}`, "error"); return; }
  renderVerdict();
  const li = document.querySelector(`#deck-list li[data-index="${deck.index}"] .v`);
  if (li) {
    const v = d.label?.verdict || "";
    li.className = `v ${v}`;
    li.textContent = v === "promote" ? "★" : v === "exclude" ? "✕" : "";
  }
  renderDeckCounts();
}

$("d-prev").onclick = () => showDecision(state.deck.index - 1);
$("d-next").onclick = () => showDecision(state.deck.index + 1);
$("v-promote").onclick = () => setVerdict("promote");
$("v-exclude").onclick = () => setVerdict("exclude");
$("v-clear").onclick = () => setVerdict(null);
$("v-accept").onclick = () => {
  const proposal = state.deck.answers?.verdict?.choice;
  if (proposal === "promote" || proposal === "exclude") setVerdict(proposal, "jev");
};

window.addEventListener("keydown", (e) => {
  if (state.mode !== "label" || !state.deck.decisions.length) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
  const actions = {
    ArrowLeft: () => showDecision(state.deck.index - 1),
    ArrowRight: () => showDecision(state.deck.index + 1),
    p: () => setVerdict("promote"),
    x: () => setVerdict("exclude"),
    c: () => setVerdict(null),
    j: () => $("v-accept").click(),
  };
  const action = actions[e.key];
  if (!action) return;
  e.preventDefault();
  action();
});

/* ── mode switching ── */
function setMode(mode) {
  state.mode = mode;
  for (const m of MODES) $(`btn-${m}`).classList.toggle("active", mode === m);
  $("shelf").classList.toggle("hidden", mode !== "replay" && mode !== "label");
  $("replay-deck").classList.toggle("hidden", mode !== "replay");
  $("bench-panel").classList.toggle("hidden", mode !== "bench");
  $("telemetry").classList.toggle("hidden", mode === "label");
  $("label-panel").classList.toggle("hidden", mode !== "label");
  $("lcd-notice").classList.remove("hidden");
  resetHud();
  if (mode !== "label") { clearTimeout(state.deck.timer); state.deck.asking?.abort(); }
  if (mode === "replay") { setPlaying(false); loadShelf(); setPlayMode(false); }
  else if (mode === "bench") { setPlaying(false); loadBench(); setPlayMode(false); }
  else if (mode === "label") {
    setPlaying(false); setPlayMode(false);
    loadShelf().then(() => { if (state.deck.runId) loadDeck(state.deck.runId); });
  }
  else { connectLive(); connectInput(); }
}

const MODES = ["live", "replay", "bench", "label"];
$("btn-live").onclick = () => setMode("live");
$("btn-replay").onclick = () => setMode("replay");
$("btn-bench").onclick = () => setMode("bench");
$("btn-label").onclick = () => setMode("label");
$("pad-select").onclick = () => setMode(MODES[(MODES.indexOf(state.mode) + 1) % MODES.length]);
$("pad-start").onclick = () => state.run && setPlaying(!state.playing);
$("t-play").onclick = () => state.run && setPlaying(!state.playing);
$("t-speed").onclick = cycleSpeed;
$("pad-a").onclick = cycleSpeed;
$("pad-b").onclick = () => { state.speed = 1; $("t-speed").textContent = "1×"; if (state.playing) setPlaying(true); };
$("t-scrub").oninput = (e) => { setPlaying(false); showFrame(parseInt(e.target.value, 10)); };

renderHud();
setMode("live");
