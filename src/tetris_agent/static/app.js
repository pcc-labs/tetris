/* GRAMBOY bench viewer: LIVE (ws) and REPLAY (cartridges over runs/). */

const $ = (id) => document.getElementById(id);
const canvas = $("screen");
const ctx = canvas.getContext("2d");
ctx.imageSmoothingEnabled = false;

const state = {
  mode: "live",          // "live" | "race" | "replay" | "bench"
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
  // A race streams one arm per slot. Every producer stamps its slot; anything
  // without one is slot 0, which is the single-screen LIVE tab — so one map
  // serves both layouts and old producers need no changes.
  slots: new Map(),
};

const blankHud = () => ({ score: 0, lines: 0, level: 0, piece: 0, holes: 0, misexec: 0 });

function slotState(slot) {
  if (!state.slots.has(slot)) {
    state.slots.set(slot, { hud: blankHud(), armMax: 0, model: "", sub: "", status: "", statusCls: "" });
  }
  return state.slots.get(slot);
}

/* ── HUD ── */
function pad(n, w) { return String(n).padStart(w, "0"); }
function renderHud() {
  const h = slotState(0).hud;
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

// `solo` is the single-screen surface: the big Game Boy plus the telemetry
// panel's ticker and banner. In RACE mode nothing is solo — every slot,
// including 0, is one tile in the grid.
// `live` marks an event arriving off the wire. A replayed run carries its own
// session-start, and only the live one may seize the tiles back — otherwise a
// replay tears itself down on its first frame.
function applyEvent(e, slot = 0, live = false) {
  const d = e.data || {};
  const s = slotState(slot);
  const solo = state.mode !== "race" && slot === 0;
  const say = (text, cls) => { if (solo) tick({ turn: e.turn ?? 0, text }, cls); };
  switch (e.event_type) {
    case "piece_spawn":
      s.hud.piece = e.turn;
      say(`spawn ${d.piece} (next ${d.next_piece})`);
      if (solo && s.armMax) $("arm-progress").textContent = `piece ${e.turn}/${s.armMax}`;
      setStatus(slot, "THINKING…", "think");
      break;
    case "placement_decision": {
      say(`plan rot=${d.rotation} col=${d.col}`);
      if (d.reason) say(`“${d.reason}”`, "reason");
      const secs = d.latency_ms != null ? ` in ${(d.latency_ms / 1000).toFixed(1)}s` : "";
      if (d.late) setStatus(slot, `TOO SLOW${secs} — dropped`, "late");
      else setStatus(slot, `placed${secs}`, "ok");
      break;
    }
    case "piece_locked":
      s.hud.lines += d.lines_delta || 0;
      s.hud.holes = d.holes ?? s.hud.holes;
      s.hud.misexec += d.misexec || 0;
      if (d.score) s.hud.score = d.score;
      s.hud.level = Math.floor(s.hud.lines / 10);
      if (d.lines_delta) say(`LINE CLEAR ×${d.lines_delta}`, "warn");
      break;
    case "stuck":
      say(`STUCK: ${d.detail}`, "bad");
      break;
    case "game_over":
      say(`game over — score ${d.fitness?.score}`, "bad");
      break;
    case "session":
      say(`session ${d.phase}${d.policy ? ` — ${d.policy}` : ""}`);
      if (d.phase === "start") {
        // Only a race owns the RACE grid. A lone live session — tetris-play,
        // or a bare --live agent — belongs to the LIVE tab and must not evict
        // the race the grid is showing.
        s.isRace = d.lane != null;
        if (live && s.isRace && state.mode === "race") stopRaceReplay();
        s.hud = blankHud();
        s.armMax = d.max_pieces || 0;
        s.model = d.model || d.policy || "";
        s.arm = d.arm || s.model;     // full identity, matching the leaderboard row
        s.runId = d.run_id || null;   // where this lane's frames land, for REPLAY
        // Harness and effort first: in a four-model race they are what tells
        // two lanes of the same model apart, and the demo has to be callable
        // out loud from across a room.
        s.sub = [
          d.harness,
          d.effort && `effort ${d.effort}`,
          d.host && `on ${d.host}`,   // which box answered; absent for a solver
          d.mode,
          d.seed != null && `seed ${d.seed}`,
        ].filter(Boolean).join(" · ");
        s.status = "";
        s.statusCls = "";
        if (solo) { setArm(d.policy); showArmBanner(d); }
      }
      if (solo && d.fitness?.policy?.cost_usd) {
        say(`cost $${d.fitness.policy.cost_usd.toFixed(4)}`, "warn");
      }
      if (d.phase === "end" && d.fitness) {
        setStatus(slot, `DONE — score ${d.fitness.score ?? 0}`, "ok");
      }
      // A browser-play session hands the keyboard to the viewer.
      if (solo && d.phase === "start" && d.surface === "browser") setPlayMode(true);
      if (solo && d.phase === "end") setPlayMode(false);
      break;
  }
  if (state.mode === "race") { if (s.isRace) renderTile(slot); }
  else if (slot === 0) renderHud();
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
  slotState(0).hud = blankHud();
  $("ticker").replaceChildren();
  renderHud();
}

/* ── arm banner: who is playing right now ── */
function setArmStatus(text, cls) {
  const el = $("arm-status");
  el.textContent = text;
  el.className = `arm-status${cls ? ` ${cls}` : ""}`;
}

// Status goes to whichever surface is showing this slot.
function setStatus(slot, text, cls) {
  const s = slotState(slot);
  s.status = text;
  s.statusCls = cls || "";
  if (state.mode === "race") renderTile(slot);
  else if (slot === 0) setArmStatus(text, cls);
}

function showArmBanner(d) {
  // Benchmark arms announce their identity on session start; human play and
  // bare agent sessions don't, and keep the plain telemetry layout.
  if (!d.model) return;
  const s = slotState(0);
  resetHud();
  $("arm-model").textContent = s.model;
  $("arm-sub").textContent = s.sub;
  $("arm-progress").textContent = s.armMax ? `piece 0/${s.armMax}` : "";
  setArmStatus("");
  $("arm-banner").classList.remove("hidden");
}

/* ── screen ── */
function drawDataUrl(url, slot = 0) {
  // Kept so a tab switch can repaint the tile it just rebuilt: a lane that has
  // finished sends no more frames, and would otherwise sit blank forever.
  slotState(slot).lastFrame = url;
  if (state.mode === "race") {
    // Race lanes only, and never over a running replay.
    if (slotState(slot).isRace && !replay.playing) paintTile(slot, url);
    return;
  }
  if (slot !== 0) return;  // a race streaming past the single-screen tab
  const img = new Image();
  img.onload = () => { ctx.drawImage(img, 0, 0, 160, 144); };
  img.src = url;
  $("lcd-notice").classList.add("hidden");
}

/* ── RACE mode: one tile per slot ── */
// Tiles are built on demand, in slot order, so the grid fills left-to-right
// however the lanes happen to announce themselves.
function raceTile(slot) {
  const grid = $("race-grid");
  let tile = state.slots.get(slot)?.tile;
  if (tile) return tile;

  const el = document.createElement("div");
  el.className = "race-tile";
  el.dataset.slot = slot;
  el.innerHTML = `
    <div class="race-arm"></div>
    <div class="race-sub"></div>
    <div class="race-lcd"><canvas width="160" height="144"></canvas></div>
    <div class="race-readouts">
      <div><label>SCORE</label><output class="r-score">000000</output></div>
      <div><label>LINES</label><output class="r-lines">0000</output></div>
      <div><label>PIECE</label><output class="r-piece">0</output></div>
    </div>
    <div class="race-status arm-status"></div>`;

  const canvas = el.querySelector("canvas");
  const ctx2 = canvas.getContext("2d");
  ctx2.imageSmoothingEnabled = false;
  tile = {
    el, ctx: ctx2,
    arm: el.querySelector(".race-arm"),
    sub: el.querySelector(".race-sub"),
    score: el.querySelector(".r-score"),
    lines: el.querySelector(".r-lines"),
    piece: el.querySelector(".r-piece"),
    status: el.querySelector(".race-status"),
  };
  slotState(slot).tile = tile;

  // Keep the DOM in slot order regardless of arrival order.
  const after = [...grid.children].find((c) => Number(c.dataset.slot) > slot);
  grid.insertBefore(el, after || null);
  $("race-empty").classList.add("hidden");
  $("race-deck").classList.remove("hidden");
  layoutRaceGrid();
  return tile;
}

// The grid takes its shape from however many lanes actually raced: three
// models go three across, four fall back to a 2x2. CSS keys off the count so
// the narrow-screen media query can still override it.
function layoutRaceGrid() {
  const grid = $("race-grid");
  grid.dataset.lanes = grid.children.length;
}

function paintTile(slot, url) {
  const tile = raceTile(slot);
  const img = new Image();
  img.onload = () => { tile.ctx.drawImage(img, 0, 0, 160, 144); };
  img.src = url;
}

function renderTile(slot) {
  const s = slotState(slot);
  const t = raceTile(slot);
  t.arm.textContent = s.model || `lane ${slot + 1}`;
  t.el.title = s.arm || s.model || "";   // full arm name on hover
  t.sub.textContent = s.sub;
  t.score.textContent = pad(s.hud.score, 6);
  t.lines.textContent = pad(s.hud.lines, 4);
  t.piece.textContent = s.armMax ? `${s.hud.piece}/${s.armMax}` : String(s.hud.piece);
  t.status.textContent = s.status;
  t.status.className = `race-status arm-status${s.statusCls ? ` ${s.statusCls}` : ""}`;
}

/* ── RACE replay: all four lanes scrubbing on one timeline ── */
// Lanes share a seed, and frames are captured on the same cadence in real
// time, so one index reads as one moment across the grid. A lane that topped
// out early holds on its last frame while the others play on — which is
// exactly what it looked like live.
const replay = {
  frames: new Map(), events: new Map(),
  index: 0, max: 0, playing: false, speed: 1, timer: null,
};

// Lanes to replay: the ones this tab watched, or — after a reload, or in a tab
// that never saw the race — the last recorded race on disk.
async function raceLanes() {
  const live = [...state.slots.entries()].filter(([, s]) => s.runId);
  if (live.length) return live.map(([slot, s]) => [slot, s.runId]);

  const races = (await (await fetch("/api/benchmarks")).json()).filter((r) => r.race && r.lane_runs?.length);
  if (!races.length) return [];
  races[0].lane_runs.forEach((lane, i) => {
    const s = slotState(i);
    s.isRace = true;
    s.model = lane.arm;   // the full arm identity, until the run's own events refine it
    s.sub = "recorded race";
  });
  return races[0].lane_runs.map((lane, i) => [i, lane.run_id]);
}

// `quiet`: an automatic restore on entering the tab, which must leave the
// empty-state message alone when there is genuinely nothing to show.
async function loadRaceReplay({ quiet = false } = {}) {
  if (!quiet) raceNote("loading…");
  const lanes = await raceLanes();
  if (!lanes.length) {
    if (!quiet) raceNote("nothing recorded — re-run the race with --record");
    return;
  }
  $("race-empty").classList.add("hidden");
  $("race-deck").classList.remove("hidden");
  replay.frames.clear();
  replay.events.clear();
  await Promise.all(lanes.map(async ([slot, runId]) => {
    const run = await (await fetch(`/api/runs/${runId}`)).json();
    if (run.frames?.length) replay.frames.set(slot, run.frames);
    replay.events.set(slot, run.events || []);
    renderTile(slot);  // a replay-only tab has no tile for this lane yet
  }));

  replay.max = Math.max(0, ...[...replay.frames.values()].map((f) => f.length));
  if (!replay.max) {
    // The run exists but holds no frames: it was recorded for grading only.
    raceNote("this race kept no frames — re-run it with --record");
    return;
  }
  $("race-scrub").max = replay.max - 1;
  raceNote(`${replay.frames.size} lanes · ${replay.max} frames`);
  showRaceFrame(0);
  setRacePlaying(true);
}

function showRaceFrame(i) {
  replay.index = Math.max(0, Math.min(i, replay.max - 1));
  for (const [slot, frames] of replay.frames) {
    // Clamp, don't blank: a finished lane keeps its final board on screen.
    const frame = frames[Math.min(replay.index, frames.length - 1)];
    paintTile(slot, frame);
    // Rebuild this lane's readouts from its own events, the way the
    // single-screen REPLAY does — the frame name carries the turn it belongs
    // to, and the readouts are meaningless frozen at zero.
    const events = replay.events.get(slot);
    if (!events?.length) continue;
    const turn = turnOfFrame(frame);
    const s = slotState(slot);
    s.hud = blankHud();
    for (const e of events) {
      if ((e.turn ?? 0) <= turn) applyEvent(e, slot);
    }
  }
  $("race-scrub").value = replay.index;
  $("race-pos").textContent = `${replay.index + 1}/${replay.max}`;
}

function setRacePlaying(playing) {
  replay.playing = playing;
  $("race-play").textContent = playing ? "❚❚" : "▶";
  clearInterval(replay.timer);
  if (!playing) return;
  replay.timer = setInterval(() => {
    if (replay.index >= replay.max - 1) { setRacePlaying(false); return; }
    showRaceFrame(replay.index + 1);
  }, 200 / replay.speed);
}

function raceNote(text) { $("race-deck-note").textContent = text; }

// A live race outranks a replay: the tiles go back to the wire, and whatever
// was loaded is dropped so REPLAY reloads the race now being run.
function stopRaceReplay() {
  if (!replay.max && !replay.playing) return;
  setRacePlaying(false);
  replay.frames.clear();
  replay.events.clear();
  replay.max = 0;
  replay.index = 0;
  $("race-scrub").max = 0;
  $("race-pos").textContent = "0/0";
  raceNote("live");
}

function cycleRaceSpeed() {
  replay.speed = replay.speed >= 4 ? 1 : replay.speed * 2;
  $("race-speed").textContent = `${replay.speed}×`;
  if (replay.playing) setRacePlaying(true);
}

function buildRaceGrid() {
  $("race-grid").replaceChildren();
  for (const s of state.slots.values()) delete s.tile;
  // Only lanes that actually announced themselves: slot 0 exists from the
  // moment the HUD first renders, and an empty grid should say so rather than
  // show a phantom tile.
  const slots = [...state.slots.entries()]
    .filter(([, s]) => s.isRace)
    .map(([slot]) => slot)
    .sort((a, b) => a - b);
  for (const slot of slots) {
    renderTile(slot);
    const last = state.slots.get(slot).lastFrame;
    if (last) paintTile(slot, last);
  }
  $("race-empty").classList.toggle("hidden", slots.length > 0);
  $("race-deck").classList.toggle("hidden", slots.length === 0);
  // A refresh must not leave the tab staring at an empty panel: with nothing
  // streaming, fall straight back to the last recorded race and play it.
  if (!slots.length) loadRaceReplay({ quiet: true });
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
    if (state.mode !== "live" && state.mode !== "race") return;
    const msg = JSON.parse(evt.data);
    const slot = msg.slot ?? 0;  // producers predating slots are the single screen
    $("power-led").classList.add("on");
    $("feed-state").textContent = "LIVE";
    $("feed-state").classList.add("live");
    if (msg.type === "frame") drawDataUrl(`data:image/png;base64,${msg.png}`, slot);
    else if (msg.type === "event") applyEvent(msg.event, slot, true);
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
  const raced = latest.race ? ` · race, ${latest.lanes ?? "?"} lanes (latency columns contended)` : "";
  $("bench-stamp").textContent = latest.recorded_at.slice(0, 19).replace("T", " ") + " UTC" + raced;
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

/* ── mode switching ── */
function setMode(mode) {
  state.mode = mode;
  for (const m of MODES) $(`btn-${m}`).classList.toggle("active", mode === m);
  $("shelf").classList.toggle("hidden", mode !== "replay");
  $("replay-deck").classList.toggle("hidden", mode !== "replay");
  $("bench-panel").classList.toggle("hidden", mode !== "bench");
  $("race-panel").classList.toggle("hidden", mode !== "race");
  // The single hero device and its telemetry make no sense beside the grid.
  $("gameboy").classList.toggle("hidden", mode === "race");
  $("telemetry").classList.toggle("hidden", mode === "race");
  $("lcd-notice").classList.remove("hidden");
  // resetHud clears slot 0, which in RACE is lane 1's tile — switching tabs
  // must not wipe a lane that is still playing.
  if (mode !== "race") resetHud();
  if (mode === "replay") { setPlaying(false); loadShelf(); setPlayMode(false); }
  else if (mode === "bench") { setPlaying(false); loadBench(); setPlayMode(false); }
  else if (mode === "race") { setPlaying(false); setPlayMode(false); buildRaceGrid(); connectLive(); }
  else { connectLive(); connectInput(); }
}

const MODES = ["live", "race", "replay", "bench"];
$("btn-live").onclick = () => setMode("live");
$("btn-race").onclick = () => setMode("race");
$("btn-replay").onclick = () => setMode("replay");
$("btn-bench").onclick = () => setMode("bench");
$("pad-select").onclick = () => setMode(MODES[(MODES.indexOf(state.mode) + 1) % MODES.length]);
$("pad-start").onclick = () => state.run && setPlaying(!state.playing);
$("t-play").onclick = () => state.run && setPlaying(!state.playing);
$("t-speed").onclick = cycleSpeed;
$("pad-a").onclick = cycleSpeed;
$("pad-b").onclick = () => { state.speed = 1; $("t-speed").textContent = "1×"; if (state.playing) setPlaying(true); };
$("t-scrub").oninput = (e) => { setPlaying(false); showFrame(parseInt(e.target.value, 10)); };

$("race-replay").onclick = loadRaceReplay;
$("race-play").onclick = () => replay.max && setRacePlaying(!replay.playing);
$("race-speed").onclick = cycleRaceSpeed;
$("race-scrub").oninput = (e) => { setRacePlaying(false); showRaceFrame(parseInt(e.target.value, 10)); };

renderHud();
setMode("live");
