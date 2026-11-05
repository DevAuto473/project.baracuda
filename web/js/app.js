// Barracuda web player: UI + game flow. The engine (WebAssembly) lives in engine-worker.js.
import { pieceSvg } from './pieces.js?v=__BUILD__';
import { t, getLang, initLang, setLang } from './i18n.js?v=__BUILD__';
import { NetViz } from './netviz.js?v=__BUILD__';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const LEVELS = [
  // minMs: the move is never played sooner than this after the player's move. The three weakest levels
  // search almost instantly, so they wait one second; otherwise the piece appears to jump by itself.
  { key: 'lvl1', depth: 1, ms: 60, minMs: 1000 },
  { key: 'lvl2', depth: 3, ms: 150, minMs: 1000 },
  { key: 'lvl3', depth: 5, ms: 400, minMs: 1000 },
  // Levels 4-6 use their whole time budget: about 1 s, 2 s and 4 s per move on a typical computer.
  { key: 'lvl4', depth: 64, ms: 1300, minMs: 0 },
  { key: 'lvl5', depth: 64, ms: 2600, minMs: 0 },
  { key: 'lvl6', depth: 64, ms: 5000, minMs: 0 },
];

const $ = (id) => document.getElementById(id);
const els = {
  board: $('board'), status: $('status'), level: $('level'), color: $('color'), moves: $('moves'),
  evalWhite: $('evalWhite'), evalLabel: $('evalLabel'), evalbar: $('evalbar'),
  thinking: $('thinking'), depth: $('depth'), nps: $('nps'), pv: $('pv'),
  promo: $('promo'), promoBox: $('promoBox'),
};

// ---------------------------------------------------------------- engine client
class EngineClient {
  constructor() {
    this.worker = new Worker('js/engine-worker.js?v=__BUILD__');
    this.pending = new Map();
    this.seq = 0;
    this.onInfo = null;
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'info') { if (this.onInfo) this.onInfo(m); return; }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      m.error ? p.reject(new Error(m.error)) : p.resolve(m.result);
    };
    this.worker.onerror = (e) => { for (const p of this.pending.values()) p.reject(new Error(e.message || 'worker error')); this.pending.clear(); };
  }
  send(msg, transfer) {
    return new Promise((resolve, reject) => {
      const id = ++this.seq;
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, ...msg }, transfer || []);
    });
  }
  init(net, useNnue) { return this.send({ cmd: 'init', net, useNnue }, net ? [net] : []); }
  call(fn, ...args) { return this.send({ cmd: 'call', fn, args }); }
}

// ---------------------------------------------------------------- state
const state = {
  startFen: START_FEN, moves: [], sans: [], fen: START_FEN, legal: [], inCheck: false,
  status: 'ongoing', human: 'w', flipped: false, last: null, selected: null,
  thinking: false, token: 0, ready: false, evalWhite: 0, drag: null, error: null,
  finished: false, thanks: null, thanksTimer: 0,
};
let engine;
let viz = null;
let collect = null;      // {type: 'googleform'|'http', ...} from config.json, or null when nothing is collected
let customFen = null;    // optional ?fen=... start position
let manifestInfo = null;

const turn = () => state.fen.split(' ')[1];
const movesStr = () => state.moves.join(' ');
const isOver = () => state.status !== 'ongoing';
const humanToMove = () => !isOver() && turn() === state.human;
const canInteract = () => state.ready && humanToMove() && !state.thinking;

function parseBoard(fen) {
  const rows = fen.split(' ')[0].split('/');
  const b = new Array(64).fill(null);
  rows.forEach((row, i) => {
    let f = 0;
    for (const ch of row) {
      if (ch >= '1' && ch <= '8') f += +ch;
      else { b[(7 - i) * 8 + f] = ch; f++; }
    }
  });
  return b; // index = rank*8 + file, a1 = 0
}
const sqName = (i) => 'abcdefgh'[i % 8] + (1 + (i >> 3));
const sqIndex = (name) => (name.charCodeAt(0) - 97) + (name.charCodeAt(1) - 49) * 8;
const isWhitePiece = (p) => p === p.toUpperCase();

function destinationsFrom(from) {
  const name = sqName(from);
  const out = new Map();
  for (const m of state.legal) if (m.startsWith(name)) out.set(m.slice(2, 4), true);
  return out;
}

// ---------------------------------------------------------------- rendering
function render() {
  const b = parseBoard(state.fen);
  const dests = state.selected != null && canInteract() ? destinationsFrom(state.selected) : new Map();
  let kingInCheck = -1;
  if (state.inCheck) kingInCheck = b.findIndex((p) => p === (turn() === 'w' ? 'K' : 'k'));
  let html = '';
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const file = state.flipped ? 7 - col : col;
      const rank = state.flipped ? row : 7 - row;
      const i = rank * 8 + file;
      const light = (file + rank) % 2 === 1;
      const cls = ['sq'];
      if (light) cls.push('l');
      if (state.last && (state.last[0] === i || state.last[1] === i)) cls.push('last');
      if (i === state.selected) cls.push('sel');
      if (i === kingInCheck) cls.push('check');
      if (state.drag && state.drag.from === i) cls.push('drag');
      let inner = '';
      if (col === 0) inner += `<span class="coord r">${rank + 1}</span>`;
      if (row === 7) inner += `<span class="coord f">${'abcdefgh'[file]}</span>`;
      if (b[i]) inner += pieceSvg(b[i]);
      if (dests.has(sqName(i))) inner += `<span class="hint${b[i] ? ' cap' : ''}"></span>`;
      html += `<div class="${cls.join(' ')}" data-sq="${i}">${inner}</div>`;
    }
  }
  els.board.innerHTML = html;
  els.evalbar.classList.toggle('flipped', state.flipped);
  renderMoves();
  renderStatus();
  renderEval();
}

function renderMoves() {
  let h = '';
  const first = Number(state.startFen.split(' ')[5] || 1);
  const blackStarts = state.startFen.split(' ')[1] === 'b';
  state.sans.forEach((san, i) => {
    const ply = i + (blackStarts ? 1 : 0);
    if (ply % 2 === 0) h += `<li class="no">${first + (ply >> 1)}.</li>`;
    else if (i === 0) h += `<li class="no">${first}.</li><li>…</li>`;
    h += `<li class="${i === state.sans.length - 1 ? 'cur' : ''}">${san}</li>`;
  });
  els.moves.innerHTML = h;
  els.moves.scrollTop = els.moves.scrollHeight;
}

function renderThinking() {
  els.thinking.textContent = t(state.thinking ? 'thinkingNow' : 'idle');
  renderViz();
}

function renderViz() {
  if (!viz) return;
  viz.setThinking(state.thinking);
  const status = $('vizStatus');
  status.classList.toggle('on', state.thinking);
  status.textContent = state.thinking ? t('viz.thinking') : netInfo.hasNet ? t('viz.idle') : '';
  $('vizNote').textContent = !netInfo.hasNet ? t('viz.none') : t(netInfo.nnue ? 'viz.used' : 'viz.notUsed');
}

function renderStatus() {
  const s = els.status;
  s.className = 'status';
  renderThinking();
  if (state.error) { s.textContent = state.error; return; }
  if (!state.ready) { s.textContent = t('loading'); return; }
  if (isOver()) {
    const [res, ...rest] = state.status.split(' ');
    const why = rest.join(' ');
    const you = (res === '1-0' && state.human === 'w') || (res === '0-1' && state.human === 'b');
    s.classList.add('over');
    s.textContent = `${res} — ${t('reason.' + why)}${res === '1/2-1/2' ? '' : you ? t('youWin') : t('engineWins')}`;
  } else if (state.thinking) {
    s.textContent = t('engineThinking');
  } else {
    s.classList.add('you');
    s.textContent = t('yourMove') + (state.inCheck ? t('inCheck') : '');
  }
}

function renderEval() {
  const cp = state.evalWhite;
  const mate = Math.abs(cp) >= 29000;
  const share = mate ? (cp > 0 ? 0.97 : 0.03) : Math.min(0.97, Math.max(0.03, 1 / (1 + Math.exp(-cp / 300))));
  els.evalWhite.style.height = `${(share * 100).toFixed(1)}%`;
  els.evalLabel.textContent = mate ? t('mate') : (Math.abs(cp) / 100).toFixed(1);
}

// ---------------------------------------------------------------- game flow
async function refresh() {
  const [fen, legal, status, chk] = await Promise.all([
    engine.call('fen', state.startFen, movesStr()),
    engine.call('legal', state.startFen, movesStr()),
    engine.call('status', state.startFen, movesStr()),
    engine.call('inCheck', state.startFen, movesStr()),
  ]);
  state.fen = fen;
  state.legal = legal ? legal.split(' ') : [];
  state.status = status;
  state.inCheck = !!chk;
  if (viz && netInfo.hasNet && !state.thinking) {
    try {
      const j = await engine.call('netView', state.startFen, movesStr());
      if (j) viz.setView(JSON.parse(j));
    } catch (_) { /* visualisation is optional */ }
  }
}

async function newGame() {
  state.token++;
  state.thinking = false;
  let side = els.color.value;
  if (side === 'r') side = Math.random() < 0.5 ? 'w' : 'b';
  state.human = side;
  state.flipped = side === 'b';
  hideThanks();
  state.finished = false;
  state.startFen = customFen || START_FEN;
  state.moves = []; state.sans = []; state.last = null; state.selected = null; state.evalWhite = 0;
  els.pv.innerHTML = '&nbsp;'; els.depth.textContent = '-'; els.nps.textContent = '-'; renderThinking();
  // clear the old game right away instead of after the engine round-trips
  state.fen = state.startFen; state.legal = []; state.status = 'ongoing'; state.inCheck = false;
  render();
  await engine.call('newGame');
  await refresh();
  render();
  maybeEngineMove();
}

async function playMove(uci, byEngine) {
  const san = await engine.call('san', state.startFen, movesStr(), uci);
  const mover = parseBoard(state.fen)[sqIndex(uci.slice(0, 2))]; // the piece being moved (FEN letter)
  state.moves.push(uci);
  state.sans.push(san);
  state.last = [sqIndex(uci.slice(0, 2)), sqIndex(uci.slice(2, 4))];
  state.selected = null;
  await refresh();
  render();
  if (byEngine && viz && mover) viz.decide(mover, san); // the chosen piece forms out of the network
  if (isOver()) handleGameOver();
  if (!byEngine) maybeEngineMove();
}

async function maybeEngineMove() {
  if (!state.ready || isOver() || turn() === state.human || state.thinking) return;
  const token = ++state.token;
  const lvl = LEVELS[+els.level.value];
  state.thinking = true;
  renderThinking();
  renderStatus();
  let res;
  const started = performance.now();
  try {
    res = await engine.call('search', state.startFen, movesStr(), lvl.ms, lvl.depth);
  } catch (err) {
    state.thinking = false;
    state.error = t('engineError') + err.message;
    renderStatus();
    return;
  }
  if (token !== state.token) return; // cancelled (new game / undo)
  // fast levels answer in a few milliseconds: keep "thinking" (and the network animation) going so
  // the move appears after a natural pause instead of jumping instantly
  const wait = (lvl.minMs || 0) - (performance.now() - started);
  if (wait > 0) {
    await new Promise((r) => setTimeout(r, wait));
    if (token !== state.token) return;
  }
  state.thinking = false;
  renderThinking();
  const [best, score] = res.split(' ');
  if (best === 'none') { await refresh(); render(); return; }
  state.evalWhite = Number(score);
  await playMove(best, true);
}

async function undo() {
  if (!state.ready || !state.moves.length) return;
  state.token++; // drop any running search
  state.thinking = false;
  state.finished = false;
  hideThanks();
  renderThinking();
  // go back to the last position where it was the human's turn
  do {
    state.moves.pop();
    state.sans.pop();
    await refresh();
  } while (state.moves.length && turn() !== state.human);
  if (!state.moves.length) state.last = null;
  else { const m = state.moves[state.moves.length - 1]; state.last = [sqIndex(m.slice(0, 2)), sqIndex(m.slice(2, 4))]; }
  state.selected = null;
  render();
  maybeEngineMove(); // when the human plays Black and undoes the first move, the engine opens again
}

// ---------------------------------------------------------------- learning from games
const shareEnabled = () => {
  try { return localStorage.getItem('barracuda-share') !== '0'; } catch (_) { return true; }
};

function gameRecord() {
  const [result] = state.status.split(' ');
  return {
    v: 1,
    id: (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2)),
    t: new Date().toISOString(),
    start: state.startFen === START_FEN ? 'startpos' : state.startFen,
    moves: movesStr(),
    result,                       // 1-0, 0-1 or 1/2-1/2 (White's point of view)
    human: state.human,
    level: +els.level.value,
    eval: manifestInfo ? manifestInfo.eval : 'classical',
    net: manifestInfo ? manifestInfo.version : 0,
  };
}

// Fire-and-forget upload of the anonymous game record (moves + result only). Returns true when sent.
async function submitGame(rec) {
  if (!collect || !shareEnabled() || !rec.moves) return false;
  try {
    if (collect.type === 'googleform') {
      await fetch(collect.formUrl, { method: 'POST', mode: 'no-cors', headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ [collect.entry]: JSON.stringify(rec) }) });
    } else if (collect.type === 'http') {
      await fetch(collect.url, { method: 'POST', mode: 'no-cors', headers: { 'Content-Type': 'text/plain;charset=UTF-8' }, body: JSON.stringify(rec) });
    } else {
      return false;
    }
    return true;
  } catch (_) {
    return false; // offline or blocked: the game is simply not saved
  }
}

const engineWon = () => {
  const [res] = state.status.split(' ');
  return (res === '1-0' && state.human === 'b') || (res === '0-1' && state.human === 'w');
};

function handleGameOver() {
  if (state.finished) return;
  state.finished = true;
  const sending = submitGame(gameRecord());
  if (!engineWon()) return;
  // let the player see the final position for a moment, then turn the board black and say thanks
  const pause = new Promise((r) => { state.thanksTimer = setTimeout(r, 1500); });
  Promise.all([sending, pause]).then(([sent]) => { if (state.finished && isOver()) showThanks(sent); });
}

function renderThanks() {
  if (!state.thanks) return;
  const { sent } = state.thanks;
  $('thanksTitle').textContent = t('thanks.title');
  // the "learned from you" message is only shown when the game really was saved for training
  $('thanksMsg').textContent = t(sent ? 'thanks.msg' : 'thanks.plain');
  $('thanksNote').textContent = sent ? t('thanks.sent') : collect && !shareEnabled() ? t('thanks.off') : '';
}
function showThanks(sent) {
  state.thanks = { sent };
  renderThanks();
  const o = $('boardOverlay');
  o.hidden = false;
  requestAnimationFrame(() => o.classList.add('show'));
}
function hideThanks() {
  clearTimeout(state.thanksTimer);
  state.thanks = null;
  const o = $('boardOverlay');
  o.classList.remove('show');
  o.hidden = true;
}

// ---------------------------------------------------------------- input
function squareAtPoint(x, y) {
  const r = els.board.getBoundingClientRect();
  const col = Math.floor(((x - r.left) / r.width) * 8);
  const row = Math.floor(((y - r.top) / r.height) * 8);
  if (col < 0 || col > 7 || row < 0 || row > 7) return null;
  return (state.flipped ? row : 7 - row) * 8 + (state.flipped ? 7 - col : col);
}

function tryMove(from, to) {
  const prefix = sqName(from) + sqName(to);
  const cands = state.legal.filter((m) => m.startsWith(prefix));
  if (!cands.length) return false;
  if (cands.length > 1) { askPromotion(cands); return true; }
  playMove(cands[0], false);
  return true;
}

function askPromotion(cands) {
  const white = state.human === 'w';
  els.promoBox.innerHTML = '';
  for (const type of ['q', 'r', 'b', 'n']) {
    const uci = cands.find((m) => m.endsWith(type));
    if (!uci) continue;
    const btn = document.createElement('button');
    btn.setAttribute('aria-label', t('promote') + ' ' + ({ q: 'Q', r: 'R', b: 'B', n: 'N' })[type]);
    btn.innerHTML = pieceSvg(white ? type.toUpperCase() : type);
    btn.onclick = () => { els.promo.hidden = true; playMove(uci, false); };
    els.promoBox.appendChild(btn);
  }
  els.promo.hidden = false;
}
els.promo.addEventListener('click', (e) => { if (e.target === els.promo) { els.promo.hidden = true; state.selected = null; render(); } });

els.board.addEventListener('pointerdown', (e) => {
  if (!canInteract() || e.button > 0) return;
  const sq = squareAtPoint(e.clientX, e.clientY);
  if (sq == null) return;
  const b = parseBoard(state.fen);
  if (state.selected != null && state.selected !== sq && destinationsFrom(state.selected).has(sqName(sq))) {
    tryMove(state.selected, sq);
    return;
  }
  const p = b[sq];
  if (p && isWhitePiece(p) === (state.human === 'w')) {
    state.selected = sq;
    startDrag(e, sq, p);
    render();
  } else {
    state.selected = null;
    render();
  }
});

function startDrag(e, from, piece) {
  const size = els.board.getBoundingClientRect().width / 8;
  const el = document.createElement('div');
  el.className = 'floating';
  el.style.width = el.style.height = size + 'px';
  el.innerHTML = pieceSvg(piece);
  el.style.left = e.clientX + 'px';
  el.style.top = e.clientY + 'px';
  document.body.appendChild(el);
  state.drag = { from, el, moved: false };
  const move = (ev) => { el.style.left = ev.clientX + 'px'; el.style.top = ev.clientY + 'px'; state.drag.moved = true; };
  const up = (ev) => {
    window.removeEventListener('pointermove', move);
    window.removeEventListener('pointerup', up);
    window.removeEventListener('pointercancel', up);
    el.remove();
    const d = state.drag;
    state.drag = null;
    const to = squareAtPoint(ev.clientX, ev.clientY);
    if (d.moved && to != null && to !== d.from && destinationsFrom(d.from).has(sqName(to))) tryMove(d.from, to);
    else render();
  };
  window.addEventListener('pointermove', move);
  window.addEventListener('pointerup', up);
  window.addEventListener('pointercancel', up);
}

// ---------------------------------------------------------------- boot
function fillLevels(selected) {
  els.level.innerHTML = LEVELS.map((l, i) => `<option value="${i}">${t(l.key)}</option>`).join('');
  if (selected != null) { els.level.value = selected; return; }
  let saved = {};
  try { saved = JSON.parse(localStorage.getItem('barracuda-settings') || '{}'); } catch (_) { /* storage unavailable */ }
  els.level.value = saved.level != null && LEVELS[saved.level] ? saved.level : 3;
  els.color.value = saved.color || 'w';
}
function saveSettings() {
  try { localStorage.setItem('barracuda-settings', JSON.stringify({ level: +els.level.value, color: els.color.value })); } catch (_) { /* ignore */ }
}

let netInfo = { manifest: null, nnue: false, hasNet: false };
function renderNet() {
  const { manifest: m, nnue, hasNet } = netInfo;
  $('netEval').textContent = t(nnue ? 'netNnue' : hasNet ? 'netClassicViz' : 'netClassic');
  $('netTrained').textContent = m && m.epochs ? t('trainedFmt', { epochs: m.epochs, m: (m.samples / 1e6).toFixed(1) }) : t('notYet');
  const elo = m ? `${m.elo}${m.elo_error ? ' ± ' + m.elo_error : ''}` : '-';
  $('netElo').textContent = m ? t('eloFmt', { elo }) : '-';
  $('netElo').title = m ? m.elo_basis : '';
  const locale = getLang() === 'ar' ? 'ar-u-nu-latn' : undefined; // keep Western digits in Arabic
  $('netUpdated').textContent = m ? new Date(m.updated).toLocaleDateString(locale, { year: 'numeric', month: 'short', day: 'numeric' }) : '-';
  $('eloBadge').textContent = m ? t('eloBadge', { elo: m.elo }) : t('eloNone');
}
function showManifest(m, nnueActive, hasNet) { netInfo = { manifest: m, nnue: nnueActive, hasNet: !!hasNet }; renderNet(); renderViz(); }

function changeLanguage(next) {
  setLang(next);
  fillLevels(+els.level.value);
  renderNet();
  render();
  renderThanks();
  if (viz) viz.kick();
  $('langBtn').textContent = t('langButton');
  $('langBtn').lang = next === 'ar' ? 'en' : 'ar';
}

async function boot() {
  initLang();
  fillLevels();
  $('langBtn').textContent = t('langButton');
  $('langBtn').lang = getLang() === 'ar' ? 'en' : 'ar';
  $('langBtn').onclick = () => changeLanguage(getLang() === 'ar' ? 'en' : 'ar');
  render();
  viz = new NetViz($('vizCanvas'));
  engine = new EngineClient();
  engine.onInfo = (m) => {
    if (!state.thinking) return;
    if (m.view) { try { viz.setView(JSON.parse(m.view)); } catch (_) { /* ignore a malformed frame */ } }
    els.depth.textContent = String(m.depth);
    els.nps.textContent = `\u2066${t('speedFmt', { n: m.npsK })}\u2069`; // isolate so Arabic pages keep the number order
    els.pv.textContent = m.pv;
    state.evalWhite = m.score;
    renderEval();
  };

  let manifest = null, net = null;
  try {
    const r = await fetch('net/manifest.json', { cache: 'no-cache' });
    if (r.ok) manifest = await r.json();
  } catch (_) { /* offline or missing: fall back to the classical evaluator */ }
  if (manifest && manifest.net) { // downloaded for the visualisation even when it does not play yet
    try {
      const r = await fetch(`net/${manifest.net}?v=${manifest.version}`);
      if (r.ok) net = await r.arrayBuffer();
    } catch (_) { /* use classical */ }
  }
  let info;
  try {
    info = await engine.init(net, !!net && manifest.eval === 'nnue');
  } catch (err) {
    state.error = t('startError') + err.message;
    renderStatus();
    return;
  }
  manifestInfo = manifest;
  showManifest(manifest, info.nnue, info.hasNet);
  try { // optional: where anonymous games are sent (see README)
    const r = await fetch('config.json', { cache: 'no-cache' });
    if (r.ok) { const c = await r.json(); collect = c && c.collect && c.collect.type ? c.collect : null; }
  } catch (_) { /* no config: nothing is collected */ }
  $('shareWrap').hidden = !collect;
  $('shareBox').checked = shareEnabled();
  $('shareBox').onchange = (e) => { try { localStorage.setItem('barracuda-share', e.target.checked ? '1' : '0'); } catch (_) { /* ignore */ } };
  // small screens: three tabs under the board (the tab bar is hidden on desktop, where everything is shown)
  const right = $('right');
  const tabs = [...document.querySelectorAll('#tabs button')];
  const selectTab = (name) => {
    right.dataset.tab = name;
    tabs.forEach((b) => b.setAttribute('aria-selected', String(b.dataset.tab === name)));
    if (viz) viz.kick();
    try { localStorage.setItem('barracuda-tab', name); } catch (_) { /* ignore */ }
  };
  tabs.forEach((b) => { b.onclick = () => selectTab(b.dataset.tab); });
  try { const saved = localStorage.getItem('barracuda-tab'); if (saved) selectTab(saved); } catch (_) { /* ignore */ }
  $('thanksNew').onclick = () => { saveSettings(); newGame(); };
  $('thanksShow').onclick = hideThanks;
  const wanted = new URLSearchParams(location.search).get('fen'); // e.g. ?fen=<FEN> starts from that position
  if (wanted) { const f = await engine.call('fen', wanted, ''); if (f && !f.startsWith('!')) customFen = f; }
  state.ready = true;
  await newGame();

  $('newGame').onclick = () => { saveSettings(); newGame(); };
  $('undo').onclick = undo;
  $('flip').onclick = () => { state.flipped = !state.flipped; render(); };
  els.level.onchange = saveSettings;
  els.color.onchange = saveSettings;
}

boot();
