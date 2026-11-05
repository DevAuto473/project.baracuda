// Neural-network visualiser: input pieces -> feature transformer -> hidden 1 -> hidden 2 -> output.
// Node brightness = activation, edge colour = weight sign (cyan +, magenta -), edge brightness = |w * a|.
// While the engine is thinking, light pulses travel along the strongest connections and each new
// search iteration flashes the graph (new data from the network is arriving).
import { t, isRtl } from './i18n.js?v=__BUILD__';
import { pieceSvg } from './pieces.js?v=__BUILD__';

const LAYER_COLORS = [[200, 200, 200], [0, 229, 255], [90, 150, 255], [232, 121, 249], [63, 230, 140]];
const CYAN = [0, 229, 255];
const MAGENTA = [232, 121, 249];
const FT = 32; // sampled feature-transformer neurons (as in the desktop dashboard)
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${Math.max(0, Math.min(1, a)).toFixed(3)})`;
const mix = (a, b, k) => [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k];

// Stand-in shown before a network is available (deterministic, clearly labelled as such).
function placeholderView() {
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const n = 14;
  const arr = (len, f) => Array.from({ length: len }, f);
  return {
    placeholder: true, n, labels: arr(n, () => ''), a1: arr(FT, () => 0.12 + rnd() * 0.2), a2: arr(32, () => 0.1 + rnd() * 0.2),
    a3: arr(32, () => 0.1 + rnd() * 0.2), a4: [0.4],
    e0: arr(n * FT, () => (rnd() - 0.5) * 1.4), e1: arr(1024, () => (rnd() - 0.5) * 1.4), e2: arr(1024, () => (rnd() - 0.5) * 1.4),
    e3: arr(32, () => (rnd() - 0.5) * 1.4), cp: 0, white: 0,
  };
}

// ---- piece silhouettes as point clouds (the decision "forms" out of the network) ----
const pointCache = new Map();
function piecePoints(letter) {
  if (!pointCache.has(letter)) {
    pointCache.set(letter, new Promise((resolve) => {
      const svg = pieceSvg(letter).replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" ');
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = c.height = 200;
        const g = c.getContext('2d');
        g.drawImage(img, 0, 0, 200, 200);
        const data = g.getImageData(0, 0, 200, 200).data;
        const pts = [];
        for (let y = 2; y < 200; y += 4) {
          for (let x = 2; x < 200; x += 4) {
            if (data[(y * 200 + x) * 4 + 3] > 120) pts.push([(x + (Math.random() - 0.5) * 2) / 200, (y + (Math.random() - 0.5) * 2) / 200]);
          }
        }
        resolve({ pts, img });
      };
      img.onerror = () => resolve({ pts: [], img: null });
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(svg);
    }));
  }
  return pointCache.get(letter);
}
const ease = (x) => (x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2);
// How solid the piece that forms from particles looks (1 = as bright as the particles can be).
const PIECE_OPACITY = 0.5;
const reduceMotion = () => window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export class NetViz {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.thinking = false;
    this.flash = 0;
    this.time = 0;
    this.raf = 0;
    this.last = 0;
    this.view = null;
    this.shown = [[], [], [], [], []];
    this.pulses = [];
    this.decision = null; // {letter, san, born, parts[], img, dissolveAt}
    this.ring = null;
    this.pos = null;
    this.setView(null);
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  setThinking(on) {
    if (on && !this.thinking && this.decision && this.decision.dissolveAt == null) this.decision.dissolveAt = this.time; // next search: the old decision fades
    this.thinking = on;
    this.kick();
  }

  // The engine has chosen a move: the moving piece assembles out of the network's neurons and connections.
  async decide(letter, san) {
    const { pts, img } = await piecePoints(letter);
    if (!pts.length) return;
    const parts = pts.map(([tx, ty]) => {
      // start on a random strong connection: the particles are pulled out of the graph itself
      let sx = null, sy = null;
      if (this.pulses.length && this.pos) {
        const e = this.pulses[(Math.random() * this.pulses.length) | 0], f = Math.random();
        const a = this.pos[e.k] && this.pos[e.k][e.s], b = this.pos[e.k + 1] && this.pos[e.k + 1][e.d];
        if (a && b) { sx = a[0] + (b[0] - a[0]) * f; sy = a[1] + (b[1] - a[1]) * f; } // fixed start point on the connection
      }
      return { tx, ty, sx, sy, delay: Math.random() * 0.45, dur: 0.8 + Math.random() * 0.7, jx: Math.random() * 6.28, hue: Math.random() < 0.55 ? 0 : 1 };
    });
    this.decision = { letter, san, born: this.time, parts, img, dissolveAt: null };
    this.ring = this.time;
    this.flash = 1;
    this.kick();
  }

  // v = parsed JSON from brc_net_view / a search info message, or null for the placeholder
  setView(v) {
    const first = !this.view;
    this.view = v || placeholderView();
    const t0 = this.view;
    const target = [new Array(t0.n).fill(1), t0.a1, t0.a2, t0.a3, t0.a4];
    this.target = target;
    for (let l = 0; l < 5; l++) {
      if (!this.shown[l] || this.shown[l].length !== target[l].length) this.shown[l] = target[l].slice();
    }
    this.edges = [t0.e0, t0.e1, t0.e2, t0.e3];
    this.sizes = [t0.n, FT, 32, 32, 1];
    this.pickPulses();
    if (!first && v) this.flash = 1;
    this.kick();
  }

  // the strongest connections carry the travelling pulses
  pickPulses() {
    this.pulses = [];
    for (let k = 0; k < 4; k++) {
      const w = this.edges[k], ns = this.sizes[k], nd = this.sizes[k + 1];
      const scored = [];
      for (let s = 0; s < ns; s++) {
        const a = this.target[k][s] ?? 0;
        for (let d = 0; d < nd; d++) scored.push([Math.abs(w[s * nd + d]) * a, s, d, w[s * nd + d]]);
      }
      scored.sort((x, y) => y[0] - x[0]);
      for (const [, s, d, wt] of scored.slice(0, k === 3 ? 12 : 40)) this.pulses.push({ k, s, d, wt, seed: (s * 31 + d * 17 + k * 7) % 97 / 97 });
    }
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    // everything is drawn in "design pixels"; u is the interface scale (see --ui-scale), so the drawing
    // shrinks with the rest of the page instead of keeping full-size text in a smaller panel
    const u = (parseFloat(getComputedStyle(document.documentElement).fontSize) || 16) / 16;
    const r = this.canvas.getBoundingClientRect();
    const cssW = Math.max(200 * u, r.width), cssH = Math.max(120 * u, r.height);
    this.w = cssW / u;
    this.h = cssH / u;
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.ctx.setTransform(dpr * u, 0, 0, dpr * u, 0, 0);
    this.kick();
  }

  kick() { if (!this.raf) this.raf = requestAnimationFrame((ts) => this.frame(ts)); }

  frame(ts) {
    this.raf = 0;
    const dt = this.last ? Math.min(0.1, (ts - this.last) / 1000) : 0.016;
    this.last = ts;
    this.time += dt;
    let moving = this.thinking || this.flash > 0.01 || !!this.decision;
    const k = Math.min(1, dt * 9);
    for (let l = 0; l < 5; l++) {
      const cur = this.shown[l], tg = this.target[l];
      for (let i = 0; i < cur.length; i++) {
        const d = tg[i] - cur[i];
        if (Math.abs(d) > 0.002) moving = true;
        cur[i] += d * k;
      }
    }
    this.flash = Math.max(0, this.flash - dt * 2.2);
    this.draw();
    if (this.decision && this.decision.dissolveAt != null && this.time - this.decision.dissolveAt > 0.8) this.decision = null;
    if (moving) this.kick(); else this.last = 0;
  }

  draw() {
    const { ctx, w, h } = this;
    const v = this.view;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);

    const fam = getComputedStyle(document.body).fontFamily;
    // below ~700 design pixels the five column titles do not fit side by side in their long form
    const narrow = w < 700;
    const small = `600 ${narrow ? 10 : 12}px ${fam}`;
    const pieceSize = Math.max(56, Math.min(h * 0.5, w * 0.17, 190));
    const left = narrow ? 14 : 70, right = w - (narrow ? 56 + pieceSize * 0.6 : 40 + pieceSize + 40), top = narrow ? 34 : 44, bottom = h - 12;
    const L = 5;
    const pos = [];
    this.pos = pos;
    let radius = 7;
    for (let l = 0; l < L; l++) {
      const n = this.sizes[l];
      const x = left + (right - left) * l / (L - 1);
      const col = [];
      for (let i = 0; i < n; i++) col.push([x, n === 1 ? (top + bottom) / 2 : top + (bottom - top) * (i + 0.5) / n]);
      pos.push(col);
      if (n > 1) radius = Math.min(radius, (bottom - top) / n * 0.38);
    }
    radius = Math.max(1.6, radius);

    // edges
    const boost = 1 + this.flash * 0.6;
    ctx.lineCap = 'round';
    for (let k = 0; k < 4; k++) {
      const w8 = this.edges[k], ns = this.sizes[k], nd = this.sizes[k + 1], src = this.shown[k];
      let maxm = 1e-6;
      for (let s = 0; s < ns; s++) for (let d = 0; d < nd; d++) maxm = Math.max(maxm, Math.abs(w8[s * nd + d]) * src[s]);
      for (let s = 0; s < ns; s++) {
        for (let d = 0; d < nd; d++) {
          const wt = w8[s * nd + d];
          const tt = Math.abs(wt) * src[s] / maxm;
          if (tt < 0.13) continue;
          const a = Math.pow(tt, 1.4) * 0.85 * boost;
          const col = wt >= 0 ? CYAN : MAGENTA;
          const [x0, y0] = pos[k][s], [x1, y1] = pos[k + 1][d];
          ctx.beginPath();
          ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
          if (tt > 0.45) { ctx.strokeStyle = rgba(col, a * 0.16); ctx.lineWidth = 4 + 3 * tt; ctx.stroke(); ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); }
          ctx.strokeStyle = rgba(col, a);
          ctx.lineWidth = 0.6 + 1.6 * tt;
          ctx.stroke();
        }
      }
    }

    // travelling pulses while the engine is thinking
    if (this.thinking) {
      for (const p of this.pulses) {
        const ph = (this.time * 0.7 + p.seed) % 1;
        const [x0, y0] = pos[p.k][p.s], [x1, y1] = pos[p.k + 1][p.d];
        const x = x0 + (x1 - x0) * ph, y = y0 + (y1 - y0) * ph;
        const col = p.wt >= 0 ? CYAN : MAGENTA;
        ctx.fillStyle = rgba(col, 0.25); ctx.beginPath(); ctx.arc(x, y, 5, 0, 7); ctx.fill();
        ctx.fillStyle = rgba(mix(col, [255, 255, 255], 0.7), 0.95); ctx.beginPath(); ctx.arc(x, y, 1.9, 0, 7); ctx.fill();
      }
    }

    // nodes
    for (let l = 0; l < L; l++) {
      const base = LAYER_COLORS[l];
      const r = l === L - 1 ? radius * 2.2 : radius;
      for (let i = 0; i < this.sizes[l]; i++) {
        const a = Math.min(1, Math.max(0, this.shown[l][i] + (l > 0 ? this.flash * 0.12 : 0)));
        const [x, y] = pos[l][i];
        if (a > 0.6) { ctx.fillStyle = rgba(base, 0.1 * a); ctx.beginPath(); ctx.arc(x, y, r * 1.9, 0, 7); ctx.fill(); }
        ctx.fillStyle = rgba(mix([8, 8, 8], base, a), 1);
        ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
        ctx.strokeStyle = rgba(base, 0.55); ctx.lineWidth = 1; ctx.stroke();
      }
    }

    // column titles
    ctx.font = small;
    ctx.textBaseline = 'alphabetic';
    ctx.direction = isRtl() ? 'rtl' : 'ltr'; // Arabic captions with embedded numbers must use an RTL paragraph
    const names = narrow ? ['viz.s0', 'viz.s1', 'viz.s2', 'viz.s3', 'viz.s4'] : ['viz.l0', 'viz.l1', 'viz.l2', 'viz.l3', 'viz.l4'];
    const subs = [t('viz.sub0', { n: v.n }), t('viz.sub1'), t('viz.sub2'), t('viz.sub2'), t('viz.sub4')];
    for (let l = 0; l < L; l++) {
      const x = left + (right - left) * l / (L - 1);
      ctx.textAlign = l === 0 && left < 40 ? 'left' : l === L - 1 && w - x < 80 ? 'right' : 'center';
      const tx = ctx.textAlign === 'left' ? 6 : ctx.textAlign === 'right' ? w - 6 : x;
      ctx.fillStyle = rgba(LAYER_COLORS[l], 1);
      ctx.fillText(t(names[l]), tx, 16);
      if (!narrow) { // captions need room; phones get the short titles only
        ctx.fillStyle = '#6e7681';
        ctx.fillText(subs[l], tx, 31);
      }
    }
    ctx.direction = 'ltr';
    // input piece labels (left) and the evaluation (right)
    if (!v.placeholder && left > 40) {
      ctx.textAlign = 'left'; ctx.fillStyle = '#6e7681';
      ctx.font = `400 ${Math.min(11, Math.max(7, (bottom - top) / v.n * 0.95))}px ui-monospace, Menlo, Consolas, monospace`;
      v.labels.forEach((lab, i) => ctx.fillText(lab, 6, pos[0][i][1] + 3));
    }
    const [ox, oy] = pos[4][0];
    this.drawDecision(ctx, pos[4][0], pieceSize, narrow, fam);
    ctx.font = `700 ${narrow ? 13 : 18}px ${fam}`;
    ctx.fillStyle = '#e0e0e0';
    const cp = v.placeholder ? '—' : t('cpFmt', { v: `${v.white > 0 ? '+' : ''}${v.white}` });
    ctx.textAlign = 'center';
    ctx.fillText(cp, ox, oy - radius * 2.2 - 12); // above the output node
  }

  drawDecision(ctx, [ox, oy], size, narrow, fam) {
    const d = this.decision;
    const cx = Math.min(this.w - size / 2 - 14, ox + 40 + size / 2 + (narrow ? 4 : 28)), cy = oy;
    const x0 = cx - size / 2, y0 = cy - size / 2;

    // shock ring when the decision is made
    if (this.ring != null) {
      const k = (this.time - this.ring) / 0.9;
      if (k >= 1) this.ring = null;
      else {
        ctx.strokeStyle = rgba(CYAN, 0.55 * (1 - k)); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(ox, oy, 10 + k * (this.w * 0.25), 0, 7); ctx.stroke();
      }
    }
    if (!d) return;

    const dis = d.dissolveAt != null ? Math.min(1, (this.time - d.dissolveAt) / 0.8) : 0;
    const age = this.time - d.born;
    const still = reduceMotion();
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    let formed = 0;
    for (const p of d.parts) {
      const k = Math.max(0, Math.min(1, still ? 1 : (age - p.delay) / p.dur));
      if (k >= 1) formed++;
      const e = ease(k);
      // start: a point on a connection of the graph
      const sx = p.sx ?? ox, sy = p.sy ?? oy;
      const tx = x0 + p.tx * size, ty = y0 + p.ty * size;
      // curved path that swings through the output neuron
      const mx = ox + (tx - ox) * 0.25, my = oy + (sy - oy) * 0.1;
      const u = 1 - e;
      let x = u * u * sx + 2 * u * e * mx + e * e * tx;
      let y = u * u * sy + 2 * u * e * my + e * e * ty;
      let a = k <= 0 ? 0 : 0.25 + 0.75 * e;
      if (k >= 1) { // formed: gentle shimmer
        x += Math.sin(this.time * 3 + p.jx) * 0.7; y += Math.cos(this.time * 2.6 + p.jx) * 0.7;
        a = 0.75 + 0.25 * Math.sin(this.time * 4 + p.jx * 3);
      }
      if (dis > 0) { x += (x - cx) * dis * 0.9 + Math.sin(p.jx) * dis * 18; y += (y - cy) * dis * 0.9 - dis * 20; a *= 1 - dis; }
      a *= PIECE_OPACITY;
      const col = p.hue ? MAGENTA : CYAN;
      ctx.fillStyle = rgba(col, a * 0.3); ctx.beginPath(); ctx.arc(x, y, 3.2, 0, 7); ctx.fill();
      ctx.fillStyle = rgba(mix(col, [255, 255, 255], 0.65), a); ctx.beginPath(); ctx.arc(x, y, 1.25, 0, 7); ctx.fill();
    }
    ctx.restore();

    // once assembled, the real piece fades in under the particles so it reads clearly
    const done = formed / d.parts.length;
    if (d.img && done > 0.85 && !dis) {
      ctx.globalAlpha = Math.min(0.28 * PIECE_OPACITY, (done - 0.85) * 2);
      ctx.drawImage(d.img, x0, y0, size, size);
      ctx.globalAlpha = 1;
    }
    if (done > 0.7) {
      ctx.globalAlpha = (1 - dis) * Math.min(1, (done - 0.7) * 3);
      ctx.font = `700 ${narrow ? 13 : 20}px ${fam}`;
      ctx.textAlign = 'center';
      ctx.fillStyle = '#ffffff';
      ctx.direction = 'ltr';
      ctx.fillText(d.san, cx, Math.min(this.h - 6, y0 + size + (narrow ? 14 : 22)));
      ctx.globalAlpha = 1;
    }
  }
}
