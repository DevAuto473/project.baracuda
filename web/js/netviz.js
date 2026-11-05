// Neural-network visualiser: input pieces -> feature transformer -> hidden 1 -> hidden 2 -> output.
// Node brightness = activation, edge colour = weight sign (cyan +, magenta -), edge brightness = |w * a|.
// While the engine is thinking, light pulses travel along the strongest connections and each new
// search iteration flashes the graph (new data from the network is arriving).
import { t, isRtl } from './i18n.js?v=__BUILD__';

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
    this.setView(null);
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  setThinking(on) { this.thinking = on; this.kick(); }

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
    const r = this.canvas.getBoundingClientRect();
    this.w = Math.max(200, r.width);
    this.h = Math.max(120, r.height);
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.kick();
  }

  kick() { if (!this.raf) this.raf = requestAnimationFrame((ts) => this.frame(ts)); }

  frame(ts) {
    this.raf = 0;
    const dt = this.last ? Math.min(0.1, (ts - this.last) / 1000) : 0.016;
    this.last = ts;
    this.time += dt;
    let moving = this.thinking || this.flash > 0.01;
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
    if (moving) this.kick(); else this.last = 0;
  }

  draw() {
    const { ctx, w, h } = this;
    const v = this.view;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);

    const fam = getComputedStyle(document.body).fontFamily;
    const small = `600 ${w < 520 ? 10 : 12}px ${fam}`;
    const narrow = w < 520;
    const left = narrow ? 14 : 70, right = w - (narrow ? 56 : 120), top = narrow ? 34 : 44, bottom = h - 12;
    const L = 5;
    const pos = [];
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
    ctx.font = `700 ${narrow ? 13 : 18}px ${fam}`;
    ctx.fillStyle = '#e0e0e0';
    const cp = v.placeholder ? '—' : `${v.white > 0 ? '+' : ''}${v.white} cp`;
    if (narrow) { // above the output node, so it never sits on top of it
      ctx.textAlign = 'right';
      ctx.fillText(cp, w - 6, oy - radius * 2.2 - 10);
    } else {
      ctx.textAlign = 'left';
      ctx.fillText(cp, ox + radius * 2.2 + 10, oy + 6);
    }
  }
}
