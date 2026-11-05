// Vector chess pieces (same shapes as the desktop dashboard), returned as inline SVG strings.
const pts = (a) => a.map(([x, y]) => `${(x * 100).toFixed(1)},${(y * 100).toFixed(1)}`).join(' ');
const n = (v) => (v * 100).toFixed(1);

function shapes(type, fill, stroke) {
  const sw = 2.8;
  const style = `fill="${fill}" stroke="${stroke}" stroke-width="${sw}" stroke-linejoin="round"`;
  const poly = (a) => `<polygon points="${pts(a)}" ${style}/>`;
  const rect = (x0, y0, x1, y1, r) =>
    `<rect x="${n(x0)}" y="${n(y0)}" width="${n(x1 - x0)}" height="${n(y1 - y0)}" rx="${n(r)}" ${style}/>`;
  const circle = (x, y, r) => `<circle cx="${n(x)}" cy="${n(y)}" r="${n(r)}" ${style}/>`;
  const base = rect(0.25, 0.77, 0.75, 0.86, 0.03);
  const shadow = `<rect x="25" y="86" width="50" height="4" fill="rgba(0,0,0,.25)"/>`;
  let body = '';
  switch (type) {
    case 'p':
      body = poly([[0.40, 0.47], [0.60, 0.47], [0.67, 0.79], [0.33, 0.79]]) + rect(0.35, 0.44, 0.65, 0.51, 0.03) +
        circle(0.5, 0.34, 0.115) + base;
      break;
    case 'n':
      body = poly([[0.30, 0.79], [0.33, 0.62], [0.43, 0.50], [0.40, 0.46], [0.30, 0.52], [0.21, 0.47], [0.22, 0.39],
        [0.38, 0.26], [0.42, 0.15], [0.49, 0.21], [0.58, 0.19], [0.69, 0.28], [0.76, 0.46], [0.74, 0.79]]) +
        `<circle cx="${n(0.46)}" cy="${n(0.31)}" r="${n(0.025)}" fill="${stroke}"/>` +
        `<line x1="${n(0.60)}" y1="${n(0.27)}" x2="${n(0.68)}" y2="${n(0.52)}" stroke="${stroke}" stroke-opacity=".5" stroke-width="2.2"/>` +
        base;
      break;
    case 'b': {
      const mitre = [];
      for (let i = 0; i <= 30; i++) {
        const a = Math.PI * i / 30, sn = Math.max(0, Math.sin(a));
        mitre.push([0.5 + 0.16 * Math.cos(a), 0.60 - 0.36 * Math.pow(sn, 1.6) - 0.04 * sn]);
      }
      body = poly([[0.40, 0.62], [0.60, 0.62], [0.66, 0.79], [0.34, 0.79]]) + poly(mitre) +
        `<line x1="${n(0.44)}" y1="${n(0.36)}" x2="${n(0.56)}" y2="${n(0.48)}" stroke="${stroke}" stroke-width="${sw}"/>` +
        circle(0.5, 0.165, 0.045) + rect(0.36, 0.58, 0.64, 0.64, 0.02) + base;
      break;
    }
    case 'r':
      body = poly([[0.34, 0.38], [0.66, 0.38], [0.69, 0.77], [0.31, 0.77]]) +
        poly([[0.27, 0.20], [0.36, 0.20], [0.36, 0.27], [0.45, 0.27], [0.45, 0.20], [0.55, 0.20], [0.55, 0.27],
          [0.64, 0.27], [0.64, 0.20], [0.73, 0.20], [0.73, 0.38], [0.27, 0.38]]) +
        rect(0.25, 0.76, 0.75, 0.86, 0.03);
      break;
    case 'q':
      body = poly([[0.30, 0.74], [0.22, 0.30], [0.37, 0.52], [0.40, 0.22], [0.50, 0.49], [0.60, 0.22], [0.63, 0.52],
        [0.78, 0.30], [0.70, 0.74]]) +
        [[0.22, 0.27], [0.40, 0.19], [0.60, 0.19], [0.78, 0.27]].map(([x, y]) => circle(x, y, 0.04)).join('') +
        rect(0.30, 0.70, 0.70, 0.77, 0.02) + rect(0.25, 0.76, 0.75, 0.86, 0.03);
      break;
    case 'k':
      body = poly([[0.31, 0.75], [0.22, 0.47], [0.30, 0.38], [0.42, 0.40], [0.50, 0.47], [0.58, 0.40], [0.70, 0.38],
        [0.78, 0.47], [0.69, 0.75]]) +
        rect(0.465, 0.12, 0.535, 0.42, 0.01) + rect(0.39, 0.19, 0.61, 0.25, 0.01) + rect(0.30, 0.70, 0.70, 0.77, 0.02) +
        rect(0.25, 0.76, 0.75, 0.86, 0.03);
      break;
  }
  return shadow + body;
}

// `piece` is a FEN letter: uppercase = white, lowercase = black.
export function pieceSvg(piece) {
  const white = piece === piece.toUpperCase();
  const fill = white ? '#FAF8F2' : '#0A0A0A';
  const stroke = white ? '#0F0F0F' : '#D7D7D7';
  return `<svg viewBox="0 0 100 100" class="piece" aria-hidden="true">${shapes(piece.toLowerCase(), fill, stroke)}</svg>`;
}
