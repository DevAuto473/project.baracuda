// Node smoke test for the WebAssembly build: node web/test/wasm_test.mjs [path/to/net.bnn]
import { createRequire } from 'node:module';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
// in a browser Worker the engine reports progress with postMessage(); emulate it under Node
const infos = [];
globalThis.postMessage = (m) => infos.push(m);
const create = require(path.join(here, '../wasm-dist/barracuda.js'));
// hand over the wasm bytes explicitly (older Emscripten + Node >= 18 would try fetch() on a file path)
const M = await create({ wasmBinary: readFileSync(path.join(here, '../wasm-dist/barracuda.wasm')) });
const f = (name, ret, args) => M.cwrap(name, ret, args);
const legal = f('brc_legal_moves', 'string', ['string', 'string']);
const status = f('brc_status', 'string', ['string', 'string']);
const fen = f('brc_fen', 'string', ['string', 'string']);
const san = f('brc_san', 'string', ['string', 'string', 'string']);
const search = f('brc_search', 'string', ['string', 'string', 'number', 'number']);
const staticEval = f('brc_static_eval', 'number', ['string', 'string']);
let failures = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name} ${extra}`); if (!ok) failures++; };

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
check('init', f('brc_init', 'number', [])() === 1, `(SIMD: ${f('brc_simd', 'string', [])()})`);
check('startpos has 20 moves', legal(START, '').split(' ').length === 20);
check('kiwipete has 48 moves', legal('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', '').split(' ').length === 48);
check('illegal move rejected', legal(START, 'e2e5').startsWith('!'));
check('fen after e4 e5', fen(START, 'e2e4 e7e5') === 'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2');
check('SAN', san(START, '', 'g1f3') === 'Nf3');
check('fool\'s mate status', status(START, 'f2f3 e7e5 g2g4 d8h4').startsWith('0-1 checkmate'));
check('stalemate status', status('7k/5Q2/6K1/8/8/8/8/8 b - - 0 1', '').includes('stalemate'));
check('threefold via move list', status(START, 'g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8').includes('threefold'));

f('brc_set_style', null, ['number'])(1); // the site plays with the strange style on
let t0 = Date.now();
let r = search('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1', '', 200, 8).split(' ');
check('finds back-rank mate', r[0] === 'd1d8' && Number(r[1]) === 30000, r.slice(0, 4).join(' '));
check('style never replaces a mate', f('brc_last_styled', 'number', [])() === 0);
let styledCount = 0;
for (const fen of [START, 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4', 'r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10', '6k1/5ppp/8/8/8/2Q5/5PPP/6K1 w - - 0 1']) {
  search(fen, '', 300, 64); if (f('brc_last_styled', 'number', [])()) styledCount++;
}
console.log(`style changed the move in ${styledCount} of 4 test positions`);
const predict = f('brc_predict', 'string', ['string', 'string']);
check('predicts that a human takes a hanging queen', predict('4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1', '') === 'd2d5');
check('prediction after a move list', /^[a-h][1-8][a-h][1-8]/.test(predict(START, 'e2e4')));
// a human model (hand-made: people like captures, checks, centre; they avoid hanging pieces)
{
  const w = Array(33).fill(0); w[6] = 2; w[7] = 1; w[8] = 1; w[23] = 0.8; w[12] = -1.5; w[15] = 1; w[21] = 0.8; w[30] = 3; w[32] = 1;
  M.FS.writeFile('/h.mdl', `BRCHUMAN 2 33 100000 0.3 0.55\n${w.join(' ')}\n`);
  check('loads a human model', f('brc_load_human', 'number', ['string'])('/h.mdl') === 1);
  M.FS.writeFile('/bad.mdl', 'nonsense');
  check('rejects a corrupt human model', f('brc_load_human', 'number', ['string'])('/bad.mdl') === 0);
  check('human model: a human takes the hanging queen', predict('4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1', '') === 'd2d5');
}
f('brc_set_opponent', null, ['number', 'number'])(100, 40);
r = search('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1', '', 200, 8).split(' ');
check('traps never replace a mate', r[0] === 'd1d8');
check('last trap is a string', typeof f('brc_last_trap', 'string', [])() === 'string');
r = search('4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1', '', 300, 12).split(' ');
check('progress messages posted', infos.length > 3 && infos.every(i => i.type === 'info' && typeof i.pv === 'string'));
check('wins the hanging queen', r[0] === 'd2d5', r.slice(0, 4).join(' '));

t0 = Date.now();
r = search(START, '', 1500, 64).split(' ');
const nodes = Number(r[3]), secs = (Date.now() - t0) / 1000;
console.log(`speed: depth ${r[2]}, ${nodes} nodes in ${secs.toFixed(2)}s = ${(nodes / secs / 1000).toFixed(0)} knps (wasm, classical)`);

const net = process.argv[2];
if (net && existsSync(net)) {
  M.FS.writeFile('/net.bnn', readFileSync(net));
  check('load network', f('brc_load_net', 'number', ['string'])('/net.bnn') === 1);
  f('brc_set_eval', null, ['number'])(1);
  const fens = [START, 'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4', '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'];
  // network visualisation data
  const netView = f('brc_net_view', 'string', ['string', 'string']);
  const v = JSON.parse(netView(START, 'e2e4'));
  check('net view shapes', v.n === v.labels.length && v.a1.length === 32 && v.a2.length === 32 && v.a3.length === 32 &&
    v.a4.length === 1 && v.e0.length === v.n * 32 && v.e1.length === 1024 && v.e2.length === 1024 && v.e3.length === 32, `(${v.n} input pieces)`);
  check('net view eval equals engine eval', v.cp === staticEval(START, 'e2e4'), `(${v.cp} cp)`);
  check('net view activations in [0,1]', [...v.a1, ...v.a2, ...v.a3].every((x) => x >= 0 && x <= 1));
  infos.length = 0;
  search(START, '', 300, 64);
  const withView = infos.filter((i) => i.view && i.view.startsWith('{'));
  check('search progress carries network views', withView.length > 3 && JSON.parse(withView[withView.length - 1].view).a1.length === 32, `(${withView.length}/${infos.length})`);
  console.log('NNUE static evals (wasm): ' + fens.map(x => staticEval(x, '')).join(' '));
  t0 = Date.now();
  r = search(START, '', 1000, 64).split(' ');
  console.log(`NNUE search: depth ${r[2]}, ${(Number(r[3]) / ((Date.now() - t0) / 1000) / 1000).toFixed(0)} knps`);
}
console.log(failures ? `\n${failures} FAILED` : '\nall wasm checks passed');
process.exit(failures ? 1 : 0);
