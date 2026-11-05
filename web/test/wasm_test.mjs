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

let t0 = Date.now();
let r = search('6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1', '', 200, 8).split(' ');
check('finds back-rank mate', r[0] === 'd1d8' && Number(r[1]) === 30000, r.slice(0, 4).join(' '));
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
  console.log('NNUE static evals (wasm): ' + fens.map(x => staticEval(x, '')).join(' '));
  t0 = Date.now();
  r = search(START, '', 1000, 64).split(' ');
  console.log(`NNUE search: depth ${r[2]}, ${(Number(r[3]) / ((Date.now() - t0) / 1000) / 1000).toFixed(0)} knps`);
}
console.log(failures ? `\n${failures} FAILED` : '\nall wasm checks passed');
process.exit(failures ? 1 : 0);
