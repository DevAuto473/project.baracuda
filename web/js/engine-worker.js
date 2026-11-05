// Runs the Barracuda WebAssembly engine off the UI thread.
// Protocol:  main -> worker  {id, cmd:'init', net: ArrayBuffer|null, useNnue: bool}
//                            {id, cmd:'call', fn, args:[...]}
//            worker -> main  {id, result} | {id, error} | {type:'info', depth, score, nodesK, npsK, timeMs, pv}
'use strict';

let api = null;

importScripts('../wasm-dist/barracuda.js?v=__BUILD__');

async function init(net, useNnue) {
  const M = await createBarracuda({
    locateFile: (f) => new URL('../wasm-dist/' + f + '?v=__BUILD__', self.location.href).href,
  });
  const w = (name, ret, args) => M.cwrap(name, ret, args);
  api = {
    legal: w('brc_legal_moves', 'string', ['string', 'string']),
    fen: w('brc_fen', 'string', ['string', 'string']),
    status: w('brc_status', 'string', ['string', 'string']),
    inCheck: w('brc_in_check', 'number', ['string', 'string']),
    netView: w('brc_net_view', 'string', ['string', 'string']),
    san: w('brc_san', 'string', ['string', 'string', 'string']),
    newGame: w('brc_new_game', null, []),
    search: w('brc_search', 'string', ['string', 'string', 'number', 'number']),
    setEval: w('brc_set_eval', null, ['number']),
    simd: w('brc_simd', 'string', []),
  };
  w('brc_init', 'number', [])();
  // The network is always loaded when published (for the visualisation); it only drives the
  // search when the manifest says it beats the classical evaluator.
  let loaded = false;
  if (net) {
    M.FS.writeFile('/net.bnn', new Uint8Array(net));
    loaded = w('brc_load_net', 'number', ['string'])('/net.bnn') === 1;
  }
  const nnue = loaded && !!useNnue;
  api.setEval(nnue ? 1 : 0);
  return { simd: api.simd(), nnue, hasNet: loaded };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    let result;
    if (m.cmd === 'init') result = await init(m.net, m.useNnue);
    else if (m.cmd === 'call') result = api[m.fn](...m.args);
    self.postMessage({ id: m.id, result });
  } catch (err) {
    self.postMessage({ id: m.id, error: String(err && err.message ? err.message : err) });
  }
};
