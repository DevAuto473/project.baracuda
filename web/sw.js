// Barracuda service worker: lets the site be installed and played without a connection.
//
// Everything the page needs (engine, scripts, fonts, images, the network) is stored when the worker
// installs, so the game runs from this copy. Each deploy stamps a new BUILD into this file, which
// makes the browser install a fresh copy and delete the old one. Keep SHELL in sync with what the
// page loads: web/test/pwa_check.mjs fails the deploy if a file is missing from it.
'use strict';

const BUILD = '__BUILD__';
const CACHE = 'barracuda-' + BUILD;

const SHELL = [
  './',
  'index.html',
  'manifest.webmanifest?v=__BUILD__',
  'css/style.css?v=__BUILD__',
  'js/app.js?v=__BUILD__',
  'js/i18n.js?v=__BUILD__',
  'js/netviz.js?v=__BUILD__',
  'js/pieces.js?v=__BUILD__',
  'js/splash.js?v=__BUILD__',
  'js/engine-worker.js?v=__BUILD__',
  'wasm-dist/barracuda.js?v=__BUILD__',
  'wasm-dist/barracuda.wasm?v=__BUILD__',
  'fonts/RocaTwo-Bold.ttf',
  'fonts/ibm-plex-sans-arabic-arabic-400-normal.woff2',
  'fonts/ibm-plex-sans-arabic-arabic-600-normal.woff2',
  'fonts/ibm-plex-sans-arabic-latin-400-normal.woff2',
  'fonts/ibm-plex-sans-arabic-latin-600-normal.woff2',
  'img/logo.webp?v=__BUILD__',
  'img/favicon-48.png?v=__BUILD__',
  'img/logo-192.png?v=__BUILD__',
  'img/icon-192.png',
  'img/icon-512.png',
  'img/icon-maskable-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(CACHE);
    // all or nothing: a half-stored copy would fail offline in confusing ways, so a missing file
    // makes the install fail and the browser tries again later
    await Promise.all(SHELL.map(async (url) => {
      const res = await fetch(url, { cache: 'reload' });
      if (!res.ok) throw new Error('cannot store ' + url + ' (' + res.status + ')');
      await cache.put(url, res);
    }));
    // optional site settings (where anonymous games are sent, if anywhere)
    try {
      const cres = await fetch('config.json', { cache: 'reload' });
      if (cres.ok) await cache.put('config.json', cres);
    } catch (_) { /* no settings file */ }
    // the published network (and the human-move model): named by net/manifest.json, requested by the
    // page as net/<file>?v=<version>
    try {
      const mres = await fetch('net/manifest.json', { cache: 'reload' });
      if (mres.ok) {
        await cache.put('net/manifest.json', mres.clone());
        const m = await mres.json();
        for (const file of [m.net, m.human]) {
          if (!file) continue;
          const url = 'net/' + file + '?v=' + m.version;
          const res = await fetch(url);
          if (res.ok) await cache.put(url, res);
        }
      }
    } catch (_) { /* offline games then use the classical evaluator */ }
    await self.skipWaiting();
  })());
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    for (const name of await caches.keys())
      if (name.startsWith('barracuda-') && name !== CACHE) await caches.delete(name);
    await self.clients.claim();
  })());
});

// Files that change without the page's URL changing: ask the network first (a few seconds at most),
// and fall back to the stored copy when there is no connection.
async function networkFirst(request) {
  const cache = await caches.open(CACHE);
  try {
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), 4000);
    const res = await fetch(request, { signal: ctl.signal });
    clearTimeout(timer);
    if (res.ok) cache.put(request, res.clone());
    return res;
  } catch (err) {
    const hit = await cache.match(request, { ignoreSearch: true });
    if (hit) return hit;
    throw err;
  }
}

self.addEventListener('fetch', (event) => {
  const request = event.request;
  if (request.method !== 'GET') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  if (request.mode === 'navigate') { // opening the site (or refreshing it): the stored page
    event.respondWith((async () => {
      const cache = await caches.open(CACHE);
      return (await cache.match('./')) || (await cache.match('index.html')) || fetch(request);
    })());
    return;
  }
  if (url.pathname.endsWith('/net/manifest.json') || url.pathname.endsWith('/config.json')) {
    event.respondWith(networkFirst(request));
    return;
  }
  event.respondWith((async () => { // everything else: the stored copy, else the network (and keep it)
    const cache = await caches.open(CACHE);
    const hit = await cache.match(request);
    if (hit) return hit;
    const res = await fetch(request);
    if (res.ok) cache.put(request, res.clone());
    return res;
  })());
});
