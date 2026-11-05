// Barracuda web player: loading screen (the badge on black).
//
// Shown while the engine loads, for at least FIRST_MS when a person arrives and RELOAD_MS when the
// page is refreshed (or opened again in the same tab). It never leaves before the engine is ready,
// and gives up waiting after MAX_MS so a failed start cannot keep the page hidden.
// To change how long it stays, edit the three numbers below.
(function () {
  'use strict';
  var FIRST_MS = 5000;   // first visit
  var RELOAD_MS = 1500;  // refresh, or another load in the same tab
  var MAX_MS = 15000;    // longest wait for the engine
  var FADE_MS = 600;     // must match the opacity transition in index.html

  var el = document.getElementById('splash');
  if (!el) return;

  var seen = false, reload = false;
  try { seen = sessionStorage.getItem('barracuda-splash') === '1'; sessionStorage.setItem('barracuda-splash', '1'); } catch (_) { /* private mode */ }
  try { var nav = performance.getEntriesByType('navigation')[0]; reload = !!nav && nav.type === 'reload'; } catch (_) { /* old browser */ }
  var quick = seen || reload;
  var ms = quick ? RELOAD_MS : FIRST_MS;
  el.style.setProperty('--splash-ms', ms + 'ms');
  if (quick) el.classList.add('fast');

  // nothing behind the screen can be reached by keyboard while it is up
  var behind = Array.prototype.slice.call(document.querySelectorAll('header.top, main.layout, footer'));
  function setInert(on) {
    behind.forEach(function (n) { if (on) n.setAttribute('inert', ''); else n.removeAttribute('inert'); });
  }
  // the page elements come after this script: wait for them before marking them
  document.addEventListener('DOMContentLoaded', function () {
    behind = Array.prototype.slice.call(document.querySelectorAll('header.top, main.layout, footer'));
    if (!done) setInert(true);
  });

  var start = performance.now(), done = false;
  function finish() {
    if (done) return;
    done = true;
    setInert(false);
    el.classList.add('out');
    setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, FADE_MS + 80);
  }
  function check() {
    if (done) return;
    var elapsed = performance.now() - start;
    var ready = window.__barracudaReady === true;
    if (elapsed >= MAX_MS || (elapsed >= ms && ready)) finish();
    else setTimeout(check, 80);
  }
  check();
})();
