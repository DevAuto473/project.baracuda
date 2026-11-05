// Checks that the site can be installed and played offline: the manifest is complete, its icons exist
// at the sizes they claim, and every file the page loads is in the service worker's SHELL list
// (a file missing from it would work online and break offline). Run after web/build.sh.
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
let failures = 0;
const check = (name, ok, detail = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${ok || !detail ? '' : ' - ' + detail}`); if (!ok) failures++; };
const read = (f) => readFileSync(join(root, f), 'utf8');
const strip = (u) => u.split('?')[0].split('#')[0];

// ---- manifest
const man = JSON.parse(read('manifest.webmanifest'));
for (const k of ['name', 'short_name', 'start_url', 'scope', 'display', 'background_color', 'theme_color', 'icons'])
  check(`manifest has ${k}`, man[k] !== undefined && man[k] !== '');
check('manifest display is standalone', man.display === 'standalone');
const pngSize = (f) => { const b = readFileSync(join(root, f)); return b.readUInt32BE(16) + 'x' + b.readUInt32BE(20); };
for (const ic of man.icons) {
  check(`icon ${ic.src} exists`, existsSync(join(root, ic.src)));
  if (existsSync(join(root, ic.src))) check(`icon ${ic.src} is ${ic.sizes}`, pngSize(ic.src) === ic.sizes, pngSize(ic.src));
}
check('an icon of 192 and one of 512 (installability)', ['192x192', '512x512'].every((z) => man.icons.some((i) => i.sizes === z)));
check('a maskable icon', man.icons.some((i) => (i.purpose || '').includes('maskable')));

// ---- the page links the manifest and has a theme colour
const html = read('index.html');
check('index.html links the manifest', /rel="manifest"/.test(html));
check('index.html has a theme colour', /name="theme-color"/.test(html));

// ---- service worker
const sw = read('sw.js');
check('sw.js is stamped with the build id on deploy', sw.includes("const BUILD = '__BUILD__'"));
const shell = [...sw.slice(sw.indexOf('const SHELL = ['), sw.indexOf('];', sw.indexOf('const SHELL = [')) + 2).matchAll(/'([^']+)'/g)].map((m) => m[1]);
check('SHELL is not empty', shell.length > 10, String(shell.length));
const shellFiles = new Set(shell.map((u) => (u === './' ? 'index.html' : strip(u))));
for (const f of shellFiles) check(`stored for offline: ${f} exists`, existsSync(join(root, f)));

// ---- everything the page loads must be stored
const need = new Map(); // file -> where it is referenced
const want = (file, from) => { const f = normalize(strip(file)).replace(/\\/g, '/'); if (!need.has(f)) need.set(f, from); };
for (const m of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
  const u = m[1];
  if (/^(https?:|data:|mailto:)/.test(u)) continue;
  want(u, 'index.html');
}
const css = read('css/style.css');
for (const m of css.matchAll(/url\("?([^")]+)"?\)/g)) if (!/^(data:|https?:)/.test(m[1])) want(join('css', m[1]), 'css/style.css');
for (const f of ['app.js', 'netviz.js', 'i18n.js', 'engine-worker.js', 'splash.js', 'pieces.js']) {
  const js = read('js/' + f);
  for (const m of js.matchAll(/from '(\.\/[^']+)'/g)) want(join('js', m[1]), 'js/' + f);
  for (const m of js.matchAll(/importScripts\('([^']+)'\)/g)) want(join('js', m[1]), 'js/' + f);
  for (const m of js.matchAll(/new Worker\('([^']+)'\)/g)) want(m[1], 'js/' + f);
}
want('wasm-dist/barracuda.wasm', 'the engine (loaded by barracuda.js)');
for (const [f, from] of need)
  check(`${f} (used by ${from}) is stored for offline`, shellFiles.has(f), 'add it to SHELL in sw.js');

console.log(failures ? `\n${failures} PWA check(s) failed` : '\nPWA checks passed');
process.exit(failures ? 1 : 0);
