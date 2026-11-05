// Guards against a stray or missing brace in web/css/style.css: browsers silently drop the rules that
// follow such a mistake (this once made the game-over screen lose all of its styling).
// Usage: node web/test/css_check.mjs
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const file = path.join(path.dirname(fileURLToPath(import.meta.url)), '../css/style.css');
const css = readFileSync(file, 'utf8').replace(/\/\*[\s\S]*?\*\//g, '').replace(/"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'/g, '""');
let depth = 0, line = 1, problem = '';
for (const ch of css) {
  if (ch === '\n') line++;
  else if (ch === '{') depth++;
  else if (ch === '}' && --depth < 0) { problem = `extra "}" near line ${line} (comments removed)`; break; }
}
if (!problem && depth !== 0) problem = `${depth} unclosed "{"`;
for (const sel of ['.board-overlay', '.board-stack', '.tabs', '.evalbar']) {
  if (!problem && !new RegExp(sel.replace('.', '\\.') + '\\s*\\{').test(css)) problem = `missing rule ${sel}`;
}
if (problem) { console.error('CSS check FAILED: ' + problem); process.exit(1); }
console.log('CSS check passed: braces balanced, key rules present');
