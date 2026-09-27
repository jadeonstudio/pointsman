import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../', import.meta.url));
function walk(dir) { return fs.readdirSync(dir, { withFileTypes: true }).flatMap(e => e.name.startsWith('.git') || e.name === 'node_modules' ? [] : e.isDirectory() ? walk(path.join(dir, e.name)) : [path.join(dir, e.name)]); }
const files = walk(root).filter(f => f.endsWith('.mjs'));
for (const file of files) {
  const check = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  if (check.status !== 0) { console.error(check.stderr); process.exit(1); }
}
console.log(`Syntax checked ${files.length} JavaScript modules.`);
