import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { ID } from './constants.mjs';

// Only directory entries and file names are inspected. Role TOML/MD file
// CONTENT is never read here; this module only reports which stems exist.
const HOST_AGENT_DIRS = {
  codex: (env, userHome) => ({ dir: path.join(env.CODEX_HOME || path.join(userHome, '.codex'), 'agents'), ext: '.toml' }),
  claude: (env, userHome) => ({ dir: path.join(env.CLAUDE_CONFIG_DIR || path.join(userHome, '.claude'), 'agents'), ext: '.md' }),
};
const MAX_ROLES = 128;

/** Lists role IDs actually present as `<role>.toml` (codex) or `<role>.md` (claude) files. Missing directory -> []. */
export function discoverHostRoles(host, { env = process.env, userHome = os.homedir() } = {}) {
  const resolver = HOST_AGENT_DIRS[host];
  if (!resolver) return [];
  const { dir, ext } = resolver(env, userHome);
  let entries;
  try { entries = fs.readdirSync(dir); } catch { return []; }
  const roles = new Set();
  for (const name of entries) {
    if (!name.endsWith(ext)) continue;
    const stem = name.slice(0, -ext.length);
    if (!ID.test(stem)) continue;
    let stat;
    try { stat = fs.statSync(path.join(dir, name)); } catch { continue; } // follows symlinks; skips broken links
    if (!stat.isFile()) continue;
    roles.add(stem);
  }
  return [...roles].sort().slice(0, MAX_ROLES);
}
