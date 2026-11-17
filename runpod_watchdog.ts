/** Independent controller: stop-only for checks/admitted production; bounded
 * deletion for an explicitly receipted, unadmitted production setup.
 * Requires Node >=22.18. Credentials come only from this checkout's .env.local.
 */
import { appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync, statSync, writeFileSync, unlinkSync } from 'node:fs';
import { spawn, execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseEnv } from 'node:util';
import { setTimeout as sleep } from 'node:timers/promises';

const ROOT = dirname(fileURLToPath(import.meta.url));
const API = 'https://api.runpod.io/v2';
export const ALLOCATION_MS = 900_000;
export const SHUTDOWN_RESERVE_MS = 60_000;
export const RETRIEVAL_MS = 180_000;
export const RETRIEVAL_SHUTDOWN_RESERVE_MS = 45_000;
export function processMatches(pid: number, expected: string): boolean {
  if (!Number.isSafeInteger(pid) || pid <= 1) return false;
  try { return execFileSync('ps', ['-p', String(pid), '-o', 'command='], { encoding: 'utf8' }).includes(expected); }
  catch { return false; }
}

export type Receipt = {
  podId: string;
  createdAt: string;
  allocationRequestedAtMs: number;
  retrievalRequestedAtMs?: number;
  hardStopAtMs?: number;
  setupStopAtMs?: number;
};
export type Pod = {
  id: string;
  createdAt: string;
  status: string;
  cost: number;
  locked: boolean;
  actions: string[];
  runtime: object | null;
};
type Event = { event: string; elapsedMs: number; [key: string]: unknown };
type Controller = {
  now: () => number;
  delay: (ms: number) => Promise<void>;
  getPod: () => Promise<Pod>;
  stopPod: () => Promise<void>;
  earlyStop: () => boolean;
  record: (event: Event) => void;
  setupExpired?: () => boolean;
  deletePrestart?: () => Promise<void>;
  deletionRecorded?: () => boolean;
};

export function validateReceipt(value: unknown): Receipt {
  if (typeof value !== 'object' || value === null) throw new Error('Invalid creation receipt');
  const r = value as Receipt;
  if (typeof r.podId !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(r.podId)) throw new Error('Invalid pod ID');
  if (typeof r.createdAt !== 'string' || !Number.isFinite(Date.parse(r.createdAt))) throw new Error('Missing creation time');
  if (!Number.isSafeInteger(r.allocationRequestedAtMs) || r.allocationRequestedAtMs <= 0) throw new Error('Missing pre-create request time');
  // Permit API second rounding, but never an unrelated/older pod receipt.
  const age = Date.parse(r.createdAt) - r.allocationRequestedAtMs;
  if (age < -1000 || age > ALLOCATION_MS) throw new Error('Creation time does not match this allocation');
  if (r.retrievalRequestedAtMs !== undefined &&
      (!Number.isSafeInteger(r.retrievalRequestedAtMs) || r.retrievalRequestedAtMs < Date.parse(r.createdAt))) {
    throw new Error('Invalid retrieval request time');
  }
  if (r.hardStopAtMs !== undefined && (!Number.isSafeInteger(r.hardStopAtMs) ||
      r.hardStopAtMs <= r.allocationRequestedAtMs || r.hardStopAtMs > r.allocationRequestedAtMs + 24 * 3600_000 ||
      r.retrievalRequestedAtMs !== undefined)) throw new Error('Invalid production hard ceiling');
  if (r.setupStopAtMs !== undefined && (!Number.isSafeInteger(r.setupStopAtMs) || r.hardStopAtMs === undefined ||
      r.setupStopAtMs > r.hardStopAtMs || r.setupStopAtMs <= r.allocationRequestedAtMs)) throw new Error('Invalid setup ceiling');
  return r;
}

export function allocationStart(r: Receipt): number {
  if (r.retrievalRequestedAtMs !== undefined) return r.retrievalRequestedAtMs;
  return Math.min(r.allocationRequestedAtMs, Date.parse(r.createdAt));
}

export function verifyIdentity(p: Pod, r: Receipt): void {
  if (p.id !== r.podId || p.createdAt !== r.createdAt) throw new Error('Pod identity differs from creation receipt');
  if (typeof p.status !== 'string' || !Number.isFinite(p.cost) || !Array.isArray(p.actions) || typeof p.locked !== 'boolean') {
    throw new Error('Incomplete pod read-back');
  }
}

export function confirmedStopped(p: Pod): boolean {
  // Live RunPod reads retain the catalog rate in cost after stopping. Lifecycle
  // and absent runtime establish release; a displayed hourly price is not usage.
  return (p.status === 'EXITED' || p.status === 'TERMINATED') && p.runtime === null && !p.actions.includes('stop');
}

export async function watch(r: Receipt, c: Controller): Promise<void> {
  validateReceipt(r);
  const start = allocationStart(r);
  const limit = r.hardStopAtMs !== undefined ? r.hardStopAtMs - start
    : r.retrievalRequestedAtMs === undefined ? ALLOCATION_MS : RETRIEVAL_MS;
  const reserve = r.retrievalRequestedAtMs === undefined ? SHUTDOWN_RESERVE_MS : RETRIEVAL_SHUTDOWN_RESERVE_MS;
  const stopAt = start + limit - reserve;
  const record = (event: string, data: Record<string, unknown> = {}) => c.record({ event, elapsedMs: c.now() - start, ...data });
  let stopping = false;
  record('armed', { podId: r.podId, stopAfterMs: stopAt - start, hardLimitMs: limit });
  for (;;) {
    if (c.setupExpired !== undefined && c.setupExpired() && c.deletePrestart !== undefined) {
      try { await c.deletePrestart(); record('prestart-deleted-verified'); return; }
      catch { record('prestart-delete-failed'); await c.delay(2000); continue; }
    }
    if (!stopping && (c.earlyStop() || c.now() >= stopAt)) {
      stopping = true;
      record('stop-requested');
    }
    let p: Pod;
    try {
      p = await c.getPod();
    } catch (error: any) {
      if (error.status === 404 && c.deletionRecorded !== undefined && c.deletionRecorded()) {
        record('deleted-receipt-verified'); return;
      }
      // A timeout, 404, auth error, or network failure is not evidence of shutdown.
      record('read-failed');
      await c.delay(2000);
      continue;
    }
    verifyIdentity(p, r);
    if (confirmedStopped(p)) {
      record('stopped-verified', { podId: p.id, status: p.status, runtime: p.runtime, reportedHourlyPrice: p.cost });
      return;
    }
    if (p.status === 'ERROR') stopping = true;
    if (stopping) {
      if (c.now() >= start + limit) record('deadline-exceeded-unverified');
      if (!p.locked && p.actions.includes('stop')) {
        try {
          await c.stopPod();
          record('stop-accepted');
        } catch {
          record('stop-failed');
        }
      } else {
        record('awaiting-stoppable-state', { status: p.status, locked: p.locked });
      }
      // Always GET again. A successful POST is not shutdown verification.
      await c.delay(2000);
    } else {
      await c.delay(Math.max(1, Math.min(2000, stopAt - c.now())));
    }
  }
}

function setting(name: string): string {
  const value = parseEnv(readFileSync(join(ROOT, '.env.local'), 'utf8'))[name];
  if (typeof value !== 'string' || value.trim() === '') throw new Error(`Add ${name} to the scripts section of code/.env.local`);
  return value;
}

function credential(): string {
  const path = join(ROOT, '.env.local');
  const key = parseEnv(readFileSync(path, 'utf8')).RUNPOD_API_KEY;
  if (typeof key !== 'string' || key.trim() === '') throw new Error('Add RUNPOD_API_KEY to the scripts section of code/.env.local; never paste it into chat');
  if ((statSync(path).mode & 0o077) !== 0) throw new Error('Make code/.env.local owner-only (chmod 600) before using the API key');
  return key;
}

async function request(key: string, path: string, method: 'GET' | 'POST' | 'DELETE'): Promise<unknown> {
  const response = await fetch(`${API}${path}`, {
    method,
    headers: { Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' },
    body: method === 'POST' ? JSON.stringify({ action: 'stop' }) : undefined,
    signal: AbortSignal.timeout(5000),
  });
  if (!response.ok) throw Object.assign(new Error(`RunPod ${method} failed (HTTP ${response.status})`), { status: response.status });
  return response.status === 204 ? null : await response.json();
}

async function checkAccount(key: string): Promise<void> {
  const result = await request(key, '/account/ssh-keys', 'GET') as { keys: string[] };
  const publicKey = readFileSync(setting('BANDNET_SSH_PUBLIC_KEY'), 'utf8').trim().split(/\s+/).slice(0, 2).join(' ');
  if (!Array.isArray(result.keys) || !result.keys.some(k => k.split(/\s+/).slice(0, 2).join(' ') === publicKey)) {
    throw new Error('This API credential does not have the prepared Mac SSH key registered');
  }
  console.log('Authenticated read succeeded; the prepared Mac SSH key is registered. Stop permission is verified separately by a live check.');
}

function loadReceipt(path: string): Receipt {
  return validateReceipt(JSON.parse(readFileSync(path, 'utf8')));
}

async function main(): Promise<void> {
  const mode = process.argv[2];
  if (mode === 'check' && process.argv.length === 3) {
    await checkAccount(credential());
    return;
  }
  if (!['arm', 'watch', 'stop'].includes(mode) || process.argv.length !== 4) {
    throw new Error('Usage: node runpod_watchdog.ts check | arm RECEIPT.json | stop RECEIPT.json');
  }
  const path = resolve(process.argv[3]);
  const receipt = loadReceipt(path);
  const stateName = receipt.retrievalRequestedAtMs === undefined ? `watchdog-${receipt.podId}`
    : `watchdog-${receipt.podId}-retrieval-${receipt.retrievalRequestedAtMs}`;
  const state = join(dirname(path), stateName);
  if (mode === 'stop') {
    // Request early shutdown from the already running controller, with no credentials in argv.
    if (!existsSync(join(state, 'armed.json'))) throw new Error('Controller is not armed; stop this pod through the MCP immediately');
    writeFileSync(join(state, 'stop-requested'), '', { mode: 0o600 });
    console.log('Early stop requested. Require stopped.json and an independent MCP read-back.');
    return;
  }
  const key = credential();
  const podPath = `/pods/${encodeURIComponent(receipt.podId)}`;
  const getPod = async () => await request(key, podPath, 'GET') as Pod;
  if (mode === 'arm') {
    if (process.platform !== 'darwin') throw new Error('The detached launcher requires macOS caffeinate');
    await checkAccount(key);
    const pod = await getPod();
    verifyIdentity(pod, receipt);
    if (receipt.hardStopAtMs === undefined && (pod.locked || !pod.actions.includes('stop'))) throw new Error('Pod cannot be stopped; inspect it through the MCP immediately');
    if (Date.now() < allocationStart(receipt) - 1000) throw new Error('Creation receipt is in the future');
    if (existsSync(state)) {
      if (receipt.hardStopAtMs === undefined) throw new Error('Short-check controller already armed');
      const armedPath = join(state, 'armed.json');
      if (existsSync(armedPath)) {
        const pid = JSON.parse(readFileSync(armedPath, 'utf8')).pid;
        if (processMatches(pid, `watch ${path}`)) throw new Error('Controller already running');
        unlinkSync(armedPath);
      }
    } else mkdirSync(state, { mode: 0o700 });
    const fd = openSync(join(state, 'controller.log'), 'a', 0o600);
    const child = spawn('/usr/bin/caffeinate', ['-is', process.execPath, fileURLToPath(import.meta.url), 'watch', path], {
      detached: true, stdio: ['ignore', fd, fd], cwd: ROOT,
    });
    closeSync(fd);
    let spawnFailed = false;
    child.on('error', () => { spawnFailed = true; });
    child.unref();
    for (let i = 0; i < 100; i++) {
      if (existsSync(join(state, 'armed.json'))) {
        console.log(`Independent controller armed for ${receipt.podId}; evidence: ${state}`);
        return;
      }
      if (spawnFailed) break;
      await sleep(100);
    }
    throw new Error('Controller did not acknowledge arming. Stop the new pod through the MCP immediately');
  }
  let signalStop = false;
  process.on('SIGTERM', () => { signalStop = true; });
  process.on('SIGINT', () => { signalStop = true; });
  // Use a monotonic lower bound too: a wall-clock rollback cannot extend this allocation.
  const wallStart = Date.now();
  const monotonicStart = performance.now();
  let shortenedStopAt = receipt.hardStopAtMs;
  await watch(receipt, {
    now: () => Math.max(Date.now(), wallStart + performance.now() - monotonicStart),
    delay: sleep,
    getPod,
    stopPod: async () => { await request(key, `${podPath}/action`, 'POST'); },
    earlyStop: () => {
      const shortened = join(state, 'shortened-deadline.json');
      if (existsSync(shortened)) {
        try {
          const value = JSON.parse(readFileSync(shortened, 'utf8')).hardStopAtMs;
          if (!Number.isSafeInteger(value) || receipt.hardStopAtMs === undefined || value > receipt.hardStopAtMs) return true;
          shortenedStopAt = shortenedStopAt === undefined ? value : Math.min(shortenedStopAt, value);
        } catch { return true; }
      }
      return signalStop || existsSync(join(state, 'stop-requested')) || (shortenedStopAt !== undefined &&
        Math.max(Date.now(), wallStart + performance.now() - monotonicStart) >= shortenedStopAt - SHUTDOWN_RESERVE_MS);
    },
    setupExpired: () => {
      if (receipt.setupStopAtMs === undefined || existsSync(join(state, 'admitted.json'))) return false;
      let ceiling = receipt.setupStopAtMs;
      const path = join(state, 'setup-deadline.json');
      if (existsSync(path)) {
        try {
          const value = JSON.parse(readFileSync(path, 'utf8')).setupStopAtMs;
          ceiling = Number.isSafeInteger(value) ? Math.min(ceiling, value) : 0;
        } catch { ceiling = 0; }
      }
      return Math.max(Date.now(), wallStart + performance.now() - monotonicStart) >= ceiling;
    },
    deletionRecorded: () => {
      const deleted = join(dirname(path), 'deleted.json');
      try { return existsSync(deleted) && JSON.parse(readFileSync(deleted, 'utf8')).podId === receipt.podId; } catch { return false; }
    },
    deletePrestart: async () => {
      if (existsSync(join(state, 'admitted.json'))) throw new Error('Already admitted');
      const before = await request(key, '/pods?includeClusterPods=true', 'GET') as any;
      if (!before.pagination?.hasNextPage && !before.pods.some((p: any) => p.id === receipt.podId)) return;
      const pod = await getPod(); verifyIdentity(pod, receipt);
      await request(key, podPath, 'DELETE');
      const inventory = await request(key, '/pods?includeClusterPods=true', 'GET') as any;
      if (inventory.pagination?.hasNextPage || inventory.pods.some((p: any) => p.id === receipt.podId)) throw new Error('Deletion not verified');
    },
    record: event => {
      appendFileSync(join(state, 'events.jsonl'), `${JSON.stringify(event)}\n`, { mode: 0o600 });
      if (event.event === 'armed') writeFileSync(join(state, 'armed.json'), JSON.stringify({ ...event, pid: process.pid }), { mode: 0o600 });
      if (event.event === 'stopped-verified') writeFileSync(join(state, 'stopped.json'), JSON.stringify(event), { mode: 0o600 });
    },
  });
}

if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(error => {
    // Never log response bodies, headers or credentials.
    console.error(error instanceof Error ? error.message : 'Controller failed');
    process.exitCode = 1;
  });
}
