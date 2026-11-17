/** Fixed-study RunPod launcher. Live mutations only through explicit launch/continue.
 * Node >=22.18 on the local machine; settings come from .env.local.
 */
import { appendFileSync, closeSync, existsSync, mkdirSync, openSync, readFileSync, readdirSync, writeSync, fsyncSync, renameSync, unlinkSync, rmdirSync } from 'node:fs';
import { spawn, execFileSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { homedir } from 'node:os';
import { randomUUID } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';
import { atomic, budget, config, credentials, EXPORT_BOUND, GPU_TYPES, invariant, keyPrefix,
  MIN_UPLOAD, awsSettings, presign, spendingLimits, validateManifest, hash, TRANSFER_CONCURRENCY, BACKGROUND_CONCURRENCY, WHOLE_LIMIT } from './production_common.ts';
import type { Manifest } from './production_common.ts';
import { confirmedStopped, verifyIdentity, processMatches } from './runpod_watchdog.ts';

const ROOT = dirname(fileURLToPath(import.meta.url));
export const GLOBAL_LOCK = join(homedir(), '.local/state/bandnet/production-launch-lock');
const REMOTE = '/workspace/bandnet-run', NODE = '/workspace/node/bin/node';
const SOURCE = ['triatomic_data.py', 'triatomic_learning.py', 'corrected_pilot.py', 'corrected_inference.py',
  'triatomic_batched.py', 'triatomic_genuine_formula.py', 'triatomic_execution.py', 'generation_policy.py',
  'generation_resources.py', 'test_triatomic_pipeline.py', 'verify_corrected_pilot.py', 'pyproject.toml', 'uv.lock',
  'production_common.ts', 'production_pod.ts'];
type Plan = { id: string; prefix: string; cap: number; reserve: number; maxHourly: number; rehearsal: boolean; resume: string | null; probeOnly?: boolean; gpus?: string[]; minRam?: number };
type State = { plan: Plan; initialBalance: number; spent: number; attempts: number; active: Attempt | null; finished: boolean; restoreBytes?: number };
type Attempt = { podId: string; createdAt: string; allocationRequestedAtMs: number; hardStopAtMs: number;
  hourly: number; storageGB: number; host: string; port: number; started: boolean; ready: boolean; upload: number; saveAtMs: number;
  uploaded: string[]; syncs: string[]; lastFits: number; dir: string;
   backgroundUpload: number; uploadedKeys?: string[]; objectPaths?: Record<string, string> };
const read = (p: string) => JSON.parse(readFileSync(p, 'utf8'));
const quote = (s: string) => `'${s.replace(/'/g, `'"'"'`)}'`;

export async function subprocess(cmd: string, args: string[], input: string | undefined, timeout: number, cwd = ROOT): Promise<string> {
  return await new Promise((resolveResult, reject) => {
    const child = spawn(cmd, args, { cwd, stdio: ['pipe', 'pipe', 'pipe'] });
    const output: Buffer[] = []; let bytes = 0, errors = '';
    child.stdout.on('data', b => { bytes += b.length; if (bytes > 32 * 1024 ** 2) child.kill('SIGKILL'); else output.push(b); });
    // Keep only a short stderr tail, with every URL redacted: a failed command can contain a signed URL.
    child.stderr.on('data', b => { errors = (errors + b.toString()).slice(-4096); });
    child.stdin.on('error', () => {}); child.stdin.end(input);
    const timer = setTimeout(() => child.kill('SIGKILL'), timeout);
    child.once('error', () => { clearTimeout(timer); reject(new Error(`${cmd} could not start`)); });
    child.once('exit', code => {
      clearTimeout(timer);
      const tail = errors.replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, '<url>').replace(/X-Amz-[A-Za-z-]+=\S+/g, '<signed>').trim().split('\n').slice(-5).join(' | ').slice(-600);
      code === 0 ? resolveResult(Buffer.concat(output).toString()) : reject(new Error(`${cmd} failed (${code})${tail ? `: ${tail}` : ''}`));
    });
  });
}
export class APIError extends Error { status: number; constructor(status: number) { super(`RunPod HTTP ${status}`); this.status = status; } }
class Live {
  env = config(ROOT);
  settings = awsSettings(ROOT);
  aws = credentials(this.settings.keyCsv);
  key: string;
  constructor() { this.key = this.env.RUNPOD_API_KEY; invariant(this.key, 'Missing RUNPOD_API_KEY'); }
  async api(path: string, method = 'GET', body?: object) {
    const r = await fetch(`https://api.runpod.io/v2${path}`, { method, headers: { Authorization: `Bearer ${this.key}`, 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body), signal: AbortSignal.timeout(20_000) });
    if (!r.ok) throw new APIError(r.status);
    return r.status === 204 ? null : await r.json();
  }
  async balance() {
    const r = await fetch('https://api.runpod.io/graphql', { method: 'POST', headers: { Authorization: `Bearer ${this.key}`, 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: 'query { myself { clientBalance } }' }), signal: AbortSignal.timeout(20_000) });
    invariant(r.ok, 'Balance read failed'); const j = await r.json();
    invariant(!j.errors && Number.isFinite(j.data?.myself?.clientBalance), 'Balance response invalid'); return j.data.myself.clientBalance as number;
  }
  async check() {
    await subprocess(process.execPath, ['runpod_watchdog.ts', 'check'], undefined, 30_000);
    // Explicit account/region/owner validation, using only a child environment.
    const result = execFileSync('aws', ['sts', 'get-caller-identity', '--output', 'json'], { encoding: 'utf8',
      env: this.awsEnv(), timeout: 30_000, stdio: ['ignore', 'pipe', 'ignore'] });
    const t = this.settings.target;
    invariant(JSON.parse(result).Account === t.owner, 'Wrong AWS account');
    execFileSync('aws', ['s3api', 'head-bucket', '--bucket', t.bucket, '--expected-bucket-owner', t.owner], {
      env: this.awsEnv(), timeout: 30_000, stdio: 'ignore' });
  }
  awsEnv() {
    return { PATH: process.env.PATH, HOME: homedir(), AWS_ACCESS_KEY_ID: this.aws.access, AWS_SECRET_ACCESS_KEY: this.aws.secret,
      AWS_DEFAULT_REGION: this.settings.target.region, AWS_EC2_METADATA_DISABLED: 'true' };
  }
  ensureEmpty(prefix: string) {
    const t = this.settings.target;
    const result = execFileSync('aws', ['s3api', 'list-objects-v2', '--bucket', t.bucket, '--expected-bucket-owner',
      t.owner, '--prefix', prefix, '--max-keys', '1', '--no-paginate', '--output', 'json'], { encoding: 'utf8',
      env: this.awsEnv(), timeout: 30_000, stdio: ['ignore', 'pipe', 'ignore'] });
    invariant(JSON.parse(result).KeyCount === 0, 'S3 prefix is not empty; choose a fresh run ID');
  }
  opts(a: Attempt) { return ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', `UserKnownHostsFile=${a.dir}/known-hosts`,
    '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']; }
  ssh(a: Attempt, command: string, input?: string, timeout = 60_000) {
    return subprocess('ssh', ['-p', String(a.port), ...this.opts(a), `root@${a.host}`, command], input, timeout);
  }
  async rpc(a: Attempt, input: object, timeout = 180_000): Promise<any> {
    return JSON.parse(await this.ssh(a, `${NODE} ${REMOTE}/code/production_pod.ts`, JSON.stringify({ root: REMOTE, ...input }), timeout));
  }
  sign(method: 'GET' | 'PUT' | 'HEAD' | 'DELETE', key: string) { return presign(this.aws, this.settings.target, method, key); }
  async object(method: 'GET' | 'HEAD' | 'DELETE', key: string) {
    const s = this.sign(method, key); return fetch(s.url, { method, headers: s.headers, signal: AbortSignal.timeout(30_000) });
  }
  async offers(maxHourly: number, gpus: readonly string[] = GPU_TYPES) {
    const offers: any[] = [];
    for (const cloud of ['SECURE', 'COMMUNITY']) {
      const c = await this.api(`/catalog/gpus?include=AVAILABILITY&product=POD&count=1&cloud=${cloud}&minCudaVersion=13.0`);
      for (const g of c.gpus) {
        const price = cloud === 'SECURE' ? g.price.secure : g.price.community;
        if (gpus.includes(g.id) && g.availability !== 'NONE' && price > 0 && price <= maxHourly) offers.push({ cloud, gpu: g.id, price });
      }
    }
    return offers.sort((a, b) => gpus.indexOf(a.gpu) - gpus.indexOf(b.gpu) || a.price - b.price);
  }
  async resumeSize(key: string) {
    const r = await this.object('GET', key); invariant(r.ok, 'Cannot size restore package');
    return validateManifest(await r.json()).files.reduce((sum, e) => sum + e.bytes, 0);
  }
  async arm(a: Attempt) {
    const path = join(a.dir, 'receipt.json'), armed = join(a.dir, `watchdog-${a.podId}`, 'armed.json');
    if (!existsSync(armed) || !processMatches(read(armed).pid, `watch ${path}`)) {
      for (let retry = 0; ; retry++) {
        try { await subprocess(process.execPath, ['runpod_watchdog.ts', 'arm', path], undefined, 30_000); return; }
        catch (e) { if (retry === 3) throw e; await sleep(2000 * 2 ** retry); }
      }
    }
  }
  setup(a: Attempt, s: State, persist: () => void) { return setup(a, s, this, persist); }
  monitor(a: Attempt, s: State, persist: () => void, log: (m: string) => void) { return monitor(a, s, this, persist, log); }
  async retrieve(a: Attempt, plan: Plan) {
    const path = join(a.dir, 'probe-cleanup.json');
    if (existsSync(path)) {
      const remaining = [];
      for (const key of read(path)) { try {
        if (!(await this.object('DELETE', key)).ok || (await this.object('HEAD', key)).status !== 404) remaining.push(key);
      } catch { remaining.push(key); } }
      atomic(path, remaining);
      invariant(remaining.length === 0, 'Probe cleanup pending; retry retrieve');
    }
    await retrieveSmall(a, plan, this);
  }
  notify(message: string) {
    try { execFileSync('osascript', ['-e', `display notification ${JSON.stringify(message)} with title "BandNet RunPod"`], { stdio: 'ignore' }); } catch { /* Persistent log remains authoritative. */ }
  }
}

/** Same snapshot transaction is exercised locally with a fake transport. */
export function updateTransferRate(a: Attempt, mode: 'foreground' | 'background', bytes: number, seconds: number) {
  if (bytes < 1024 ** 3 || seconds <= 0) return;
  const field = mode === 'foreground' ? 'upload' : 'backgroundUpload';
  const measured = bytes / seconds;
  // Adopt slowdown immediately; increase cautiously. Modes never contaminate one another.
  a[field] = measured < a[field] ? measured : .75 * a[field] + .25 * measured;
}
export async function sync(a: Attempt, plan: Plan, kind: Manifest['kind'], label: string, io: Pick<Live, 'rpc' | 'sign'>,
                           persist: () => void) {
  if (kind !== 'final') invariant(Date.now() + 60_000 < a.saveAtMs, 'Mid-run snapshot deferred before scheduled stop');
  const m = validateManifest(await io.rpc(a, { op: 'snapshot', kind }, Math.min(600_000,
    kind === 'final' ? 600_000 : a.saveAtMs - Date.now())));
  if (a.uploadedKeys === undefined) a.uploadedKeys = [];
  if (a.objectPaths === undefined) a.objectPaths = {};
  invariant(a.backgroundUpload > 0 && a.upload > 0, 'Missing mode-specific transfer measurements; fresh setup required');
  for (const e of m.files) for (const c of e.chunks) {
    if (e.chunks.length === 1) {
      if (a.objectPaths[e.path] === undefined) a.objectPaths[e.path] = c.sha256;
      c.key = a.objectPaths[e.path] === c.sha256 ? `files/${e.path}` : `revisions/${c.sha256}/${e.path}`;
    } else c.key = `chunks/${c.sha256}`;
  }
  persist();
  const commit = async (manifest: Manifest, name: string) => {
    invariant(Date.now() < a.hardStopAtMs - 180_000, 'No time to commit export manifest');
    await io.rpc(a, { op: 'put-manifest', manifest, signed: io.sign('PUT', `${plan.prefix}snapshots/${name}.json`) });
    atomic(join(a.dir, `${name}.json`), manifest); a.syncs = a.syncs.filter(previous => previous !== name); a.syncs.push(name); persist();
  };
  const small = m.files.filter(e => e.bytes < 16 * 1024 ** 2 && !e.path.includes('-data/'));
  const uploadFiles = async (files: Manifest['files']) => {
    const concurrency = kind === 'final' ? TRANSFER_CONCURRENCY : BACKGROUND_CONCURRENCY;
    const queued = new Set<string>();
    const pending = files.flatMap(e => e.chunks.map((c, index) => ({ file: e.path, index, c })))
      .filter(v => {
        const key = v.c.key!;
        if (a.uploadedKeys!.includes(key) || queued.has(key)) return false;
        queued.add(key); return true;
      });
    while (pending.length) {
      invariant(Date.now() < a.hardStopAtMs - 180_000, 'Export deadline reached; earlier snapshots retained');
      const batch = []; let bytes = 0;
      const mode = kind === 'final' ? 'foreground' : 'background';
      const rate = kind === 'final' ? a.upload : a.backgroundUpload;
      // Up to three whole-object waves or five minutes of chunks, in one process.
      while (pending.length && batch.length < 512 && (batch.length < concurrency || bytes < Math.min(3 * concurrency * WHOLE_LIMIT, rate * 300))) {
        const item = pending.shift()!; batch.push(item); bytes += item.c.bytes;
      }
      if (kind !== 'final') invariant(Date.now() + (2 * bytes / rate + 120) * 1000 < a.saveAtMs,
        'Mid-run save deferred to preserve scheduled stop');
      const start = performance.now();
      const deadlineAtMs = Math.floor(Math.min(Date.now() + Math.min(3300_000, (2 * bytes / rate + 120) * 1000),
        a.hardStopAtMs - 180_000, kind === 'final' ? a.hardStopAtMs : a.saveAtMs));
      // Leave five seconds for SSH transit, without comparing clocks across hosts.
      const budgetMs = deadlineAtMs - Date.now() - 5000;
      invariant(budgetMs > 0, 'No transfer budget remains');
      let result;
      try { result = await io.rpc(a, { op: 'upload-batch', mode, budgetMs,
        items: batch.map(v => ({ file: v.file, index: v.index,
        key: v.c.key, signed: io.sign('PUT', `${plan.prefix}${v.c.key}`) })) }, budgetMs + 5000); }
      catch (error) {
        // Lost replies can hide deadline failures; conservatively reduce only this mode.
        if (kind === 'final') a.upload /= 2; else a.backgroundUpload /= 2;
        persist(); throw error;
      }
      let transferred = 0;
      for (const r of result.results) {
        const item = batch.find(v => v.c.key === r.key); invariant(item !== undefined, 'Unknown batch acknowledgement');
        if (!a.uploadedKeys!.includes(r.key)) a.uploadedKeys!.push(r.key);
        if (!a.uploaded.includes(item.c.sha256)) a.uploaded.push(item.c.sha256);
        transferred += r.bytes;
      }
      if (!result.failed) updateTransferRate(a, mode, transferred, (performance.now() - start) / 1000);
      else if (kind === 'final') a.upload /= 2; else a.backgroundUpload /= 2;
      persist(); invariant(!result.failed, `Batch interrupted; confirmed objects retained for retry: ${result.error}`);
    }
  };
  await uploadFiles(small);
  await commit({ ...m, kind: 'progress', files: small }, 'results');
  await uploadFiles(m.files.filter(e => !small.includes(e)).sort((x, y) =>
    Number(x.path.includes('-data/')) - Number(y.path.includes('-data/')) ||
    Number(x.path.endsWith('/latest.pt')) - Number(y.path.endsWith('/latest.pt')) || x.bytes - y.bytes));
  await commit(m, label);
  persist();
  return m;
}
// Live S3 restores ran at 80 MB/s on one pod and under 31 MB/s on another with a
// similar upload probe, so download time is not derived from the upload rate.
const RESTORE_FLOOR = 20_000_000;
async function restore(a: Attempt, key: string, live: Live) {
  invariant(/^3Band-corrected\/[a-zA-Z0-9_-]+\/snapshots\/[a-zA-Z0-9_-]+\.json$/.test(key), 'Invalid resume key');
  const response = await live.object('GET', key); invariant(response.ok, 'Cannot read resume manifest');
  const m = validateManifest(await response.json()); invariant(m.kind !== 'progress', 'Progress-only snapshot is not resumable');
  invariant(['production/protocol.json', 'production/progress.json', 'source.tar'].every(p => m.files.some(e => e.path === p)),
    'Snapshot is not a production resume package');
  const prefix = key.split('/').slice(0, 2).join('/') + '/';
  let items: any[] = [], bytes = 0;
  const flush = async () => {
    if (items.length) {
      const setupPath = join(a.dir, `watchdog-${a.podId}`, 'setup-deadline.json');
      const deadlineAtMs = Math.floor(Math.min(Date.now() + Math.min(3300_000, (bytes / RESTORE_FLOOR + 300) * 1000),
        read(setupPath).setupStopAtMs - 60_000, a.hardStopAtMs - 60_000));
      invariant(deadlineAtMs > Date.now(), 'No restore allowance remains');
      const budgetMs = deadlineAtMs - Date.now() - 5000;
      invariant(budgetMs > 0, 'No restore transfer budget remains');
      await live.rpc(a, { op: 'restore-batch', items, budgetMs }, budgetMs + 5000);
    }
    items = []; bytes = 0;
  };
  for (const e of m.files) {
    if (!e.path.startsWith('production/')) continue;
    if (e.bytes === 0) await live.ssh(a, `mkdir -p ${quote(dirname(join(REMOTE, e.path)))} && touch ${quote(join(REMOTE, e.path))}`);
    for (let index = 0; index < e.chunks.length; index++) {
      // Ranges supply concurrency even for a single large object. Keep receipts
      // incremental instead of acknowledging most of the package in one RPC.
      if (items.length >= 128 || (items.length > 0 && bytes + e.chunks[index].bytes > 8 * 1024 ** 3)) await flush();
      items.push({ entry: e, index, signed: live.sign('GET', `${prefix}${e.chunks[index].key === undefined ? `chunks/${e.chunks[index].sha256}` : e.chunks[index].key}`) });
      bytes += e.chunks[index].bytes;
    }
  }
  await flush();
  // Keep this launch's source.tar and executable checkout. Frozen scientific
  // sources remain under production/sources; Python admits only the recorded v4
  // scheduling migration, and records each execution's source bytes.
  atomic(join(a.dir, 'restored.json'), m);
}
async function setup(a: Attempt, s: State, live: Live, persist: () => void) {
  for (let i = 0; i < 60; i++) {
    const p = await live.api(`/pods/${a.podId}`);
    if (p.status === 'RUNNING' && p.ssh?.direct) { a.host = p.ssh.direct.host; a.port = p.ssh.direct.port; persist(); break; }
    await sleep(5000);
  }
  invariant(a.host !== '', 'No direct SSH endpoint within five minutes');
  await live.ssh(a, `mkdir -p ${REMOTE}/code`);
  await subprocess('git', ['diff', '--quiet', 'HEAD', '--', ...SOURCE.filter(p => p.endsWith('.py') || p === 'uv.lock' || p === 'pyproject.toml')], undefined, 30_000);
  await subprocess('git', ['bundle', 'create', join(a.dir, 'source.bundle'), 'HEAD'], undefined, 30_000);
  await subprocess('tar', ['--no-xattrs', '--no-mac-metadata', '-cf', join(a.dir, 'source.tar'), ...SOURCE, '-C', a.dir, 'source.bundle'], undefined, 30_000);
  await subprocess('scp', ['-P', String(a.port), ...live.opts(a), join(a.dir, 'source.tar'), `root@${a.host}:${REMOTE}/source.tar`], undefined, 60_000);
  await live.ssh(a, `set -eu; tar --no-same-owner -xf ${REMOTE}/source.tar -C ${REMOTE}/code; git clone --no-checkout -q ${REMOTE}/code/source.bundle ${REMOTE}/repository; mv ${REMOTE}/repository/.git ${REMOTE}/code/.git; cd /workspace; curl -fsSLO https://nodejs.org/dist/v22.18.0/node-v22.18.0-linux-x64.tar.gz; curl -fsSLO https://nodejs.org/dist/v22.18.0/SHASUMS256.txt; grep ' node-v22.18.0-linux-x64.tar.gz$' SHASUMS256.txt | sha256sum -c -; mkdir -p node; tar --no-same-owner -xf node-v22.18.0-linux-x64.tar.gz -C node --strip-components=1`, undefined, 180_000);
  const probe = await measureProbe(a, s.plan, live);
  if (probe.passed) { a.upload = probe.upload; a.backgroundUpload = probe.backgroundUpload; persist(); }
  if (s.plan.probeOnly === true) { a.ready = true; persist(); return; }
  invariant(probe.passed, 'Disk/hash/S3 probe failed; inspect probe.json');
  if (s.plan.resume !== null) {
    invariant(s.restoreBytes !== undefined, 'Restore size was not established before allocation');
    const receipt = read(join(a.dir, 'receipt.json'));
    atomic(join(a.dir, `watchdog-${a.podId}`, 'setup-deadline.json'), { setupStopAtMs:
      Math.min(receipt.setupStopAtMs, a.allocationRequestedAtMs + 45 * 60_000 + Math.floor(s.restoreBytes / RESTORE_FLOOR * 1000)) });
  }
  if (s.plan.resume !== null) await restore(a, s.plan.resume, live);
  // Never copy .env.local: an explicit nonsecret allowlist is rendered for the pod.
  const controls = ['BANDNET_GENERATION_CPU_RESERVE=3', 'BANDNET_GENERATION_RAM_RESERVE_MIB=16384', 'BANDNET_GENERATION_TUNING_SECONDS=5',
    'BANDNET_PRODUCTION_DEVICE=cuda:0', 'BANDNET_PRODUCTION_BATCH_SIZE=1024', 'BANDNET_PRODUCTION_TORCH_THREADS=8',
    'BANDNET_PRODUCTION_CHECKPOINT_STEPS=5000', 'BANDNET_PRODUCTION_MAX_SECONDS=600', 'BANDNET_TIMING_IO_ROWS=32768'].join('\n') + '\n';
  await live.ssh(a, `cat > ${REMOTE}/code/.env.local; chmod 600 ${REMOTE}/code/.env.local`, controls);
  await live.ssh(a, `set -eu; curl -LsSf https://astral.sh/uv/install.sh | sh; cd ${REMOTE}/code; $HOME/.local/bin/uv sync --frozen --extra training`, undefined, 900_000);
  await live.ssh(a, `cd ${REMOTE}/code && .venv/bin/python -`, `from pathlib import Path
from generation_resources import detect_resources
r = detect_resources()
p = Path('.env.local')
s = p.read_text().replace('BANDNET_PRODUCTION_TORCH_THREADS=8', 'BANDNET_PRODUCTION_TORCH_THREADS=' + str(min(8, r.cpu_budget)))
s = s.replace('BANDNET_GENERATION_CPU_RESERVE=3', 'BANDNET_GENERATION_CPU_RESERVE=' + str(min(3, max(1, r.cpu_budget - 1))))
p.write_text(s)
`);
  await live.ssh(a, `cd ${REMOTE}/code && .venv/bin/python -B corrected_pilot.py --gpu-preflight --output ${REMOTE}/preflight > ${REMOTE}/preflight.log 2>&1`, undefined, 660_000);
  a.ready = true; persist();
}
export async function measureProbe(a: Attempt, plan: Plan, live: Pick<Live, 'rpc' | 'sign' | 'object'>) {
  const cleanupPath = join(a.dir, 'probe-cleanup.json');
  const keys = Array.from({ length: 40 }, (_, i) => `${plan.prefix}probe/${a.podId}/${randomUUID()}-${i}`);
  const previous: string[] = existsSync(cleanupPath) ? read(cleanupPath) : [];
  // Durable before issuing any PUT, including probes that fail partway through.
  atomic(cleanupPath, [...previous, ...keys]);
  const start = performance.now();
  let probe: any;
  try {
    const budgetMs = Math.floor(Math.min(480_000, a.hardStopAtMs - Date.now() - 90_000));
    invariant(budgetMs > 0, 'No probe budget remains');
    invariant(budgetMs > 30_000, 'Insufficient time for bounded probe');
    // Pod cancellation/draining and JSON reply get 30 seconds of the RPC window.
    probe = await live.rpc(a, { op: 'probe', urls: keys.map(key => live.sign('PUT', key)), budgetMs: budgetMs - 30_000 }, budgetMs + 5000);
    probe.roundTripSeconds = (performance.now() - start) / 1000;
    invariant(Number.isFinite(probe.elapsedSeconds) && probe.elapsedSeconds > 0, 'Missing pod probe timing');
    probe.sshOverheadSeconds = Math.max(0, probe.roundTripSeconds - probe.elapsedSeconds);
    for (const wave of probe.waves) {
      invariant(Number.isFinite(wave.seconds) && wave.seconds > 0 && Number.isFinite(wave.bytes) && wave.bytes >= 0, 'Invalid wave measurement');
      const measuredBytes = wave.partial === true ? wave.sentBytes : wave.bytes;
      invariant(Number.isFinite(measuredBytes) && measuredBytes >= 0, 'Invalid sent-byte measurement');
      wave.effectiveRate = measuredBytes / (wave.seconds + probe.sshOverheadSeconds / probe.waves.length);
    }
    probe.upload = probe.waves.length === 3 ? Math.min(probe.waves[0].effectiveRate, probe.waves[2].effectiveRate) : 0;
    probe.backgroundUpload = probe.waves.length >= 2 ? probe.waves[1].effectiveRate : 0;
    probe.passed = !probe.failed && probe.waves.length === 3 && probe.waves.every((w: any, i: number) =>
      i === 2 && w.partial === true ? w.sentBytes >= 1024 ** 3 :
      !w.failed && w.bytes === (i === 2 ? 16 : 1) * 1024 ** 3 && w.objects === (i === 2 ? 8 : 16)) && probe.write >= 300e6 && probe.sha >= 400e6 &&
      probe.upload >= MIN_UPLOAD && probe.backgroundUpload > 0;
  } catch (e) {
    probe = { passed: false, error: String((e as Error).message).replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, '<url>').slice(0, 600) };
  }
  atomic(join(a.dir, 'probe.json'), probe);
  // Deletion happens after pod deletion in retrieve(), avoiding paid cleanup time
  // and any in-flight PUT completing after a premature DELETE.
  return probe;
}
export async function stop(a: Attempt, live: Pick<Live, 'api'>, log: (s: string) => void) {
  // Stop-only, including success. Deletion is exclusively the pre-start path below.
  for (let i = 0; i < 60; i++) {
    try {
      const p = await live.api(`/pods/${a.podId}`); verifyIdentity(p, a);
      if (confirmedStopped(p)) { atomic(join(a.dir, 'stopped.json'), { podId: p.id, status: p.status }); return; }
      if (!p.locked && p.actions.includes('stop')) await live.api(`/pods/${a.podId}/action`, 'POST', { action: 'stop' });
    } catch { log('Stop/read failed; independent watchdog remains armed'); }
    await sleep(2000);
  }
  throw new Error('STOP NOT VERIFIED; watchdog still armed, manual intervention required');
}
export async function removePrestart(a: Attempt, live: Pick<Live, 'api'>, delay = sleep) {
  invariant(!a.started, 'Never delete after production admission');
  let deleted = false;
  for (let i = 0; i < 60; i++) {
    try { await live.api(`/pods/${a.podId}`, 'DELETE'); } catch { /* Verify even after an ambiguous DELETE. */ }
    try {
      const inventory = await live.api('/pods?includeClusterPods=true');
      if (!inventory.pagination?.hasNextPage && !inventory.pods.some((p: any) => p.id === a.podId)) { deleted = true; break; }
    } catch { /* Retry bounded cleanup. */ }
    await delay(2000);
  }
  invariant(deleted, 'Deletion unverified; no further allocation permitted');
  atomic(join(a.dir, 'deleted.json'), { podId: a.podId });
  const armed = join(a.dir, `watchdog-${a.podId}`, 'armed.json');
  if (existsSync(armed) && processMatches(read(armed).pid, `watch ${join(a.dir, 'receipt.json')}`)) {
    try { process.kill(read(armed).pid, 'SIGKILL'); } catch { /* Own controller already exited. */ }
  }
}
export async function reconcileAllocation(name: string, dir: string, live: Pick<Live, 'api'>, delay = sleep, polls = 24) {
  let lastReadSucceeded = false;
  for (let poll = 0; poll < polls; poll++) {
    let inventory;
    try { inventory = await live.api('/pods?includeClusterPods=true'); lastReadSucceeded = true; }
    catch { lastReadSucceeded = false; await delay(5000); continue; }
    invariant(!inventory.pagination?.hasNextPage, 'Ambiguous paginated allocation inventory');
    const matches = inventory.pods.filter((p: any) => p.name === name);
    if (matches.length > 1) {
      for (const p of matches) await removePrestart({ podId: p.id, started: false, dir } as Attempt, live, delay);
      throw new Error('Duplicate allocations reconciled and deleted');
    }
    if (matches.length === 1) return matches[0];
    invariant(inventory.pods.length === 0, 'Unrelated allocation appeared'); await delay(5000);
  }
  invariant(lastReadSucceeded, 'Allocation remains ambiguous; continue this run after connectivity returns');
  return null;
}
export async function monitor(a: Attempt, s: State, live: Pick<Live, 'rpc' | 'sign' | 'balance'>,
  persist: () => void, log: (s: string) => void, pollingMs = 15_000) {
  let failures = 0, pendingBalance: number | null = null, interrupting = false, lastUnsaved = EXPORT_BOUND;
  for (;;) {
    if (Date.now() >= a.hardStopAtMs - 180_000) { log('Export guard reached; stopping with committed snapshots retained'); return; }
    // Never extend the receipt ceiling. Balance read-back can only shorten it.
    const elapsedSpend = a.hourly * (Date.now() - a.allocationRequestedAtMs) / 3600_000;
    const estimated = s.initialBalance - s.spent - elapsedSpend;
    let balance = estimated;
    try {
      const observed = await live.balance();
      invariant(Number.isFinite(observed) && observed >= 0, 'Invalid balance');
      if (observed >= estimated - .10) { balance = Math.min(estimated, observed); pendingBalance = null; }
      else if (estimated - observed > Math.max(.25, a.hourly * 5 / 60)) { pendingBalance = null; log('Balance is impossible at this run rate; ignoring observation'); }
      else if (pendingBalance !== null && Math.abs(observed - pendingBalance) <= .10) { balance = observed; pendingBalance = null; }
      else { pendingBalance = observed; log('Unexpected balance drop awaits confirmation'); }
    } catch { log('Live balance unavailable; elapsed-price ceiling retained'); }
    const lowerCeiling = Math.floor(Date.now() + Math.max(0, balance - s.plan.reserve) / a.hourly * 3600_000);
    if (lowerCeiling < a.hardStopAtMs) {
      a.hardStopAtMs = lowerCeiling;
      atomic(join(a.dir, `watchdog-${a.podId}`, 'shortened-deadline.json'), { hardStopAtMs: lowerCeiling });
      persist();
    }
    atomic(join(a.dir, 'money.json'), { estimated, observedConservative: balance, elapsedSpend, hourly: a.hourly,
      hardStopAtMs: a.hardStopAtMs, saveAtMs: a.saveAtMs, uploadBytesPerSecond: a.upload });
    let st;
    try { st = await live.rpc(a, { op: 'status' }, 30_000); failures = 0; }
    catch { failures++; log('Pod contact lost; retrying within money ceiling'); await sleep(Math.min(60_000, pollingMs * 2 ** Math.min(failures, 3))); continue; }
    if (st.exit !== null && !st.alive) {
      try {
        await sync(a, s.plan, 'final', 'final', live, persist);
        log(`Writer exit ${st.exit.code}; audit ${st.exit.audit}; final manifest committed`); return;
      } catch { log('Final save interrupted; retrying final snapshot'); await sleep(Math.min(60_000, pollingMs * 2)); continue; }
    }
    invariant(st.alive, 'Job disappeared without exit evidence');
    try { lastUnsaved = (await live.rpc(a, { op: 'inventory', uploadedKeys: a.uploadedKeys === undefined ? [] : a.uploadedKeys,
      objectPaths: a.objectPaths === undefined ? {} : a.objectPaths }, 30_000)).unsaved; }
    catch { log('Inventory unavailable; retaining previous unsaved estimate'); }
    // Actual current bytes plus space for a checkpoint being committed as SIGINT lands.
    const saveSeconds = 2 * (lastUnsaved + (s.plan.rehearsal ? 2e9 : 10e9)) / a.upload + 1020;
    a.saveAtMs = Math.floor(a.hardStopAtMs - saveSeconds * 1000); persist();
    try { await live.rpc(a, { op: 'schedule-stop', saveAtMs: a.saveAtMs }, 30_000); }
    catch { log('Cannot confirm pod save deadline; defer mid-run uploads'); await sleep(pollingMs); continue; }
    if (interrupting || Date.now() >= a.saveAtMs) {
      interrupting = true;
      try { await live.rpc(a, { op: 'interrupt' }, 30_000); } catch { log('Cooperative stop request will be retried'); }
      await sleep(pollingMs); continue;
    }
    if (st.progress !== null) {
      const n = Object.keys(st.progress.fits).length;
      try {
        if (n > a.lastFits) { await sync(a, s.plan, 'progress', `progress-${n}`, live, persist); a.lastFits = n; persist(); }
        for (const [threshold, label] of [[1, 'main'], [9, 'middle']] as const) {
          if (n >= threshold && !a.syncs.includes(label)) { await sync(a, s.plan, 'completed-fits', label, live, persist); log(`Committed ${label} snapshot`); }
        }
      } catch { log('Mid-run save deferred; training continues and save retries next cycle'); await sleep(Math.min(60_000, pollingMs * 2)); }
    }
    if (Date.now() >= a.hardStopAtMs - 120_000) throw new Error('Independent shutdown deadline reached');
    await sleep(pollingMs);
  }
}
async function retrieveSmall(a: Attempt, plan: Plan, live: Live) {
  // Runs AFTER verified GPU stop; Mac retrieval cannot consume the export budget.
  const label = a.syncs.at(-1); if (label === undefined) return;
  const m = validateManifest(read(join(a.dir, `${label}.json`)));
  for (const e of m.files.filter(e => !e.path.includes('-data/') && !e.path.endsWith('/latest.pt') && e.bytes < 512 * 1024 ** 2)) {
    const dest = join(a.dir, 'results', e.path); mkdirSync(dirname(dest), { recursive: true });
    const fd = openSync(`${dest}.tmp`, 'w', 0o600);
    try {
      for (const c of e.chunks) {
    const r = await live.object('GET', `${plan.prefix}${c.key === undefined ? `chunks/${c.sha256}` : c.key}`); invariant(r.ok, 'Mac retrieval failed');
        const b = Buffer.from(await r.arrayBuffer()); invariant(b.length === c.bytes && hash(b) === c.sha256, 'Mac retrieval checksum mismatch');
        invariant(writeSync(fd, b) === b.length, 'Short Mac write');
      }
      fsyncSync(fd);
    } finally { closeSync(fd); }
    renameSync(`${dest}.tmp`, dest);
  }
}
export async function supervise(dir: string, live: any = new Live(), delay = sleep, reconciliationPolls = 24, lockPath = GLOBAL_LOCK) {
  const lock = join(dir, 'supervisor.json');
  if (existsSync(lock)) invariant(!processMatches(read(lock).pid, `supervise ${read(join(dir, 'state.json')).plan.id}`), 'Supervisor already running');
  // Exclusive mkdir avoids two simultaneous reattach invocations.
  const mutex = join(dir, 'mutex');
  if (existsSync(mutex)) {
    invariant(existsSync(lock) && !processMatches(read(lock).pid, `supervise ${read(join(dir, 'state.json')).plan.id}`), 'Unowned supervisor mutex; inspect before retry');
    execFileSync('rmdir', [mutex]);
  }
  mkdirSync(mutex); atomic(lock, { pid: process.pid });
  const log = (m: string) => appendFileSync(join(dir, 'supervisor.log'), `${m}\n`);
  const statePath = join(dir, 'state.json');
  let s: State = read(statePath); const persist = () => atomic(statePath, s);
  const global = lockPath; mkdirSync(dirname(global), { recursive: true });
  try { mkdirSync(global); atomic(join(global, 'owner.json'), { id: s.plan.id, pid: process.pid, directory: dir }); }
  catch {
    if (!existsSync(join(global, 'owner.json')) || read(join(global, 'owner.json')).id !== s.plan.id || read(join(global, 'owner.json')).directory !== dir) {
      rmdirSync(mutex); throw new Error('Another run owns the global launch lock');
    }
    atomic(join(global, 'owner.json'), { id: s.plan.id, pid: process.pid, directory: dir });
  }
  const adopt = async (pod: any, requested: number, available: number, cap: number) => {
    try {
    invariant(Number.isFinite(pod.cost) && pod.cost > 0, 'Missing allocated hourly price; inspect allocation intent');
    invariant(Number.isFinite(pod.disk) && pod.disk > 0 && Number.isFinite(pod.mounts?.persistent?.size) && pod.mounts.persistent.size > 0,
      'Missing allocated storage sizes; inspect allocation intent');
    const storageGB = pod.disk + pod.mounts.persistent.size;
    const hourly = pod.cost + storageGB * .1 / 720;
    const limits = budget(available, s.plan.reserve, cap, hourly, requested, MIN_UPLOAD, s.plan.rehearsal ? 2e9 : EXPORT_BOUND);
    const adir = join(dir, `attempt-${++s.attempts}-${pod.id}`); mkdirSync(adir, { recursive: true, mode: 0o700 });
    const receipt = { podId: pod.id, createdAt: pod.createdAt, allocationRequestedAtMs: requested, hardStopAtMs: limits.hardStopAtMs,
      setupStopAtMs: Math.min(limits.hardStopAtMs, requested + 45 * 60_000 +
        (s.plan.resume === null ? 0 : Math.floor(2 * s.restoreBytes! / MIN_UPLOAD * 1000))) };
    atomic(join(adir, 'receipt.json'), receipt);
    atomic(join(adir, 'placement.json'), { podId: pod.id, name: pod.name, gpu: pod.gpu,
      dataCenterId: pod.dataCenterId, hourly, storageGB });
    s.active = { ...receipt, hourly, storageGB, host: '', port: 0, started: false, ready: false, upload: MIN_UPLOAD,
       backgroundUpload: MIN_UPLOAD * BACKGROUND_CONCURRENCY / TRANSFER_CONCURRENCY,
       saveAtMs: limits.saveAtMs, uploaded: [], syncs: [], lastFits: 0, dir: adir }; persist();
    await live.arm(s.active);
    if (existsSync(join(dir, 'allocation-intent.json'))) atomic(join(dir, 'allocation-intent.json'), { ...read(join(dir, 'allocation-intent.json')), pending: false });
    } catch (error) {
      const cleanup = s.active === null ? { podId: pod.id, started: false, dir } as Attempt : s.active;
      await removePrestart(cleanup, live, delay); s.active = null;
      if (existsSync(join(dir, 'allocation-intent.json'))) atomic(join(dir, 'allocation-intent.json'), { ...read(join(dir, 'allocation-intent.json')), pending: false });
      persist(); throw error;
    }
  };
  try {
    invariant(!s.finished, 'Run already finished; inspect results instead of relaunching');
    if (s.active === null) {
      const pendingPath = join(dir, 'allocation-intent.json');
      if (existsSync(pendingPath) && read(pendingPath).pending === true) {
        const intent = read(pendingPath), recovered = await reconcileAllocation(intent.name, dir, live, delay, reconciliationPolls);
        if (recovered !== null) await adopt(recovered, intent.requested, Math.min(await live.balance(), s.initialBalance - s.spent), s.plan.cap - s.spent);
        else atomic(pendingPath, { ...intent, pending: false });
      }
    }
    if (s.active === null) {
      for (let i = 0; ; i++) { try { await live.check(); break; } catch (e) { if (i === 3) throw e; await delay(5000 * (i + 1)); } }
      if (s.plan.resume !== null && s.restoreBytes === undefined) { s.restoreBytes = await live.resumeSize(s.plan.resume); persist(); }
    }
    if (s.active === null) {
      const inv = await live.api('/pods?includeClusterPods=true');
      invariant(!inv.pagination?.hasNextPage, 'Paginated inventory requires manual review');
      if (inv.pods.length !== 0) {
        const intentPath = join(dir, 'allocation-intent.json');
        invariant(existsSync(intentPath) && inv.pods.length === 1, 'Existing pods require manual review');
        const intent = read(intentPath), p = inv.pods[0];
        invariant(p.name === intent.name && Math.abs(Date.parse(p.createdAt) - intent.requested) < 300_000,
          'Inventory does not match this allocation intent');
        await adopt(await live.api(`/pods/${p.id}`), intent.requested, Math.min(await live.balance(), s.initialBalance - s.spent), s.plan.cap - s.spent);
        log('Recovered allocation from pre-create intent');
      }
      if (s.initialBalance === 0) { s.initialBalance = await live.balance(); persist(); }
    }
    let offers: any[] = [];
    let offerIndex = 0;
    while (s.attempts < 3 || s.active !== null) {
      if (s.active === null) {
        if (offers.length === 0) offers = await live.offers(s.plan.maxHourly, s.plan.gpus);
        invariant(offerIndex < offers.length, 'No remaining acceptable offers');
        const offer = offers[offerIndex++]; let pod: any = null; let requested = 0;
        const available = Math.min(await live.balance(), s.initialBalance - s.spent);
        const cap = s.plan.cap - s.spent;
        invariant(available > s.plan.reserve && cap > 0, 'Budget exhausted before placement');
        for (const vcpu of [16, null]) {
          for (let retry = 0; retry < 3 && pod === null; retry++) {
          requested = Date.now();
          // Allocation intent is durable BEFORE POST. An ambiguous POST is never retried.
          atomic(join(dir, 'allocation-intent.json'), { name: `bandnet-${s.plan.id}-${s.attempts + 1}`, requested, pending: true });
          try { pod = await live.api('/pods', 'POST', { name: `bandnet-${s.plan.id}-${s.attempts + 1}`, cloud: offer.cloud,
            image: 'runpod/base:1.0.2-ubuntu2404', gpu: { id: offer.gpu, count: 1, minCudaVersion: '13.0', ...(vcpu === null ? {} : { minVcpuCountPerGpu: vcpu }), ...(s.plan.minRam === undefined ? {} : { minRamPerGpu: s.plan.minRam }) },
            disk: 30, mounts: { persistent: { size: 220, path: '/workspace' } }, ports: ['22/tcp'], startSsh: true, startJupyter: false }); break;
          } catch (e) {
            if (e instanceof APIError && [400, 402, 403, 429].includes(e.status)) {
              atomic(join(dir, 'allocation-intent.json'), { ...read(join(dir, 'allocation-intent.json')), pending: false });
              if (e.status === 402) throw e;
              if (e.status === 429) { await delay(5000 * (retry + 1)); continue; }
              break;
            }
            // A timeout/5xx may have allocated successfully. Reconcile before another POST.
            pod = await reconcileAllocation(`bandnet-${s.plan.id}-${s.attempts + 1}`, dir, live, delay, reconciliationPolls);
            if (pod === null) atomic(join(dir, 'allocation-intent.json'), { ...read(join(dir, 'allocation-intent.json')), pending: false });
          }
          }
          if (pod !== null) break;
        }
        if (pod === null) continue;
        try { await adopt(pod, requested, available, cap); } catch (e) {
          if (s.active !== null) throw e;
          if (s.plan.probeOnly === true) { s.finished = true; persist(); throw e; }
          log(String(e)); continue;
        }
      }
      const a = s.active;
      try {
        await live.arm(a);
        const admitted = join(a.dir, `watchdog-${a.podId}`, 'admitted.json');
        if (existsSync(admitted)) { a.started = true; persist(); }
        invariant(a.hourly <= s.plan.maxHourly + a.storageGB * .1 / 720, 'Allocated GPU price exceeded ceiling');
        invariant(a.storageGB === 250, 'Allocated storage differs from requested 30+220 GB');
        if (!a.ready) await live.setup(a, s, persist);
        if (s.plan.probeOnly === true) { log('Probe-only speed check ended; inspect probe.json; deleting unadmitted pod'); return; }
        if (!a.started) {
          const b = budget(Math.min(await live.balance(), s.initialBalance - s.spent - a.hourly * (Date.now() - a.allocationRequestedAtMs) / 3600_000),
            s.plan.reserve, s.plan.cap - s.spent - a.hourly * (Date.now() - a.allocationRequestedAtMs) / 3600_000,
            a.hourly, Date.now(), a.upload, s.plan.rehearsal ? 2e9 : EXPORT_BOUND);
          a.saveAtMs = Math.min(b.hardStopAtMs, a.hardStopAtMs) - 1020_000;
          if (s.plan.rehearsal) a.saveAtMs = Math.min(a.saveAtMs, Date.now() + 120_000);
          invariant(a.saveAtMs > Date.now() + 60_000, 'Insufficient compute time after safe export allowance');
          // From this durable admission onward even an ambiguous launch must KEEP disk.
          const setupDeadlinePath = join(a.dir, `watchdog-${a.podId}`, 'setup-deadline.json');
          const setupDeadline = existsSync(setupDeadlinePath) ? read(setupDeadlinePath).setupStopAtMs : read(join(a.dir, 'receipt.json')).setupStopAtMs;
          invariant(Date.now() < setupDeadline - 60_000, 'Setup ceiling reached before admission');
          atomic(admitted, { admitted: true }); a.started = true; persist();
          await live.rpc(a, { op: 'start', spec: { saveAtMs: a.saveAtMs,
            initialSaveAtMs: Math.min(a.saveAtMs, a.hardStopAtMs - (2 * (s.plan.rehearsal ? 2e9 : EXPORT_BOUND) / a.upload + 1020) * 1000),
            resume: s.plan.resume !== null, rehearsal: s.plan.rehearsal } });
        }
        await live.monitor(a, s, persist, log);
      } catch (e) {
        log(e instanceof Error ? e.message : 'Attempt failed');
        if (s.plan.probeOnly === true && !existsSync(join(a.dir, 'probe.json'))) atomic(join(a.dir, 'probe.json'), {
          passed: false, error: 'Setup failed before measurement; inspect supervisor.log' });
        if (a.started) {
          try { await live.rpc(a, { op: 'interrupt' }, 30_000); await live.monitor(a, s, persist, log); }
          catch { try { await sync(a, s.plan, 'completed-fits', 'failure', live, persist); } catch { log('No new coherent failure snapshot; earlier snapshots retained'); } }
        }
      } finally {
        if (a.started) await stop(a, live, log);
        s.spent += a.hourly * (Date.now() - a.allocationRequestedAtMs) / 3600_000;
        if (!a.started) {
          await removePrestart(a, live, delay); s.active = null;
           if (s.plan.probeOnly === true) { s.finished = true; live.notify(`Speed probe ${s.plan.id} ended; inspect probe.json. Pod deletion verified.`); }
          persist(); try { await live.retrieve(a, s.plan); } catch { log('Probe cleanup remains pending on Mac'); }
        }
        else {
          s.finished = true; persist(); log('Pod stopped and retained. Manual review required.');
          live.notify(`Run ${s.plan.id} stopped. Results and retained storage need review.`);
          try { await live.retrieve(a, s.plan); } catch { log('S3 snapshot retained; small Mac retrieval incomplete'); }
        }
      }
      if (s.finished) return;
    }
  } finally {
    rmdirSync(mutex);
    const pending = existsSync(join(dir, 'allocation-intent.json')) && read(join(dir, 'allocation-intent.json')).pending === true;
    if ((s.active === null && !pending) || s.finished) {
      unlinkSync(join(global, 'owner.json')); rmdirSync(global);
      if (!s.finished) live.notify(`Run ${s.plan.id} ended before admission. Check the supervisor log.`);
    }
    else live.notify(`Run ${s.plan.id} needs attention. Shutdown is not yet verified.`);
  }
}
async function main() {
  const [mode, id, ...args] = process.argv.slice(2);
  if (mode === 'check') { const live = new Live(); await live.check(); console.log(JSON.stringify({ balance: await live.balance(), offers: await live.offers(spendingLimits(ROOT).maxHourly) }, null, 2)); return; }
  invariant(['launch', 'continue', 'supervise', 'retrieve'].includes(mode) && /^[a-zA-Z0-9_-]+$/.test(id), 'Usage: node production_launcher.ts check | launch ID [--rehearsal --cap DOLLARS] [--resume KEY] [--gpu TYPE ...] [--min-ram GB] | continue ID | retrieve ID');
  const dir = join(ROOT, 'artifacts', 'production', id);
  if (mode === 'retrieve') {
    const s = read(join(dir, 'state.json')); invariant(s.finished, 'Run has not finished');
    const live = new Live(); await live.check();
    if (s.active !== null) await live.retrieve(s.active, s.plan);
    else for (const entry of readdirSync(dir)) if (entry.startsWith('attempt-'))
      await live.retrieve({ dir: join(dir, entry), syncs: [] } as unknown as Attempt, s.plan);
    return;
  }
  if (mode === 'launch') {
    invariant(!existsSync(dir), 'Run ID already exists; use continue');
    const rehearsal = args.includes('--rehearsal'), probeOnly = args.includes('--probe-only'), ri = args.indexOf('--resume'), ci = args.indexOf('--cap');
    const resume = ri === -1 ? null : args[ri + 1]; invariant(resume !== undefined, 'Missing resume key');
    const gi = args.flatMap((a, i) => a === '--gpu' ? [i] : []), gpus = gi.map(i => args[i + 1]);
    invariant(gpus.every(g => GPU_TYPES.includes(g)) && new Set(gpus).size === gpus.length, '--gpu must name distinct established GPU types');
    const mi = args.indexOf('--min-ram'), minRam = mi === -1 ? undefined : Number(args[mi + 1]);
    invariant(minRam === undefined || Number.isInteger(minRam) && minRam >= 1 && minRam <= 65536, '--min-ram is whole GB of host RAM');
    invariant(args.every((a, i) => a === '--rehearsal' || a === '--probe-only' || a === '--resume' || a === '--cap' || a === '--gpu' || a === '--min-ram' || i === ri + 1 && ri !== -1 || i === ci + 1 && ci !== -1 || i === mi + 1 && mi !== -1 || gi.includes(i - 1)), 'Unknown launch option');
    invariant(!(rehearsal && resume !== null), 'Rehearsal is never production resume');
    invariant(!probeOnly || rehearsal, '--probe-only requires --rehearsal');
    const limits = spendingLimits(ROOT);
    const cap = ci === -1 ? (probeOnly ? limits.probeCap : rehearsal ? limits.rehearsalCap : limits.runCap) : Number(args[ci + 1]);
    invariant(!probeOnly || cap <= limits.probeCap, 'Probe-only cap must not exceed BANDNET_RUNPOD_PROBE_CAP_USD');
    invariant(Number.isFinite(cap) && cap > 0 && cap <= limits.runCap && (ci === -1 || rehearsal), '--cap is a positive rehearsal cap, at most BANDNET_RUNPOD_RUN_CAP_USD');
    const plan: Plan = { id, prefix: keyPrefix(`3Band-corrected/${rehearsal ? 'rehearsal-' : ''}${id}/`), cap, reserve: limits.reserve, maxHourly: limits.maxHourly, rehearsal, resume, probeOnly, ...(gpus.length ? { gpus } : {}), ...(minRam === undefined ? {} : { minRam }) };
    const live = new Live(); await live.check();
    live.ensureEmpty(plan.prefix);
    mkdirSync(dir, { recursive: true, mode: 0o700 });
    atomic(join(dir, 'state.json'), { plan, initialBalance: 0, spent: 0, attempts: 0, active: null, finished: false });
  }
  if (mode === 'supervise') return supervise(dir);
  invariant(existsSync(join(dir, 'state.json')), 'Unknown run ID');
  const fd = openSync(join(dir, 'supervisor.out'), 'a', 0o600);
  const child = spawn('/usr/bin/caffeinate', ['-is', process.execPath, fileURLToPath(import.meta.url), 'supervise', id], {
    cwd: ROOT, detached: true, stdio: ['ignore', fd, fd] }); closeSync(fd); child.unref();
  console.log(`Detached supervisor requested. Evidence: ${dir}`);
}
if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(() => { console.error('Launcher failed. Inspect retained supervisor evidence; never assume a pod stopped.'); process.exitCode = 1; });
}
