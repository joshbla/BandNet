import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync, existsSync, readdirSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';
import { atomic, budget, hash, validateManifest, presign } from './production_common.ts';
import { rpc, restoreChunk, RESTORE_RANGE_BYTES, RESTORE_CONCURRENCY, snapshot, command, auditDeadline, probeTransfers, transferPool, upload } from './production_pod.ts';
import { monitor, removePrestart, stop, sync, supervise, APIError, updateTransferRate, measureProbe } from './production_launcher.ts';
import { validateReceipt, watch } from './runpod_watchdog.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
const out = join(HERE, 'artifacts/production-local-rehearsal'); mkdirSync(out, { recursive: true });
function fixture(root: string) {
  atomic(join(root, 'job-spec.json'), { saveAtMs: Date.now() + 12 * 3600_000 });
  for (const d of ['production/sources', 'production/executions', 'production/main', 'production/main-data', 'production/study-active', 'production/study-active-data']) mkdirSync(join(root, d), { recursive: true });
  writeFileSync(join(root, 'source.tar'), 'scientific-source-fixture');
  atomic(join(root, 'production/protocol.json'), { fixedStudy: true });
  atomic(join(root, 'production/progress.json'), { status: 'running', fits: { main: { checkpoint_sha256: hash('model') } }, evaluations: {}, attempts: [{ status: 'running' }] });
  writeFileSync(join(root, 'production/sources/science.py'), 'locked science');
  atomic(join(root, 'production/executions/preflight.json'), { qualified: true });
  writeFileSync(join(root, 'production/main/best.pt'), 'model');
  writeFileSync(join(root, 'production/main/latest.pt'), 'optimizer');
  writeFileSync(join(root, 'production/main-data/bands.npy'), Buffer.alloc(1024 * 1024, 17));
  atomic(join(root, 'production/main-data/manifest.json'), { complete: true });
  writeFileSync(join(root, 'production/study-active/latest.pt'), 'half-written, deliberately unsafe');
  writeFileSync(join(root, 'production/study-active-data/bands.npy'), 'partial');
}
function state(root: string) {
  const now = Date.now();
  const a: any = { dir: join(root, 'mac'), podId: 'simulated', createdAt: new Date(now).toISOString(), allocationRequestedAtMs: now,
    hardStopAtMs: now + 10 * 3600_000, hourly: 4, host: 'simulated', port: 22, started: true, ready: true,
    upload: 100e6, backgroundUpload: 25e6, saveAtMs: now + 8 * 3600_000, uploaded: [], syncs: [], lastFits: 0 };
  mkdirSync(a.dir, { recursive: true });
  const s: any = { plan: { id: 'local', prefix: '3Band-corrected/local/', cap: 80, reserve: 10, maxHourly: 5, rehearsal: false, resume: null },
    initialBalance: 100, spent: 0, attempts: 1, active: a, finished: false };
  return s;
}
async function storage(options: { badETag?: boolean; holdMs?: number } = {}) {
  const objects = new Map<string, Buffer>(), puts: string[] = [];
  let active = 0, peak = 0;
  const server = createServer(async (req, res) => {
    const path = req.url!;
    if (req.method === 'PUT') {
      peak = Math.max(peak, ++active);
      const chunks: Buffer[] = []; for await (const chunk of req) chunks.push(chunk);
      // Like S3 with presigned URLs: an unsigned Content-MD5 is refused, and a single-part ETag is the MD5.
      if (req.headers['content-md5'] !== undefined) { active--; res.statusCode = 403; res.end(); return; }
      const body = Buffer.concat(chunks);
      objects.set(path, body); puts.push(path);
      if (options.holdMs !== undefined) await sleep(options.holdMs);
      res.setHeader('ETag', `"${options.badETag === true ? '0'.repeat(32) : createHash('md5').update(body).digest('hex')}"`); active--; res.end();
    } else if (objects.has(path)) {
      const body = objects.get(path)!;
      if (req.headers.range !== undefined) {
        const match = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range); assert(match);
        const start = Number(match[1]), end = Number(match[2]);
        res.writeHead(206, { 'Content-Range': `bytes ${start}-${end}/${body.length}`, 'Content-Length': end-start+1 });
        res.end(body.subarray(start, end+1));
      } else res.end(body);
    }
    else { res.statusCode = 404; res.end(); }
  });
  server.listen(0, '127.0.0.1'); await once(server, 'listening');
  const address = server.address(); assert(address && typeof address === 'object');
  const base = `http://127.0.0.1:${address.port}`;
  return { objects, puts, server, base, peak: () => peak };
}
function local(root: string, base: string, balance = 100) {
  return { rpc: async (_a: any, input: any) => rpc({ ...input, root }),
    sign: (_method: string, key: string) => ({ url: `${base}/${key}`, headers: {} }),
    balance: async () => balance, opts: () => [] };
}
async function waitFor(check: () => boolean) { for (let i = 0; i < 200; i++) { if (check()) return; await sleep(20); } throw new Error('Fixture timeout'); }

if (process.argv[2] === 'sim-supervisor') {
  const root = resolve(process.argv[3]), base = process.argv[4], s = JSON.parse(readFileSync(join(root, 'state.json'), 'utf8'));
  await monitor(s.active, s, local(root, base), () => atomic(join(root, 'state.json'), s), () => {}, 20, false);
  atomic(join(root, 'supervisor-done.json'), { done: true });
} else {
  test('money cutoff reserves twice the bounded transfer plus 15 minutes and stop latency', () => {
    const b = budget(60, 10, 80, 4, 1000, 100e6, 110e9);
    assert.equal(b.dollars, 50); assert.equal(b.transferSeconds, 1100);
    assert.equal(b.hardStopAtMs - b.saveAtMs, 3220_000);
    assert.equal(budget(60, 10, 2, 4, 1000, 100e6, 2e9).dollars, 2);
    assert.throws(() => budget(10, 10, 80, 4, 1000, 100e6));
    atomic(join(out, 'arithmetic.json'), b);
  });
  test('completed-fit snapshot is coherent, excludes active state, deduplicates, and restores', async () => {
    const root = mkdtempSync(join(out, 'snapshot-')); fixture(root); const s = state(root), a = s.active;
    const store = await storage();
    try {
      const io = local(root, store.base), save = () => atomic(join(root, 'state.json'), s);
      const m = await sync(a, s.plan, 'completed-fits', 'main', io as any, save);
      assert(m.files.every(e => !e.path.includes('study-active')));
      assert(m.files.some(e => e.path.endsWith('/latest.pt')));
      const puts = store.puts.filter(p => !p.includes('/snapshots/')).length;
      await sync(a, s.plan, 'completed-fits', 'same', io as any, save);
      assert.equal(store.puts.filter(p => !p.includes('/snapshots/')).length, puts);
      const resumed = join(root, 'fresh-pod'); mkdirSync(resumed);
      const restoreItems = m.files.flatMap(entry => entry.chunks.map((c, index) => ({ entry, index, signed: io.sign('GET', `${s.plan.prefix}${c.key}`) })));
      const restored = await rpc({ root: resumed, op: 'restore-batch', items: restoreItems, budgetMs: 60_000 });
      assert.equal(restored.restored, restoreItems.length);
      assert.equal(readFileSync(join(resumed, 'production/main/latest.pt'), 'utf8'), 'optimizer');
      assert.equal(readFileSync(join(resumed, 'production/main-data/bands.npy')).length, 1024 * 1024);
      assert.equal(JSON.parse(readFileSync(join(resumed, 'production/progress.json'), 'utf8')).status, 'paused');
      assert(!existsSync(join(resumed, 'production/study-active')));
      const corrupt = structuredClone(m.files[0]); corrupt.chunks[0].sha256 = '0'.repeat(64);
      await assert.rejects(restoreChunk(resumed, corrupt, 0, io.sign('GET', `${s.plan.prefix}${m.files[0].chunks[0].key}`)));
    } finally { store.server.close(); }
  });
  test('normal finish and mid-run sync survive supervisor SIGKILL and reattach', async () => {
    const root = mkdtempSync(join(out, 'reattach-')); fixture(root); const s = state(root);
    // Independent fake pod writer remains alive when the Mac supervisor is killed.
    const writer = spawn(process.execPath, ['-e', 'setInterval(()=>{},1000)'], { stdio: 'ignore' });
    atomic(join(root, 'job.json'), { pid: writer.pid, spec: {} }); atomic(join(root, 'state.json'), s);
    const store = await storage(); let child: ReturnType<typeof spawn> | null = null;
    try {
      const args = [fileURLToPath(import.meta.url), 'sim-supervisor', root, store.base];
      child = spawn(process.execPath, args, { stdio: ['ignore', 'ignore', 'inherit'] });
      await waitFor(() => existsSync(join(s.active.dir, 'main.json')));
      child.kill('SIGKILL'); await once(child, 'exit');
      process.kill(writer.pid!, 0);
      const before = store.puts.filter(p => !p.includes('/snapshots/')).length;
      child = spawn(process.execPath, args, { stdio: ['ignore', 'ignore', 'inherit'] });
      writer.kill('SIGTERM'); await once(writer, 'exit'); atomic(join(root, 'exit.json'), { code: 0, audit: 0 });
      await once(child, 'exit');
      assert(existsSync(join(root, 'supervisor-done.json')));
      assert(existsSync(join(s.active.dir, 'final.json')));
      assert(store.puts.filter(p => !p.includes('/snapshots/')).length > before);
      atomic(join(root, 'evidence.json'), { supervisorKilled: true, independentWriterSurvived: true, reattached: true, finalCommitted: true });
    } finally { child?.kill('SIGKILL'); writer.kill('SIGKILL'); store.server.close(); }
  });
  test('money-based early cutoff saves completed fits; post-start failure cannot delete', async () => {
    const root = mkdtempSync(join(out, 'cutoff-')); fixture(root); const s = state(root), store = await storage();
    try {
      await assert.rejects(monitor(s.active, s, local(root, store.base) as any, () => {}, () => {}, 1, false), /Job disappeared/);
      await sync(s.active, s.plan, 'completed-fits', 'failure', local(root, store.base) as any, () => {});
      s.active.hardStopAtMs = Date.now() + 20 * 60_000;
      const base = local(root, store.base); let interrupted = false;
      const cutting = { ...base, rpc: async (a: any, v: any) => {
        if (v.op === 'status' && !interrupted) return { alive: true, exit: null, progress: null };
        if (v.op === 'interrupt') { interrupted = true; atomic(join(root, 'exit.json'), { code: 1, audit: null }); return {}; }
        return base.rpc(a, v);
      } };
      await monitor(s.active, s, cutting as any, () => {}, () => {}, 1);
      assert(interrupted); assert(existsSync(join(s.active.dir, 'final.json')));
      let deletes = 0, stopped = false;
      const io: any = { api: async (_p: string, method = 'GET') => {
        if (method === 'DELETE') deletes++;
        if (method === 'POST') stopped = true;
        return { id: s.active.podId, createdAt: s.active.createdAt, cost: 4, status: stopped ? 'EXITED' : 'RUNNING',
          runtime: stopped ? null : {}, locked: false, actions: stopped ? [] : ['stop'] };
      } };
      await assert.rejects(removePrestart(s.active, io), /Never delete/); assert.equal(deletes, 0);
      await stop(s.active, io, () => {}); assert(stopped); assert.equal(deletes, 0);
      atomic(join(root, 'evidence.json'), { earlyCutoffSnapshot: true, stopped: true, deleted: false });
    } finally { store.server.close(); }
  });
  test('snapshot commit failure never publishes manifest and retries only missing chunks', async () => {
    const root = mkdtempSync(join(out, 'failed-upload-')); fixture(root); const s = state(root), store = await storage();
    try {
      const good = local(root, store.base);
      const bad: any = { ...good, rpc: async (a: any, v: any) => {
        if (v.op === 'upload-batch') { const partial = await good.rpc(a, { ...v, items: v.items.slice(0, 2) }); return { ...partial, failed: true }; }
        return good.rpc(a, v);
      } };
      await assert.rejects(sync(s.active, s.plan, 'completed-fits', 'main', bad, () => {}));
      assert(!store.objects.has('/3Band-corrected/local/snapshots/main.json'));
      const before = s.active.uploaded.length; assert.equal(before, 2);
      await sync(s.active, s.plan, 'completed-fits', 'main', good as any, () => {});
      assert(store.objects.has('/3Band-corrected/local/snapshots/main.json'));
    } finally { store.server.close(); }
  });
  test('final refuses a live writer and manifest rejects traversal', () => {
    const root = mkdtempSync(join(out, 'consistency-')); fixture(root);
    atomic(join(root, 'job.json'), { pid: process.pid }); atomic(join(root, 'exit.json'), { code: 0 });
    assert.throws(() => snapshot(root, 'final'), /exited writer/);
    assert.throws(() => validateManifest({ schema: 'bandnet-snapshot-v1', kind: 'final', fits: [], files: [{ path: '../escape', bytes: 0, chunks: [] }] }));
  });
  test('parallel snapshots exclude every active fit until all writers exit', () => {
    const root = mkdtempSync(join(out, 'parallel-snapshot-')); fixture(root);
    for (const name of ['study-second', 'execution-sources/execution']) mkdirSync(join(root, 'production', name), { recursive: true });
    writeFileSync(join(root, 'production/study-second/latest.pt'), 'second paused cursor');
    writeFileSync(join(root, 'production/execution-sources/execution/corrected_pilot.py'), 'reviewed scheduler');
    const progress = JSON.parse(readFileSync(join(root, 'production/progress.json'), 'utf8'));
    progress.active_fits = ['study-active', 'study-second'];
    atomic(join(root, 'production/progress.json'), progress);
    atomic(join(root, 'job.json'), { pid: process.pid });
    const mid = snapshot(root, 'completed-fits');
    assert(!mid.files.some(e => e.path.includes('study-active') || e.path.includes('study-second')));
    assert(mid.files.some(e => e.path.includes('execution-sources/')));
    assert.deepEqual(JSON.parse(readFileSync(join(root, 'snapshot-meta/progress.json'), 'utf8')).active_fits, []);
    atomic(join(root, 'job.json'), { pid: 2147483647 });
    atomic(join(root, 'exit.json'), { code: 1 });
    const final = snapshot(root, 'final');
    for (const name of progress.active_fits) assert(final.files.some(e => e.path === `production/${name}/latest.pt`));
  });
  test('exited coordinator cannot leave a fit process writing into a final snapshot', async () => {
    const root = mkdtempSync(join(out, 'process-group-'));
    const heartbeat = join(root, 'heartbeat');
    const worker = `const fs=require('node:fs'); setInterval(()=>fs.appendFileSync(${JSON.stringify(heartbeat)},'.'),20)`;
    const parent = `const {spawn}=require('node:child_process'); const fs=require('node:fs');
      spawn(process.execPath,['-e',${JSON.stringify(worker)}],{stdio:'ignore'});
      setInterval(()=>{if(fs.existsSync(${JSON.stringify(heartbeat)}))process.exit(0)},20)`;
    assert.equal(await command(process.execPath, ['-e', parent], root, join(root, 'job.log'), Date.now()+5000, 'training'), 0);
    await sleep(100);
    const stopped = readFileSync(heartbeat, 'utf8');
    await sleep(100);
    assert.equal(readFileSync(heartbeat, 'utf8'), stopped);
  });
  test('production controller uses receipt-derived long deadline, not 900 seconds', async () => {
    const start = 1000000, events: any[] = []; let now = start, stopped = false;
    const receipt = { podId: 'test', createdAt: new Date(start).toISOString(), allocationRequestedAtMs: start, hardStopAtMs: start + 10 * 3600_000 };
    await watch(receipt, { now: () => now, delay: async ms => { now += ms; }, earlyStop: () => false,
      getPod: async () => ({ id: 'test', createdAt: receipt.createdAt, cost: 4, locked: false, actions: stopped ? [] : ['stop'], status: stopped ? 'EXITED' : 'RUNNING', runtime: stopped ? null : {} }),
      stopPod: async () => { stopped = true; }, record: e => events.push(e) });
    assert(events.find(e => e.event === 'stop-requested').elapsedMs >= 10 * 3600_000 - 60_000);
  });
  test('transient status, mid-save and final-save failures retry without stopping training', async () => {
    const root = mkdtempSync(join(out, 'retry-')); fixture(root); const s = state(root), store = await storage();
    let statusCalls = 0, failedMid = false, finalFailures = 0, interrupted = false;
    const base = local(root, store.base);
    const io = { ...base, rpc: async (a: any, v: any) => {
      if (v.op === 'interrupt') { interrupted = true; throw Error('Unexpected stop'); }
      if (v.op === 'status') {
        statusCalls++;
        if (statusCalls <= 4) throw Error('Mac network outage');
        if (statusCalls < 9) return { alive: true, exit: null, progress: { fits: { main: {} } } };
        atomic(join(root, 'exit.json'), { code: 0, audit: 0 });
      }
      if (v.op === 'upload-batch' && !failedMid) { failedMid = true; throw Error('Transient PUT'); }
      if (v.op === 'put-manifest' && v.signed.url.includes('/snapshots/final.json') && finalFailures++ < 2) throw Error('Transient final PUT');
      return base.rpc(a, v);
    } };
    try { await monitor(s.active, s, io as any, () => {}, () => {}, 1); assert(failedMid); assert(!interrupted); assert(finalFailures >= 3); }
    finally { store.server.close(); }
  });
  test('two impossible zero balances do not shorten the money ceiling', async () => {
    const root = mkdtempSync(join(out, 'balance-')); fixture(root); const s = state(root), ceiling = s.active.hardStopAtMs, store = await storage();
    let reads = 0, calls = 0; const base = local(root, store.base);
    const io = { ...base, balance: async () => ++reads <= 2 ? 0 : 100, rpc: async (a: any, v: any) => {
      if (v.op === 'status' && ++calls < 3) return { alive: true, exit: null, progress: null };
      if (v.op === 'status') atomic(join(root, 'exit.json'), { code: 0, audit: 0 });
      if (v.op === 'interrupt') throw Error('Bogus balance caused interruption');
      return base.rpc(a, v);
    } };
    try { await monitor(s.active, s, io as any, () => {}, () => {}, 1); assert.equal(s.active.hardStopAtMs, ceiling); }
    finally { store.server.close(); }
  });
  test('supervise placement, ambiguous adoption, admission, cleanup and resume use the real state machine', async () => {
    for (const scenario of ['normal', '400', '402', 'ambiguous', 'setup-failure', 'arm-failure', 'malformed', 'reattach', 'global-lock', 'resume', 'probe-only', 'probe-failure']) {
      const root = mkdtempSync(join(out, `supervise-${scenario}-`)), dir = join(root, 'run'); mkdirSync(dir);
      const s = state(root); s.active = null; s.initialBalance = 100;
      if (scenario === 'resume') s.plan.resume = '3Band-corrected/old/snapshots/final.json';
      if (scenario.startsWith('probe-')) { s.plan.probeOnly = true; s.plan.rehearsal = true; s.plan.cap = 1; }
      atomic(join(dir, 'state.json'), s);
      let creates = 0, deletes = 0, arms = 0, setups = 0, admitted = 0, offers = 0, checks = 0, lag = 0;
      let pod: any = null;
      const makePod = () => ({ id: `pod${creates}`, name: `bandnet-local-${creates}`, createdAt: new Date().toISOString(), cost: 4,
        disk: 30, mounts: { persistent: { size: 220 } }, status: 'RUNNING', actions: ['stop'], locked: false, runtime: {} });
      const io: any = {
        check: async () => { checks++; }, balance: async () => 100,
        resumeSize: async () => 100e9 + 7,
        offers: async () => { offers++; return [{ gpu: 'NVIDIA H100 NVL', cloud: 'SECURE' }, { gpu: 'NVIDIA H100 PCIe', cloud: 'SECURE' }]; },
        api: async (path: string, method = 'GET', body: any) => {
          if (path.startsWith('/pods?')) {
            if (lag-- > 0) return { pods: [] };
            return { pods: pod === null ? [] : [pod] };
          }
          if (path === '/pods' && method === 'POST') {
            creates++;
            if (scenario === '402') throw new APIError(402);
            if (scenario === '400' && creates === 1) throw new APIError(400);
            pod = makePod(); pod.name = body.name;
            if (scenario === 'malformed') pod.cost = NaN;
            if (scenario === 'ambiguous' && creates === 1) { lag = 2; throw new APIError(500); }
            return pod;
          }
          if (method === 'DELETE') { deletes++; pod = null; return null; }
          if (method === 'POST') { pod.status = 'EXITED'; pod.runtime = null; pod.actions = []; return pod; }
          return pod;
        },
        arm: async (a: any) => { arms++; if (scenario === 'arm-failure' && arms === 1) throw Error('arm failed');
          atomic(join(a.dir, `watchdog-${a.podId}`, 'armed.json'), { pid: 0 }); },
        setup: async (a: any, state: any, save: () => void) => {
          setups++; if (scenario === 'setup-failure' && setups === 1) throw Error('bad placement');
          if (scenario === 'probe-failure') throw Error('slow failed probe');
          if (scenario === 'resume') {
            const receipt = JSON.parse(readFileSync(join(a.dir, 'receipt.json'), 'utf8'));
            assert.equal(receipt.setupStopAtMs - a.allocationRequestedAtMs, 45 * 60_000 + Math.floor(2 * (100e9 + 7) / 20e6 * 1000));
            assert.equal(validateReceipt(receipt), receipt);
          }
          a.ready = true; a.upload = 100e6; save();
        },
        rpc: async (a: any, input: any) => { if (input.op === 'start') { assert(a.started); assert(existsSync(join(a.dir, `watchdog-${a.podId}`, 'admitted.json'))); admitted++; } return {}; },
        monitor: async () => {
          if (scenario === 'global-lock') {
            const second = join(root, 'other'); mkdirSync(second);
            atomic(join(second, 'state.json'), { ...s, plan: { ...s.plan, id: 'other' }, active: null });
            await assert.rejects(supervise(second, io, async () => {}, 1, join(root, 'global-lock')), /global launch lock/);
            assert(!existsSync(join(second, 'mutex')));
          }
          pod.status = 'EXITED'; pod.runtime = null; pod.actions = [];
        },
        retrieve: async () => {}, notify: () => {}, sign: () => ({ url: '', headers: {} }),
      };
      if (scenario === 'reattach') {
        pod = makePod(); const old = state(root).active; old.podId = pod.id; old.createdAt = pod.createdAt;
        s.active = old; atomic(join(dir, 'state.json'), s);
      }
      if (scenario === '402' || scenario === 'malformed') await assert.rejects(supervise(dir, io, async () => {}, 3, join(root, 'global-lock')));
      else await supervise(dir, io, async () => {}, 3, join(root, 'global-lock'));
      if (scenario === 'normal' || scenario === 'ambiguous') { assert.equal(creates, 1); assert.equal(admitted, 1); assert.equal(deletes, 0); }
      if (scenario === '400') assert.equal(creates, 2);
      if (scenario === '402') { assert.equal(creates, 1); assert.equal(admitted, 0); }
      if (scenario === 'setup-failure' || scenario === 'arm-failure') { assert.equal(deletes, 1); assert.equal(admitted, 1); }
      if (scenario === 'malformed') assert.equal(deletes, 2);
      if (scenario === 'reattach') { assert.equal(checks, 0); assert.equal(offers, 0); assert.equal(creates, 0); assert(arms > 0); }
      if (scenario.startsWith('probe-')) { assert.equal(creates, 1); assert.equal(deletes, 1); assert.equal(admitted, 0); assert(JSON.parse(readFileSync(join(dir, 'state.json'), 'utf8')).finished); }
      atomic(join(root, 'evidence.json'), { scenario, creates, deletes, arms, setups, admitted, checks, offers });
    }
  });
  test('setup watchdog deletes an unadmitted provisioning pod at its setup deadline', async () => {
    let now = 1000000, deleted = false;
    const receipt = { podId: 'setup', createdAt: new Date(now).toISOString(), allocationRequestedAtMs: now,
      hardStopAtMs: now + 10 * 3600_000, setupStopAtMs: now + 45 * 60_000 };
    await watch(receipt, { now: () => now, delay: async ms => { now += ms; }, earlyStop: () => false,
      setupExpired: () => now >= receipt.setupStopAtMs, deletePrestart: async () => { deleted = true; },
      getPod: async () => ({ id: 'setup', createdAt: receipt.createdAt, cost: 4, locked: false, actions: ['start', 'terminate'], status: 'INITIALIZING', runtime: null }),
      stopPod: async () => { throw Error('Provisioning cannot stop'); }, record: () => {} });
    assert(deleted); assert.equal(now, receipt.setupStopAtMs);
  });
  test('prestart DELETE retries and waits for eventual inventory absence without STOP', async () => {
    const root = mkdtempSync(join(out, 'delete-')); const a = state(root).active; a.started = false;
    let deletes = 0, reads = 0;
    const io: any = { api: async (_path: string, method = 'GET') => {
      assert.notEqual(method, 'POST');
      if (method === 'DELETE') { if (++deletes === 1) throw new APIError(500); return null; }
      return { pods: ++reads < 3 ? [{ id: a.podId }] : [] };
    } };
    await removePrestart(a, io, async () => {}); assert.equal(deletes, 3); assert(existsSync(join(a.dir, 'deleted.json')));
  });
  test('repeated probes cannot reuse receipts, and real-save throughput replaces the probe', async () => {
    const seen = new Set(); const urls = Array.from({ length: 16 }, () => ({ url: 'http://local', headers: {} }));
    const send = async (items: any[]) => ({ failed: false, results: items.map(item => {
      const bytes = seen.has(item.key) ? 0 : 64 * 1024 ** 2; seen.add(item.key); return { key: item.key, bytes };
    }) });
    assert.equal(await probeTransfers(urls, send), 1024 ** 3);
    assert.equal(await probeTransfers(urls, send), 1024 ** 3); assert.equal(seen.size, 32);
    await assert.rejects(probeTransfers(urls, async () => ({ failed: false, results: urls.map(() => ({ bytes: 0 })) })));
    const a = state(mkdtempSync(join(out, 'rate-'))).active;
    a.upload = 1e12;
    updateTransferRate(a, 'foreground', 1024 ** 3, 100); assert.equal(a.upload, 1024 ** 3 / 100);
    assert.equal(a.backgroundUpload, 25e6);
    updateTransferRate(a, 'background', 1024 ** 3, 200); assert.equal(a.backgroundUpload, 1024 ** 3 / 200);
    assert.equal(a.upload, 1024 ** 3 / 100);
  });
  test('pod enforces the scheduled training stop, while audit ignores leftover training interruption', async () => {
    const root = mkdtempSync(join(out, 'scheduled-'));
    atomic(join(root, 'job-spec.json'), { saveAtMs: Date.now() + 5000 });
    await rpc({ root, op: 'schedule-stop', saveAtMs: Date.now() + 100 });
    const code = await command(process.execPath, ['-e', 'process.on("SIGINT",()=>process.exit(0));setInterval(()=>{},1000)'], root, join(root, 'job.log'), Date.now() + 5000, 'training');
    assert.equal(code, 0); assert(existsSync(join(root, 'interrupt.json')));
    await rpc({ root, op: 'schedule-stop', saveAtMs: Date.now() + 4000 });
    const audit = await command(process.execPath, ['-e', 'setTimeout(()=>process.exit(0),1200)'], root, join(root, 'audit.log'), Date.now() + 5000, 'audit');
    assert.equal(audit, 0);
    atomic(join(root, 'job.json'), { pid: process.pid }); atomic(join(root, 'python.json'), { pid: process.pid, phase: 'audit' });
    await rpc({ root, op: 'interrupt' }); // Must return before any attempt to signal this test runner.
  });
  test('audit may exceed ten minutes but cannot spend the final-save reserve', async () => {
    const root = mkdtempSync(join(out, 'audit-deadline-'));
    const ceiling = Date.now() + 3600_000;
    atomic(join(root, 'job-spec.json'), { saveAtMs: ceiling });
    assert.equal(auditDeadline(root, ceiling), ceiling);
    await rpc({ root, op: 'schedule-stop', saveAtMs: ceiling - 600_000 });
    assert(auditDeadline(root, ceiling) - Date.now() > 600_000);
    await rpc({ root, op: 'schedule-stop', saveAtMs: ceiling + 600_000 });
    assert.equal(auditDeadline(root, ceiling), ceiling);
    // A live balance/upload revision must reach an already-running auditor,
    // even with an old immutable command ceiling and no interrupt RPC.
    const running = command(process.execPath, ['-e', 'setInterval(()=>{},1000)'], root,
      join(root, 'audit.log'), ceiling, 'audit');
    await rpc({ root, op: 'schedule-stop', saveAtMs: Date.now() + 100 });
    assert.equal(await running, 137);
  });
  test('background batches use their own rate, exceed 32 chunks, and shrink after failure', async () => {
    const root = mkdtempSync(join(out, 'mode-budget-')), s = state(root), a = s.active;
    a.upload = 240e6; a.backgroundUpload = 60e6;
    const bytes = 64 * 1024 ** 2;
    const files = Array.from({ length: 400 }, (_, i) => ({ path: `production/main-data/${i}.bin`, bytes,
      chunks: [{ bytes, offset: 0, sha256: hash(String(i)) }] }));
    const batches: any[] = []; let failed = false;
    const io: any = { sign: () => ({}), rpc: async (_a: any, input: any) => {
      if (input.op === 'snapshot') return { schema: 'bandnet-snapshot-v1', kind: 'completed-fits', fits: ['main'], files };
      if (input.op !== 'upload-batch') return {};
      batches.push(input);
      assert.equal(input.mode, 'background'); assert.equal(input.deadlineAtMs, undefined);
      if (!failed) { failed = true; return { failed: true, error: 'deadline', results: [] }; }
      return { failed: false, results: input.items.map((i: any) => ({ key: i.key, bytes: 0 })) };
    } };
    await assert.rejects(sync(a, s.plan, 'completed-fits', 'main', io, () => {}), /deadline/);
    assert.equal(a.upload, 240e6); assert.equal(a.backgroundUpload, 30e6);
    assert(batches[0].items.length > 32);
    assert(batches[0].budgetMs > 2 * batches[0].items.length * bytes / 60e6 * 1000);
    await sync(a, s.plan, 'completed-fits', 'main', io, () => {});
    assert(batches[1].items.length < batches[0].items.length); assert.equal(a.upload, 240e6);
  });
  test('probe journals every key before RPC and excludes disk/hash from rate', async () => {
    const s = state(mkdtempSync(join(out, 'probe-accounting-'))), a = s.active;
    const io: any = { sign: () => ({}), rpc: async (_a: any, input: any, timeout: number) => {
      assert(timeout - input.budgetMs >= 30_000);
      assert.equal(JSON.parse(readFileSync(join(a.dir, 'probe-cleanup.json'), 'utf8')).length, 40);
      return { elapsedSeconds: 100, write: 400e6, sha: 500e6, failed: false,
        waves: [0, 1, 2].map(i => ({ bytes: (i === 2 ? 16 : 1) * 1024 ** 3, seconds: i === 2 ? 80 : i === 1 ? 20 : 5, objects: i === 2 ? 8 : 16, failed: false })) };
    } };
    const probe = await measureProbe(a, s.plan, io);
    assert(probe.passed); assert.equal(probe.sshOverheadSeconds, 0);
    assert.equal(probe.upload, 16 * 1024 ** 3 / 80); assert.equal(probe.backgroundUpload, 1024 ** 3 / 20);
    io.rpc = async () => ({ elapsedSeconds: 450, write: 400e6, sha: 500e6, failed: false,
      waves: [{ bytes: 1024 ** 3, seconds: 35, objects: 16, failed: false },
        { bytes: 1024 ** 3, seconds: 40, objects: 16, failed: false },
        { bytes: 0, sentBytes: 9e9, seconds: 350, objects: 0, failed: true, partial: true }] });
    const partial = await measureProbe(a, s.plan, io);
    assert(partial.passed); assert.equal(partial.upload, 9e9 / 350);
    io.rpc = async () => { throw Error('https://secret.example/?X-Amz-Signature=hidden'); };
    const failure = await measureProbe(a, s.plan, io);
    assert(!failure.passed); assert(!failure.error.includes('hidden'));
    assert.equal(JSON.parse(readFileSync(join(a.dir, 'probe-cleanup.json'), 'utf8')).length, 120);
  });
  test('stream deadline retains conservative sent bytes without creating an S3 receipt', async () => {
    const root = mkdtempSync(join(out, 'partial-stream-')), body = Buffer.alloc(64 * 1024 ** 2, 7);
    writeFileSync(join(root, 'payload'), body);
    const manifest: any = { schema: 'bandnet-snapshot-v1', kind: 'progress', fits: [], files: [
      { path: 'payload', bytes: body.length, chunks: [{ offset: 0, bytes: body.length, sha256: hash(body) }] }] };
    const server = createServer((req, res) => {
      req.on('data', () => { req.pause(); setTimeout(() => req.resume(), 2); });
      req.on('end', () => res.end());
    });
    server.listen(0, '127.0.0.1'); await once(server, 'listening');
    const address: any = server.address(); let sent = 0;
    try {
      await assert.rejects(upload(root, 'payload', 0, { url: `http://127.0.0.1:${address.port}/partial`, headers: {} },
        { manifest, paths: { payload: join(root, 'payload') } }, Date.now() + 800, bytes => { sent = bytes; }), /abort/i);
      assert(sent > 0 && sent < body.length); assert(!existsSync(join(root, 'upload-receipts')));
    } finally { server.closeAllConnections(); server.close(); }
  });
  test('early S3 rejection stops the stream promptly and retains only the XML Code', async () => {
    const root = mkdtempSync(join(out, 'early-response-')); fixture(root); snapshot(root, 'completed-fits');
    const server = createServer((_req, res) => {
      res.writeHead(403, { 'Content-Type': 'application/xml' });
      res.write('<Error><Code>AccessDenied</Code><Message>https://secret.example</Message></Error>');
      // Deliberately never end the response.
    });
    server.listen(0, '127.0.0.1'); await once(server, 'listening'); const address: any = server.address();
    const start = performance.now();
    try {
      await assert.rejects(upload(root, 'production/main-data/bands.npy', 0,
        { url: `http://127.0.0.1:${address.port}/rejected`, headers: {} }, undefined, Date.now() + 10_000),
        error => /HTTP 403 \(AccessDenied\)/.test(String(error)) && !String(error).includes('secret'));
      assert(performance.now() - start < 2000);
    } finally { server.closeAllConnections(); server.close(); }
  });
  test('restore retries a dropped S3 connection and still verifies the checksum', async () => {
    const root = mkdtempSync(join(out, 'dropped-')), body = Buffer.from('restored bytes');
    const entry = { path: 'payload', bytes: body.length, chunks: [{ offset: 0, bytes: body.length, sha256: hash(body) }] };
    let requests = 0;
    const server = createServer((req, res) => {
      requests++;
      if (requests === 1) { req.socket.destroy(); return; }
      res.writeHead(206, { 'Content-Range': `bytes 0-${body.length-1}/${body.length}` });
      res.end(body);
    });
    server.listen(0, '127.0.0.1'); await once(server, 'listening'); const address: any = server.address();
    try {
      await restoreChunk(root, entry, 0, { url: `http://127.0.0.1:${address.port}/payload`, headers: {} }, true, Date.now() + 30_000);
      assert.equal(requests, 2); assert.deepEqual(readFileSync(join(root, 'payload')), body);
    } finally { server.closeAllConnections(); server.close(); }
  });
  test('presigned GET permits unsigned Range headers', () => {
    const signed = presign({ access: 'fixture', secret: 'fixture' }, { bucket: 'fixture-bucket', owner: '000000000000', region: 'us-west-1' }, 'GET', '3Band-corrected/fixture/chunk');
    assert.equal(new URL(signed.url).searchParams.get('X-Amz-SignedHeaders'), 'host;x-amz-expected-bucket-owner');
    assert(!Object.keys(signed.headers).some(key => key.toLowerCase() === 'range'));
  });
  test('multi-range restore retries only a dropped range and preserves chunk file offsets', async () => {
    const root = mkdtempSync(join(out, 'ranged-')), prefix = Buffer.from('preserved-prefix');
    const body = Buffer.alloc(RESTORE_RANGE_BYTES + 8193, 0x53);
    body.fill(0xa7, RESTORE_RANGE_BYTES);
    const requests = new Map<number, number>();
    const server = createServer((req, res) => {
      const match = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range!); assert(match);
      const start = Number(match[1]), end = Number(match[2]);
      requests.set(start, requests.has(start) ? requests.get(start)! + 1 : 1);
      res.writeHead(206, { 'Content-Range': `bytes ${start}-${end}/${body.length}`, 'Content-Length': end-start+1 });
      if (start === RESTORE_RANGE_BYTES && requests.get(start) === 1) {
        res.write(body.subarray(start, start+100)); setImmediate(() => res.destroy()); return;
      }
      res.end(body.subarray(start, end+1));
    });
    server.listen(0, '127.0.0.1'); await once(server, 'listening'); const address: any = server.address();
    const entry = { path: 'payload', bytes: prefix.length+body.length, chunks: [
      { offset: 0, bytes: prefix.length, sha256: hash(prefix) },
      { offset: prefix.length, bytes: body.length, sha256: hash(body) }] };
    writeFileSync(join(root, 'payload'), prefix);
    try {
      await restoreChunk(root, entry, 1, { url: `http://127.0.0.1:${address.port}/payload?secret=private`, headers: {} }, false, Date.now()+30_000);
      const actual = readFileSync(join(root, 'payload'));
      assert.deepEqual(actual.subarray(0, prefix.length), prefix);
      assert.deepEqual(actual.subarray(prefix.length), body);
      assert.equal(requests.get(0), 1); assert.equal(requests.get(RESTORE_RANGE_BYTES), 2);
    } finally { server.closeAllConnections(); server.close(); }
  });
  test('restore rejects ignored ranges, wrong Content-Range, 4xx, and corrupted disk bytes without retry', async () => {
    for (const mode of ['ignored', 'wrong-range', 'denied', 'corrupt']) {
      const root = mkdtempSync(join(out, 'bad-range-')), body = Buffer.from('payload'); let requests = 0;
      const server = createServer((_req, res) => {
        requests++;
        if (mode === 'denied') { res.writeHead(403); res.end('denied'); return; }
        if (mode !== 'ignored') res.writeHead(206, { 'Content-Range': `bytes ${mode === 'wrong-range' ? 1 : 0}-${body.length-1}/${body.length}` });
        res.end(mode === 'corrupt' ? Buffer.alloc(body.length) : body);
      });
      server.listen(0, '127.0.0.1'); await once(server, 'listening'); const address: any = server.address();
      try {
        const entry = { path: 'payload', bytes: body.length, chunks: [{ offset: 0, bytes: body.length, sha256: hash(body) }] };
        await assert.rejects(restoreChunk(root, entry, 0, { url: `http://127.0.0.1:${address.port}/private?X-Amz-Signature=secret`, headers: {} }), error => {
          const message = String(error);
          assert(!message.includes('secret') && !message.includes('http://'));
          return mode === 'corrupt' ? /checksum mismatch/.test(message) : mode === 'denied' ? /HTTP 403/.test(message) : /ranged GET response/.test(message);
        });
        assert.equal(requests, 1);
      } finally { server.closeAllConnections(); server.close(); }
    }
  });
  test('restore batch uses one global 32-stream pool', async () => {
    const root = mkdtempSync(join(out, 'restore-pool-')); let active = 0, peak = 0;
    const server = createServer(async (_req, res) => {
      peak = Math.max(peak, ++active); await sleep(50);
      res.writeHead(206, { 'Content-Range': 'bytes 0-0/1' }); active--; res.end('x');
    });
    server.listen(0, '127.0.0.1'); await once(server, 'listening'); const address: any = server.address();
    try {
      const items = Array.from({ length: 40 }, (_, i) => ({
        entry: { path: `file-${i}`, bytes: 1, chunks: [{ offset: 0, bytes: 1, sha256: hash('x') }] },
        index: 0, signed: { url: `http://127.0.0.1:${address.port}/payload`, headers: {} } }));
      const result = await rpc({ root, op: 'restore-batch', items, budgetMs: 30_000 });
      assert.equal(result.restored, 40); assert.equal(peak, RESTORE_CONCURRENCY);
    } finally { server.closeAllConnections(); server.close(); }
  });
  test('relative batch budgets reject expired work and restore keeps the actual error', async () => {
    const root = mkdtempSync(join(out, 'relative-')), store = await storage();
    try {
      const entry = { path: 'missing', bytes: 1, chunks: [{ bytes: 1, offset: 0, sha256: hash('x') }] };
      await assert.rejects(rpc({ root, op: 'restore-batch', budgetMs: 1000,
        items: [{ entry, index: 0, signed: { url: `${store.base}/absent`, headers: {} } }] }), /HTTP 404/);
      await assert.rejects(rpc({ root, op: 'restore-batch', budgetMs: 0, items: [] }), /relative batch budget/);
    } finally { store.server.close(); }
  });
  test('a deleted allocation controller exits on 404 only with its deletion receipt', async () => {
    const now = 1000000; let record = '';
    await watch({ podId: 'deleted', createdAt: new Date(now).toISOString(), allocationRequestedAtMs: now }, {
      now: () => now, delay: async () => { throw Error('Should not keep polling'); }, earlyStop: () => false,
      getPod: async () => { throw Object.assign(Error('gone'), { status: 404 }); }, stopPod: async () => {},
      deletionRecorded: () => true, record: e => { record = e.event; },
    });
    assert.equal(record, 'deleted-receipt-verified');
  });
  test('an interrupted final save leaves the newest results manifest last in commit order', async () => {
    const root = mkdtempSync(join(out, 'commit-order-')); fixture(root); const s = state(root), store = await storage();
    try {
      const good = local(root, store.base);
      await sync(s.active, s.plan, 'completed-fits', 'main', good as any, () => {});
      atomic(join(root, 'exit.json'), { code: 0, audit: 0 });
      const failed = { ...good, rpc: async (a: any, v: any) => {
        if (v.op === 'put-manifest' && v.signed.url.endsWith('/snapshots/final.json')) throw Error('Lost final commit');
        return good.rpc(a, v);
      } };
      await assert.rejects(sync(s.active, s.plan, 'final', 'final', failed as any, () => {}));
      assert.equal(s.active.syncs.at(-1), 'results');
      const latest = JSON.parse(readFileSync(join(s.active.dir, 'results.json'), 'utf8'));
      assert(latest.files.some((e: any) => e.path === 'exit.json'));
    } finally { store.server.close(); }
  });
  test('mid-run snapshot is deferred rather than crossing the scheduled save time', async () => {
    const root = mkdtempSync(join(out, 'defer-')), s = state(root);
    s.active.saveAtMs = Date.now() + 30_000;
    const io: any = { rpc: async () => { throw Error('Should not start snapshot'); }, sign: () => ({}) };
    await assert.rejects(sync(s.active, s.plan, 'completed-fits', 'main', io, () => {}), /deferred/);
  });
  test('transfer pool bounds concurrency, drains siblings on failure and stops scheduling', async () => {
    for (const limit of [2, 8]) {
      let active = 0, maximum = 0;
      const result = await transferPool(Array.from({ length: 24 }, (_, i) => i), limit, async i => {
        maximum = Math.max(maximum, ++active); await sleep(5); active--; return i;
      });
      assert.equal(maximum, limit); assert.equal(active, 0); assert.equal(result.results.length, 24); assert(!result.failed);
    }
    let active = 0, started = 0;
    const failed = await transferPool(Array.from({ length: 40 }, (_, i) => i), 8, async i => {
      started++; active++; await sleep(i === 0 ? 1 : 10); active--; if (i === 0) throw Error('fault'); return i;
    });
    assert(failed.failed); assert.equal(active, 0); assert.equal(started, 8); assert.equal(failed.results.length, 7);
  });
  test('parallel restore writes one file out of order without truncating sibling ranges', async () => {
    const root = mkdtempSync(join(out, 'parallel-restore-')), store = await storage();
    try {
      const buffers = Array.from({ length: 16 }, (_, i) => Buffer.alloc(16384, i));
      const entry = { path: 'large-array.npy', bytes: 16 * 16384, chunks: buffers.map((b, i) => ({ bytes: b.length, offset: i * b.length, sha256: hash(b) })) };
      buffers.forEach((b, i) => store.objects.set(`/part-${i}`, b));
      const items = entry.chunks.map((_c, index) => ({ entry, index, signed: { url: `${store.base}/part-${index}`, headers: {} } })).reverse();
      await rpc({ root, op: 'restore-batch', items, budgetMs: 60_000 });
      assert.deepEqual(readFileSync(join(root, entry.path)), Buffer.concat(buffers));
    } finally { store.server.close(); }
  });
  test('real HTTP uploads overlap within mode limits and reject bad ETags before receipts', async () => {
    for (const [mode, limit] of [['foreground', 8], ['background', 2]] as const) {
      const root = mkdtempSync(join(out, 'http-parallel-')); fixture(root);
      const manifest = snapshot(root, 'completed-fits');
      const options = { badETag: true, holdMs: 30 }, store = await storage(options);
      const items = manifest.files.flatMap(e => e.chunks.map((_c, index) => ({ key: `${e.path}-${index}`, file: e.path, index,
        signed: { url: `${store.base}/${e.path}-${index}`, headers: {} } })));
      try {
        const bad = await rpc({ root, op: 'upload-batch', mode, budgetMs: 60_000, items });
        assert(bad.failed); assert.equal(bad.results.length, 0);
        assert(!existsSync(join(root, 'upload-receipts')));
        options.badETag = false;
        const good = await rpc({ root, op: 'upload-batch', mode, budgetMs: 60_000, items });
        assert(!good.failed); assert.equal(good.results.length, items.length);
        assert(store.peak() > 1); assert(store.peak() <= limit);
        const before = store.puts.length;
        const retry = await rpc({ root, op: 'upload-batch', mode, budgetMs: 60_000, items });
        assert(retry.results.every((r: any) => r.bytes === 0)); assert.equal(store.puts.length, before);
      } finally { store.server.close(); }
    }
  });
}
