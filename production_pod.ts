/** Credential-free pod worker. RPC over SSH stdin; never prints signed URLs. */
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, renameSync,
  statSync, writeFileSync, writeSync, unlinkSync, fsyncSync, createReadStream } from 'node:fs';
import { spawn } from 'node:child_process';
import { createHash, randomUUID } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';
import { request as httpsRequest } from 'node:https';
import { request as httpRequest } from 'node:http';
import { pipeline } from 'node:stream/promises';
import { CHUNK, WHOLE_LIMIT, TRANSFER_CONCURRENCY, BACKGROUND_CONCURRENCY, atomic, hash, invariant, safePath, validateManifest } from './production_common.ts';
import type { Entry, Manifest } from './production_common.ts';

const json = (p: string) => JSON.parse(readFileSync(p, 'utf8'));
function walk(root: string, prefix = ''): string[] {
  const found: string[] = [];
  for (const e of readdirSync(join(root, prefix), { withFileTypes: true })) {
    invariant(!e.isSymbolicLink(), 'Snapshot symlink refused');
    const p = prefix === '' ? e.name : `${prefix}/${e.name}`;
    if (e.isDirectory()) found.push(...walk(root, p));
    else if (e.isFile()) found.push(p);
  }
  return found.sort();
}
function alive(pid: number) { try { process.kill(pid, 0); return true; } catch { return false; } }
function groupHasWriters(group: number) {
  if (process.platform !== 'linux') {
    try { process.kill(-group, 0); return true; } catch { return false; }
  }
  // Orphaned zombies can remain under a container PID 1 without writing anything.
  for (const pid of readdirSync('/proc').filter(p => /^\d+$/.test(p))) {
    try {
      const stat = readFileSync(`/proc/${pid}/stat`, 'utf8');
      const fields = stat.slice(stat.lastIndexOf(')') + 2).split(' ');
      if (Number(fields[2]) === group && fields[0] !== 'Z' && fields[0] !== 'X') return true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT' && (error as NodeJS.ErrnoException).code !== 'ESRCH') throw error;
    }
  }
  return false;
}
export function status(root: string) {
  const job = existsSync(join(root, 'job.json')) ? json(join(root, 'job.json')) : null;
  const running = job !== null && alive(job.pid);
  const exit = !running && existsSync(join(root, 'exit.json')) ? json(join(root, 'exit.json')) : null;
  let progress = existsSync(join(root, 'production/progress.json')) ? json(join(root, 'production/progress.json')) : null;
  if (progress === null && job?.spec?.rehearsal && existsSync(join(root, 'production/report.json'))) progress = { fits: {}, evaluations: {} };
  return { job, exit, alive: running, progress };
}
export function snapshot(root: string, kind: Manifest['kind']): Manifest {
  const state = status(root);
  invariant(state.progress !== null, 'No production state to export');
  if (kind === 'final') invariant(state.exit !== null && !state.alive, 'Final snapshot requires an exited writer');
  if (kind === 'final' && existsSync(join(root, 'python.json'))) {
    const python = json(join(root, 'python.json'));
    if (python.group !== undefined) invariant(!groupHasWriters(python.group), 'Final snapshot requires all fit writers to exit');
  }
  const fits = Object.keys(state.progress.fits).sort();
  // A completed fit's model/data/history never changes again. Exclude ALL active
  // fit/evaluation files. Construct a coherent earlier fit-boundary progress record.
  // SIGSTOP of an arbitrary write is NOT an application-consistent snapshot.
  const selected = walk(join(root, 'production')).filter(p => {
    if (kind === 'final') return !p.endsWith('.tmp') && !(state.job?.spec?.rehearsal && p.startsWith('bulk/'));
    if (p === 'progress.json' || p === 'protocol.json' || p.startsWith('sources/') || p.startsWith('executions/') || p.startsWith('execution-sources/')) return true;
    return fits.some(f => p.startsWith(`${f}/`) && p.split('/').length === 2 ||
      p.startsWith(`${f}-data/`) || p === `${f}-preflight.json` || p === `${f}-calibrations.json`);
  }).filter(p => kind !== 'progress' || (!p.includes('-data/') && !p.endsWith('latest.pt')));
  mkdirSync(join(root, 'snapshot-meta'), { recursive: true });
  if (kind !== 'final') {
    const copy = structuredClone(state.progress);
    copy.evaluations = {}; copy.active_fits = []; copy.status = 'paused';
    atomic(join(root, 'snapshot-meta/progress.json'), copy);
  }
  const cachePath = join(root, 'hash-cache.json');
  const cache: Record<string, { signature: string; entry: Entry }> = existsSync(cachePath) ? json(cachePath) : {};
  const files: Entry[] = [];
  const paths: Record<string, string> = {};
  for (const rel of selected) paths[`production/${rel}`] = kind !== 'final' && rel === 'progress.json'
    ? join(root, 'snapshot-meta/progress.json') : join(root, 'production', rel);
  for (const rel of ['source.tar', 'job.log', 'audit.json', 'audit.log', 'exit.json']) {
    // Running logs are not immutable; collect them only after the writer exits.
    if ((rel === 'source.tar' || kind === 'final') && existsSync(join(root, rel))) paths[rel] = join(root, rel);
  }
  for (const [rel, path] of Object.entries(paths)) {
    const st = statSync(path), signature = `${st.size}:${st.mtimeMs}:${st.ino}`;
    if (cache[rel] !== undefined && cache[rel].signature === signature) { files.push(cache[rel].entry); continue; }
    const entry: Entry = { path: rel, bytes: st.size, chunks: [] }, fd = openSync(path, 'r');
    if (st.size === 0) entry.chunks.push({ sha256: hash(Buffer.alloc(0)), bytes: 0, offset: 0 });
    try {
      const unit = st.size <= WHOLE_LIMIT ? WHOLE_LIMIT : CHUNK;
      for (let offset = 0; offset < st.size; offset += unit) {
        const bytes = Math.min(unit, st.size - offset), digest = createHash('sha256');
        for (let at = offset; at < offset + bytes; at += CHUNK) {
          const buf = Buffer.alloc(Math.min(CHUNK, offset + bytes - at));
          invariant(readSync(fd, buf, 0, buf.length, at) === buf.length, 'Short snapshot read'); digest.update(buf);
        }
        entry.chunks.push({ sha256: digest.digest('hex'), bytes, offset });
      }
    } finally { closeSync(fd); }
    const after = statSync(path);
    invariant(st.size === after.size && st.mtimeMs === after.mtimeMs, 'Snapshot file changed during hash');
    cache[rel] = { signature, entry }; files.push(entry); atomic(cachePath, cache);
  }
  atomic(cachePath, cache); atomic(join(root, 'snapshot-paths.json'), paths);
  const m: Manifest = { schema: 'bandnet-snapshot-v1', kind, files, fits };
  atomic(join(root, 'snapshot.json'), m);
  return validateManifest(m);
}
type Signed = { url: string; headers: Record<string, string> };
export async function transferPool<T, R>(items: T[], concurrency: number, run: (item: T) => Promise<R>) {
  invariant(Number.isInteger(concurrency) && concurrency >= 1 && concurrency <= TRANSFER_CONCURRENCY, 'Invalid transfer concurrency');
  return boundedPool(items, concurrency, run);
}
async function boundedPool<T, R>(items: T[], concurrency: number, run: (item: T) => Promise<R>) {
  let cursor = 0, failed = false, error: string | null = null;
  const errors: string[] = [];
  const results: R[] = [];
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (!failed && cursor < items.length) {
      const item = items[cursor++];
      try { results.push(await run(item)); } catch (e) {
        failed = true;
        const message = String((e as Error).message).replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, '<url>')
          .replace(/X-Amz-[A-Za-z-]+=\S+/g, '<signed>').slice(0, 300);
        errors.push(message); if (error === null) error = message;
      }
    }
  }));
  // All in-flight operations settle before reporting failure. Successful siblings
  // retain their receipts; a retry sends only the remaining objects.
  return { results, failed, error, errors };
}
export async function probeTransfers(urls: Signed[], send: (items: any[]) => Promise<any>) {
  invariant(urls.length === 16, 'Probe requires sixteen complete PUTs');
  const nonce = randomUUID();
  const batch = await send(urls.map((signed, i) => ({ key: `probe-${nonce}-${i}`, file: 'probe', index: 0, signed })));
  const bytes = batch.results.reduce((n: number, r: any) => n + r.bytes, 0);
  invariant(!batch.failed && bytes === 1024 ** 3 && batch.results.length === 16, `Probe did not upload a full GiB${batch.error ? `: ${batch.error}` : ''}`);
  return bytes;
}
export async function transfer(method: string, signed: Signed, body?: Buffer, timeoutMs = 600_000) {
  // URLs are bearer capabilities and never appear in logs or error messages.
  // Presigned URLs reject unsigned Content-MD5, so integrity is checked against S3's single-part ETag instead.
  const r = await fetch(signed.url, { method, headers: { ...signed.headers }, body,
    signal: AbortSignal.timeout(Math.max(1, Math.floor(timeoutMs))) });
  if (!r.ok) { await r.body?.cancel(); throw new Error(`Object ${method} failed: HTTP ${r.status}`); }
  if (method === 'PUT' && body !== undefined) checkETag(r, createHash('md5').update(body).digest('hex'));
  return r;
}
function checkETag(r: Response, md5: string) {
  invariant(r.headers.get('etag')?.replace(/"/g, '') === md5, 'Uploaded object checksum mismatch');
}
export async function upload(root: string, file: string, index: number, signed: Signed,
                             pinned?: { manifest: Manifest; paths: Record<string, string> }, deadlineAtMs = Date.now() + 600_000,
                             sent?: (bytes: number) => void) {
  const m = pinned === undefined ? validateManifest(json(join(root, 'snapshot.json'))) : pinned.manifest;
  const e = m.files.find(e => e.path === file); invariant(e !== undefined, 'Unknown snapshot file');
  const c = e.chunks[index]; invariant(c !== undefined, 'Unknown chunk');
  if (c.bytes === 0) { await transfer('PUT', signed, Buffer.alloc(0), deadlineAtMs - Date.now()); return { bytes: 0, seconds: 0 }; }
  const paths = pinned === undefined ? json(join(root, 'snapshot-paths.json')) : pinned.paths;
  const digest = createHash('sha256'), md5 = createHash('md5');
  const signal = AbortSignal.timeout(Math.max(1, Math.floor(deadlineAtMs - Date.now())));
  for await (const b of createReadStream(paths[file], { start: c.offset, end: c.offset + c.bytes - 1, highWaterMark: 1024 ** 2, signal })) { digest.update(b); md5.update(b); }
  invariant(digest.digest('hex') === c.sha256, 'Snapshot bytes changed before upload');
  const start = performance.now();
  // Native streaming HTTP avoids undici's independent header timeout on long PUTs.
  // The batch abort signal is the single bounded lifetime, including hashing.
  const url = new URL(signed.url), request = url.protocol === 'https:' ? httpsRequest : httpRequest;
  const req = request(url, { method: 'PUT', headers: { ...signed.headers, 'Content-Length': String(c.bytes) }, signal });
  let socket: import('node:net').Socket | undefined, initialBytes = 0;
  req.once('socket', s => { socket = s; initialBytes = s.bytesWritten; });
  const response = new Promise<string>((resolveResponse, reject) => {
    let rejectedResponse = false;
    req.on('error', error => { if (!rejectedResponse) reject(error); });
    req.once('response', res => {
      if (res.statusCode === undefined || res.statusCode < 200 || res.statusCode >= 300) {
        rejectedResponse = true;
        // Stop sending immediately. A small XML body already accompanying the
        // headers can still supply a diagnostic Code; never wait for a hung body.
        let body = '';
        res.on('data', b => { if (body.length < 4096) body += b.toString().slice(0, 4096 - body.length); });
        const fail = () => {
          const code = body.match(/<Code>([A-Za-z0-9_-]{1,80})<\/Code>/)?.[1];
          reject(new Error(`Object PUT failed: HTTP ${res.statusCode}${code === undefined ? '' : ` (${code})`}`));
        };
        res.on('error', fail); res.once('end', fail); res.once('close', fail);
        res.resume(); queueMicrotask(() => { req.destroy(); fail(); }); return;
      }
      res.on('error', reject); res.resume();
      res.once('end', () => {
        if (res.statusCode === undefined || res.statusCode < 200 || res.statusCode >= 300) reject(new Error(`Object PUT failed: HTTP ${res.statusCode}`));
        else resolveResponse(typeof res.headers.etag === 'string' ? res.headers.etag.replace(/"/g, '') : '');
      });
    });
  });
  const completed = await Promise.allSettled([response, pipeline(createReadStream(paths[file], {
    start: c.offset, end: c.offset + c.bytes - 1, highWaterMark: 1024 ** 2, signal }), req)]);
  // Socket writes are not S3 receipts. Discount 8 MiB per in-flight stream for
  // headers and local buffering; only the disposable probe uses this lower bound.
  if (sent !== undefined) sent(socket === undefined ? 0 : Math.min(c.bytes, Math.max(0, socket.bytesWritten - initialBytes - 8 * 1024 ** 2)));
  for (const result of completed) if (result.status === 'rejected') throw result.reason;
  const etag = (completed[0] as PromiseFulfilledResult<string>).value;
  invariant(etag === md5.digest('hex'), 'Uploaded object checksum mismatch');
  return { bytes: c.bytes, seconds: (performance.now() - start) / 1000 };
}
export const RESTORE_RANGE_BYTES = 64 * 1024 ** 2;
export const RESTORE_CONCURRENCY = 32;
const RESTORE_ATTEMPTS = 5;
type RestoreItem = { entry: Entry; index: number; signed: Signed };
class RestoreProtocolError extends Error {}
function restoreError(error: unknown) {
  const cause = (error as Error & { cause?: unknown }).cause;
  return `${(error as Error).message}${cause instanceof Error ? ` (${cause.message})` : ''}`
    .replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, '<url>').replace(/X-Amz-[A-Za-z-]+=\S+/g, '<signed>');
}
async function restoreRange(root: string, item: RestoreItem, start: number, bytes: number, deadlineAtMs: number) {
  const c = item.entry.chunks[item.index], end = start + bytes - 1;
  // Range offsets refer to the S3 chunk object; destination offsets include the
  // chunk's position in its file. Retries overwrite only this disjoint interval.
  for (let attempt = 1; ; attempt++) {
    let fd: number | undefined;
    let response: Response | undefined;
    try {
      invariant(Date.now() < deadlineAtMs, 'Restore deadline exceeded');
      response = await fetch(item.signed.url, {
        method: 'GET', headers: { ...item.signed.headers, ...(bytes === 0 ? {} : { Range: `bytes=${start}-${end}` }) },
        signal: AbortSignal.timeout(Math.max(1, Math.floor(deadlineAtMs - Date.now()))) });
      if (!response.ok) throw new Error(`Object GET failed: HTTP ${response.status}`);
      if (bytes !== 0 && (response.status !== 206 || response.headers.get('content-range') !== `bytes ${start}-${end}/${c.bytes}`)) {
        throw new RestoreProtocolError(`Invalid ranged GET response: HTTP ${response.status}; expected exact Content-Range`);
      }
      const contentLength = response.headers.get('content-length');
      if (contentLength !== null && Number(contentLength) !== bytes) throw new RestoreProtocolError('Ranged GET Content-Length mismatch');
      invariant(response.body !== null, 'Empty download');
      fd = openSync(join(root, item.entry.path), 'r+');
      let length = 0;
      for await (const b of response.body as any) {
        invariant(Date.now() < deadlineAtMs, 'Restore deadline exceeded');
        if (length + b.length > bytes) throw new RestoreProtocolError('Ranged GET exceeded requested length');
        invariant(writeSync(fd, b, 0, b.length, c.offset + start + length) === b.length, 'Short restore write');
        length += b.length;
      }
      invariant(length === bytes, 'Short ranged download');
      return;
    } catch (error) {
      const message = restoreError(error);
      if (error instanceof RestoreProtocolError || attempt >= RESTORE_ATTEMPTS || Date.now() + 2000 * attempt >= deadlineAtMs || /HTTP 4\d\d/.test(message)) throw new Error(message);
    } finally {
      if (fd !== undefined) closeSync(fd);
      if (response?.body !== null && response?.body !== undefined && !response.body.locked) await response.body.cancel().catch(() => {});
    }
    await sleep(2000 * attempt);
  }
}
async function restoreItems(root: string, items: RestoreItem[], truncate: boolean, deadlineAtMs: number) {
  const ranges: { item: RestoreItem; start: number; bytes: number }[] = [];
  const identities = new Set<string>();
  // Truncate before any worker starts, never inside a range worker.
  for (const item of items) {
    safePath(item.entry.path);
    const c = item.entry.chunks[item.index]; invariant(c !== undefined, 'Invalid restore chunk');
    const identity = `${item.entry.path}:${item.index}`;
    invariant(!identities.has(identity), 'Duplicate restore chunk'); identities.add(identity);
    const p = join(root, item.entry.path); mkdirSync(dirname(p), { recursive: true });
    if (truncate && item.index === 0) closeSync(openSync(p, 'w'));
    if (c.bytes === 0) ranges.push({ item, start: 0, bytes: 0 });
    for (let start = 0; start < c.bytes; start += RESTORE_RANGE_BYTES) ranges.push({ item, start, bytes: Math.min(RESTORE_RANGE_BYTES, c.bytes - start) });
  }
  const batch = await boundedPool(ranges, RESTORE_CONCURRENCY, async range => {
    await restoreRange(root, range.item, range.start, range.bytes, deadlineAtMs);
  });
  invariant(!batch.failed, `Restore batch failed: ${batch.error}`);
  // Hash the actual destination bytes after every range has settled. A successful
  // network response alone is never a chunk receipt. Bound disk readers separately.
  const verified = await boundedPool(items, TRANSFER_CONCURRENCY, async item => {
    const c = item.entry.chunks[item.index], p = join(root, item.entry.path), digest = createHash('sha256');
    let length = 0;
    invariant(Date.now() < deadlineAtMs, 'Restore deadline exceeded');
    if (c.bytes > 0) for await (const b of createReadStream(p, { start: c.offset, end: c.offset + c.bytes - 1,
      highWaterMark: 1024 ** 2, signal: AbortSignal.timeout(Math.max(1, Math.floor(deadlineAtMs - Date.now()))) })) {
      length += b.length; digest.update(b);
    }
    invariant(length === c.bytes && digest.digest('hex') === c.sha256, 'Restored checksum mismatch');
    const fd = openSync(p, 'r+'); try { fsyncSync(fd); } finally { closeSync(fd); }
  });
  invariant(!verified.failed, `Restore batch failed: ${verified.error}`);
  return { restored: items.length };
}
export async function restoreChunk(root: string, e: Entry, index: number, signed: Signed, truncate = true, deadlineAtMs = Date.now() + 600_000) {
  await restoreItems(root, [{ entry: e, index, signed }], truncate, deadlineAtMs);
}
export function auditDeadline(root: string, ceiling: number) {
  const schedule = join(root, 'save-deadline.json');
  // Audit uses the remaining paid compute window, never the final-save reserve.
  // The supervisor can shorten it as the balance or measured upload rate changes.
  return existsSync(schedule) ? Math.min(ceiling, json(schedule).saveAtMs) : ceiling;
}
export async function command(cmd: string, args: string[], cwd: string, log: string, deadline: number, phase: 'training' | 'audit') {
  const fd = openSync(log, 'a', 0o600);
  const child = spawn(cmd, args, { cwd, detached: true, stdio: ['ignore', fd, fd] }); closeSync(fd);
  // Spawned fit processes share this isolated process group. Forced deadlines
  // must stop every writer, not just the coordinator, before a final snapshot.
  const stopGroup = (signal: NodeJS.Signals) => {
    if (child.pid === undefined) return;
    try { process.kill(-child.pid, signal); } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ESRCH') throw error;
    }
  };
  atomic(join(dirname(log), 'python.json'), { pid: child.pid, group: child.pid, cmd, args, phase });
  let interruptSeen: number | null = null;
  let auditStopSeen: number | null = null;
  const interruptTimer = setInterval(() => {
    if (phase === 'audit') {
      if (Date.now() >= auditDeadline(dirname(log), deadline) && auditStopSeen === null) {
        auditStopSeen = Date.now(); stopGroup('SIGTERM');
      }
      if (auditStopSeen !== null && Date.now() - auditStopSeen >= 30_000) stopGroup('SIGKILL');
      return;
    }
    const schedule = join(dirname(log), 'save-deadline.json');
    if (existsSync(schedule) && Date.now() >= json(schedule).saveAtMs && !existsSync(join(dirname(log), 'interrupt.json'))) {
      child.kill('SIGINT'); atomic(join(dirname(log), 'interrupt.json'), { requested: true });
    }
    if (existsSync(join(dirname(log), 'interrupt.json'))) {
      if (interruptSeen === null) interruptSeen = Date.now();
      if (Date.now() - interruptSeen > 180_000) stopGroup('SIGTERM');
      if (Date.now() - interruptSeen > 210_000) stopGroup('SIGKILL');
    }
  }, 1000);
  const timer = setTimeout(() => stopGroup('SIGTERM'), Math.max(1, deadline - Date.now()));
  const killer = setTimeout(() => stopGroup('SIGKILL'), Math.max(1, deadline - Date.now() + 30_000));
  try { return await new Promise<number>(res => { child.once('error', () => res(127)); child.once('exit', code => res(code === null ? 137 : code)); }); }
  finally {
    clearTimeout(timer); clearTimeout(killer); clearInterval(interruptTimer); stopGroup('SIGKILL');
    const drainDeadline = Date.now() + 30_000;
    while (child.pid !== undefined && groupHasWriters(child.pid) && Date.now() < drainDeadline) await sleep(50);
    if (child.pid !== undefined) invariant(!groupHasWriters(child.pid), 'Fit writer group did not exit');
  }
}
async function job(root: string) {
  const spec = json(join(root, 'job-spec.json'));
  atomic(join(root, 'job.json'), { pid: process.pid, spec });
  const seconds = Math.floor((spec.saveAtMs - Date.now()) / 1000);
  invariant(seconds > 0, 'Compute deadline already passed');
  const env = readFileSync(join(root, 'code/.env.local'), 'utf8').replace(/^BANDNET_PRODUCTION_MAX_SECONDS=.*$/m, `BANDNET_PRODUCTION_MAX_SECONDS=${seconds}`);
  writeFileSync(join(root, 'code/.env.local'), env, { mode: 0o600 });
  let code = 127, audit: number | null = null;
  try {
    code = await command(join(root, 'code/.venv/bin/python'), ['-B', 'corrected_pilot.py',
      ...(spec.rehearsal ? ['--timing-check'] : ['--production', '--preflight', join(root, 'preflight')]),
      '--output', join(root, 'production'), ...(spec.resume ? ['--resume'] : [])],
      join(root, 'code'), join(root, 'job.log'), spec.saveAtMs + 180_000, 'training');
    if (code === 0) audit = await command(join(root, 'code/.venv/bin/python'), ['-B', 'verify_corrected_pilot.py', spec.rehearsal ? '--timing' : '--production',
      '--run', join(root, 'production'), '--report', join(root, 'audit.json')], join(root, 'code'), join(root, 'audit.log'), spec.saveAtMs, 'audit');
  } finally { atomic(join(root, 'exit.json'), { code, audit }); }
}
export async function rpc(input: any) {
  const root = resolve(input.root);
  const batchDeadline = () => {
    invariant(Number.isFinite(input.budgetMs) && input.budgetMs > 0 && input.budgetMs <= 3300_000, 'Invalid relative batch budget');
    return Date.now() + Math.floor(input.budgetMs);
  };
  switch (input.op) {
    case 'status': return status(root);
    case 'schedule-stop': {
      invariant(Number.isSafeInteger(input.saveAtMs) && input.saveAtMs > 0, 'Invalid save deadline');
      const spec = json(join(root, 'job-spec.json'));
      atomic(join(root, 'save-deadline.json'), { saveAtMs: Math.min(input.saveAtMs, spec.saveAtMs) });
      return { saved: true };
    }
    case 'snapshot': return snapshot(root, input.kind);
    case 'upload': return upload(root, input.file, input.index, input.signed);
    case 'upload-batch': {
      const pinned = { manifest: validateManifest(json(join(root, 'snapshot.json'))), paths: json(join(root, 'snapshot-paths.json')) };
      invariant(['foreground', 'background'].includes(input.mode), 'Missing upload mode');
      const deadline = batchDeadline();
      let sentBytes = 0;
      const batch = await transferPool(input.items, input.mode === 'foreground' ? TRANSFER_CONCURRENCY : BACKGROUND_CONCURRENCY, async (item: any) => {
        const entry = pinned.manifest.files.find(e => e.path === item.file), chunk = entry?.chunks[item.index];
        invariant(chunk !== undefined, 'Unknown upload chunk');
        const receipt = join(root, 'upload-receipts', `${hash(item.key)}.json`);
        if (existsSync(receipt)) {
          invariant(json(receipt).sha256 === chunk.sha256, 'Receipt content identity changed');
          return { key: item.key, bytes: 0, seconds: 0 };
        }
        invariant(Date.now() < deadline, 'Batch deadline exceeded');
        const measured = await upload(root, item.file, item.index, item.signed, pinned, deadline, bytes => { sentBytes += bytes; });
        // Per-object files avoid lost receipts when an old disconnected worker
        // and a retry briefly overlap; there is no shared read/modify/write map.
        const temporary = `${receipt}.${randomUUID()}`;
        atomic(temporary, { key: item.key, sha256: chunk.sha256, ...measured }); renameSync(temporary, receipt);
        return { key: item.key, ...measured };
      });
      return { ...batch, sentBytes, deadlineReached: Date.now() >= deadline };
    }
    case 'inventory': {
      const known = new Set(input.uploadedKeys), cachePath = join(root, 'hash-cache.json');
      const cache = existsSync(cachePath) ? json(cachePath) : {}; let unsaved = 0;
      if (existsSync(join(root, 'production'))) for (const rel of walk(join(root, 'production'))) {
        const st = statSync(join(root, 'production', rel)), saved = cache[`production/${rel}`];
        const signature = `${st.size}:${st.mtimeMs}:${st.ino}`;
        if (saved === undefined || saved.signature !== signature) unsaved += st.size;
        else for (const c of saved.entry.chunks) {
          const path = `production/${rel}`;
          const key = saved.entry.chunks.length === 1 ? (input.objectPaths[path] === c.sha256 ? `files/${path}` : `revisions/${c.sha256}/${path}`) : `chunks/${c.sha256}`;
          if (!known.has(key)) unsaved += c.bytes;
        }
      }
      return { unsaved };
    }
    case 'interrupt': {
      const state = status(root); if (!state.alive) return state;
      const child = json(join(root, 'python.json'));
      if (child.phase === 'audit') return state;
      const cmdline = readFileSync(`/proc/${child.pid}/cmdline`, 'utf8');
      invariant(cmdline.includes('corrected_pilot.py') && !cmdline.includes('verify_corrected_pilot.py'), 'Refusing signal to unrelated PID');
      const parent = readFileSync(`/proc/${child.pid}/status`, 'utf8').match(/^PPid:\s+(\d+)/m);
      invariant(parent !== null && Number(parent[1]) === state.job.pid, 'Python PID no longer belongs to this worker');
      if (!existsSync(join(root, 'interrupt.json'))) { process.kill(child.pid, 'SIGINT'); atomic(join(root, 'interrupt.json'), { requested: true }); }
      return state;
    }
    case 'put-manifest': {
      const m = validateManifest(input.manifest);
      await transfer('PUT', input.signed, Buffer.from(JSON.stringify(m))); return { committed: true };
    }
    case 'restore': return restoreChunk(root, input.entry, input.index, input.signed);
    case 'restore-batch': {
      const deadline = batchDeadline();
      return restoreItems(root, input.items, true, deadline);
    }
    case 'probe': {
      invariant(input.urls.length === 40, 'Probe requires 16 foreground, 16 background and 8 large PUT links');
      const deadline = batchDeadline(), start = performance.now(), waves: any[] = [];
      const b = Buffer.alloc(CHUNK, 0x6d), p = join(root, 'probe.bin');
      let write = 0, sha = 0, error: string | null = null;
      try {
        const fd = openSync(p, 'w');
        try { for (let i = 0; i < 32; i++) { invariant(Date.now() < deadline, 'Probe disk deadline'); invariant(writeSync(fd, b) === b.length, 'Short probe write'); } fsyncSync(fd); }
        finally { closeSync(fd); }
        write = 2 * 1024 ** 3 / ((performance.now() - start) / 1000);
        const h = performance.now(), digest = createHash('sha256');
        for (let i = 0; i < 32; i++) digest.update(b);
        sha = 2 * 1024 ** 3 / ((performance.now() - h) / 1000);
        const largeHash = digest.digest('hex'), smallHash = hash(b);
        for (const [name, mode, offset, count, size, checksum] of [
          ['foreground', 'foreground', 0, 16, CHUNK, smallHash],
          ['background', 'background', 16, 16, CHUNK, smallHash],
          ['large', 'foreground', 32, 8, 2 * 1024 ** 3, largeHash],
        ] as const) {
          atomic(join(root, 'snapshot-paths.json'), { probe: p });
          atomic(join(root, 'snapshot.json'), { schema: 'bandnet-snapshot-v1', kind: 'progress', fits: [], files: [
            { path: 'probe', bytes: size, chunks: [{ sha256: checksum, bytes: size, offset: 0 }] }] });
          const t = performance.now(), nonce = randomUUID();
          const batch = await rpc({ root, op: 'upload-batch', mode, budgetMs: deadline - Date.now(),
            items: input.urls.slice(offset, offset + count).map((signed: Signed, i: number) => ({ key: `probe-${nonce}-${i}`, file: 'probe', index: 0, signed })) });
          const seconds = (performance.now() - t) / 1000, bytes = batch.results.reduce((n: number, r: any) => n + r.bytes, 0);
          const partial = name === 'large' && batch.failed && batch.deadlineReached &&
            batch.errors.length > 0 && batch.errors.every((error: string) => /abort|deadline|timeout/i.test(error)) && batch.sentBytes >= 1024 ** 3;
          waves.push({ name, concurrency: mode === 'foreground' ? 8 : 2, objectBytes: size, objects: batch.results.length,
            bytes, sentBytes: batch.sentBytes, partial, seconds,
            bytesPerSecond: (partial ? batch.sentBytes : bytes) / seconds, failed: batch.failed, error: batch.error });
          invariant(partial || !batch.failed && bytes === count * size, `Probe ${name} incomplete: ${batch.error}`);
        }
      } catch (e) { error = String((e as Error).message).replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, '<url>').slice(0, 600); }
      finally { if (existsSync(p)) unlinkSync(p); }
      return { write, sha, waves, failed: error !== null, error, elapsedSeconds: (performance.now() - start) / 1000 };
    }
    case 'start': {
      if (existsSync(join(root, 'job-spec.json'))) return status(root);
      invariant(Number.isFinite(input.spec.initialSaveAtMs) && input.spec.initialSaveAtMs <= input.spec.saveAtMs, 'Missing initial autonomous save deadline');
      atomic(join(root, 'job-spec.json'), input.spec);
      atomic(join(root, 'save-deadline.json'), { saveAtMs: input.spec.initialSaveAtMs });
      const fd = openSync(join(root, 'worker.log'), 'a', 0o600);
      const child = spawn(process.execPath, [fileURLToPath(import.meta.url), 'job', root], { detached: true, stdio: ['ignore', fd, fd] });
      closeSync(fd); child.unref();
      for (let i = 0; i < 50 && !existsSync(join(root, 'job.json')); i++) await sleep(100);
      invariant(existsSync(join(root, 'job.json')), 'Pod job did not acknowledge start'); return status(root);
    }
    default: throw new Error('Unknown pod operation');
  }
}
if (process.argv[1] !== undefined && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    if (process.argv[2] === 'job') await job(resolve(process.argv[3]));
    else { const chunks: Buffer[] = []; for await (const b of process.stdin) chunks.push(b); console.log(JSON.stringify(await rpc(JSON.parse(Buffer.concat(chunks).toString())))); }
  } catch (e) {
    // Messages are written without URLs; redact any that slip through.
    console.error(`Pod worker failed: ${String((e as Error)?.message).replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, '<url>').slice(0, 300)}`);
    process.exitCode = 1;
  }
}
