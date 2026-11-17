/** Mac-side S3 access for scoring completed fits without a pod. Node >=22.18.
 *
 *   node scoring_restore.ts check <prefix> <manifest>
 *   node scoring_restore.ts restore <prefix> <manifest> <destination> <fit>...
 *
 * `check` confirms every chunk object of a committed snapshot manifest exists
 * with its recorded size. `restore` downloads only what scoring needs: shared
 * run records plus each named fit's records, best checkpoint and data, except
 * that training bands are reduced to their first 64 MiB. Scoring qualifies a
 * checkpoint on the first training rows only, verified against the dataset
 * manifest's own band checksum. Resume checkpoints (`latest.pt`) are skipped.
 */
import { closeSync, createReadStream, existsSync, mkdirSync, openSync, statSync, writeSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { TRANSFER_CONCURRENCY, atomic, awsSettings, credentials, hash, invariant, keyPrefix, presign, safePath, validateManifest } from './production_common.ts';
import type { Entry, Manifest } from './production_common.ts';
import { restoreChunk, transferPool } from './production_pod.ts';

export const TRAIN_PREFIX_BYTES = 64 * 1024 ** 2;
const { target, keyCsv } = awsSettings(dirname(fileURLToPath(import.meta.url)));
const aws = credentials(keyCsv);
const sign = (method: 'GET' | 'HEAD', key: string) => presign(aws, target, method, key);

async function manifest(prefix: string, name: string) {
  const s = sign('GET', keyPrefix(prefix) + safePath(name));
  const r = await fetch(s.url, { headers: s.headers, signal: AbortSignal.timeout(60_000) });
  invariant(r.ok, `Manifest GET failed: HTTP ${r.status}`);
  const bytes = Buffer.from(await r.arrayBuffer());
  return { manifest: validateManifest(JSON.parse(bytes.toString()) as Manifest), sha256: hash(bytes) };
}

function chunkKey(prefix: string, c: Entry['chunks'][number]) {
  invariant(c.key !== undefined, 'Snapshot chunk has no object key');
  return prefix + safePath(c.key);
}

export function selectedEntries(m: Manifest, fits: string[]) {
  invariant(fits.length > 0 && fits.every(f => m.fits.includes(f)), 'Every requested fit must be complete in the snapshot');
  const common = ['production/protocol.json', 'production/progress.json'];
  const shared = ['production/sources/', 'production/executions/', 'production/execution-sources/'];
  const owned = (path: string) => fits.some(f => path.startsWith(`production/${f}/`) || path.startsWith(`production/${f}-data/`)
    || path === `production/${f}-preflight.json` || path === `production/${f}-calibrations.json`);
  return m.files.filter(e => (common.includes(e.path) || shared.some(s => e.path.startsWith(s)) || owned(e.path))
    && !e.path.endsWith('/latest.pt'));
}

async function check(prefix: string, name: string) {
  const { manifest: m, sha256 } = await manifest(prefix, name);
  const chunks = m.files.flatMap(e => e.chunks.map(c => ({ path: e.path, c })));
  const missing: string[] = [];
  const result = await transferPool(chunks, TRANSFER_CONCURRENCY, async ({ path, c }) => {
    const s = sign('HEAD', chunkKey(prefix, c));
    const r = await fetch(s.url, { method: 'HEAD', headers: s.headers, signal: AbortSignal.timeout(60_000) });
    if (!r.ok || Number(r.headers.get('content-length')) !== c.bytes) missing.push(`${path} @${c.offset}: HTTP ${r.status}`);
  });
  invariant(!result.failed, `Check failed: ${result.error}`);
  const report = { prefix, manifest: name, manifest_sha256: sha256, kind: m.kind, fits: m.fits, files: m.files.length,
    chunks: chunks.length, bytes: m.files.reduce((n, e) => n + e.bytes, 0), missing };
  console.log(JSON.stringify(report, null, 2));
  invariant(missing.length === 0, 'Snapshot is incomplete in S3');
}

async function sha256File(path: string) {
  const digest = createHash('sha256');
  for await (const b of createReadStream(path, { highWaterMark: 1024 ** 2 })) digest.update(b);
  return digest.digest('hex');
}

async function restorePrefix(prefix: string, destination: string, e: Entry) {
  const c = e.chunks[0], bytes = Math.min(TRAIN_PREFIX_BYTES, e.bytes);
  invariant(c.bytes >= bytes, 'Training prefix spans more than one snapshot chunk');
  const s = sign('GET', chunkKey(prefix, c));
  const r = await fetch(s.url, { headers: { ...s.headers, Range: `bytes=0-${bytes - 1}` }, signal: AbortSignal.timeout(600_000) });
  invariant(r.status === 206 && r.headers.get('content-range') === `bytes 0-${bytes - 1}/${c.bytes}`, `Prefix GET failed: HTTP ${r.status}`);
  const path = join(destination, `${e.path.slice('production/'.length)}.prefix`);
  mkdirSync(dirname(path), { recursive: true });
  const fd = openSync(path, 'w', 0o600);
  let length = 0;
  try {
    for await (const b of r.body as any) {
      invariant(length + b.length <= bytes && writeSync(fd, b, 0, b.length, length) === b.length, 'Prefix write failed');
      length += b.length;
    }
  } finally { closeSync(fd); }
  invariant(length === bytes, 'Short prefix download');
  return { path: e.path, prefix_bytes: bytes, file_bytes: e.bytes, sha256: await sha256File(path),
    verification: 'first dataset chunk band checksum, checked by score_completed_fits.py' };
}

async function restore(prefix: string, name: string, destination: string, fits: string[]) {
  invariant(!existsSync(destination) && statSync(dirname(destination)).isDirectory(), 'Destination must be new in an existing directory');
  const { manifest: m, sha256 } = await manifest(prefix, name);
  const entries = selectedEntries(m, fits);
  const whole = entries.filter(e => !/-data\/train\.bands\.npy$/.test(e.path));
  const prefixes = entries.filter(e => /-data\/train\.bands\.npy$/.test(e.path));
  invariant(prefixes.length === fits.length, 'Each fit needs exactly one training band file');
  // Chunks of one file would race on its truncation; only main training bands are chunked.
  invariant(whole.every(e => e.chunks.length === 1), 'Unexpected multi-chunk scoring file');
  mkdirSync(destination, { mode: 0o700 });
  const started = performance.now();
  // Restored paths drop the snapshot's `production/` root so the directory is a run root.
  const items = whole.flatMap(e => e.chunks.map((c, index) => ({ e: { ...e, path: e.path.slice('production/'.length) }, index, c })));
  const result = await transferPool(items, TRANSFER_CONCURRENCY, async ({ e, index, c }) => {
    await restoreChunk(destination, e, index, sign('GET', chunkKey(prefix, c)), index === 0, Date.now() + 3 * 3600_000);
  });
  invariant(!result.failed, `Restore failed: ${result.error}`);
  const trained = [];
  for (const e of prefixes) trained.push(await restorePrefix(prefix, destination, e));
  const bytes = whole.reduce((n, e) => n + e.bytes, 0) + trained.reduce((n, t) => n + t.prefix_bytes, 0);
  atomic(join(destination, 'scoring-restore.json'), { schema: 'bandnet-scoring-restore-v1', prefix, manifest: name,
    manifest_sha256: sha256, fits, files: whole.map(e => ({ path: e.path, bytes: e.bytes, sha256: e.chunks.length === 1 ? e.chunks[0].sha256 : null })),
    training_band_prefixes: trained, downloaded_bytes: bytes, seconds: (performance.now() - started) / 1000 });
  console.log(JSON.stringify({ destination, fits, downloaded_bytes: bytes }));
}

const [command, ...args] = process.argv.slice(2);
if (command === 'check' && args.length === 2) await check(args[0], args[1]);
else if (command === 'restore' && args.length >= 4) await restore(args[0], args[1], resolve(args[2]), args.slice(3));
else throw new Error('usage: check <prefix> <manifest> | restore <prefix> <manifest> <destination> <fit>...');
