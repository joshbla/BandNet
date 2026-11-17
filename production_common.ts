/** Shared production-launch contracts. No cloud side effects on import. Node >=22.18. */
import { createHash, createHmac } from 'node:crypto';
import { readFileSync, writeFileSync, renameSync, mkdirSync, statSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { parseEnv } from 'node:util';

export const CHUNK = 64 * 1024 ** 2;
export const WHOLE_LIMIT = 5_000_000_000;
// One Node process bounds hashing to one event loop/core; streams share a small
// fixed worker pool instead of spawning a process per object.
export const TRANSFER_CONCURRENCY = 8;
export const BACKGROUND_CONCURRENCY = 2;
export const SAFETY_SECONDS = 900, STOP_SECONDS = 120;
export const EXPORT_BOUND = 110 * 10 ** 9;
export const MIN_UPLOAD = 20 * 10 ** 6;
export const GPU_TYPES = ['NVIDIA H100 NVL', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 PCIe',
  'NVIDIA H200', 'NVIDIA H200 NVL', 'NVIDIA A100-SXM4-80GB', 'NVIDIA A100 80GB PCIe'];
export function invariant(ok: unknown, message: string): asserts ok { if (!ok) throw new Error(message); }
export const hash = (b: Buffer | string) => createHash('sha256').update(b).digest('hex');
export function atomic(path: string, value: unknown) {
  mkdirSync(dirname(path), { recursive: true, mode: 0o700 });
  writeFileSync(`${path}.tmp`, JSON.stringify(value, null, 2), { mode: 0o600 });
  renameSync(`${path}.tmp`, path);
}
export function config(root: string) {
  const p = resolve(root, '.env.local');
  invariant((statSync(p).mode & 0o077) === 0, '.env.local must be owner-only');
  return parseEnv(readFileSync(p, 'utf8'));
}
export type S3Target = { bucket: string; owner: string; region: string };
// Account, bucket and credential location are local settings, never source constants.
export function awsSettings(root: string): { target: S3Target; keyCsv: string } {
  const env = config(root);
  const need = (name: string) => {
    const value = env[name];
    invariant(typeof value === 'string' && value.trim() !== '', `Add ${name} to the scripts section of code/.env.local`);
    return value;
  };
  return { target: { bucket: need('BANDNET_S3_BUCKET'), owner: need('BANDNET_AWS_ACCOUNT_ID'), region: need('BANDNET_S3_REGION') },
    keyCsv: need('BANDNET_AWS_KEY_CSV') };
}
export type SpendingLimits = { reserve: number; maxHourly: number; runCap: number; rehearsalCap: number; probeCap: number };
// Spending limits are local settings, never source constants.
export function spendingLimits(root: string): SpendingLimits {
  const env = config(root);
  const need = (name: string) => {
    const value = Number(env[name]);
    invariant(typeof env[name] === 'string' && env[name].trim() !== '' && Number.isFinite(value) && value > 0,
      `Add a positive ${name} to the scripts section of code/.env.local`);
    return value;
  };
  return { reserve: need('BANDNET_RUNPOD_BALANCE_RESERVE_USD'), maxHourly: need('BANDNET_RUNPOD_MAX_GPU_HOURLY_USD'),
    runCap: need('BANDNET_RUNPOD_RUN_CAP_USD'), rehearsalCap: need('BANDNET_RUNPOD_REHEARSAL_CAP_USD'),
    probeCap: need('BANDNET_RUNPOD_PROBE_CAP_USD') };
}
export function keyPrefix(s: string) {
  invariant(/^3Band-corrected\/[a-zA-Z0-9_-]+\/$/.test(s), 'Prefix must be a new named child of 3Band-corrected/');
  return s;
}
export function safePath(s: string) {
  invariant(s.length > 0 && !s.startsWith('/') && !s.split('/').some(x => x === '..' || x === '' || x === '.') && !s.includes('\\'), 'Unsafe relative path');
  return s;
}
export type Chunk = { sha256: string; bytes: number; offset: number; key?: string };
export type Entry = { path: string; bytes: number; chunks: Chunk[] };
export type Manifest = { schema: 'bandnet-snapshot-v1'; kind: 'completed-fits' | 'final' | 'progress'; files: Entry[]; fits: string[] };
export function validateManifest(m: Manifest) {
  invariant(m.schema === 'bandnet-snapshot-v1' && ['completed-fits', 'final', 'progress'].includes(m.kind), 'Invalid snapshot');
  const seen = new Set<string>();
  for (const e of m.files) {
    safePath(e.path); invariant(!seen.has(e.path), 'Duplicate file'); seen.add(e.path);
    invariant(Number.isSafeInteger(e.bytes) && e.bytes >= 0, 'Invalid size');
    let offset = 0;
    for (const c of e.chunks) {
      invariant(/^[a-f0-9]{64}$/.test(c.sha256) && c.offset === offset && Number.isSafeInteger(c.bytes) && c.bytes >= 0 &&
        (c.bytes > 0 || e.bytes === 0 && e.chunks.length === 1) && c.bytes <= WHOLE_LIMIT, 'Invalid chunk');
      if (c.key !== undefined) safePath(c.key);
      offset += c.bytes;
    }
    invariant(offset === e.bytes, 'Incomplete file');
  }
  return m;
}
export function budget(balance: number, reserve: number, cap: number, hourly: number, start: number, uploadBytesPerSecond: number, unsaved = EXPORT_BOUND) {
  invariant([balance, reserve, cap, hourly, start, uploadBytesPerSecond, unsaved].every(Number.isFinite), 'Nonfinite budget');
  invariant(reserve > 0 && balance > reserve && cap > 0 && hourly > 0 && uploadBytesPerSecond > 0 && unsaved >= 0, 'Invalid budget');
  const dollars = Math.min(balance - reserve, cap);
  const hardStopAtMs = Math.floor(start + Math.min(24 * 3600_000, dollars / hourly * 3600_000));
  const transferSeconds = unsaved / uploadBytesPerSecond;
  const saveAtMs = Math.floor(hardStopAtMs - (2 * transferSeconds + SAFETY_SECONDS + STOP_SECONDS) * 1000);
  return { dollars, hourly, transferSeconds, hardStopAtMs, saveAtMs };
}
export function credentials(path: string): { access: string; secret: string } {
  invariant((statSync(path).mode & 0o077) === 0, 'AWS credential CSV must be owner-only');
  const rows = readFileSync(path, 'utf8').replace(/^\uFEFF/, '').trim().split(/\r?\n/).map(r => r.split(',').map(v => v.replace(/^"|"$/g, '')));
  invariant(rows.length === 2, 'Expected single IAM credential CSV row');
  const a = rows[0].indexOf('Access key ID'), s = rows[0].indexOf('Secret access key');
  invariant(a >= 0 && s >= 0 && rows[1][a] && rows[1][s], 'Unexpected IAM CSV columns');
  return { access: rows[1][a], secret: rows[1][s] };
}
const enc = (v: string) => encodeURIComponent(v).replace(/[!'()*]/g, c => '%' + c.charCodeAt(0).toString(16).toUpperCase());
// SigV4 query signing, bounded to the configured bucket/region/owner. Each chunk is <5 GB.
export function presign(c: { access: string; secret: string }, t: S3Target, method: 'GET' | 'PUT' | 'HEAD' | 'DELETE', key: string, now = new Date()) {
  invariant(key.startsWith('3Band-corrected/'), 'Refusing historical prefix'); safePath(key);
  const stamp = now.toISOString().replace(/[:-]|\.\d{3}/g, ''), day = stamp.slice(0, 8);
  const scope = `${day}/${t.region}/s3/aws4_request`, host = `${t.bucket}.s3.${t.region}.amazonaws.com`;
  const path = '/' + key.split('/').map(enc).join('/');
  const fields: Record<string, string> = { 'X-Amz-Algorithm': 'AWS4-HMAC-SHA256', 'X-Amz-Credential': `${c.access}/${scope}`,
    'X-Amz-Date': stamp, 'X-Amz-Expires': '3600', 'X-Amz-SignedHeaders': 'host;x-amz-expected-bucket-owner' };
  const query = Object.keys(fields).sort().map(k => `${enc(k)}=${enc(fields[k])}`).join('&');
  const canonical = `${method}\n${path}\n${query}\nhost:${host}\nx-amz-expected-bucket-owner:${t.owner}\n\nhost;x-amz-expected-bucket-owner\nUNSIGNED-PAYLOAD`;
  const hmac = (key: Buffer | string, text: string) => createHmac('sha256', key).update(text).digest();
  const signing = hmac(hmac(hmac(hmac('AWS4' + c.secret, day), t.region), 's3'), 'aws4_request');
  const signature = createHmac('sha256', signing).update(`AWS4-HMAC-SHA256\n${stamp}\n${scope}\n${hash(canonical)}`).digest('hex');
  return { url: `https://${host}${path}?${query}&X-Amz-Signature=${signature}`, headers: { 'x-amz-expected-bucket-owner': t.owner } };
}
