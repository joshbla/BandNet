/** Small real-S3 correctness check, run only with an explicit flag. Never a bandwidth test.
 * Writes only a unique child of 3Band-corrected/rehearsal-smoke/ and deletes it.
 */
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { dirname, join } from 'node:path';
import { homedir } from 'node:os';
import { fileURLToPath } from 'node:url';
import { setTimeout as sleep } from 'node:timers/promises';
import { atomic, awsSettings, credentials, hash, invariant, presign } from './production_common.ts';
import { rpc } from './production_pod.ts';

invariant(process.argv[2] === '--confirm-small-s3-write', 'Requires --confirm-small-s3-write');
const root = dirname(fileURLToPath(import.meta.url));
const { target, keyCsv } = awsSettings(root);
const { bucket: BUCKET, owner: OWNER, region: REGION } = target;
const aws = credentials(keyCsv);
const env = { PATH: process.env.PATH, HOME: homedir(), AWS_ACCESS_KEY_ID: aws.access, AWS_SECRET_ACCESS_KEY: aws.secret,
  AWS_DEFAULT_REGION: REGION, AWS_EC2_METADATA_DISABLED: 'true' };
const identity = JSON.parse(execFileSync('aws', ['sts', 'get-caller-identity', '--output', 'json'], { env, encoding: 'utf8', timeout: 30_000, stdio: ['ignore', 'pipe', 'ignore'] }));
invariant(identity.Account === OWNER, 'Wrong AWS account');
execFileSync('aws', ['s3api', 'head-bucket', '--bucket', BUCKET, '--expected-bucket-owner', OWNER], { env, timeout: 30_000, stdio: 'ignore' });
const id = randomUUID(), prefix = `3Band-corrected/rehearsal-smoke/${id}/`;
const dir = join(root, 'artifacts/production-local-rehearsal/s3-smoke', id);
mkdirSync(dir, { recursive: true });
const files = Array.from({ length: 16 }, (_, index) => {
  const path = `payload-${index}.bin`, body = Buffer.alloc(64 * 1024, index);
  writeFileSync(join(dir, path), body);
  return { path, bytes: body.length, chunks: [{ bytes: body.length, offset: 0, sha256: hash(body) }] };
});
atomic(join(dir, 'snapshot.json'), { schema: 'bandnet-snapshot-v1', kind: 'progress', fits: [], files });
atomic(join(dir, 'snapshot-paths.json'), Object.fromEntries(files.map(f => [f.path, join(dir, f.path)])));
const keys = files.map(f => prefix + f.path), deleted: string[] = [];
let passed = false, error: string | null = null;
try {
  const items = files.map((f, i) => ({ key: keys[i], file: f.path, index: 0, signed: presign(aws, target, 'PUT', keys[i]) }));
  const result = await rpc({ root: dir, op: 'upload-batch', mode: 'foreground', items, budgetMs: 120_000 });
  invariant(!result.failed && result.results.length === 16, `Live upload failed: ${result.error}`);
  const retry = await rpc({ root: dir, op: 'upload-batch', mode: 'foreground', items, budgetMs: 120_000 });
  invariant(!retry.failed && retry.results.every((r: any) => r.bytes === 0), 'Receipts did not prevent repeated uploads');
  const restored = join(dir, 'restored'); mkdirSync(restored);
  await rpc({ root: restored, op: 'restore-batch', budgetMs: 120_000,
    items: files.map((entry, i) => ({ entry, index: 0, signed: presign(aws, target, 'GET', keys[i]) })) });
  for (const f of files) invariant(hash(readFileSync(join(restored, f.path))) === f.chunks[0].sha256, 'Live restore mismatch');
  passed = true;
} catch (e) {
  // Record failure without printing a transport error that might contain a URL.
  passed = false;
  error = String((e as Error).message).replace(/[a-z][a-z0-9+.-]*:\/\/\S+/gi, '<url>').replace(/X-Amz-[A-Za-z-]+=\S+/g, '<signed>').slice(0, 600);
} finally {
  for (const key of keys) {
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
    const signed = presign(aws, target, 'DELETE', key);
    const response = await fetch(signed.url, { method: 'DELETE', headers: signed.headers, signal: AbortSignal.timeout(30_000) });
    invariant(response.ok, 'Smoke cleanup DELETE failed');
    const check = presign(aws, target, 'HEAD', key);
    invariant((await fetch(check.url, { method: 'HEAD', headers: check.headers, signal: AbortSignal.timeout(30_000) })).status === 404, 'Smoke deletion unverified');
    deleted.push(key); break;
      } catch { await sleep(1000); }
    }
  }
  atomic(join(dir, 'result.json'), { node: process.version, passed, error, bytes: 1024 * 1024, objects: keys.length, deleted: deleted.length,
    prefix, boundary: 'Small real-S3 correctness/ETag/restore/receipt test; not pod throughput' });
}
invariant(passed && deleted.length === keys.length, `Smoke test or cleanup incomplete; inspect ${join(dir, 'result.json')}`);
console.log(JSON.stringify({ node: process.version, passed, objects: keys.length, bytes: 1024 * 1024, deleted: deleted.length, evidence: dir }));
