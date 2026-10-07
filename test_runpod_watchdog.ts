import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ALLOCATION_MS, SHUTDOWN_RESERVE_MS, RETRIEVAL_MS, RETRIEVAL_SHUTDOWN_RESERVE_MS, allocationStart, confirmedStopped, validateReceipt, verifyIdentity, watch } from './runpod_watchdog.ts';
import type { Pod, Receipt } from './runpod_watchdog.ts';

// Artificial epoch used only to exercise elapsed-time logic, not a run timestamp.
const start = 1_000_000;
const receipt: Receipt = { podId: 'created-by-this-test', createdAt: new Date(start).toISOString(), allocationRequestedAtMs: start };
const running = (): Pod => ({ id: receipt.podId, createdAt: receipt.createdAt, status: 'RUNNING', cost: 1.59, locked: false, actions: ['stop'], runtime: { uptime: 1 } });
const stopped = (): Pod => ({ ...running(), status: 'EXITED', cost: 1.59, actions: ['start'], runtime: null });

function fixture() {
  let now = start;
  let pod = running();
  const events: { event: string; elapsedMs: number }[] = [];
  const stopTimes: number[] = [];
  const controller = {
    now: () => now,
    delay: async (ms: number) => { now += ms; assert.ok(now < start + 1_000_000, 'Controller failed to finish'); },
    getPod: async () => pod,
    stopPod: async () => { stopTimes.push(now); pod = stopped(); },
    earlyStop: () => false,
    record: (e: { event: string; elapsedMs: number }) => { events.push(e); },
  };
  return { controller, events, stopTimes, setTime: (t: number) => { now = t; }, setPod: (p: Pod) => { pod = p; } };
}

test('receipt rejects unrelated or missing creation evidence', () => {
  assert.equal(validateReceipt(receipt), receipt);
  for (const bad of [null, {}, { ...receipt, podId: '../another' }, { ...receipt, allocationRequestedAtMs: start + 2000 }, { ...receipt, allocationRequestedAtMs: 0 }]) {
    assert.throws(() => validateReceipt(bad));
  }
  assert.equal(allocationStart({ ...receipt, allocationRequestedAtMs: start + 999 }), start);
  assert.throws(() => verifyIdentity({ ...running(), id: 'foreign-pod' }, receipt));
});

test('deadline includes provisioning and reserves a minute for actual shutdown', async () => {
  assert.equal(ALLOCATION_MS, 900_000);
  const f = fixture();
  f.setTime(start + 200_000); // Setup already consumed part of the 15 minutes.
  await watch(receipt, f.controller);
  assert.deepEqual(f.stopTimes, [start + ALLOCATION_MS - SHUTDOWN_RESERVE_MS]);
  assert.equal(f.events.at(-1)?.event, 'stopped-verified');
});

test('explicit early success/failure request stops immediately', async () => {
  const f = fixture();
  f.controller.earlyStop = () => true;
  await watch(receipt, f.controller);
  assert.deepEqual(f.stopTimes, [start]);
});

test('separately approved retrieval preserves creation identity and uses its short restart deadline', async () => {
  const f = fixture();
  const retrieval = { ...receipt, retrievalRequestedAtMs: start + 100_000 };
  f.setTime(retrieval.retrievalRequestedAtMs + 20_000);
  assert.equal(RETRIEVAL_MS, 180_000);
  assert.equal(allocationStart(retrieval), retrieval.retrievalRequestedAtMs);
  await watch(retrieval, f.controller);
  assert.deepEqual(f.stopTimes, [retrieval.retrievalRequestedAtMs + RETRIEVAL_MS - RETRIEVAL_SHUTDOWN_RESERVE_MS]);
  assert.equal(f.events.at(-1)?.event, 'stopped-verified');
  assert.throws(() => validateReceipt({ ...retrieval, retrievalRequestedAtMs: 0 }), /retrieval request/);
  assert.throws(() => validateReceipt({ ...retrieval, retrievalRequestedAtMs: NaN }), /retrieval request/);
  assert.throws(() => verifyIdentity({ ...running(), id: 'foreign-pod' }, retrieval), /identity/);
});

test('arming late does not reset the deadline', async () => {
  const f = fixture();
  f.setTime(start + ALLOCATION_MS + 1);
  await watch(receipt, f.controller);
  assert.equal(f.stopTimes[0], start + ALLOCATION_MS + 1);
  assert.ok(f.events.some(e => e.event === 'deadline-exceeded-unverified'));
});

test('retries read errors and stop errors and verifies with a fresh GET', async () => {
  const f = fixture();
  f.controller.earlyStop = () => true;
  let reads = 0;
  let stops = 0;
  f.controller.getPod = async () => {
    reads++;
    if (reads < 3) throw new Error('network unavailable');
    return stops >= 2 ? stopped() : running();
  };
  f.controller.stopPod = async () => {
    stops++;
    if (stops === 1) throw new Error('HTTP 503');
  };
  await watch(receipt, f.controller);
  assert.equal(reads, 5);
  assert.equal(stops, 2);
  assert.equal(f.events.at(-1)?.event, 'stopped-verified');
});

test('requires stopped lifecycle and absent runtime; published rate may remain positive', async () => {
  const f = fixture();
  f.controller.earlyStop = () => true;
  let reads = 0;
  f.controller.getPod = async () => {
    reads++;
    if (reads === 1) return running();
    if (reads === 2) return { ...stopped(), runtime: { uptime: 1 } };
    return stopped();
  };
  await watch(receipt, f.controller);
  assert.equal(reads, 3);
  assert.equal(confirmedStopped({ ...running(), status: 'ERROR', cost: 0 }), false);
  assert.equal(confirmedStopped({ ...stopped(), cost: 3.49 }), true);
  assert.equal(confirmedStopped({ ...running(), cost: 0 }), false);
});

test('never mutates a pod whose identity differs from the creation receipt', async () => {
  const f = fixture();
  f.controller.earlyStop = () => true;
  f.setPod({ ...running(), createdAt: new Date(start + 1).toISOString() });
  await assert.rejects(watch(receipt, f.controller), /identity/);
  assert.equal(f.stopTimes.length, 0);
});

test('provisioning is stoppable; a temporarily locked pod is retried without an unlock', async () => {
  const f = fixture();
  f.controller.earlyStop = () => true;
  let reads = 0;
  f.controller.getPod = async () => {
    reads++;
    if (reads === 1) return { ...running(), locked: true };
    if (reads === 2) return { ...running(), status: 'PROVISIONING' };
    return stopped();
  };
  await watch(receipt, f.controller);
  assert.equal(f.stopTimes.length, 1);
  assert.ok(f.events.some(e => e.event === 'awaiting-stoppable-state'));
});
