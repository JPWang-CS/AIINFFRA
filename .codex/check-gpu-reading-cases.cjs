// Recompute tutorial examples without CUDA.
const assert = require('node:assert/strict');
const blocks = Math.ceil(1000 / 256);
assert.equal(blocks, 4);
assert.equal(1000 - (blocks - 1) * 256, 232);
assert.equal(blocks * 256 - 1000, 24);
assert.equal(3 * 256 + 231, 999);
const vectorValue = i => (i % 97) * 0.25 - (i % 53) * 0.5;
assert.equal(vectorValue(256), -6.5);
assert.equal(vectorValue(999), -15.25);

const hits = new Map(), validPerBlock = [];
for (let by = 0; by < 2; by++) {
  for (let bx = 0; bx < 2; bx++) {
    let valid = 0;
    for (let ty = 0; ty < 2; ty++) {
      for (let tx = 0; tx < 4; tx++) {
        const row = by * 2 + ty, col = bx * 4 + tx;
        if (row < 3 && col < 5) {
          const index = row * 5 + col;
          hits.set(index, (hits.get(index) || 0) + 1);
          valid++;
        }
      }
    }
    validPerBlock.push(valid);
  }
}
assert.deepEqual(validPerBlock, [8, 2, 4, 1]);
assert.equal(hits.size, 15);
assert([...hits.values()].every(count => count === 1));
const warpRows = width => new Set(Array.from({length: 32}, (_, tid) => Math.floor(tid / width)));
assert.deepEqual([...warpRows(16)], [0, 1]);
assert.deepEqual([...warpRows(32)], [0]);

const sectors = (stride, offset = 0) =>
  new Set(Array.from({length: 32}, (_, lane) => Math.floor((offset + 4 * stride * lane) / 32))).size;
assert.equal(sectors(1), 4);
assert.equal(sectors(2), 8);
assert.equal(sectors(32), 32);
assert.equal(sectors(1, 4), 5);
const bankCount = stride => new Set(Array.from({length: 32}, (_, lane) => stride * lane % 32)).size;
assert.equal(bankCount(32), 1);
assert.equal(bankCount(33), 32);

function occupancy(registers, shared) {
  const resident = Math.min(2048 / 256, 64 / 8,
    Math.floor(65536 / (256 * registers)), Math.floor(98304 / shared), 32);
  return [resident, resident * 8, resident * 8 / 64];
}
assert.deepEqual(occupancy(64, 32768), [3, 24, 0.375]);
assert.deepEqual(occupancy(96, 49152), [2, 16, 0.25]);
const [m, reduction, k] = [8192, 6144, 4096];
const flops = 2 * m * reduction * k;
const bytes = 4 * (m * reduction + reduction * k + m * k);
assert.equal(flops, 412316860416);
assert.equal(bytes, 436207616);
assert(Math.abs(flops / bytes - 945.2307692307693) < 1e-9);
assert.equal(840 / 12, 70);
assert.equal(((22.362 / 21.208 - 1) * 100).toFixed(2), '5.44');
const tokenTimes = [120, 160, 220];
assert.equal(tokenTimes[0], 120);
assert.equal((tokenTimes.at(-1) - tokenTimes[0]) / (tokenTimes.length - 1), 50);
assert.equal(tokenTimes.at(-1), 220);
assert(Math.abs(1 / (0.8 + 0.2 / 2) - 1.1111111111111112) < 1e-12);

console.log('PASS: vector bounds, 2D exact coverage, warp rows, sectors, banks, occupancy, Roofline and request-time arithmetic.');
console.log('CPU arithmetic models only; no CUDA execution, hardware scheduling simulation or performance measurement.');
