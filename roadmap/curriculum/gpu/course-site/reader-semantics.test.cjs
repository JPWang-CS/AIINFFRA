const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {papers, prepare} = require('./paper-catalog.cjs');
const repo = path.resolve(__dirname, '../../../..');
const read = file => fs.readFileSync(path.join(repo, file), 'utf8');
const prepared = new Map(papers.map(p => [p.id, prepare(read(p.source), p, repo)]));

// Guard concrete editorial regressions, not a substitute for technical review.
assert(prepared.get(26).includes('求职时应能现场核对的三个问题'));
for (const id of [27, 28, 29, 30]) {
  assert(prepared.get(id).includes('求职核对题'), 'Interview exercises lost in publication: ' + id);
}
for (const [id, text] of prepared) {
  assert(!/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(text), 'Control character in paper ' + id);
}
assert(!prepared.get(24).includes('89.6'), 'Unsubstantiated SuperGLUE table returned');
assert(prepared.get(24).includes('T5-XXL') && prepared.get(24).includes('TPUv4'));
assert(!prepared.get(25).includes('p[:, :, None] * c_kv[None, :, :]'),
  'MLA accumulation must reduce over the token axis');
assert(prepared.get(25).includes('tl.dot(p, c_kv)'));
assert(!prepared.get(33).includes('~0.1×'), 'Numerical accuracy is not a fixed bit-width multiplier');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const payload = JSON.parse(html.match(/<script id="paper-content-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
for (const [chapter, sections] of [[20,[4]],[21,[5]],[24,[8]],[25,[31]],
  [30,[11]],[33,[7,8]],[35,[6]],[36,[16,17,18,29,30]],[37,[6]],[38,[21,22,23]]]) {
  for (const section of sections) {
    assert(payload[chapter].includes('id="chapter-' + chapter + '-section-' + section + '"'),
      'Renamed or consolidated section lost its bookmark: ' + chapter + '/' + section);
  }
}

const near = (a, b) => assert(Math.abs(a - b) < 1e-12, a + ' != ' + b);
const kv = (L, B, S, H, D, bytes) => 2 * L * B * S * H * D * bytes;
assert.equal(kv(32, 4, 8192, 8, 128, 2), 4 * 2 ** 30);
assert.equal(kv(32, 4, 8192, 32, 128, 2), 16 * 2 ** 30);
assert.equal(kv(61, 1, 131072, 128, 128, 2), 488 * 2 ** 30);
assert.equal(kv(61, 1, 131072, 8, 128, 2), 30.5 * 2 ** 30);
near(61 * 131072 * (512 + 64) * 2 / 2 ** 30, 8.578125);
assert.equal(Math.ceil(17 / 16) * 16 - 17, 15);

// Merge two local softmax states after rescaling both numerator and denominator.
const alpha = Math.exp(-Math.log(3));
near((2 * alpha + 10) / (alpha + 1), 8);

// A one-dimensional Gated DeltaNet example catches decay applied too late.
const state = 2, key = 1, value = 3, decay = .5, beta = .25;
const decayed = decay * state;
near(decayed + beta * (value - decayed * key) * key,
  decay * state * (1 - beta * key * key) + beta * value * key);
assert.notEqual(decayed + beta * (value - state * key) * key, 1.5);

// Exact two-token rejection sampling, enumerated instead of Monte Carlo.
const p = [.5, .5], q = [.8, .2];
const accepted = q.map((mass, i) => mass * Math.min(1, p[i] / mass));
const residual = p.map((mass, i) => Math.max(0, mass - q[i]));
const rejectMass = 1 - accepted.reduce((a, b) => a + b, 0);
const residualMass = residual.reduce((a, b) => a + b, 0);
accepted.forEach((mass, i) => near(mass + rejectMass * residual[i] / residualMass, p[i]));

console.log('PASS: reviewed paper publication, GQA/MLA capacity, page tail, online merge, gated delta update and rejection sampling examples.');
