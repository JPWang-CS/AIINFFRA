const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const repo = path.resolve(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'pdf-coverage.json'), 'utf8'));
assert.equal(data.sources.length, 3);
const counts = {};
for (const source of data.sources) {
  const bytes = fs.readFileSync(path.join(repo, source.path));
  const hash = crypto.createHash('sha256').update(bytes).digest('hex').toUpperCase();
  assert.equal(hash, source.sha256, 'PDF changed; re-audit before updating hash: ' + source.path);
  const sections = new Set();
  for (const unit of source.units) {
    assert(!sections.has(unit.section), 'Duplicate coverage section');
    sections.add(unit.section);
    assert(unit.pdf_pages.length === 2);
    const [first, last] = unit.pdf_pages;
    assert(first >= 1 && last >= first && last <= source.pages, 'Invalid page interval');
    assert(unit.status in data.status_definitions, 'Unknown coverage status');
    assert.notEqual(unit.status, 'pending_write', 'Pending write must not be claimed as delivered');
    counts[unit.status] = (counts[unit.status] || 0) + 1;
    if (unit.target) {
      const target = path.resolve(repo, unit.target);
      assert(!path.relative(repo, target).startsWith('..'));
      assert(fs.existsSync(target), 'Missing tutorial: ' + unit.target);
      assert(!/^(?:PATH|NOW)\.md$/i.test(unit.target), 'Coverage is not learning progress');
    } else {
      assert.equal(unit.status, 'not_integrated');
    }
  }
}
console.log('PASS: original PDF hashes, coverage intervals and tutorial targets.');
console.log(JSON.stringify(counts));
console.log('Structural tracking only; topic presence is not exhaustive semantic coverage.');
