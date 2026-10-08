const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const site = __dirname;
const repo = path.resolve(site, '../../../..');
const html = fs.readFileSync(path.join(site, 'index.html'), 'utf8');
const app = fs.readFileSync(path.join(site, 'app.js'), 'utf8');
const now = fs.readFileSync(path.join(repo, 'NOW.md'), 'utf8');

for (const name of ['theme.js', 'styles.css', 'vendor/katex/katex.min.css', 'app.js']) {
  assert(fs.existsSync(path.join(site, name)), `Missing offline asset: ${name}`);
  assert(html.includes(`"${name}`), `Generated entry does not reference ${name}`);
}
for (const id of ['course-search-data', 'paper-content-data']) {
  assert(html.includes(`id="${id}"`), `Content would need a runtime request: ${id}`);
}
assert(!/\bfetch\s*\(|\bXMLHttpRequest\b/.test(app), 'Reader must not fetch local content at runtime');
assert(/const localFile = location\.protocol === 'file:'/.test(app), 'Missing local-file navigation branch');
assert(/if \(localFile\) \{\s*if \(replace\) location\.replace\(url\.href\);\s*else location\.assign\(url\.href\);/.test(app),
  'Local-file navigation must avoid history.pushState');
assert(!/127\.0\.0\.1:8765/.test(now), 'Current course entry must not depend on a temporary server');
const currentPractice = now.match(/^- \[当前实践：[^\]]+\]\(([^)]+)\)/m);
assert(currentPractice, 'NOW must expose a Markdown link for 当前实践');
const route = currentPractice[1];
const routeMatch = route.match(/^\.\/(roadmap\/curriculum\/gpu\/course-site\/index\.html)(?:\?([^#]*))?(?:#(.+))?$/);
assert(routeMatch, '当前实践 must use the repository-relative static course index');
const query = new URLSearchParams(routeMatch[2] || '');
const chapter = query.get('chapter');
const fragment = routeMatch[3];
assert(chapter && /^\d+$/.test(chapter), '当前实践 link must identify a numeric chapter');
assert(fragment, '当前实践 link must identify a section fragment');
assert(html.includes(`<article class="chapter" id="chapter-${chapter}" data-chapter="${chapter}"`),
  `Current chapter article is missing: ${chapter}`);
assert(html.includes(`id="${fragment}"`), `Current section fragment is missing: ${fragment}`);
assert(now.includes('./roadmap/curriculum/gpu/course-site/index.html?chapter=25'));

console.log('PASS: static entry/assets, embedded search/paper content, local-file navigation branch and server-free NOW links.');
console.log('Source/asset check only; local file opening still needs a browser check.');
