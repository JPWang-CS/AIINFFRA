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
const currentFocus = now.match(/^- \[(?:当前实践|当前阅读)：[^\]]+\]\(([^)]+)\)/m);
assert(currentFocus, 'NOW must expose a Markdown link for 当前实践 or 当前阅读');

function isValidCourseRoute(route) {
  const routeMatch = route.match(/^\.\/(roadmap\/curriculum\/gpu\/course-site\/index\.html)(?:\?([^#]*))?(?:#(.+))?$/);
  if (!routeMatch) return false;
  const chapter = new URLSearchParams(routeMatch[2] || '').get('chapter');
  const fragment = routeMatch[3];
  if (!chapter || !/^\d+$/.test(chapter)) return false;

  const articleStart = html.indexOf(`<article class="chapter" id="chapter-${chapter}" data-chapter="${chapter}"`);
  if (articleStart < 0) return false;
  const articleEnd = html.indexOf('</article>', articleStart);
  if (articleEnd < 0) return false;
  const article = html.slice(articleStart, articleEnd + '</article>'.length);
  if (!fragment) return true;
  return new RegExp(`^chapter-${chapter}-section-\\d+$`).test(fragment)
    && article.includes(`id="${fragment}"`);
}

const currentRoute = currentFocus[1];
assert(isValidCourseRoute(currentRoute), 'Current focus must target a valid static course chapter or section');
assert(isValidCourseRoute('./roadmap/curriculum/gpu/course-site/index.html?chapter=4'),
  'A valid chapter-level entry without a fragment must be accepted');
const chapter4 = html.match(/<article class="chapter" id="chapter-4" data-chapter="4"[\s\S]*?<\/article>/)?.[0];
const validSection = chapter4?.match(/\bid="(chapter-4-section-\d+)"/)?.[1];
assert(validSection, 'Chapter 4 must expose a section anchor for route validation');
assert(isValidCourseRoute(`./roadmap/curriculum/gpu/course-site/index.html?chapter=4#${validSection}`),
  'A valid section-level entry must be accepted');
assert(!isValidCourseRoute('./roadmap/curriculum/gpu/course-site/index.html?chapter=999'),
  'A route to a missing chapter must be rejected');
const chapter25 = html.match(/<article class="chapter" id="chapter-25" data-chapter="25"[\s\S]*?<\/article>/)?.[0];
const wrongChapterSection = chapter25?.match(/\bid="(chapter-25-section-\d+)"/)?.[1];
assert(wrongChapterSection, 'Chapter 25 must expose a section anchor for wrong-chapter validation');
assert(!isValidCourseRoute(
  `./roadmap/curriculum/gpu/course-site/index.html?chapter=4#${wrongChapterSection}`),
  'A fragment from a different chapter must be rejected');
assert(now.includes('./roadmap/curriculum/gpu/course-site/index.html?chapter=25'));

console.log('PASS: static entry/assets, embedded search/paper content, local-file navigation and valid chapter/section focus routes.');
console.log('Source/asset check only; local file opening still needs a browser check.');
