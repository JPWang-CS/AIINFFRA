const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM} = require('jsdom');
const {pathToFileURL, fileURLToPath} = require('node:url');

const site = __dirname;
const repo = path.resolve(site, '../../../..');
const index = path.join(site, 'index.html');
const htmlFiles = [];
function walk(dir) {
  for (const entry of fs.readdirSync(dir, {withFileTypes:true})) {
    if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(file); else if (/\.html$/i.test(entry.name)) htmlFiles.push(file);
  }
}
walk(site);
assert(fs.existsSync(index), 'Missing index.html');
const domCache = new Map();
function dom(file) { if (!domCache.has(file)) domCache.set(file, new JSDOM(fs.readFileSync(file, 'utf8')).window.document); return domCache.get(file); }
function within(file, parent) { const rel = path.relative(parent, file); return rel && !rel.startsWith('..') && !path.isAbsolute(rel); }
const isExternal = value => /^(?:[a-z][a-z0-9+.-]*:|\/\/|data:)/i.test(value);
const routeChapters = new Set(); let checked = 0; let fragmentCount = 0; let docsTocCount = 0;
const publishedChapters = new Set([...dom(index).querySelectorAll('article.chapter[data-chapter]')].map(node => Number(node.dataset.chapter)));
assert(publishedChapters.size > 0, 'No published chapters');
for (const file of htmlFiles) {
  const sourceDoc = dom(file);
  for (const node of sourceDoc.querySelectorAll('[href],[src]')) {
    const attr = node.hasAttribute('href') ? 'href' : 'src'; const value = node.getAttribute(attr);
    if (!value || isExternal(value)) continue;
    const rawPath = value.split(/[?#]/, 1)[0];
    assert(!/[\\\s\u0080-\uffff]/.test(rawPath), 'Unencoded local URL path in ' + path.relative(site, file) + ': ' + value);
    let targetUrl;
    try { targetUrl = new URL(value, pathToFileURL(file).href); } catch (error) { throw new Error('Invalid local URL in ' + path.relative(site, file) + ': ' + value + ' (' + error.message + ')'); }
    let target;
    try { target = fileURLToPath(targetUrl); } catch (error) { throw new Error('Local URL is not a file URL in ' + value + ': ' + error.message); }
    assert(fs.existsSync(target), 'Missing local target from ' + path.relative(site, file) + ': ' + value + ' -> ' + target);
    assert(within(target, site) || within(target, repo), 'Local target escapes repository: ' + value);
    assert(!target.toLowerCase().endsWith('.md'), 'Generated site still links to raw Markdown: ' + value);
    const chapterValue = targetUrl.searchParams.get('chapter');
    if (chapterValue !== null) {
      assert.equal(path.normalize(target), path.normalize(index), 'chapter route must target site/index.html: ' + value);
      const chapter = Number(chapterValue); assert(Number.isInteger(chapter) && publishedChapters.has(chapter), 'Invalid chapter route: ' + value); routeChapters.add(chapter);
      if (targetUrl.hash) { const anchor = decodeURIComponent(targetUrl.hash.slice(1)); const targetDoc = dom(target); const owner = targetDoc.querySelector('[data-chapter="' + chapter + '"]'); const targetNode = targetDoc.getElementById(anchor); assert(owner && targetNode && owner.contains(targetNode), 'Anchor does not belong to chapter ' + chapter + ': ' + value); fragmentCount++; }
    } else if (targetUrl.hash) {
      const anchor = decodeURIComponent(targetUrl.hash.slice(1));
      if (/\.html$/i.test(target)) { assert(dom(target).getElementById(anchor), 'Missing HTML anchor in ' + path.relative(site, target) + ': ' + value); fragmentCount++; }
      else if (/\.pdf$/i.test(target)) { assert(/^page=\d+$/i.test(anchor), 'PDF fragment must be page=N: ' + value); fragmentCount++; }
      else throw new Error('Fragment target is neither HTML nor supported PDF: ' + value);
    }
    if (path.relative(site, file).startsWith('docs' + path.sep) && node.closest('.chapter-outline') && targetUrl.hash) docsTocCount++;
    checked++;
  }
}
for (const n of publishedChapters) assert(routeChapters.has(n), 'No generated route coverage for chapter ' + n);
assert(docsTocCount > 0, 'No attached-document TOC fragments were audited');
console.log('PASS: audited ' + htmlFiles.length + ' HTML artifacts and ' + checked + ' local href/src targets; verified full-file URL resolution, encoded paths, target files, raw-Markdown exclusion, chapter routes, target-DOM fragments (' + fragmentCount + ') and docs TOCs (' + docsTocCount + ').');
