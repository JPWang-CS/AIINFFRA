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
const metadataCache = new Map();
function metadata(file) {
  if (metadataCache.has(file)) return metadataCache.get(file);
  const window = new JSDOM(fs.readFileSync(file, 'utf8')).window;
  try {
    const document = window.document;
    const ids = new Set();
    const anchorOwner = new Map();
    for (const node of document.querySelectorAll('[id]')) {
      const id = node.id;
      if (!id || ids.has(id)) continue;
      ids.add(id);
      const chapter = node.closest('article.chapter[data-chapter]');
      anchorOwner.set(id, chapter ? Number(chapter.dataset.chapter) : null);
    }
    const publishedChapters = new Set(
      [...document.querySelectorAll('article.chapter[data-chapter]')].map(node => Number(node.dataset.chapter))
    );
    const links = [...document.querySelectorAll('[href],[src]')].map(node => ({
      value: node.getAttribute(node.hasAttribute('href') ? 'href' : 'src'),
      inDocsToc: Boolean(node.closest('.chapter-outline'))
    }));
    const result = {ids, anchorOwner, publishedChapters, links};
    metadataCache.set(file, result);
    return result;
  } finally {
    window.close();
  }
}
function within(file, parent) { const rel = path.relative(parent, file); return rel && !rel.startsWith('..') && !path.isAbsolute(rel); }
const isExternal = value => /^(?:[a-z][a-z0-9+.-]*:|\/\/|data:)/i.test(value);
const routeChapters = new Set(); let checked = 0; let fragmentCount = 0; let docsTocCount = 0;
const publishedChapters = metadata(index).publishedChapters;
assert(publishedChapters.size > 0, 'No published chapters');
for (const file of htmlFiles) {
  const sourceMetadata = metadata(file);
  const inDocs = path.relative(site, file).startsWith('docs' + path.sep);
  for (const {value, inDocsToc} of sourceMetadata.links) {
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
      if (targetUrl.hash) {
        const anchor = decodeURIComponent(targetUrl.hash.slice(1));
        const targetMetadata = metadata(target);
        assert(targetMetadata.ids.has(anchor) && targetMetadata.anchorOwner.get(anchor) === chapter, 'Anchor does not belong to chapter ' + chapter + ': ' + value);
        fragmentCount++;
      }
    } else if (targetUrl.hash) {
      const anchor = decodeURIComponent(targetUrl.hash.slice(1));
      if (/\.html$/i.test(target)) { assert(metadata(target).ids.has(anchor), 'Missing HTML anchor in ' + path.relative(site, target) + ': ' + value); fragmentCount++; }
      else if (/\.pdf$/i.test(target)) { assert(/^page=\d+$/i.test(anchor), 'PDF fragment must be page=N: ' + value); fragmentCount++; }
      else throw new Error('Fragment target is neither HTML nor supported PDF: ' + value);
    }
    if (inDocs && inDocsToc && targetUrl.hash) docsTocCount++;
    checked++;
  }
}
for (const n of publishedChapters) assert(routeChapters.has(n), 'No generated route coverage for chapter ' + n);
assert(docsTocCount > 0, 'No attached-document TOC fragments were audited');
console.log('PASS: audited ' + htmlFiles.length + ' HTML artifacts and ' + checked + ' local href/src targets; verified full-file URL resolution, encoded paths, target files, raw-Markdown exclusion, chapter routes, target-DOM fragments (' + fragmentCount + ') and docs TOCs (' + docsTocCount + ').');
