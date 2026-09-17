const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');

const site = __dirname;
const themeSource = fs.readFileSync(path.join(site, 'theme.js'), 'utf8');
const index = fs.readFileSync(path.join(site, 'index.html'), 'utf8');

assert.match(index, /<script src="theme\.js\?v=[a-f0-9]{12}"><\/script><link/);
assert.match(index, /id="theme-toggle"/);
assert.match(fs.readFileSync(path.join(site, 'styles.css'), 'utf8'), /:root\[data-theme="dark"\]/);

function boot(storage) {
  const dom = new JSDOM('<!doctype html><html><head></head><body><button id="theme-toggle">浅色</button></body></html>', {
    url: 'http://127.0.0.1/theme.html', runScripts: 'outside-only', pretendToBeVisual: true
  });
  if (storage !== undefined) {
    Object.defineProperty(dom.window, 'localStorage', {configurable: true, value: storage});
  }
  dom.window.eval(themeSource);
  dom.window.document.dispatchEvent(new dom.window.Event('DOMContentLoaded'));
  return dom;
}

const first = boot({getItem: () => null, setItem: () => {}});
assert.equal(first.window.document.documentElement.dataset.theme, 'dark');
first.window.document.querySelector('#theme-toggle').click();
assert.equal(first.window.document.documentElement.dataset.theme, 'light');

let saved;
const persisted = boot({getItem: () => saved || null, setItem: (_key, value) => { saved = value; }});
persisted.window.document.querySelector('#theme-toggle').click();
assert.equal(saved, 'light');
const reloaded = boot({getItem: () => saved, setItem: () => {}});
assert.equal(reloaded.window.document.documentElement.dataset.theme, 'light');

const unavailable = boot({getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); }});
assert.equal(unavailable.window.document.documentElement.dataset.theme, 'dark');
unavailable.window.document.querySelector('#theme-toggle').click();
assert.equal(unavailable.window.document.documentElement.dataset.theme, 'light');

console.log('theme tests passed');
