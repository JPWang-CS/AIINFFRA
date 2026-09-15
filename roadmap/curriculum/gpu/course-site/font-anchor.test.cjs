const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM} = require('jsdom');
const html = '<div class="sidebar"></div><button id="menu-button"></button>'
  + '<input id="search"><div id="search-results"></div><div id="progress-bar"></div><nav id="page-toc"></nav>'
  + '<a data-part-link="gpu" data-chapter-link="1" href="?chapter=1">GPU</a>'
  + '<a data-part-link="papers" data-chapter-link="39" href="?chapter=39">Papers</a>'
  + '<main class="content"><article class="chapter" data-chapter="1" data-part="gpu" data-title="GPU">'
  + '<div class="markdown-body"><h2 id="gpu">GPU</h2></div></article>'
  + '<article class="chapter" data-chapter="39" data-part="papers" data-title="Paper">'
  + '<div class="markdown-body" data-paper-pending="true"><span id="math"></span></div></article></main>'
  + '<script id="paper-content-data" type="application/json">{"39":"<h2 id=\\"math\\">Math</h2>"}</script>';
async function run(action, expected) {
  const dom = new JSDOM(html, {url:'http://localhost/index.html?chapter=39#math', runScripts:'outside-only'});
  const w=dom.window;
  let resolveFonts, scrolls=0;
  const ready=new Promise(resolve=>{resolveFonts=resolve;});
  Object.defineProperty(w.document,'fonts',{value:{status:'loading',ready}});
  w.HTMLElement.prototype.scrollIntoView=()=>{scrolls++;};
  w.scrollTo=()=>{};
  w.requestAnimationFrame=fn=>{fn();return 1;};
  w.eval(fs.readFileSync(path.join(__dirname,'app.js'),'utf8'));
  assert.equal(scrolls,1);
  if(action==='interaction')w.dispatchEvent(new w.Event('wheel'));
  if(action==='navigate')w.document.querySelector('[data-part-link="gpu"]').click();
  resolveFonts();
  await Promise.resolve(); await Promise.resolve();
  assert.equal(scrolls,expected,action);
  dom.window.close();
}
(async()=>{
  await run('settle',2);
  await run('interaction',1);
  await run('navigate',1);
  console.log('PASS: font-ready anchor settlement, user-scroll cancellation and navigation cancellation.');
})().catch(error=>{console.error(error);process.exitCode=1;});
