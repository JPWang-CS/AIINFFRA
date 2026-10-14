const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {JSDOM,VirtualConsole} = require('jsdom');
const html=fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const app=fs.readFileSync(path.join(__dirname,'app.js'),'utf8');
const base='http://127.0.0.1:8765/roadmap/curriculum/gpu/course-site/index.html';
const payload=JSON.parse(html.match(/<script id="paper-content-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const {papers,categoryOrder,prepare}=require('./paper-catalog.cjs');
const repo=path.resolve(__dirname,'..','..','..','..');
const anchorRegistry=JSON.parse(fs.readFileSync(path.join(repo,'.codex','course-section-anchors.json'),'utf8'));
const legacyAnchors=JSON.parse(fs.readFileSync(path.join(repo,'.codex','paper-legacy-anchors.json'),'utf8'));
assert.equal(anchorRegistry.chapters['31']['2|12. 数字账（记住这组数）|1'],18,
  'Renamed V4 section must retain its old title alias');
assert.equal(anchorRegistry.chapters['31']['2|12. 架构配置与成本对比|1'],18,
  'Renamed V4 section must reuse the old stable section ID');
const examPaper=papers.find(paper=>paper.id===39);
const preparedExam=prepare(fs.readFileSync(path.join(repo,examPaper.source),'utf8'),examPaper,repo);
assert(preparedExam.includes("Agent's Last Exam"),'prepare must preserve the official benchmark name');
assert(!preparedExam.includes("智能体's Last Exam"),'prepare must not translate part of the official benchmark name');
assert.equal(Object.keys(payload).length,papers.length);
assert.deepEqual(Object.keys(legacyAnchors.chapters),Array.from({length:21},(_,i)=>String(20+i)));
for(const [chapter,ranges] of Object.entries(legacyAnchors.chapters)){
  const body=payload[String(chapter)];
  assert(body,`Missing legacy paper payload ${chapter}`);
  const ids=new Set([...JSDOM.fragment(body).querySelectorAll('[id]')].map(node=>node.id));
  for(const [first,last] of ranges){
    for(let section=first;section<=last;section++){
      assert(ids.has(`chapter-${chapter}-section-${section}`),
        `HEAD-published legacy anchor was not preserved: ${chapter}/${section}`);
    }
  }
}
assert(!legacyAnchors.chapters['23'].some(([first,last])=>first<=31&&31<=last),
  'Historical registry-only chapter 23/31 was never published and is not a legacy URL');
for(const [id,body] of Object.entries(payload)){
  const doc=JSDOM.fragment(body);
  for(const word of ['WIP','等待验收','待讨论','Agent 整合','当前进度','当前状态：','本笔记作为当前学习节']){
    assert(!doc.textContent.includes(word),'Paper '+id+' contains '+word);
  }
  for(const a of doc.querySelectorAll('a[href]')){
    assert(!/\.(?:py|cu|cuh|cpp|sh|md)(?:[?#]|$)|\.pdf#page=/i.test(a.getAttribute('href')),
      'Paper '+id+' needs inline content: '+a.getAttribute('href'));
    const href=a.getAttribute('href');
    if(/^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(href))continue;
    const url=new URL(href,base);
    // Reader routes are resolved below against hydrated headings.
    if(url.pathname===new URL(base).pathname){
      const chapter=url.searchParams.get('chapter');
      assert(chapter && html.includes('id="chapter-'+chapter+'"'),'Unknown paper target '+href);
      if(url.hash) assert(html.includes('id="'+decodeURIComponent(url.hash.slice(1))+'"'),'Missing anchor '+href);
    }else{
      const pathname=decodeURIComponent(href.split(/[?#]/)[0]);
      assert(fs.existsSync(path.resolve(__dirname,pathname)),'Missing paper asset '+href);
    }
  }
  const ids=[...doc.querySelectorAll('[id]')].map(el=>el.id);
  assert.equal(ids.length,new Set(ids).size,'Duplicate paper headings '+id);
}
const errors=[];
const vc=new VirtualConsole();
vc.on('jsdomError',error=>errors.push(error));
const dom=new JSDOM(html,{url:base+'?chapter=13',runScripts:'outside-only',virtualConsole:vc,pretendToBeVisual:true});
const w=dom.window,d=w.document;
w.scrollTo=()=>{};
w.HTMLElement.prototype.scrollIntoView=function(){w.lastScrolled=this.id;};
let copied='';
Object.defineProperty(w.navigator,'clipboard',{value:{writeText:async text=>{copied=text;}}});
w.eval(app);
const click=selector=>{const el=d.querySelector(selector);assert(el,selector);el.click();return el;};
const current=id=>{
  assert.equal(new URL(w.location.href).searchParams.get('chapter'),String(id));
  assert.equal([...d.querySelectorAll('.chapter')].filter(el=>!el.hidden)[0].dataset.chapter,String(id));
};
(async()=>{
  click('[data-part-link="papers"]');
  current(41);
  assert(d.querySelector('[data-nav-chapter="41"]').closest('.paper-category').open);
  assert(d.querySelector('#chapter-41 .markdown-body').textContent.includes('next-token'));
  click('[data-nav-chapter="25"] summary a');
  current(25);
  assert(d.querySelector('[data-nav-chapter="25"]').closest('.paper-category').open);
  assert(d.querySelector('#chapter-25 .katex'));
  assert(d.querySelector('#chapter-25 .markdown-body').textContent.includes('def mla_content'));
  click('#chapter-25 .copy-code');
  await new Promise(r=>setTimeout(r,5));
  assert(copied.length>0,'Hydrated code must be copyable');
  const sub=click('[data-nav-chapter="23"] a[data-section-link]');
  current(23);
  assert.equal(w.lastScrolled,sub.dataset.sectionLink,'Scroll to hydrated heading, not detached placeholder');
  assert.equal(d.getElementById(sub.dataset.sectionLink).tagName,'H2');
  const group=d.querySelector('[data-nav-chapter="23"]').closest('.paper-category');
  const before=w.location.href;
  group.querySelector(':scope > summary').click();
  assert(!group.open);assert.equal(w.location.href,before);
  click('[data-nav-chapter="26"] summary a');
  current(26);
  assert(d.querySelector('#chapter-26 .markdown-body').textContent.includes('def selected_attention'));
  click('[data-part-link="operators"]');current(13);
  click('[data-part-link="papers"]');current(26);
  // All paper leaves hydrate in place, and every sidebar subsection resolves.
  for(const id of Object.keys(payload)){
    click('[data-nav-chapter="'+id+'"] summary a');current(id);
    for(const a of d.querySelectorAll('[data-nav-chapter="'+id+'"] a[data-section-link]')){
      const heading=d.getElementById(a.dataset.sectionLink);
      assert(heading && /^H[23]$/.test(heading.tagName),'Broken paper section '+a.href);
    }
  }
  assert(!d.querySelector('#chapter-19 .chapter-switch [data-chapter-link="20"]'));
  assert.equal(d.querySelectorAll('.paper-category').length,categoryOrder.length);
  assert.deepEqual([...d.querySelectorAll('.paper-category > summary')].map(node=>node.textContent),categoryOrder);
  for(const paper of papers){
    const peers=papers.filter(p=>p.category===paper.category).sort((a,b)=>a.order-b.order);
    const at=peers.findIndex(p=>p.id===paper.id);
    const expected=[peers[at-1]?.id,peers[at+1]?.id].filter(Boolean).map(String);
    const actual=[...d.querySelectorAll(`#chapter-${paper.id} .chapter-switch [data-chapter-link]`)]
      .map(node=>node.dataset.chapterLink);
    assert.deepEqual(actual,expected,`Previous/next paper links must follow category order for ${paper.id}`);
  }
  assert(d.querySelector('#chapter-39 .markdown-body').textContent.includes('def sinkhorn_step'));
  assert(d.querySelector('#chapter-39 .markdown-body').textContent.includes('16384'));
  assert(d.querySelector('#chapter-40 .markdown-body').textContent.includes('def clipped_objective'));
  assert(d.querySelector('#chapter-4 .markdown-body').textContent.includes('cudaFreeAsync'));
  assert.equal(d.querySelectorAll('#chapter-32 .chapter-switch [data-chapter-link]').length,0,
    'Standalone MoE category has no cross-category page turn');
  click('#chapter-31 .chapter-switch [data-chapter-link="39"]');current(39);
  click('#chapter-39 .chapter-switch [data-chapter-link="31"]');current(31);
  assert.equal(errors.length,0,errors.map(e=>e.message).join('\n'));
  for(const [source,id] of [...papers.map(p=>[p.source,p.id]),['notes/algorithms/README.md',41]]){
    const redirect=fs.readFileSync(path.join(__dirname,'docs',source.replace(/\.md$/,'.html')),'utf8');
    assert(redirect.includes('location.replace(') && redirect.includes('?chapter='+id),'Legacy paper redirect '+source);
    assert(!redirect.includes('window.open'),'Do not open paper in another window');
  }
  dom.window.close();
  console.log(`PASS: ${papers.length} papers, ${categoryOrder.length} ordered categories, in-reader routing, folding, hydrated formula/code, copying, category pagination and all subsection anchors.`);
})().catch(error=>{dom.window.close();console.error(error);process.exitCode=1;});
