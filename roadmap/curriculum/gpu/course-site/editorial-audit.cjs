const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const html = fs.readFileSync(path.join(__dirname,'index.html'),'utf8');
const papers = JSON.parse(html.match(/<script id="paper-content-data" type="application\/json">([\s\S]*?)<\/script>/)[1]);
const articles = [...html.matchAll(/<article class="chapter" id="chapter-(\d+)"[\s\S]*?<\/article>/g)];
const forbidden = ['从问题、公式与数据流理解算法，结合关键代码阅读。','等待验收','本轮生成','Agent 整合',
  '真正掌握机制','挂靠','学习挂载点','主线 A 挂载顺序','给 A4/A5 铺路','不伪造平台成绩','不能伪造多卡结果',
  'FP8 在生产里是默认选项','必须从 HBM 读 ~1.37TB','AdamW 修正 4 里的一个细节'];
const hits = [];
for (const match of articles) {
  const id=match[1];
  const prose=(match[0]+'\n'+(papers[id]||''))
    .replace(/<pre\b[\s\S]*?<\/pre>/g,'')
    .replace(/<[^>]+>/g,'');
  for(const phrase of forbidden)if(prose.includes(phrase))hits.push({chapter:Number(id),phrase});
  assert(!/^\s*(?:~~~|```)(?:python|cpp|cuda|bash)\s*$/m.test(prose),
    `Chapter ${id} contains an unrendered code fence`);
}
assert.equal(articles.length,39);
assert.equal(hits.length,0,JSON.stringify(hits,null,2));
const v41Body = papers['39'];
assert.equal([...v41Body.matchAll(/<h3\b[^>]*>多模态张量路径：patch token 到语言 token/g)].length,1,
  'Keep the multimodal walkthrough once, under the architecture section');
assert(!v41Body.includes('F_tile = block_l(mixed)'),
  'Mega-mHC does not fuse the full Transformer sublayer into each feature tile');
assert(v41Body.includes('copysign'), 'Retain the exact signed-root gate convention');
const llamaDetail = html.match(/<details>\s*<summary>完整本地 Llama 校准、保存\/加载与评估程序<\/summary>([\s\S]*?)<\/details>/);
assert(llamaDetail && /<pre\b/.test(llamaDetail[1]) && /<code\b/.test(llamaDetail[1]),
  'The complete Llama program must render as code, not raw Markdown inside HTML');
assert(!llamaDetail[1].includes('~~~python'), 'No literal fence in the Llama program');
console.log('PASS: 39 reader articles checked for retired planning/status boilerplate.');
console.log('Editorial guard only; technical constraints are retained and prose still needs human review.');
