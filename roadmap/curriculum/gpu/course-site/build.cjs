const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const {Marked} = require('marked');
const katex = require('katex');
const hljs = require('highlight.js');

const site = __dirname;
const gpuRoot = path.resolve(site, '..');
const repo = path.resolve(site, '../../../..');
const anchorRegistryFile = path.join(repo,'.codex/course-section-anchors.json');
const anchorRegistry = JSON.parse(fs.readFileSync(anchorRegistryFile,'utf8'));
const docsRoot = path.join(site, 'docs');
const assetRevision = name => crypto.createHash('sha256').update(fs.readFileSync(path.join(site,name))).digest('hex').slice(0,12);
const styleRevision = assetRevision('styles.css');
const appRevision = assetRevision('app.js');
const themeRevision = assetRevision('theme.js');
const chapters = [
  ['01-gpu-hardware-map-and-generations', '从 CUDA 程序看 GPU 的整体结构', '硬件全貌、编程模型、完整程序与地址映射。', '01-introduction/programming-model.html'],
  ['02-cuda-execution-and-scheduling', '线程执行、指令调度与计算管线', '从驻留线程到就绪指令，理解并行度、依赖与数值计算。', '02-basics/writing-cuda-kernels.html'],
  ['03-registers-and-memory-system', '寄存器、存储层次与数据访问', '从变量活跃区间到逐线程地址，分析复用、合并访问和存储冲突。', '02-basics/writing-cuda-kernels.html'],
  ['04-synchronization-and-asynchronous-execution', '同步、线程协作与异步流水线', '建立正确的数据依赖，再组织计算与搬运的重叠。', '02-basics/asynchronous-execution.html'],
  ['05-performance-analysis-and-optimization', '资源模型、性能分析与优化方法', '用资源预算、性能上界和真实实验解释优化结果。', '02-basics/writing-cuda-kernels.html'],
  ['../operators', '完整 GPU 算子体系', '九类算子的原理、实现与实践对比。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/01-memory-and-layout', '访存与布局算子', '从数据搬运、向量化访问到高效转置。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/02-reduction-and-norm', '并行归约、Softmax 与归一化', '从两级归约、稳定 Softmax 到 RMSNorm 和 LayerNorm。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/03-gemm', 'GEMM：从分块实现到性能优化', '从地址与尾块开始，理解分块复用、矩阵指令、流水线和实测证据。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/04-activation-and-fusion', 'Activation 与 Fusion：从 SwiGLU 点算子到 MLP 的物化边界', '从激活公式、地址布局和稳定 sigmoid，走到 pointwise、epilogue 与完整 MLP 的融合边界。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/05-prefill-attention', 'Prefill Attention：从 Softmax 加权和到在线分块', '从矩形题面、方形教学 baseline 和 online softmax 出发，理解 Q 驻留、KV 扫描、尾块与并行组织。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/06-decode-paged-attention', 'Decode 与 PagedAttention：分页寻址、归约与缓存管理', '从单 Query、GQA、非连续页表走到 split-KV、写时复制和缓存压缩账本。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/07-quantized-operators', '量化算子：编码、布局与低精度计算', '从舍入、scale 与 INT4 打包，走到量化算法、GEMM 路径及性能对比。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/08-moe', 'MoE：路由、重排、专家计算与合并', '沿 token 到专家再回到 token 的数据路径，验证索引、权重和性能。', '02-basics/writing-cuda-kernels.html'],
  ['../operators/09-sampling-kv', 'Sampling 与 KV 辅助算子', '选择、随机采样、投机验证与缓存提交的代码和边界。', '02-basics/writing-cuda-kernels.html'],
  ['../model-analysis', '模型 GPU 执行分析：Prefill 与 Decode', '从算子形状、FLOPs、容量与时间线，解释模型级性能。', null],
  ['../model-analysis/mini-transformer', 'Mini Transformer：完整前向与算子替换', '用同一组权重对照整段、逐 token、分块输入及算子替换。', null],
  ['../systems', 'vLLM：请求、调度、缓存与算子接入', '确认实际 backend，建立同条件的服务实验。', null],
  ['../systems/multi-gpu', '多 GPU：分片、通信与端到端收益', '从 TP 数学到 NCCL、EP/PP/CP，分析容量与通信代价。', null]
];
const {papers, prepare: preparePaper} = require('./paper-catalog.cjs');
const paperByFile = new Map(papers.map(p => [path.resolve(repo,p.source), p]));
const paperById = new Map(papers.map(p => [p.id,p]));
chapters.push(...papers.map(p => [p.source,p.title,'',null]));
const chapterFile = dir => dir.endsWith('.md') ? path.resolve(repo,dir) : path.resolve(gpuRoot,dir,'README.md');
const publicationText = file => paperByFile.has(file) ? preparePaper(sourceText(file),paperByFile.get(file),repo) : sourceText(file);
// ID 6 is retired; retain all other public URLs.
const chapterFiles = new Map(chapters.flatMap(([dir], i) => i === 5 ? [] : [[chapterFile(dir), i + 1]]));
const operatorDirectory = path.resolve(gpuRoot, '../operators/README.md');
const pages = new Map();
const assets = new Map();
const metadataMarked = new Marked({gfm: true});
const allowedRoots = new Set(['lessons', 'notes', 'papers', 'solutions', 'reference', 'roadmap', 'weekly', 'templates', 'scripts']);
const allowedAsset = /\.(?:png|jpe?g|gif|svg|webp|bmp|cu|c|cc|cpp|h|hh|hpp|cuh|py|sh|txt|pdf)$/i;
const esc = text => String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const posix = value => value.split(path.sep).join('/');
const encodedPath = value => posix(value).split('/').map(encodeURIComponent).join('/');
const insideRepo = file => { const relative = path.relative(repo, file); return relative && !relative.startsWith('..') && !path.isAbsolute(relative); };
const repoRelative = file => posix(path.relative(repo, file));
function allowedFile(file) {
  const parts = repoRelative(file).split('/');
  if (parts.some(part => part.startsWith('.'))) return false;
  if (/^downloads\/[^/]+\.pdf$/i.test(repoRelative(file))) return true;
  if (parts.length === 1) return parts[0].toLowerCase() === 'readme.md';
  return allowedRoots.has(parts[0]) && (/\.md$/i.test(file) || allowedAsset.test(file));
}
const sourceText = file => fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n');
function external(href) { return /^(?:[a-z][a-z0-9+.-]*:|\/\/)/i.test(href); }
function localTarget(href, fromFile) {
  const rawPath = href.split(/[?#]/, 1)[0];
  if (!rawPath) return null;
  let file = path.resolve(path.dirname(fromFile), decodeURIComponent(rawPath));
  if (fs.existsSync(file) && fs.statSync(file).isDirectory()) { const readme = path.join(file, 'README.md'); if (fs.existsSync(readme)) file = readme; }
  return fs.existsSync(file) ? fs.realpathSync(file) : file;
}
function stripInline(value) {
  return value.replace(/!\[([^]]*)\]\([^)]*\)/g, '$1').replace(/\[([^]]+)\]\([^)]*\)/g, '$1')
    .replace(/[`*_~]/g, '').replace(/<[^>]*>/g, '').trim();
}
function slug(value, used) {
  const base = stripInline(value).toLowerCase().replace(/[^\p{L}\p{N}\s_-]/gu, '').replace(/\s/g, '-').replace(/^-+|-+$/g, '') || 'section';
  let id = base; let n = 2;
  while (used.has(id)) id = base + '-' + n++;
  used.add(id); return id;
}
function headings(text, chapter) {
  const used = new Set(); const out = []; let number = 0;
  const registry = chapter ? (anchorRegistry.chapters[String(chapter)] ||= {}) : null;
  let nextId = registry ? Math.max(-1,...Object.values(registry)) + 1 : 0;
  const occurrences = new Map();
  const walk = tokens => {
    for (const token of tokens || []) {
      if (token.type === 'heading') {
        const plain = stripInline(token.text);
        let id;
        if (!chapter) id = slug(token.text,used);
        else if (!registry) id = 'chapter-'+chapter+'-section-'+number++;
        else {
          const base = token.depth+'|'+plain;
          const occurrence = (occurrences.get(base)||0)+1;
          occurrences.set(base,occurrence);
          const key = base+'|'+occurrence;
          if (!(key in registry)) registry[key] = nextId++;
          id = 'chapter-'+chapter+'-section-'+registry[key];
        }
        out.push({text:plain,level:token.depth,id});
      }
      if (token.tokens) walk(token.tokens);
    }
  };
  walk(metadataMarked.lexer(text)); return out;
}
function markdownTargets(text) {
  const out = []; const re = /!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+["'][^)]*["'])?\s*\)/g;
  for (const m of text.matchAll(re)) out.push(m[1] || m[2]);
  return out;
}
function registerTarget(href, fromFile) {
  if (external(href) || href.startsWith('#')) return;
  const target = localTarget(href, fromFile);
  if (!target || !insideRepo(target) || !fs.existsSync(target) || !allowedFile(target)) return;
  if (/\.md$/i.test(target)) discover(target);
  else assets.set(target, 'docs/' + repoRelative(target));
}
function discover(file) {
  file = fs.realpathSync(file); if (!allowedFile(file) || pages.has(file)) return;
  const text = publicationText(file);
  pages.set(file, {file, text, output: 'docs/' + repoRelative(file).replace(/\.md$/i, '.html'), title: (text.match(/^#\s+(.+)$/m) || [,'附属文档'])[1], headings: headings(text)});
  for (const href of markdownTargets(text)) registerTarget(href, file);
}
for (const [dir] of chapters) discover(chapterFile(dir));
discover(path.join(repo, 'notes', 'algorithms', 'README.md'));
const chapterHeadings = new Map(chapters.map(([dir], i) => {
  const file = fs.realpathSync(chapterFile(dir));
  const text = publicationText(file).replace(/^# .*\n/, '').replace(/\n## 章节导航[\s\S]*$/, '').trim();
  return [file, headings(text, i + 1)];
}));

let context = null; let mathCount = 0; let checkedSourceBlocks = 0;
function math(text, displayMode) { mathCount++; return katex.renderToString(text, {displayMode, throwOnError: true, trust: false, strict: 'error', output: 'htmlAndMathml'}); }
function anchorFor(file, hash, chapter) {
  if (!hash) return '';
  const wanted = decodeURIComponent(hash).replace(/^#/, '').toLowerCase(); const list = chapter ? chapterHeadings.get(file) : pages.get(file)?.headings;
  const hit = list && list.find(h => h.id.toLowerCase() === wanted || slug(h.text, new Set()).toLowerCase() === wanted || h.text.toLowerCase() === wanted);
  if (!hit && chapter) {
    const aliases = anchorRegistry.chapters[String(chapter)] || {};
    for (const [key, number] of Object.entries(aliases)) {
      const title = key.slice(key.indexOf('|')+1,key.lastIndexOf('|'));
      const id = 'chapter-'+chapter+'-section-'+number;
      if ((slug(title,new Set()) === wanted || title.toLowerCase() === wanted)
          && list?.some(h=>h.id===id)) return '#'+id;
    }
  }
  return '#' + encodeURIComponent(hit ? hit.id : wanted);
}
function relativeUrl(fromOutput, targetOutput) {
  const rel = posix(path.relative(path.dirname(path.join(site, fromOutput)), path.join(site, targetOutput)));
  return encodedPath(rel || path.basename(targetOutput));
}
function repoUrl(fromOutput, targetFile) {
  return encodedPath(posix(path.relative(path.dirname(path.join(site, fromOutput)), targetFile)));
}
function rewriteLink(href, fromFile, fromOutput, kind) {
  if (external(href)) return href;
  if (/(?:^|[?&])chapter=6(?:&|#|$)/.test(href)) href = href.replace(/chapter=6(?=&|#|$)/, 'chapter=7').replace(/#.*$/, '');
  const match = /^(.*?)(\?[^#]*)?(#.*)?$/.exec(href); const rawPath = match[1]; const query = match[2] || ''; const hash = match[3] || '';
  if (!rawPath) return kind === 'chapter' ? '?chapter=' + context.chapter + (hash ? anchorFor(fromFile, hash, context.chapter) : '') : hash;
  const target = localTarget(rawPath, fromFile);
  if (!target || !insideRepo(target) || !fs.existsSync(target)) return href;
  if (target === path.join(site, 'index.html')) return relativeUrl(fromOutput, 'index.html') + query + hash;
  if (fs.statSync(target).isDirectory()) return repoUrl(fromOutput, target) + (rawPath.endsWith('/') ? '/' : '') + query + hash;
  if (target === path.join(repo,'notes/algorithms/README.md')) return relativeUrl(fromOutput,'index.html') + '?chapter=25';
  if (target === operatorDirectory) {
    const topic = decodeURIComponent(hash).match(/^#([1-9])-/);
    return (kind === 'chapter' ? '' : relativeUrl(fromOutput, 'index.html')) + '?chapter=' + (topic ? Number(topic[1]) + 6 : 7);
  }
  if (!allowedFile(target)) return null;
  if (/\.md$/i.test(target)) {
    const chapter = chapterFiles.get(target);
    if (chapter) return (kind === 'chapter' ? '' : relativeUrl(fromOutput, 'index.html')) + '?chapter=' + chapter + anchorFor(target, hash, chapter);
    discover(target); return relativeUrl(fromOutput, pages.get(target).output) + anchorFor(target, hash);
  }
  if (!allowedFile(target)) throw new Error('Disallowed local asset: ' + repoRelative(target));
  if (/^downloads[\\/].+\.pdf$/i.test(repoRelative(target))) return repoUrl(fromOutput, target) + query + hash;
  const output = assets.get(target) || ('docs/' + repoRelative(target)); assets.set(target, output);
  return relativeUrl(fromOutput, output) + query + hash;
}
const md = new Marked({gfm: true});
md.use({renderer: {
  blockquote({tokens}) {
    const html = this.parser.parse(tokens); const note = /^<p>\[!(IMPORTANT|WARNING|TIP|NOTE)\][ \t]*([^\n<]*)(?:\n|<br\s*\/?>)/.exec(html);
    if (!note) return '<blockquote>' + html + '</blockquote>';
    const labels = {IMPORTANT:'核心结论', WARNING:'边界条件', TIP:'实践复盘', NOTE:'理解提示'}; const label = note[2].trim() || labels[note[1]];
    return '<aside class="callout callout-' + note[1].toLowerCase() + '" aria-label="' + esc(label) + '"><div class="callout-title">' + esc(label) + '</div><div class="callout-body">' + html.replace(note[0], '<p>') + '</div></aside>';
  },
  heading({tokens, depth, text}) {
    const title = this.parser.parseInline(tokens); const plain = stripInline(text || title);
    const id = context.kind === 'chapter'
      ? chapterHeadings.get(context.file)[context.headingNumber++].id
      : slug(text || plain, context.usedHeadings);
    return '<h' + depth + ' id="' + esc(id) + '">' + title + '<a class="heading-anchor" href="' + (context.kind === 'chapter' ? '?chapter=' + context.chapter + '#' + encodeURIComponent(id) : '#' + encodeURIComponent(id)) + '" aria-label="链接到' + esc(plain) + '">#</a></h' + depth + '>\n';
  },
  link({href, title, tokens}) { const label = this.parser.parseInline(tokens); const target = rewriteLink(href, context.file, context.output, context.kind); return target === null ? label : '<a href="' + esc(target) + '"' + (title ? ' title="' + esc(title) + '"' : '') + '>' + label + '</a>'; },
  image({href, title, text}) { const target = rewriteLink(href, context.file, context.output, context.kind); if (target === null) return '<span class="unavailable-asset">' + esc(text || href) + '</span>'; return '<img src="' + esc(target) + '" alt="' + esc(text || '') + '"' + (title ? ' title="' + esc(title) + '"' : '') + '>'; },
  code({text, lang}) {
    const language = (lang || 'text').split(/\s/)[0]; const selection = /\{([\d,\s-]+)\}/.exec(lang || ''); const focus = new Set();
    if (selection) for (const part of selection[1].split(',')) { const [a,b] = part.trim().split('-').map(Number); for (let n=a; n <= (b || a); n++) if (n > 0 && n <= text.split('\n').length) focus.add(n); }
    let highlighted = hljs.getLanguage(language) ? hljs.highlight(text, {language, ignoreIllegals:true}).value : esc(text); const open = [];
    highlighted = highlighted.split('\n').map((line, i) => { const prefix = open.join(''); for (const tag of line.matchAll(/<span\b[^>]*>|<\/span>/g)) tag[0].startsWith('</') ? open.pop() : open.push(tag[0]); return '<span class="code-line' + (focus.has(i+1) ? ' is-focus' : '') + '" data-line="' + (i+1) + '">' + prefix + line + '</span>'.repeat(open.length) + '</span>'; }).join('\n');
    return '<div class="code-panel"><div class="code-caption"><span>' + esc(language.toUpperCase()) + '</span><button class="copy-code" type="button" aria-label="复制本段代码">复制</button></div><pre tabindex="0"><code class="hljs language-' + esc(language) + '">' + highlighted + '</code></pre></div>\n';
  }
}, extensions: [
  {name:'displayMath', level:'block', start:src => src.indexOf('$$'), tokenizer(src) { const m = /^\$\$[ \t]*\n([\s\S]+?)\n[ \t]*\$\$[ \t]*(?:\n|$)/.exec(src); if (m) return {type:'displayMath', raw:m[0], text:m[1]}; }, renderer:token => '<div class="equation">' + math(token.text, true) + '</div>\n'},
  {name:'inlineMath', level:'inline', start:src => src.indexOf('$'), tokenizer(src) { const m = /^\$([^\s$](?:\\.|[^$\n])*?)\$(?!\d)/.exec(src); if (m) return {type:'inlineMath', raw:m[0], text:m[1]}; }, renderer:token => math(token.text, false)}
]});
function checkEmbeddedSources(text, dir) {
  let fence = null;
  let mathOpen = false;
  for (const [lineIndex, line] of text.split('\n').entries()) {
    const marker = line.match(/^\s*(\x60{3,}|~{3,})/);
    if (marker) {
      if (!fence) fence = marker[1][0];
      else if (marker[1][0] === fence) fence = null;
      continue;
    }
    if (fence) continue;
    if (/^\s*\$\$(?!\s*$)/.test(line)) throw new Error('Display-math $$ must be on a line by itself: ' + dir + ':' + (lineIndex + 1));
    if (/^\s*\$\s*$/.test(line)) throw new Error('Orphan single-dollar display delimiter: ' + dir);
    if (/^\s*\$\$\s*$/.test(line)) mathOpen = !mathOpen;
  }
  if (mathOpen) throw new Error('Unclosed display math: ' + dir);
  const refs = [...text.matchAll(/<!-- source-check:\s*([^\n]+?)\s*-->\s*\n(~~~|`{3})[^\n]*\n([\s\S]*?)\n\2/g)];
  for (const m of refs) {
    if (/^def\s.+:\s*$/.test(m[3].trim()) && !m[3].trim().includes('\n')) {
      throw new Error('Embedded Python excerpt contains only a function signature: ' + m[1]);
    }
  }
  const parsedSourceStarts = new Set(refs.map(match => match.index));
  const misplacedSource = [...text.matchAll(/<!-- source-check:([^\n]*?)-->/g)]
    .find(match => !parsedSourceStarts.has(match.index));
  if (misplacedSource) {
    const line = text.slice(0, misplacedSource.index).split('\n').length;
    throw new Error('Source marker must immediately precede its code fence (inside details): '
      + dir + ':' + line + ': ' + misplacedSource[1].trim());
  }
  if ((text.match(/<!-- source-check:/g) || []).length !== refs.length) {
    throw new Error('Malformed source-check comment: ' + dir);
  }
  for (const m of refs) { const file = /^(solutions|reference)\//.test(m[1].trim()) ? path.resolve(repo, m[1].trim()) : path.resolve(path.dirname(paperByFile.get(chapterFile(dir))?.override ? path.resolve(repo,paperByFile.get(chapterFile(dir)).override) : chapterFile(dir)), m[1].trim()); const rel = path.relative(repo, file); if (rel.startsWith('..') || !/\.(cu|c|cpp|h|hpp|cuh|py|ptx|inc|sh)$/.test(file)) throw new Error('Invalid embedded source path: ' + m[1]); if (!sourceText(file).includes(m[3].trim())) throw new Error('Embedded source differs from original: ' + m[1]); checkedSourceBlocks++; }
}
const firstFile = path.join(gpuRoot, chapters[0][0], 'README.md'); const example = sourceText(path.join(gpuRoot, chapters[0][0], 'examples/vector_add_walkthrough.cu')).trim(); const firstRaw = sourceText(firstFile); const embedded = firstRaw.match(/<!-- BEGIN vector_add_walkthrough\.cu -->\n~~~cpp\n([\s\S]*?)\n~~~\n<!-- END vector_add_walkthrough\.cu -->/);
if (!embedded || embedded[1].trim() !== example) throw new Error('CUDA example and chapter differ');
const searchData = []; const sidebarParts = [[], [], [], []]; const paperPayload = {}; const paperGroups = new Map();
function render(text, info, kind, chapter) {
  context = {file:info.file, output:kind === 'chapter' ? 'index.html' : info.output, kind, chapter, headingNumber:0, usedHeadings:new Set()};
  return md.parse(text).replace(/\b(src|href)="([^"]+)"/g, (whole, attr, href) => {
    const target = rewriteLink(href, context.file, context.output, context.kind);
    return target === null ? whole : attr + '="' + esc(target) + '"';
  });
}
const articles = chapters.map(([dir, title, intro, guide], i) => {
  if (i === 5) return ''; // Retired overview.
  const paper = paperById.get(i+1);
  const file = fs.realpathSync(chapterFile(dir)); const info = pages.get(file); const raw = info.text.replace(/^# .*\n/, '').replace(/\n## 章节导航[\s\S]*$/, '').trim(); checkEmbeddedSources(raw, dir); const html = render(raw, info, 'chapter', i+1);
  const outline = [...html.matchAll(/<h2 id="([^"]+)">([\s\S]*?)<\/h2>/g)].map(m => '<li><a href="?chapter=' + (i+1) + '#' + m[1] + '">' + stripInline(m[2]).replace(/#$/, '') + '</a></li>').join('');
  const side = [...html.matchAll(/<h([23]) id="([^"]+)">([\s\S]*?)<\/h\1>/g)].map(m => '<a class="nav-subchapter level-' + m[1] + '" href="?chapter=' + (i+1) + '#' + m[2] + '" data-chapter-link="' + (i+1) + '" data-section-link="' + m[2] + '">' + esc(stripInline(m[3]).replace(/#$/, '')) + '</a>').join('');
  const part = i < 5 ? 0 : i < 15 ? 1 : i < 19 ? 2 : 3;
  const number = paper ? '' : String(i < 5 ? i+1 : i < 15 ? i-5 : i-14).padStart(2,'0');
  const leaf = '<details class="nav-chapter" data-nav-chapter="' + (i+1) + '"><summary><a href="?chapter=' + (i+1) + '" data-chapter-link="' + (i+1) + '">' + (number ? '<span>'+number+'</span>' : '') + '<strong>' + esc(title) + '</strong></a></summary><nav aria-label="' + esc(title) + '子课程">' + side + '</nav></details>';
  if (paper) { if (!paperGroups.has(paper.category)) paperGroups.set(paper.category,[]); paperGroups.get(paper.category).push(leaf); }
  else sidebarParts[part].push(leaf);
  searchData.push({chapter:i+1, title, text:raw.replace(/<!--[\s\S]*?-->/g, '')});
  const peers = paper ? papers.filter(p => p.category === paper.category) : [];
  const peerIndex = paper ? peers.findIndex(p => p.id === paper.id) : -1;
  const previousIndex = paper ? (peers[peerIndex-1]?.id ? peers[peerIndex-1].id-1 : -1) : i === 6 ? 4 : i - 1;
  const nextIndex = paper ? (peers[peerIndex+1]?.id ? peers[peerIndex+1].id-1 : chapters.length) : i === 18 ? chapters.length : i === 4 ? 6 : i + 1;
  const previous = previousIndex >= 0 ? '<a href="?chapter=' + (previousIndex+1) + '" data-chapter-link="' + (previousIndex+1) + '"><small>上一章</small><span>' + esc(chapters[previousIndex][1]) + '</span></a>' : '<span></span>';
  const next = nextIndex < chapters.length ? '<a class="next" href="?chapter=' + (nextIndex+1) + '" data-chapter-link="' + (nextIndex+1) + '"><small>下一章</small><span>' + esc(chapters[nextIndex][1]) + '</span></a>' : '<span></span>';
  const breadcrumb = paper ? '论文与算法 <span>/</span> ' + esc(paper.category) : i < 5 ? 'GPU 架构与性能基础 <span>/</span> 第 ' + (i+1) + ' 章' : i < 15 ? '完整 GPU 算子体系 <span>/</span> ' + '第 ' + (i - 5) + ' 章' : '模型与系统应用 <span>/</span> 第 ' + (i-14) + ' 章';
  if (paper) paperPayload[i+1] = html;
  const content = paper ? [...html.matchAll(/<h[1-6] id="([^"]+)"/g)].map(m => '<span hidden id="'+m[1]+'"></span>').join('') : html;
  return '<article class="chapter" id="chapter-' + (i+1) + '" data-chapter="' + (i+1) + '" data-part="' + (paper ? 'papers' : i < 5 ? 'gpu' : i < 15 ? 'operators' : 'systems') + '" data-title="' + esc(title) + '"' + (i ? ' hidden' : '') + '><header class="article-header"><div class="breadcrumb">' + breadcrumb + '</div><h1>' + esc(title) + '</h1>' + (intro ? '<p>' + esc(intro) + '</p>' : '') + '</header><nav class="chapter-outline" aria-label="本章目录"><p>目录</p><ol>' + outline + '</ol></nav><div class="markdown-body"' + (paper ? ' data-paper-pending="true"' : '') + '>' + content + '</div><nav class="chapter-switch" aria-label="章节翻页">' + previous + next + '</nav></article>';
}).join('\n');
sidebarParts[3] = [...paperGroups].map(([name,leaves]) => '<details class="paper-category"><summary>'+esc(name)+'</summary>'+leaves.join('')+'</details>');
const sidebar = '<section class="sidebar-part" data-sidebar-part="gpu"><div class="nav-group">GPU 架构与性能基础</div>' + sidebarParts[0].join('') + '</section><section class="sidebar-part" data-sidebar-part="operators"><div class="nav-group">完整 GPU 算子体系</div>' + sidebarParts[1].join('') + '</section><section class="sidebar-part" data-sidebar-part="systems"><div class="nav-group">模型与系统应用</div>' + sidebarParts[2].join('') + '</section><section class="sidebar-part" data-sidebar-part="papers"><div class="nav-group">论文与算法</div>' + sidebarParts[3].join('') + '</section>';
const index = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="结合 CUDA 手册、工程代码和真实实验的 GPU 硬件与性能课程"><title>GPU 与 CUDA · AIINFFRA</title><script src="theme.js?v=' + themeRevision + '"></script><link rel="stylesheet" href="vendor/katex/katex.min.css"><link rel="stylesheet" href="styles.css?v=' + styleRevision + '"></head><body><a class="skip-link" href="#main">跳到正文</a><div class="reading-progress" aria-hidden="true"><span id="progress-bar"></span></div><header class="topbar"><a class="site-name" href="?chapter=1" data-chapter-link="1">AIINFFRA</a><nav aria-label="课程导航"><a href="?chapter=1" data-chapter-link="1" data-part-link="gpu">GPU 与 CUDA</a><a href="?chapter=7" data-chapter-link="7" data-part-link="operators">算子与优化</a><a href="?chapter=16" data-chapter-link="16" data-part-link="systems">模型与系统</a><a href="?chapter=25" data-chapter-link="25" data-part-link="papers">论文与算法</a><a href="https://docs.nvidia.com/cuda/cuda-programming-guide/index.html" target="_blank" rel="noreferrer">官方手册 ↗</a></nav><div class="topbar-actions"><button id="theme-toggle" class="theme-toggle" type="button" aria-label="切换到浅色模式">浅色</button><button id="menu-button" class="menu-button" aria-label="打开章节导航" aria-controls="sidebar" aria-expanded="false">目录</button></div></header><aside id="sidebar" class="sidebar"><div class="section-label">课程目录</div><label class="search"><span>检索</span><input id="search" type="search" placeholder="搜索课程内容" aria-label="搜索课程内容"></label><nav id="chapter-nav" aria-label="章节">' + sidebar + '</nav><div id="search-results" class="search-results" aria-live="polite" hidden></div><div class="sidebar-resources"><a href="../../../../downloads/cuda-programming-guide.pdf">本地 CUDA 手册 ↗</a><a href="https://triton-lang.org/main/">Triton 文档 ↗</a></div></aside><main id="main" class="content" tabindex="-1">' + articles + '</main><aside class="toc"><div class="section-label">本页目录</div><nav id="page-toc"></nav></aside><script id="course-search-data" type="application/json">' + JSON.stringify(searchData).replace(/</g, '\\u003c') + '</script><script id="paper-content-data" type="application/json">' + JSON.stringify(paperPayload).replace(/</g, '\\u003c') + '</script><script src="app.js?v=' + appRevision + '"></script></body></html>';
const visible = index.replace(/<script[\s\S]*?<\/script>/gi, '').replace(/<[^>]*>/g, ''); const management = ['WIP','Agent','本轮生成','本轮修改','本轮更新','旧稿','待讨论','等待验收','正文状态']; const leaked = management.filter(word => visible.includes(word)); if (leaked.length) throw new Error('Management text in tutorial: ' + leaked.join(', ')); if (!index.includes('<code>__global__</code>') || index.includes('<strong>global</strong>')) throw new Error('Broken inline CUDA qualifier');
fs.rmSync(docsRoot, {recursive:true, force:true}); fs.mkdirSync(docsRoot, {recursive:true});
 for (const info of pages.values()) { const output = path.join(site, info.output); fs.mkdirSync(path.dirname(output), {recursive:true}); const body = render(info.text, info, 'doc'); const css = relativeUrl(info.output, 'styles.css') + '?v=' + styleRevision; const theme = relativeUrl(info.output, 'theme.js') + '?v=' + themeRevision; const katexCss = relativeUrl(info.output, 'vendor/katex/katex.min.css'); const reader = relativeUrl(info.output, 'reader.js'); const back = relativeUrl(info.output, 'index.html'); const toc = info.headings.filter(h => h.level === 2 || h.level === 3).map(h => '<li><a href="#' + encodeURIComponent(h.id) + '">' + esc(h.text) + '</a></li>').join(''); fs.writeFileSync(output, '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + esc(info.title) + ' · AIINFFRA 附属文档</title><script src="' + theme + '"></script><link rel="stylesheet" href="' + katexCss + '"><link rel="stylesheet" href="' + css + '"></head><body class="doc-page"><header class="topbar"><a class="site-name" href="' + back + '?chapter=1">AIINFFRA</a><nav><a href="' + back + '?chapter=1">返回课程</a></nav><button id="theme-toggle" class="theme-toggle" type="button" aria-label="切换到浅色模式">浅色</button></header><main class="content doc-content"><div class="breadcrumb">附属阅读文档 · ' + esc(repoRelative(info.file)) + '</div><nav class="chapter-outline" aria-label="附属文档目录"><p>目录</p><ol>' + toc + '</ol></nav><div class="markdown-body">' + body + '</div></main><script src="' + reader + '"></script></body></html>'); }
// Keep bookmarked paper URLs in the same course reader.
const paperRedirects = papers.map(p => [p.source,p.id]);
paperRedirects.push(['notes/algorithms/README.md',25],['papers/attention/flash-attention.md',22]);
for (const [source,id] of paperRedirects) {
  const output='docs/'+source.replace(/\.md$/i,'.html');
  const target=relativeUrl(output,'index.html')+'?chapter='+id;
  const file=path.resolve(repo,source);
  const mapped=chapterHeadings.get(chapterFile(paperById.get(id).source)) || [];
  const usedSlugs=new Set();
  const anchors=Object.fromEntries(mapped.flatMap(h=>[[slug(h.text,usedSlugs),h.id],[h.id,h.id]]));
  const oldHeads=pages.get(file)?.headings || [];
  const placeholders=oldHeads.map(h=>'<span hidden id="'+esc(h.id)+'"></span>').join('');
  const redirectScript='const m='+JSON.stringify(anchors)+';let h;try{h=decodeURIComponent(location.hash.slice(1));}catch{}location.replace('+JSON.stringify(target)+'+(m[h]?"#"+m[h]:""));';
  const out=path.join(site,output);
  fs.mkdirSync(path.dirname(out),{recursive:true});
  fs.writeFileSync(out,'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>论文与算法 · AIINFFRA</title></head><body>'+placeholders+'<p><a href="'+esc(target)+'">在课程中阅读</a></p><script>'+redirectScript+'</script></body></html>');
}
for (const [file, output] of assets) { if (/^downloads[\\/].+\.pdf$/i.test(repoRelative(file))) continue; const target = path.join(site, output); fs.mkdirSync(path.dirname(target), {recursive:true}); fs.copyFileSync(file, target); }
const vendor = path.join(site, 'vendor', 'katex'); fs.mkdirSync(vendor, {recursive:true}); const katexRoot = path.dirname(require.resolve('katex/package.json')); fs.copyFileSync(path.join(katexRoot, 'dist/katex.min.css'), path.join(vendor, 'katex.min.css')); fs.cpSync(path.join(katexRoot, 'dist/fonts'), path.join(vendor, 'fonts'), {recursive:true}); fs.copyFileSync(path.join(katexRoot, 'LICENSE'), path.join(vendor, 'LICENSE')); fs.writeFileSync(path.join(site, 'index.html'), index);
fs.writeFileSync(anchorRegistryFile,JSON.stringify(anchorRegistry,null,2)+'\n');
console.log('Built ' + chapterFiles.size + ' chapters; generated ' + pages.size + ' readable Markdown pages and ' + assets.size + ' local assets; rendered ' + mathCount + ' formulas; checked ' + checkedSourceBlocks + ' original-source excerpts.');
