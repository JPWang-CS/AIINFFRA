const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const expectedChapters = 39;
const quantChapter = html.match(/<article\b[^>]*id="chapter-13"[\s\S]*?<\/article>/)?.[0];
assert(quantChapter,'Quantization chapter missing');
assert(quantChapter.includes('id="chapter-13-section-10">4. 校准、缩放搜索与误差补偿'));
assert(quantChapter.includes('id="chapter-13-section-11">SmoothQuant：把激活幅度迁移到权重'));
assert(quantChapter.includes('id="chapter-13-section-12">GPTQ：把一次舍入的影响分配给剩余权重'));
for (const token of ['gptq_quantize_blocked','awq_style_search','packed_int4_matmul',
  'plain_kernel','quantization_lab.py --demo','packed_int4_gpu.py --benchmark',
  '0.00283880','0.00697122']) {
  assert(quantChapter.includes(token),'Missing integrated quantization content: '+token);
}
const styles = fs.readFileSync(path.join(__dirname,'styles.css'),'utf8');
assert(styles.includes('svg:not(.katex svg)'), 'Image sizing must exclude KaTeX stretchy SVGs');
assert(/styles\.css\?v=[a-f0-9]{12}/.test(html) && /app\.js\?v=[a-f0-9]{12}/.test(html),
  'Changed reader assets must bypass stale browser caches');
assert.equal((html.match(/<article class="chapter"/g) || []).length, expectedChapters);
assert(/id="chapter-4-section-13">9\. Stream-ordered allocation/.test(html),
  'Previously delivered stream-allocation bookmark must keep its meaning');
for (const [chapter,count] of [[4,28],[9,35]]) {
  for (let section=0; section<count; section++) {
    assert(html.includes('id="chapter-'+chapter+'-section-'+section+'"'),
      'Previously published section anchor disappeared: '+chapter+'/'+section);
  }
}
assert(!html.includes('id="chapter-6"') && !html.includes('data-nav-chapter="6"'), 'Retired overview must not be published');
assert(!html.includes('<span>序</span>'), 'No operator preface');
assert(html.includes('class="callout callout-important"'));
assert(html.includes('class="callout callout-warning"'));
assert(html.includes('class="code-line is-focus"'));
assert(!html.replace(/<script[\s\S]*?<\/script>/g, '').includes('[!IMPORTANT]'));
assert(html.includes('vendor/katex/katex.min.css'));
assert(html.includes('class="katex"'));
assert(!html.includes('<strong>global</strong>'));
const body = html.replace(/<script[\s\S]*?<\/script>/g, '');
for (const match of body.matchAll(/<a\b[^>]*href="([^"]+)"/g)) {
  assert(!/\.(?:py|cu|cuh|cpp|cc|c|h|hpp|sh)(?:[?#]|$)/i.test(match[1]),
    'Show code inline instead of linking to source files: ' + match[1]);
  assert(!/\.pdf#page=/i.test(match[1]), 'Keep exact PDF locations in the maintenance index');
}
for (const word of ['正文状态', '等待验收', 'WIP', 'Agent', '旧稿', '待讨论', '本轮生成', '本轮修改', '本轮更新']) {
  assert(!body.replace(/<[^>]*>/g, '').includes(word), 'Management text: ' + word);
}
const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
assert.equal(new Set(ids).size, ids.length, 'Duplicate HTML ids');
const script = html.match(/<script id="course-search-data" type="application\/json">([\s\S]*?)<\/script>/);
const searchData = JSON.parse(script[1]);
assert.equal(searchData.length, expectedChapters);
const decode = text => text.replace(/<[^>]*>/g, '').replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&#39;|&#x27;/g, "'").replace(/&amp;/g, '&');
const codeBlocks = [...html.matchAll(/<code class="hljs language-cpp">([\s\S]*?)<\/code>/g)].map(match => decode(match[1]).trim());
const originalExample = fs.readFileSync(path.resolve(__dirname, '../01-gpu-hardware-map-and-generations/examples/vector_add_walkthrough.cu'), 'utf8').replace(/\r\n/g, '\n').trim();
assert(codeBlocks.includes(originalExample), 'Rendered line markup changed the copyable CUDA source');
const naiveSource = fs.readFileSync(path.resolve(__dirname, '../../../../solutions/cuda/gemm/naive_float.cu'), 'utf8').replace(/\r\n/g, '\n');
const originalNaiveKernel = naiveSource.slice(naiveSource.indexOf('__global__'), naiveSource.indexOf('extern "C"')).trim();
assert(codeBlocks.some(code => code.includes(originalNaiveKernel)), 'User GEMM kernel must be readable inline, not just linked');
assert(codeBlocks.some(code => code.includes('As[threadIdx.x][threadIdx.y]') && code.includes('__syncthreads()')), 'Tiled GEMM loading and synchronization must be inline');
assert(body.includes('diagram-frame'), 'Missing engineering diagrams');
assert(body.includes('host-device.svg') && body.includes('gpu-resource-map.svg'), 'Hardware overview needs rendered architecture figures');
assert(!body.includes('主机一侧                                GPU 设备一侧'), 'Do not render the host/device diagram as a text code block');
assert.equal((body.match(/class="chapter-outline"/g) || []).length, expectedChapters, 'Each reading page needs an inline contents');
assert(body.includes('class="heading-anchor"'), 'Missing section permalinks');
assert.equal((body.match(/data-sidebar-part="gpu"/g) || []).length, 1);
assert.equal((body.match(/data-sidebar-part="operators"/g) || []).length, 1);
assert.equal((body.match(/class="nav-chapter"/g) || []).length, expectedChapters);
assert(body.includes('data-nav-chapter="1"') && body.includes('data-nav-chapter="7"') && body.includes('data-nav-chapter="8"') && body.includes('data-nav-chapter="9"') && body.includes('data-nav-chapter="10"') && body.includes('data-nav-chapter="11"'));
assert(body.includes('<span>02</span><strong>并行归约、Softmax 与归一化</strong>'), 'Operator chapter 2 must be numbered 02');
assert(body.includes('完整 GPU 算子体系 <span>/</span> 第 2 章'), 'Operator chapter 2 breadcrumb must be correct');
assert(body.includes('<span>03</span><strong>GEMM：从分块实现到性能优化</strong>'), 'Operator chapter 3 must be numbered 03');
assert(body.includes('完整 GPU 算子体系 <span>/</span> 第 3 章'), 'Operator chapter 3 breadcrumb must be correct');
assert(body.includes('412316860416') && body.includes('131072B'), 'GEMM evidence ledger must be rendered');
assert(body.includes('<span>04</span><strong>Activation 与 Fusion：从 SwiGLU 点算子到 MLP 的物化边界</strong>'), 'Operator chapter 4 must be numbered 04');
assert(body.includes('完整 GPU 算子体系 <span>/</span> 第 4 章'), 'Operator chapter 4 breadcrumb must be correct');
const ledgerText = decode(body).replace(/,/g, '');
assert(ledgerText.includes('450887680') && ledgerText.includes('270532608') && ledgerText.includes('180355072'), 'Activation bytes ledger must be rendered');
const fusionSource = fs.readFileSync(path.resolve(__dirname, '../../operators/04-activation-and-fusion/README.md'), 'utf8');
assert(!/(?<!\\)\bqquad\b/.test(fusionSource) && !fusionSource.includes('$odot$'), 'Activation formula command lost its backslash');
assert(body.includes('reference/cuda/include/activations.cuh') && body.includes('aten.silu.out'), 'Activation reference and split baseline must be rendered');
assert(body.includes('data-section-link='), 'Sidebar is missing rendered subsection anchors');
assert(body.includes('<span>05</span><strong>Prefill Attention：从 Softmax 加权和到在线分块</strong>'), 'Operator chapter 5 must be numbered 05');
assert(body.includes('完整 GPU 算子体系 <span>/</span> 第 5 章'), 'Operator chapter 5 breadcrumb must be correct');
assert(body.includes('Softmax Attention') && body.includes('BLOCK_KV'), 'Prefill shape and tile contract must be rendered');
assert(body.includes('validate_prefill.py --benchmark'), 'Prefill benchmark command must be visible in generated chapter 11');
assert(body.includes('https://leetgpu.com/challenges') && body.includes('6_softmax_attention'), 'Prefill LeetGPU platform and public-source links must be rendered');
// Verify hand-calculation examples independently of CUDA execution.
for (const [id, text] of [[16,'def block_ledger'],[17,'def forward'],[18,'def allocate_tokens'],[19,'def tensor_parallel_mlp']]) {
  const article = html.match(new RegExp('<article\\b[^>]*id="chapter-'+id+'"[\\s\\S]*?<\\/article>'));
  assert(article && article[0].includes('data-part="systems"'), 'Missing systems chapter '+id);
  assert(decode(article[0]).includes(text), 'Missing inline implementation '+id);
}
assert(body.includes('data-part-link="systems"') && body.includes('data-sidebar-part="systems"'));
assert(!body.includes('回到开篇'), 'Final chapter should not force a reading loop');
const decodeArticle = html.match(/<article\b[^>]*id="chapter-12"[\s\S]*?<\/article>/);
assert(decodeArticle, 'Decode article must be published');
for (const required of ['_paged_decode_kernel', 'validate_paged_decode.py --benchmark',
  'https://leetgpu.com/challenges', '890', '70272', 'Copy-on-Write']) {
  assert(decodeArticle[0].includes(required), 'Decode body missing: ' + required);
}
assert(!decodeArticle[0].includes('最后写 offset 0'), 'COW must preserve the logical page offset');
assert.equal((512 * 4 / 8 + 512 / 16 + 128 * 4 / 8 + 128 / 32) * 2.5, 890);
assert.equal(61 * (512 + 64) * 2, 70272);
assert(body.includes('cpu_metrics_case.py') && body.includes('cpu_mhc_case.py'), 'PDF cases must be integrated inline');
const gemmArticle = html.match(/<article\b[^>]*id="chapter-9"[\s\S]*?<\/article>/)[0];
for (const required of ['mma-output-layout.svg','acc[i][j] += ar[i] * br[j]',
  'def mma16816_f16_coords', 'def pipeline_protocol',
  'C[out_m * K + out_k] = Cs[x]', 'wmma_padded.cu -o wmma_padded']) {
  assert(decode(gemmArticle).includes(required) || gemmArticle.includes(required), 'Missing GEMM depth: '+required);
}
const syncArticle = html.match(/<article\b[^>]*id="chapter-4"[\s\S]*?<\/article>/)[0];
for (const required of ['pipeline-lifecycle.svg','__pipeline_wait_prior(STAGES - 1)',
  'const float next = buffer[t % STAGES][next_lane]', 'asm volatile("trap;")']) {
  assert(decode(syncArticle).includes(required) || syncArticle.includes(required), 'Missing pipeline contract: '+required);
}
assert(!decode(gemmArticle).includes('值得追问的现象'), 'Use concrete source-linked explanations, not narrative filler');
const repositoryRoot = path.resolve(__dirname, '../../../..');
const sourceRegistry = JSON.parse(fs.readFileSync(path.join(repositoryRoot,
  '.codex/course-source-index.json'), 'utf8'));
const gemmSources = sourceRegistry.continuous_reading_source_archive[
  'roadmap/curriculum/operators/03-gemm/README.md'];
for (const source of ['notes/triton/matmul-performance-analysis.md',
  'notes/triton/logs/2026-08-29-matmul-k256-s3-nsys.txt',
  'notes/triton/logs/2026-08-30-matmul-k256-s2-nsys.txt']) {
  assert(gemmSources.includes(source), 'GEMM measurement provenance missing from maintenance registry: ' + source);
  assert(fs.existsSync(path.join(repositoryRoot, source)), 'GEMM original record missing: ' + source);
}
assert(decodeArticle[0].includes('cache_accounting.py') && decodeArticle[0].includes('cache_sources'),
  'PDF mechanisms must include the executable accounting/source model, not only PDF links');
for (const [id, snippets] of [
  [13,['def pack_int4','def block_scale_kernel','SmoothQuant','GPTQ','validate_baselines.py quant']],
  [14,['def group_routes','def gate_kernel','validate_baselines.py moe']],
  [15,['def nucleus','def argmax_partial','def inverse_cdf','validate_baselines.py sampling']]
]) {
  const article = html.match(new RegExp('<article\\b[^>]*id="chapter-'+id+'"[\\s\\S]*?<\\/article>'));
  assert(article, 'Missing chapter '+id);
  const readable = decode(article[0]);
  for (const part of snippets) assert(readable.includes(part), 'Missing inline lesson content '+id+': '+part);
  assert(article[0].includes('https://leetgpu.com/challenges'), 'Missing practice entry '+id);
  if (id === 14) assert(article[0].includes('moe-flow.svg'), 'MoE dataflow diagram missing');
}
const sectors = stride => new Set(Array.from({length:32}, (_,lane) => Math.floor(4*stride*lane/32))).size;
assert.deepEqual([1,2,32].map(sectors), [4,8,32]);
const banks = width => new Set(Array.from({length:32},(_,lane)=>(lane*width)%32)).size;
assert.deepEqual([32,33].map(banks), [1,32]);
for (const n of [1,3,4,5,257,1027]) {
  const threads = Math.ceil(Math.ceil(n / 4) / 256) * 256;
  for (const aligned of [true, false]) {
    const writes = Array(n).fill(0);
    for (let tid = 0; tid < threads; tid++) {
      if (aligned) {
        const vectors = Math.floor(n / 4);
        for (let v = tid; v < vectors; v += threads) for (let j=0;j<4;j++) writes[v*4+j]++;
        for (let i = vectors*4+tid; i<n; i+=threads) writes[i]++;
      } else {
        for (let i=tid;i<n;i+=threads) writes[i]++;
      }
    }
    assert(writes.every(count => count === 1), 'Copy body/tail coverage mismatch');
  }
}
for (const [rows,cols] of [[1,1],[3,5],[32,32],[37,65]]) {
  const output = Array(rows*cols).fill(-1);
  for(let by=0;by<Math.ceil(rows/32);by++) for(let bx=0;bx<Math.ceil(cols/32);bx++){
    const shared=Array.from({length:32},()=>Array(32).fill(0));
    for(let y=0;y<32;y++) for(let x=0;x<32;x++){
      const r=by*32+y,c=bx*32+x;
      if(r<rows&&c<cols) shared[y][x]=r*cols+c;
    }
    for(let y=0;y<32;y++) for(let x=0;x<32;x++){
      const r=bx*32+y,c=by*32+x;
      if(r<cols&&c<rows) output[r*rows+c]=shared[x][y];
    }
  }
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++)assert.equal(output[c*rows+r],r*cols+c);
}
console.log('PASS: ' + expectedChapters + ' pages, emphasis, highlighted code, formulas, links, unique anchors, search data and address arithmetic.');
console.log('These are static/content checks, not CUDA compilation, GPU correctness or browser visual tests.');
// Keep the full audit as the default; use the quick pass while editing content.
if (!process.argv.includes('--content-only')) require('./link-audit.cjs');
