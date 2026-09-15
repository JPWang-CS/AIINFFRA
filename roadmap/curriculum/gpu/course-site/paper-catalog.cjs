const fs = require('node:fs');
const path = require('node:path');
const introductions = require('./paper-introductions.cjs');
const papers = [
  [20,'数学与并行算法','Online Softmax','notes/algorithms/online-softmax.md'],
  [21,'数学与并行算法','Parallel Reduce','notes/algorithms/parallel-reduce.md'],
  [22,'Attention','FlashAttention：分块与在线归约','notes/algorithms/flash-attention-mechanism.md'],
  [23,'Attention','FlashAttention-2','notes/algorithms/flash-attention-2.md'],
  [24,'Attention','MQA 与 GQA','papers/attention/gqa.md'],
  [25,'Attention','MLA：低维缓存与权重吸收','notes/algorithms/mla-deepseek.md'],
  [26,'Attention','DSA：索引与稀疏选择','notes/algorithms/dsa-sparse-attention.md','roadmap/curriculum/papers/dsa.md'],
  [27,'Attention','Gated DeltaNet 与线性注意力','notes/algorithms/gdn-linear-attention.md'],
  [28,'Attention','FlashAttention 与可编程注意力专题','notes/algorithms/fa4-flexattention.md'],
  [29,'Attention','低精度与结构复用 Attention 专题','notes/algorithms/attention-2026-sage3-kascade.md'],
  [30,'模型架构','DeepSeek：参数、KV 与计算量','notes/algorithms/deepseek-v32-handcalc.md'],
  [31,'模型架构','DeepSeek-V4 架构专题','notes/algorithms/deepseek-v4.md'],
  [32,'模型架构','MoE 路由与专家','notes/algorithms/moe-inference.md'],
  [33,'量化与低精度','INT8 与 FP8','notes/algorithms/quantization-int8-fp8.md'],
  [34,'推理系统','Speculative Decoding','notes/algorithms/speculative-decoding.md'],
  [35,'推理系统','Prefill / Decode 分离','notes/algorithms/pd-disaggregation.md'],
  [36,'推理系统','PagedAttention','papers/inference/paged-attention.md'],
  [37,'训练与优化','Adam、AdamW 与 Muon','notes/algorithms/optimizers-adam.md'],
  [38,'训练与优化','ZeRO 与模型状态分片','papers/training/zero-paper.md'],
  [39,'模型架构','DeepSeek-V4.1-Flash：缓存、索引与训练推理协同','roadmap/curriculum/papers/deepseek-v41.md'],
  [40,'训练与优化','Scaling Law、强化学习与长任务推理','roadmap/curriculum/papers/inference-workloads.md'],
].map(([id,category,title,source,override])=>({id,category,title,source,override}));

function prepare(text, paper, repo) {
  if (paper.override) text=fs.readFileSync(path.join(repo,paper.override),'utf8');
  text=text.replace(/\r\n/g,'\n');
  const bibliography = text.match(/^\*\*(?:Authors|Venue)\*\*:.*$/gm) || [];
  if (introductions[paper.id]) {
    text=text.replace(/^# [^\n]+\n[\s\S]*?(?=^## )/m,
      ()=>'# '+paper.title+'\n\n'+introductions[paper.id]+'\n\n'
        +(bibliography.length?bibliography.join('\n')+'\n\n':''));
  }
  text=text.replace(/自检题和当前状态/g,'自检题')
    .replace(/^当前状态：\s*\n\s*(~~~|\x60{3})[\s\S]*?\n\1\s*\n/gm,'')
    .replace(/^后续实现顺序：\s*\n\s*(~~~|\x60{3})[\s\S]*?\n\1\s*\n/gm,'');
  text=text.split(/(?=^## )/m).filter(part=>!/^## (?:\d+\.\s*)?(?:与我何干|为什么这篇论文重要|学习路线|当前进度|实践计划|后续计划|下一步：从看懂到自己写)/.test(part)).join('');
  let fence=null;
  const known=new Set(papers.map(p=>path.resolve(repo,p.source)));
  text=text.split('\n').map(line=>{
    const marker=line.match(/^\s*(\x60{3,}|~{3,})/);
    if(marker){if(!fence)fence=marker[1][0];else if(fence===marker[1][0])fence=null;return line;}
    if(fence)return line;
    if(/^\s*>?\s*(?:\*\*)?(?:状态|优先级|实现|挂靠|前置|代码映射|唯一入口|代码状态)(?:\*\*)?\s*[:：]/.test(line))return '';
    if(/(?:Agent.*(?:整合|整理|草稿)|\bWIP\b|生产验证|给\s*A\d.*铺路)/.test(line))return '';
    line=line.replace(/\bAgent\b(?!'s Last Exam\b)/g,'智能体').replace(/[🚧✅]/gu,'');
    return line.replace(/\[([^\]\n]+)\]\(([^)\n]+)\)/g,(all,label,target)=>{
      if(/\.(?:py|cu|cuh|cpp|cc|c|h|hpp|sh)(?:[?#]|$)/i.test(target))return label;
      if(/\.pdf#page=/.test(target))return '['+label.replace(/(?:PDF|打印|印刷|第)?\s*\d+(?:[–-]\d+)?\s*页/g,'').trim()+']('+target.split('#')[0]+')';
      if(!/^(?:https?:|#)/.test(target)&&/\.md(?:#|$)/.test(target)) {
        const destination=path.resolve(path.dirname(path.join(repo,paper.source)),target.split('#')[0]);
        if(!known.has(destination))return label.replace(/\x60/g,'');
      }
      return all;
    });
  }).join('\n');
  if(paper.id===36) text=text.replace(/^> \*\*一句话\*\*.*$/m,
    '> PagedAttention 用软件页表组织非连续 KV。它降低分配浪费并支持缓存复用，不改变普通 Attention 的数学；具体容量和速度收益依赖工作负载。');
  if(paper.id===38) text=text.replace(/^> \*\*一句话\*\*.*$/m,
    '> ZeRO 分阶段分片优化器状态、梯度和参数。各阶段的显存与通信不同，不能用一个固定倍数概括所有配置。')
    .replace(/^.*Meta 抄.*$/m,'').replace(/^.*后续所有大模型训练框架.*$/m,'');
  if(paper.id===20) text=text.replace(/^\*下一条建议学：.*$/m,'');
  if(paper.id===23) text=text.replace('七题能不看笔记讲清楚，才算真正掌握机制。','');
  if(paper.id===33) text=text.replace(/^\*{1,2}(?:理论线下一步|下一步)[：:][^\n]+$/gm,'');
  if(paper.id===30) {
    text=text.replace('，主线 A 第 3 步','')
      .replace(/## 7\. 做完之后[\s\S]*$/, '## 7. 核对三笔账\n\n- 权重容量使用总参数和存储精度；单步权重访问还受共享层、batch 命中的专家集合和缓存复用影响。\n- KV 按实际保存的 latent、位置分支和元数据计数，再乘有效 token 数。\n- FLOPs、容量和读写量分别统计；用算术强度与实际时间线判断瓶颈。\n')
      .replace(/\x60{3}text\ndecode 每 token：[\s\S]*?\x60{3}/,
        '一次 Decode 的矩阵工作可按激活参数粗估，但读取量要按本批次实际访问的共享层与专家权重计算。1.37 TB 是本例 BF16 总权重容量，不是每 token 必然读取的字节数。Prefill 还包含 Attention、数据搬运和调度；其瓶颈需要结合输入长度、batch 和硬件判断。')
      .replace('| Decode 每 token | ≈ 74 GFLOP + 读 1.37TB | memory-bound → 量化 + wideEP |',
        '| Decode 每 token | 线性层约 74 GFLOP；读取量按实际权重访问计算 | 结合 batch、专家覆盖与带宽分析 |')
      .replace('| 权重显存（BF16） | ≈ 1.37 TB | 必须多卡 + EP + FP8/FP4 |',
        '| 权重显存（BF16） | ≈ 1.37 TB | 先计算容量，再选择分片、精度与卸载方案 |')
      .replace('| Prefill FLOPs（4096 token） | ≈ 303 TFLOP | prefill 计算密集 → PD 分离 |',
        '| Prefill FLOPs（4096 token） | 线性层约 303 TFLOP | 结合 shape、batch 与 Attention 工作量分析 |')
      .replace('**一句话结论**：权重 1.37TB 决定"必须多卡 + EP + 量化"；量化到 FP8 直接少一半卡，这就是 FP8 在生产里是默认选项的原因。',
        '1.37 TB 是仅权重容量。精度、分片、卸载和设备容量共同决定放置方案，KV、激活与通信缓冲需要额外空间。');
  }
  if(paper.id===31) {
    text=text.replace(/^所以安排：.*$/m,'')
      .replace('## 13. 与 V3.2 的关系 & 学习挂载点','## 13. 与 V3.2 的机制差异')
      .replace('## 14. 验证入口（等学到这里再深挖）','## 14. 参考实现')
      .replace('等主线走到这一步，重开一张 V4 手算工作纸。','这两套结构应分别计算缓存和访问量。')
      .replace(/^- 主线 A 挂载顺序：.*$/m,'')
      .replace(/^- 主线 B（Qwen3\.5 GDN）和 V4 的对比点留到第 3 步：.*$/m,
        '- GDN 的递归状态与压缩稀疏 KV 使用不同的历史表示，需要分别分析状态管理、检索成本与信息保留。');
  }
  if(paper.id===37) {
    text=text.replace('朴素 SGD 有几个硬伤：','比较更新规则时，可以先观察以下问题：')
      .replace('**稀疏特征吃亏**：低频参数更新太少，永远学不到位','**稀疏更新**：不同参数被访问和更新的频率不同，更新幅度需要与训练分布一起分析')
      .replace('**显存贵**：Adam 需要为每个参数额外存两个状态，大模型训练一半显存花在优化器状态上','**状态容量**：Adam 通常额外维护一阶和二阶统计，按状态精度和分片方式计算容量');
    text=text.replace('Adam 解决 1-3（自适应 + 动量），AdamW 修正 4 里的一个细节（weight decay 耦合问题）。',
      'Adam 使用动量和逐参数二阶统计调整更新；AdamW 将 weight decay 与自适应梯度更新解耦，不减少 Adam 的两份统计状态。');
  }
  if(paper.id===25) {
    const file='roadmap/curriculum/papers/examples/attention_checks.py';
    const code=fs.readFileSync(path.join(repo,file),'utf8').replace(/\r\n/g,'\n').match(/def mla_content[\s\S]*?(?=\n\ndef selected_attention)/)[0].trim();
    text += '\n\n## 把吸收公式与代码逐项对齐\n\n'
      + '这里只检查 content 分支与 Value 的线性重排，暂不加入 RoPE。q[h,d] 是当前 Query，latent[t,c] 是历史低维缓存，wk[h,c,d] 将 latent 映射到完整 Key；wv[h,c,v] 映射到 Value。\n\n'
      + '$$\nK_h=CW_h^K,\\qquad q_hK_h^T=(q_h(W_h^K)^T)C^T,\n$$\n\n'
      + '右式先改变当前 Query，再读 latent；不是把历史完整 K 解压后仍存两份。Value 路径同样可先在 latent 上加权，再乘投影矩阵。\n\n'
      + '<!-- source-check: ../../roadmap/curriculum/papers/examples/attention_checks.py -->\n~~~python\n'+code+'\n~~~\n\n'
      + '两份 scores 都是 [H,T]，两份 out 都是 [H,dv]。einsum 中 t 是历史位置，c 是 latent 维，d 是 Key 维，v 是 Value 维。权重吸收依赖线性关系；位置分支、归一化与量化顺序须另外核对，不能任意跨过非线性。\n';
  }
  if(paper.id===23) text += '\n\n## 用数值例子核对在线状态\n\n'
    + '第一块只有 score=0、Value=2，第二块只有 score=ln(3)、Value=10。局部输出分别是 2 与 10，但整体不是平均值 6。\n\n'
    + '$$\n\\alpha=e^{0-\\ln3}=\\frac13,\\quad \\ell=\\frac13+1,\\quad U=\\frac23+10,\\quad O=U/\\ell=8.\n$$\n\n'
    + 'alpha 必须同时缩放旧分母和旧输出分子。FA1 与 FA2 都有 Q/KV 分块；FA2 还改进 Q 块在 CTA 间的并行和 warp 内分工，不能总结为“FA1 一行、FA2 一块”。\n';
  return text.trim()+'\n';
}
module.exports={papers,prepare};
