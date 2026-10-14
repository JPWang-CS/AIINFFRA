const fs = require('node:fs');
const path = require('node:path');
const introductions = require('./paper-introductions.cjs');
const categoryOrder = [
  '模型基础与完整结构',
  '注意力、序列建模与长上下文',
  'MoE 与条件计算',
  '训练、优化与后训练',
  '生成、推理与检索算法',
  '量化与模型压缩',
  '多模态与生成模型',
  '高效计算与系统论文',
];
// Keep IDs and source paths stable. Display order is explicit per category.
const papers = [
  [20,'高效计算与系统论文',1,'Online Softmax','notes/algorithms/online-softmax.md'],
  [21,'高效计算与系统论文',2,'Parallel Reduce','notes/algorithms/parallel-reduce.md'],
  [22,'高效计算与系统论文',3,'FlashAttention：分块与在线归约','notes/algorithms/flash-attention-mechanism.md'],
  [23,'高效计算与系统论文',4,'FlashAttention-2','notes/algorithms/flash-attention-2.md'],
  [24,'注意力、序列建模与长上下文',1,'MQA 与 GQA','papers/attention/gqa.md'],
  [25,'注意力、序列建模与长上下文',2,'MLA：低维缓存与权重吸收','notes/algorithms/mla-deepseek.md'],
  [26,'注意力、序列建模与长上下文',3,'DSA：索引与稀疏选择','notes/algorithms/dsa-sparse-attention.md','roadmap/curriculum/papers/dsa.md'],
  [27,'注意力、序列建模与长上下文',5,'Gated DeltaNet 与线性注意力','notes/algorithms/gdn-linear-attention.md'],
  [28,'高效计算与系统论文',5,'FlashAttention 与可编程注意力专题','notes/algorithms/fa4-flexattention.md'],
  [29,'高效计算与系统论文',6,'低精度与结构复用 Attention 专题','notes/algorithms/attention-2026-sage3-kascade.md'],
  [30,'模型基础与完整结构',7,'DeepSeek：参数、KV 与计算量','notes/algorithms/deepseek-v32-handcalc.md'],
  [31,'模型基础与完整结构',8,'DeepSeek-V4 架构专题','notes/algorithms/deepseek-v4.md'],
  [32,'MoE 与条件计算',1,'MoE 路由与专家','notes/algorithms/moe-inference.md'],
  [33,'量化与模型压缩',1,'INT8 与 FP8','notes/algorithms/quantization-int8-fp8.md'],
  [34,'生成、推理与检索算法',2,'Speculative Decoding','notes/algorithms/speculative-decoding.md'],
  [35,'高效计算与系统论文',7,'Prefill / Decode 分离','notes/algorithms/pd-disaggregation.md'],
  [36,'高效计算与系统论文',8,'PagedAttention','papers/inference/paged-attention.md'],
  [37,'训练、优化与后训练',2,'Adam、AdamW 与训练状态','notes/algorithms/optimizers-adam.md'],
  [38,'高效计算与系统论文',9,'ZeRO 与模型状态分片','papers/training/zero-paper.md'],
  [39,'模型基础与完整结构',9,'DeepSeek-V4.1-Flash：缓存、索引与训练推理协同','roadmap/curriculum/papers/deepseek-v41.md'],
  [40,'训练、优化与后训练',5,'Scaling Law、强化学习与长任务推理','roadmap/curriculum/papers/inference-workloads.md'],
  [41,'模型基础与完整结构',1,'语言模型：从 token 到 next-token 目标','papers/foundations/01-language-modeling.md'],
  [42,'模型基础与完整结构',2,'Attention：从相似度到加权读取','papers/foundations/02-attention.md'],
  [43,'模型基础与完整结构',3,'Transformer Block：残差、归一化与 MLP','papers/foundations/03-transformer-block.md'],
  [44,'模型基础与完整结构',4,'位置编码：绝对位置、相对位置与 RoPE','papers/foundations/04-position-encoding.md'],
  [45,'模型基础与完整结构',5,'完整 Decoder：一次前向与逐 token 解码','papers/foundations/05-complete-decoder.md'],
  [46,'模型基础与完整结构',6,'Transformer 家族：结构变化与权衡','papers/foundations/06-transformer-families.md'],
  [47,'注意力、序列建模与长上下文',4,'Mamba：输入选择的状态空间模型','papers/topics/ssm-mamba.md'],
  [48,'训练、优化与后训练',1,'训练目标、优化与 Scaling Law','papers/topics/training-and-scaling.md'],
  [49,'训练、优化与后训练',3,'LoRA：冻结底座上的低秩增量','papers/topics/lora-adaptation.md'],
  [50,'训练、优化与后训练',4,'对齐与强化学习：目标函数如何改变回答','papers/topics/alignment-and-rl.md'],
  [51,'生成、推理与检索算法',1,'采样与搜索：从概率分布到候选决策','papers/topics/sampling-and-search.md'],
  [52,'生成、推理与检索算法',3,'检索与记忆：把外部证据接入生成','papers/topics/retrieval-and-memory.md'],
  [53,'量化与模型压缩',2,'GPTQ：校准输入上的权重重构','papers/topics/gptq.md'],
  [54,'量化与模型压缩',3,'AWQ：用激活重要性选择权重量化缩放','papers/topics/awq.md'],
  [55,'量化与模型压缩',4,'SmoothQuant：把激活离群值迁移到权重','papers/topics/smoothquant.md'],
  [56,'多模态与生成模型',1,'视觉语言模型：从图像 patch 到文本 token','papers/topics/vision-language.md'],
  [57,'多模态与生成模型',2,'Diffusion 与 Flow Matching：从训练噪声到生成轨迹','papers/topics/diffusion-and-flow.md'],
  [58,'高效计算与系统论文',10,'并行 Scan 与 Selection：前缀、统计量和 Top-k','papers/topics/scan-and-selection.md'],
  [59,'量化与模型压缩',5,'蒸馏与压缩：迁移函数、删结构与低秩近似','papers/topics/compression-and-distillation.md'],
  [60,'注意力、序列建模与长上下文',6,'长上下文：位置外推、局部注意力与记忆压缩','papers/topics/long-context-design.md'],
].map(([id,category,order,title,source,override])=>({id,category,order,title,source,override}));

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
  // Numerical and algorithmic corrections belong in the source note.
  // Do not replace the tail of chapter 30: it contains reviewed derivations
  // and interview exercises that must survive publication.
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
module.exports={papers,categoryOrder,prepare};
