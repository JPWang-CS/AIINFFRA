#!/usr/bin/env python3
"""Fetch and rank recent AI-infrastructure papers from arXiv.

The script is intentionally stdlib-only. It writes only when --write-inbox or
--output is supplied; otherwise it prints Markdown to stdout.

Examples:
  python scripts/research_watch.py --topic distributed --days 14
  python scripts/research_watch.py --topic kernels --days 7 --write-inbox
  python scripts/research_watch.py --query 'all:"expert parallel"' --max 30
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = "AIINFFRA-Research-Watch/2.0"

TOPICS = {
    "kernels": {
        "categories": ["cs.DC", "cs.AR", "cs.PF", "cs.LG"],
        "terms": [
            "gpu kernel", "cuda", "triton", "flash attention", "gemm",
            "tensor core", "kernel fusion", "gpu compiler", "cutlass",
        ],
    },
    "inference": {
        "categories": ["cs.DC", "cs.LG", "cs.CL"],
        "terms": [
            "llm inference", "serving", "kv cache", "paged attention",
            "speculative decoding", "prefill", "decode", "mixture of experts",
        ],
    },
    "distributed": {
        "categories": ["cs.DC", "cs.NI", "cs.PF", "cs.LG"],
        "terms": [
            "collective communication", "multi-gpu", "multi-node", "nccl",
            "rdma", "expert parallel", "tensor parallel", "pipeline parallel",
            "communication overlap", "gpu cluster",
        ],
    },
    "training": {
        "categories": ["cs.DC", "cs.LG", "cs.CL"],
        "terms": [
            "large language model training", "fsdp", "zero", "distributed training",
            "context parallel", "fp8 training", "optimizer", "checkpoint",
        ],
    },
    "architecture": {
        "categories": ["cs.AR", "cs.DC", "cs.PF"],
        "terms": [
            "gpu architecture", "accelerator", "memory hierarchy", "interconnect",
            "nvlink", "infiniband", "roce", "tensor memory",
        ],
    },
}


@dataclass
class Paper:
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    published: datetime
    updated: datetime
    categories: list[str]
    score: int = 0
    matched_terms: tuple[str, ...] = ()


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clean_text(text: str | None) -> str:
    return " ".join((text or "").split())


def short_authors(authors: list[str], limit: int = 3) -> str:
    if len(authors) <= limit:
        return ", ".join(authors)
    return ", ".join(authors[:limit]) + " et al."


def query_for(topic: str, custom_query: str | None) -> tuple[str, list[str]]:
    if custom_query:
        return custom_query, []
    cfg = TOPICS[topic]
    category_query = " OR ".join(f"cat:{c}" for c in cfg["categories"])
    term_query = " OR ".join(f'all:"{term}"' for term in cfg["terms"])
    return f"({category_query}) AND ({term_query})", list(cfg["terms"])


def fetch(query: str, max_results: int) -> list[Paper]:
    params = urllib.parse.urlencode(
        {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
    )
    url = f"{ARXIV_API}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        xml_data = response.read()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_data)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ns):
        raw_id = clean_text(entry.findtext("atom:id", namespaces=ns))
        arxiv_id = raw_id.rstrip("/").split("/")[-1]
        authors = [
            clean_text(node.findtext("atom:name", namespaces=ns))
            for node in entry.findall("atom:author", ns)
        ]
        categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", ns)]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=clean_text(entry.findtext("atom:title", namespaces=ns)),
                summary=clean_text(entry.findtext("atom:summary", namespaces=ns)),
                authors=authors,
                published=parse_time(entry.findtext("atom:published", namespaces=ns) or "1970-01-01T00:00:00Z"),
                updated=parse_time(entry.findtext("atom:updated", namespaces=ns) or "1970-01-01T00:00:00Z"),
                categories=categories,
            )
        )
    return papers


def rank(papers: list[Paper], terms: list[str], days: int, min_score: int) -> list[Paper]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    ranked: list[Paper] = []
    for paper in papers:
        if days > 0 and max(paper.published, paper.updated) < cutoff:
            continue
        title = paper.title.lower()
        body = f"{paper.title} {paper.summary}".lower()
        matched = tuple(term for term in terms if term.lower() in body)
        title_hits = sum(1 for term in matched if term.lower() in title)
        paper.score = 3 * title_hits + len(matched)
        paper.matched_terms = matched
        if not terms or paper.score >= min_score:
            ranked.append(paper)
    return sorted(ranked, key=lambda item: (item.score, item.updated), reverse=True)


def existing_ids(inbox_dir: Path) -> set[str]:
    ids: set[str] = set()
    if not inbox_dir.exists():
        return ids
    pattern = re.compile(r"arxiv\.org/abs/([0-9.]+(?:v\d+)?)", re.IGNORECASE)
    for path in inbox_dir.glob("*.md"):
        ids.update(pattern.findall(path.read_text(encoding="utf-8")))
    return ids


def render(papers: list[Paper], topic: str, query: str, skipped: int) -> str:
    now = datetime.now().astimezone()
    lines = [
        f"# Research inbox — {now:%Y-%m-%d}",
        "",
        f"> topic: `{topic}` · generated: `{now.isoformat(timespec='seconds')}`",
        f"> query: `{query}`",
        f"> new: {len(papers)} · skipped existing: {skipped}",
        "> 自动抓取只代表“待筛选”，不代表事实已核验或已进入学习计划。",
        "",
        "| Score | Paper | Date | Categories | Matched | Triage |",
        "|------:|-------|------|------------|---------|--------|",
    ]
    for paper in papers:
        url = f"https://arxiv.org/abs/{paper.arxiv_id}"
        cats = ", ".join(paper.categories[:3])
        matched = ", ".join(paper.matched_terms[:4]) or "custom query"
        title = paper.title.replace("|", "\\|")
        lines.append(
            f"| {paper.score} | [{title}]({url})<br>{short_authors(paper.authors)} "
            f"| {paper.updated:%Y-%m-%d} | {cats} | {matched} | P0/P1/P2/skip |"
        )
    lines.extend(
        [
            "",
            "## 筛选动作",
            "",
            "1. 核对 arXiv 版本、作者主页/官方代码和发表状态。",
            "2. 写一句与当前 PATH 节点的关系；无关系的 P2 不进入主线。",
            "3. P0/P1 移入 `papers/README.md` 或专题 watchlist；其余留 inbox。",
            "4. 论文性能数字必须记录 GPU、dtype、shape、baseline 与统计口径。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    # Windows terminals often default to GBK; arXiv metadata is Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", choices=sorted(TOPICS), default="distributed")
    parser.add_argument("--query", help="Raw arXiv API search_query; bypasses topic query")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--max", type=int, default=100, dest="max_results")
    parser.add_argument("--min-score", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-inbox", action="store_true")
    parser.add_argument("--include-existing", action="store_true")
    parser.add_argument("--delay", type=float, default=3.0, help="Polite delay before request")
    args = parser.parse_args()

    query, terms = query_for(args.topic, args.query)
    try:
        if args.delay > 0:
            time.sleep(args.delay)
        papers = rank(fetch(query, args.max_results), terms, args.days, args.min_score)
    except (urllib.error.URLError, TimeoutError, ET.ParseError) as exc:
        print(f"research_watch: network/API error: {exc}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[1]
    inbox_dir = repo_root / "papers" / "inbox"
    seen = set() if args.include_existing else existing_ids(inbox_dir)
    fresh = [paper for paper in papers if paper.arxiv_id not in seen]
    markdown = render(fresh, args.topic, query, len(papers) - len(fresh))

    output = args.output
    if args.write_inbox:
        output = inbox_dir / f"{datetime.now():%Y-%m-%d}-{args.topic}.md"
    if output:
        output = output if output.is_absolute() else repo_root / output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown, encoding="utf-8")
        print(output)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
