#!/usr/bin/env python3
"""Fetch recent AI Infra papers from arxiv. No dependencies beyond stdlib.

Usage:
  python fetch_papers.py --days 7                        # Last 7 days
  python fetch_papers.py --days 3 --titles-only          # Titles only, quick scan
  python fetch_papers.py --query "flash attention gpu"   # Keyword search
  python fetch_papers.py --category cs.LG --days 1       # Specific category

Output: markdown table, ready to paste into papers/README.md
"""

import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

# ---- Config ----
ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = 50
SLEEP_SEC = 3  # arxiv rate limit: be polite

# AI Infra relevant categories
DEFAULT_CATEGORIES = [
    "cs.DC",   # Distributed, Parallel, and Cluster Computing
    "cs.LG",   # Machine Learning
    "cs.CL",   # Computation and Language (LLM papers)
    "cs.AR",   # Architecture
    "cs.PF",   # Performance
]

# AI Infra keyword filters (title/abstract must contain at least one)
INFRA_KEYWORDS = [
    "attention", "transformer", "llm", "gpu", "cuda", "triton",
    "inference", "training", "distributed", "parallel", "nccl",
    "quantization", "quantize", "kv cache", "pagedattention",
    "flash attention", "mixture of experts", "moe", "lora",
    "fine-tuning", "finetuning", "speculative decoding",
    "memory optimization", "throughput", "latency", "bandwidth",
    "kernel", "gemm", "softmax", "layernorm", "rmsnorm",
    "tensor", "accelerator", "npu", "tpu", "fp8", "fp16", "int8", "int4",
    "deepseek", "llama", "mistral", "qwen", "yi ", "gemma",
    "mamba", "state space", "ssm", "linear attention",
    "mixture of experts", "deep learning", "neural network",
    "hpc", "data parallelism", "model parallelism", "pipeline parallelism",
    "fsdp", "deepspeed", "megatron", "vllm", "sglang", "tensorrt",
    "agent", "tool use", "function calling", "rag", "retrieval augmented",
    "mcp", "model context protocol", "react",
]


def build_query(categories, keywords, days, custom_query):
    """Build arxiv API query string."""
    if custom_query:
        return custom_query

    parts = []
    # Category filter
    cat_filter = " OR ".join(f"cat:{c}" for c in categories)
    parts.append(f"({cat_filter})")

    # Keyword filter (OR)
    if keywords:
        kw_filter = " OR ".join(f'all:"{kw}"' for kw in keywords[:20])  # limit to avoid huge query
        parts.append(f"({kw_filter})")

    return " AND ".join(parts)


def parse_date(arxiv_date_str):
    """Parse arxiv date format: 2026-05-27T12:00:00Z or similar."""
    # Strip timezone suffix and parse
    date_str = arxiv_date_str.replace("Z", "+00:00")
    return datetime.fromisoformat(date_str)


def fetch(query, max_results=MAX_RESULTS):
    """Fetch papers from arxiv API, return list of dicts."""
    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })
    url = f"{ARXIV_API}?{params}"
    print(f"Fetching: {url}", file=sys.stderr)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AIInfra-Paper-Fetcher/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(f"Network error: {e}", file=sys.stderr)
        print("Tip: arxiv API requires internet access. Check VPN/proxy settings.", file=sys.stderr)
        return []
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return []

    # Parse Atom XML
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(data)
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        summary = entry.find("atom:summary", ns)
        published = entry.find("atom:published", ns)
        arxiv_id = entry.find("atom:id", ns)
        authors = entry.findall("atom:author/atom:name", ns)
        primary_cat = entry.find("arxiv:primary_category", ns)

        papers.append({
            "title": title.text.strip().replace("\n", " ") if title is not None else "?",
            "summary": summary.text.strip().replace("\n", " ") if summary is not None else "",
            "published": published.text if published is not None else "",
            "id": arxiv_id.text.strip() if arxiv_id is not None else "",
            "authors": [a.text for a in authors],
            "category": primary_cat.get("term") if primary_cat is not None else "",
        })

    print(f"Fetched {len(papers)} papers", file=sys.stderr)
    return papers


def filter_recent(papers, days):
    """Keep only papers from the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    recent = []
    for p in papers:
        try:
            pub_date = parse_date(p["published"])
            if pub_date > cutoff:
                recent.append(p)
        except (ValueError, TypeError):
            # Can't parse date, include anyway
            recent.append(p)
    return recent


def short_authors(authors, max_n=3):
    """First N authors + et al."""
    if len(authors) <= max_n:
        return ", ".join(authors)
    return ", ".join(authors[:max_n]) + " et al."


def clean_id(arxiv_id):
    """Extract short arxiv ID from full URL."""
    # http://arxiv.org/abs/2501.xxxx
    parts = arxiv_id.split("/")
    return parts[-1] if parts else arxiv_id


def find_category(paper_index_path):
    """Guess best category for a paper based on title/abstract keywords."""
    # This is a heuristic; user should manually verify
    category_map = {
        "attention": "attention",
        "kv cache": "inference",
        "pagedattention": "inference",
        "inference": "inference",
        "serving": "inference",
        "quantiz": "inference",
        "speculative decoding": "inference",
        "training": "training",
        "distributed": "training",
        "parallel": "training",
        "fsdp": "training",
        "deepspeed": "training",
        "megatron": "training",
        "moe": "training",
        "mixture of experts": "training",
        "agent": "agents",
        "tool": "agents",
        "rag": "agents",
        "retrieval": "agents",
        "mcp": "agents",
        "react": "agents",
        "compiler": "compiler",
        "triton": "compiler",
        "mlir": "compiler",
        "tvm": "compiler",
        "cuda": "cuda",
        "gpu kernel": "cuda",
        "gemm": "cuda",
    }
    # Not called in current flow; reserved for future use
    return "uncategorized"


def main():
    parser = argparse.ArgumentParser(description="Fetch AI Infra papers from arxiv")
    parser.add_argument("--days", type=int, default=3, help="Days to look back")
    parser.add_argument("--categories", nargs="*", default=None, help="arxiv categories")
    parser.add_argument("--query", type=str, default=None, help="Custom search query")
    parser.add_argument("--titles-only", action="store_true", help="Only show titles")
    parser.add_argument("--max", type=int, default=MAX_RESULTS, help="Max results")
    parser.add_argument("--no-filter", action="store_true", help="Don't filter by keywords")
    args = parser.parse_args()

    categories = args.categories if args.categories else DEFAULT_CATEGORIES
    keywords = None if args.no_filter else INFRA_KEYWORDS

    query = build_query(categories, keywords, args.days, args.query)
    papers = fetch(query, args.max)

    if args.days > 0 and not args.query:
        papers = filter_recent(papers, args.days)

    if not papers:
        print("No papers found.")
        return

    # Output as markdown
    today = datetime.now().strftime("%Y-%m-%d")
    print(f"\n## Fetched {today} ({len(papers)} papers)\n")
    print("| # | Title | Authors | Date | ID |")
    print("|---|-------|---------|------|----|")

    for i, p in enumerate(papers, 1):
        pub_date = p["published"][:10] if p["published"] else "?"
        authors = short_authors(p["authors"])
        arxiv_id = clean_id(p["id"])
        url = f"https://arxiv.org/abs/{arxiv_id}"
        title = p["title"][:80] + ("..." if len(p["title"]) > 80 else "")

        if args.titles_only:
            print(f"| {i} | [{title}]({url}) | {authors} | {pub_date} | {arxiv_id} |")
        else:
            summary = p["summary"][:120] + ("..." if len(p["summary"]) > 120 else "")
            print(f"| {i} | [{title}]({url}) | {authors} | {pub_date} | {arxiv_id} |")
            print(f"  | | *{summary}* | | | |")


if __name__ == "__main__":
    main()
