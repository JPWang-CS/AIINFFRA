#!/usr/bin/env python3
"""Check local Markdown links without network access.

Usage:
  python scripts/check_markdown_links.py
  python scripts/check_markdown_links.py roadmap papers notes/distributed
"""

from __future__ import annotations

import re
import sys
import urllib.parse
from pathlib import Path


LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def markdown_files(root: Path, inputs: list[str]) -> list[Path]:
    if not inputs:
        return sorted(root.rglob("*.md"))
    files: list[Path] = []
    for raw in inputs:
        path = (root / raw).resolve()
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.suffix.lower() == ".md" and path.exists():
            files.append(path)
        else:
            print(f"input not found: {raw}", file=sys.stderr)
    return files


def local_target(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith(("#", "http://", "https://", "mailto:")):
        return None
    target = urllib.parse.unquote(target.split("#", 1)[0])
    if not target:
        return None
    return (source.parent / target).resolve()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures: list[tuple[Path, int, str]] = []
    checked = 0
    for path in markdown_files(root, sys.argv[1:]):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"non-utf8 markdown: {path.relative_to(root)}", file=sys.stderr)
            continue
        for line_no, line in enumerate(text.splitlines(), 1):
            for match in LINK_RE.finditer(line):
                target = local_target(path, match.group(1))
                if target is None:
                    continue
                checked += 1
                if not target.exists():
                    failures.append((path.relative_to(root), line_no, match.group(1)))

    for path, line_no, target in failures:
        print(f"{path}:{line_no}: missing {target}")
    print(f"checked={checked} missing={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
