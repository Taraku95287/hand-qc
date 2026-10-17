#!/usr/bin/env python
"""Keep the bilingual docs honest.

    python scripts/check_docs.py
    python scripts/check_docs.py --fix-hint

Every doc that has a `*.zh-CN.md` sibling is a pair. A translation drifts when
a number is retyped, when a table loses a row, or when a link is rewritten and
the other side is forgotten. This checks four things per pair, and one thing
across the whole repo:

1. both files exist
2. each side carries a switcher line pointing at the other
3. the **multiset of numeric tokens is identical** — the single most useful
   check, because a mistyped AUC or percentage is invisible on review
4. the block structure matches: same number of headings, same number of rows
   per table, byte-identical contents inside fenced code blocks

plus: every relative markdown link in any doc resolves to a real path.

Exits non-zero on any failure, so CI catches it.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

REPO = Path(__file__).resolve().parent.parent

# English doc -> Chinese doc. Add a row when a new pair appears.
PAIRS = (
    ("README.md", "README.zh-CN.md"),
    ("MODEL_CARD.md", "MODEL_CARD.zh-CN.md"),
    ("docs/RESULTS.md", "docs/RESULTS.zh-CN.md"),
    ("docs/FINDINGS.md", "docs/FINDINGS.zh-CN.md"),
    ("docs/DATA_AND_WEIGHTS.md", "docs/DATA_AND_WEIGHTS.zh-CN.md"),
    ("training/README.md", "training/README.zh-CN.md"),
    ("weights/README.md", "weights/README.zh-CN.md"),
)

# Deliberately not paired: legal text (translation would not be authoritative)
# and the research log (Chinese only, written first, has no English original).
UNPAIRED = ("LICENSE", "LICENSE-MODEL.md", "docs/RESEARCH_LOG.md")

NUM = re.compile(r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
FENCE = re.compile(r"^```")
HEADING = re.compile(r"^(#{1,6}) ")
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SWITCHER = re.compile(r"^\*\*(English|简体中文)\*\*\s*\|\s*\[", re.MULTILINE)


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def split_fences(text: str) -> tuple[list[str], list[str]]:
    """Return (prose_lines, fenced_block_contents)."""
    prose, blocks, current, in_fence = [], [], [], False
    for line in text.splitlines():
        if FENCE.match(line):
            if in_fence:
                blocks.append("\n".join(current))
                current = []
            in_fence = not in_fence
            continue
        (current if in_fence else prose).append(line)
    if current:
        blocks.append("\n".join(current))
    return prose, blocks


def numeric_tokens(text: str) -> Counter:
    """Numeric tokens outside of link targets and HTML comments."""
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]()", text)          # drop link targets
    return Counter(NUM.findall(text))


def table_row_counts(prose: list[str]) -> list[int]:
    counts, run = [], 0
    for line in prose:
        if line.lstrip().startswith("|"):
            run += 1
        elif run:
            counts.append(run)
            run = 0
    if run:
        counts.append(run)
    return counts


def heading_count(prose: list[str]) -> int:
    return sum(1 for line in prose if HEADING.match(line))


def check_pair(en_rel: str, zh_rel: str) -> list[str]:
    problems: list[str] = []
    en, zh = REPO / en_rel, REPO / zh_rel
    if not en.exists():
        return [f"{en_rel}: missing"]
    if not zh.exists():
        return [f"{zh_rel}: missing (pair with {en_rel})"]

    en_text, zh_text = read(en), read(zh)

    zh_name = Path(zh_rel).name
    en_name = Path(en_rel).name
    en_head = "\n".join(en_text.splitlines()[:12])
    zh_head = "\n".join(zh_text.splitlines()[:12])
    if not (SWITCHER.search(en_head) and zh_name in en_head):
        problems.append(f"{en_rel}: no switcher line pointing at {zh_name}")
    if not (SWITCHER.search(zh_head) and en_name in zh_head):
        problems.append(f"{zh_rel}: no switcher line pointing at {en_name}")

    en_nums, zh_nums = numeric_tokens(en_text), numeric_tokens(zh_text)
    missing = en_nums - zh_nums
    extra = zh_nums - en_nums
    if missing:
        problems.append(f"{zh_rel}: {sum(missing.values())} numeric token(s) "
                        f"present in English but not Chinese: "
                        f"{sorted(missing)[:8]}")
    if extra:
        problems.append(f"{zh_rel}: {sum(extra.values())} numeric token(s) "
                        f"in Chinese but not English: {sorted(extra)[:8]}")

    en_prose, en_blocks = split_fences(en_text)
    zh_prose, zh_blocks = split_fences(zh_text)

    if len(en_blocks) != len(zh_blocks):
        problems.append(f"{zh_rel}: {len(zh_blocks)} code fences vs "
                        f"{len(en_blocks)} in {en_rel}")
    else:
        for i, (a, b) in enumerate(zip(en_blocks, zh_blocks)):
            if a != b:
                problems.append(f"{zh_rel}: code fence #{i + 1} differs from "
                                f"{en_rel} (code must not be translated)")

    en_h, zh_h = heading_count(en_prose), heading_count(zh_prose)
    if en_h != zh_h:
        problems.append(f"{zh_rel}: {zh_h} headings vs {en_h} in {en_rel}")

    en_t, zh_t = table_row_counts(en_prose), table_row_counts(zh_prose)
    if en_t != zh_t:
        problems.append(f"{zh_rel}: table row counts {zh_t} vs {en_t} in {en_rel}")

    return problems


def check_links() -> list[str]:
    problems: list[str] = []
    for path in sorted(REPO.rglob("*.md")):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if ".pytest_cache" in path.parts:
            continue
        text = read(path).replace("`", "")
        for target in LINK.findall(text):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                problems.append(f"{path.relative_to(REPO)}: broken link -> {target}")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links-only", action="store_true",
                    help="skip the pair checks")
    args = ap.parse_args(argv)

    problems: list[str] = []

    if not args.links_only:
        print(f"checking {len(PAIRS)} doc pair(s)")
        for en_rel, zh_rel in PAIRS:
            found = check_pair(en_rel, zh_rel)
            status = "ok" if not found else f"{len(found)} problem(s)"
            print(f"  {en_rel:<28} <-> {zh_rel:<34} {status}")
            problems += found

    link_problems = check_links()
    print(f"checking relative links: "
          f"{'ok' if not link_problems else str(len(link_problems)) + ' broken'}")
    problems += link_problems

    if problems:
        print("\nFAIL")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
