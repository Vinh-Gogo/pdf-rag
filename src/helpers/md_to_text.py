"""Batch convert cleaned Markdown pages to plain text using pandoc.

This script scans page_<n> folders under a base directory, looks for
`page_cleared_<n>.md` (fallback: `page_<n>.md`), and produces `.txt` files
in the specified output directory, e.g. `src/data/contents/page_<n>_cleared.txt`.

Requirements:
  - pandoc must be installed and available on PATH.

Usage:
  python -m src.helpers.md_to_text \
    --base_dir src/data/markdown \
    --out_dir src/data/contents \
    --pattern "page_*" \
    --overwrite

Options:
  --dry_run   Show actions without writing files.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Tuple
import re


def check_pandoc() -> None:
    try:
        subprocess.run(["pandoc", "-v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        raise SystemExit("pandoc not found. Please install pandoc and ensure it's on PATH.")


def find_markdown_pages(base_dir: Path, pattern: str) -> Iterable[Tuple[int, Path]]:
    """Yield (page_index, md_path) from page_<n> folders.

    Prefer page_cleared_<n>.md; fall back to page_<n>.md if cleared not found.
    """
    import re

    for sub in base_dir.glob(pattern):
        if not sub.is_dir():
            continue
        m = re.search(r"(\d+)$", sub.name)
        if not m:
            continue
        idx = int(m.group(1))
        cleared = sub / f"page_cleared_{idx}.md"
        raw = sub / f"page_{idx}.md"
        if cleared.exists():
            yield idx, cleared
        elif raw.exists():
            yield idx, raw


IMAGE_INLINE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
IMAGE_LINE_MD_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
IMAGE_TAG_HTML_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)


def strip_image_links_from_text(text: str) -> str:
    """Remove Markdown/HTML image links from text.

    - Removes standalone image lines (e.g., `![alt](url)`)
    - Removes inline image occurrences within a line
    - Removes HTML <img ...> tags
    - Collapses repeated blank lines introduced by removals
    """
    lines = []
    for line in text.splitlines():
        # Skip lines that are just an image
        if IMAGE_LINE_MD_RE.match(line):
            continue
        # Remove inline images and HTML <img> tags
        cleaned = IMAGE_INLINE_MD_RE.sub("", line)
        cleaned = IMAGE_TAG_HTML_RE.sub("", cleaned)
        lines.append(cleaned)

    # Collapse multiple consecutive blank lines
    collapsed: list[str] = []
    blank = False
    for l in lines:
        if l.strip() == "":
            if not blank:
                collapsed.append("")
            blank = True
        else:
            collapsed.append(l)
            blank = False

    return "\n".join(collapsed).rstrip() + "\n"


def convert_one(md_path: Path, txt_path: Path, dry_run: bool = False) -> bool:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[DRY RUN] pandoc -s {md_path} -o {txt_path}")
        return True
    try:
        # Follow the provided example command. pandoc chooses writer by output extension.
        result = subprocess.run(["pandoc", "-s", str(md_path), "-o", str(txt_path)], check=True)

        # Post-process to strip any remaining image links that might have been
        # preserved by upstream tools or input peculiarities.
        try:
            raw = txt_path.read_text(encoding="utf-8", errors="ignore")
            cleaned = strip_image_links_from_text(raw)
            if cleaned != raw:
                txt_path.write_text(cleaned, encoding="utf-8")
        except Exception as e:  # best-effort cleanup; don't fail conversion
            print(f"! warning: post-process strip images failed for {txt_path}: {e}")

        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"✗ pandoc failed for {md_path}: {e}")
        return False


def run(base_dir: Path, out_dir: Path, pattern: str, overwrite: bool, dry_run: bool) -> None:
    check_pandoc()

    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    pages = sorted(find_markdown_pages(base_dir, pattern), key=lambda t: t[0])
    if not pages:
        print("No markdown pages found.")
        return

    print(f"Found {len(pages)} pages. Converting to text…")
    changed = 0
    skipped = 0
    for idx, md_path in pages:
        # Name output following input stem, but place under out_dir
        # Example: page_161/page_161_cleared.md -> contents/page_161_cleared.txt
        out_name = md_path.stem + ".txt"
        txt_path = out_dir / out_name

        if txt_path.exists() and not overwrite:
            print(f"[page {md_path}] skip (exists, no overwrite)")
            skipped += 1
            continue

        ok = convert_one(md_path, txt_path, dry_run=dry_run)
        if ok:
            print(f"[page {md_path}] -> {txt_path}")
            changed += 1
        else:
            print(f"[page {md_path}] failed: {md_path}")

    print("-" * 80)
    print(f"Done. Changed: {changed}, Skipped: {skipped}, Dry-run: {dry_run}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch convert cleaned Markdown to text using pandoc")
    p.add_argument("--base_dir", default="src/data/markdown", help="Base directory with page_<n> folders")
    p.add_argument("--out_dir", default="src/data/contents", help="Output directory for .txt files")
    p.add_argument("--pattern", default="page_*", help="Glob pattern for page folders")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt files")
    p.add_argument("--dry_run", action="store_true", help="List actions without writing files")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(Path(args.base_dir), Path(args.out_dir), args.pattern, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
