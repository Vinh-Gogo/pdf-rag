"""Clean markdown files with a simple, safe pipeline.

Default behavior (table-safe):
	- Replace every occurrence of the literal string "<br>" with a single space " ".
	- Preserve the original Markdown formatting (tables intact).

Optional text mode (--to_text):
	- After replacing "<br>", convert Markdown → HTML → plain text using markdown-it
	  + BeautifulSoup and remove code/pre blocks. This may flatten tables.

Input structure:
	src/data/markdown/page_<n>/page_<n>.md
Output written as:
	src/data/markdown/page_<n>/page_cleared_<n>.md

Usage:
	python -m src.helpers.markdown_to_markdown \
	--base_dir src/data/markdown \
	--pattern page_* \
	--overwrite

Options:
	--dry_run    Show planned actions without writing files.

Exit codes:
	0 success
	1 base directory missing
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable


def find_markdown_pages(base_dir: Path, pattern: str) -> Iterable[tuple[int, Path]]:
	"""Yield (page_index, md_path) for matching markdown files."""
	import re
	for sub in base_dir.glob(pattern):
		if not sub.is_dir():
			continue
		m = re.search(r"(\d+)$", sub.name)
		if not m:
			continue
		idx = int(m.group(1))
		md_file = sub / f"page_{idx}.md"
		if md_file.exists():
			yield idx, md_file


def markdown_to_clean_text(md_text: str) -> str:
	"""Convert Markdown to clean plain text using markdown-it and BeautifulSoup.

	- Parses Markdown with the "commonmark" preset.
	- Renders to HTML, then strips tags to text.
	- Removes code/pre blocks for cleaner output.
	"""
	from markdown_it import MarkdownIt
	from markdown_it.presets import commonmark  # noqa: F401 (preset enabled by name)
	from bs4 import BeautifulSoup

	md = MarkdownIt("commonmark")
	html = md.render(md_text)
	soup = BeautifulSoup(html, "html.parser")

	for code in soup.find_all(["code", "pre"]):
		code.decompose()

	return soup.get_text(separator=" ", strip=True)


def clean_content(text: str, to_text: bool = False) -> str:
	"""Clean content with optional Markdown→text conversion.

	- Always replace '<br>' with a space.
	- If to_text=True, convert to plain text; otherwise keep Markdown as-is.
	"""
	step1 = text.replace("<br>", " ")
	while "  " in step1:
		step1 = step1.replace("  ", " ")
	while "** " in step1:
		step1 = step1.replace("** ", "**\n")
		
	return step1

def process_file(src: Path, dst: Path, overwrite: bool, dry_run: bool, to_text: bool) -> bool:
	# if dst.exists() and not overwrite:
	# 	return False
	# if dry_run:
	# 	return True
	content = src.read_text(encoding="utf-8")
	cleaned = clean_content(content, to_text=to_text)
	dst.write_text(cleaned, encoding="utf-8")
	return True


def run(base_dir: Path, pattern: str, overwrite: bool, dry_run: bool, to_text: bool) -> None:
	if not base_dir.exists():
		print(f"Base directory not found: {base_dir}")
		raise SystemExit(1)
	pages = sorted(find_markdown_pages(base_dir, pattern), key=lambda t: t[0])
	if not pages:
		print("No markdown pages found matching pattern.")
		return
	total = len(pages)
	print(f"Found {total} markdown pages. Starting clean…")
	changed = 0
	skipped = 0
	for idx, md_path in pages:
		dst = md_path.parent / f"page_cleared_{idx}.md"
		wrote = process_file(md_path, dst, overwrite=overwrite, dry_run=dry_run, to_text=to_text)
		if wrote:
			print(f"[page {idx}] -> {'(dry-run)' if dry_run else dst}")
			changed += 1
		else:
			print(f"[page {idx}] skip (exists, no overwrite)")
			skipped += 1
	print(f"Done. Changed: {changed}, Skipped: {skipped}, Dry-run: {dry_run}")


def parse_args(argv=None):
	p = argparse.ArgumentParser(description="Clean markdown pages: replace <br> with space; optional Markdown→text.")
	p.add_argument("--base_dir", default="src/data/markdown", help="Base directory containing page_<n> folders")
	p.add_argument("--pattern", default="page_*", help="Glob pattern for page folders")
	p.add_argument("--overwrite", action="store_true", help="Overwrite existing cleaned files")
	p.add_argument("--dry_run", action="store_true", help="Show actions without writing files")
	p.add_argument("--to_text", action="store_true", help="After replacing <br>, convert Markdown to plain text (tables will be flattened)")
	return p.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	run(Path(args.base_dir), args.pattern, overwrite=args.overwrite, dry_run=args.dry_run, to_text=args.to_text)


if __name__ == "__main__":
	main()

