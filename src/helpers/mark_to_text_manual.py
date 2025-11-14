"""Manual Markdown -> text converter preserving tables and paragraph breaks.

Features:
  - Remove image markdown blocks and inline images: ![alt](path) and <img ...> tags.
  - Preserve table rows exactly (retain '|').
  - Group contiguous table lines into a single block separated by one blank line
	from other paragraphs.
  - Paragraphs separated by exactly one blank line (\n\n) in output.
  - No pandoc dependency; works directly on source markdown.

Usage:
  python -m src.helpers.mark_to_text_manual \
	--base_dir src/data/markdown \
	--out_dir src/data/contents \
	--pattern "page_*" \
	--overwrite

Options:
  --dry_run    Show planned conversions without writing.

The output file name mirrors the input markdown stem with .txt extension.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Tuple, List


IMAGE_INLINE_MD_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
IMAGE_LINE_MD_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]*\)\s*$")
IMAGE_TAG_HTML_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)

TABLE_ROW_RE = re.compile(r"^\|.*\|\s*$")


def find_markdown_pages(base_dir: Path, pattern: str) -> Iterable[Tuple[int, Path]]:
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


def strip_images(line: str) -> str:
	if IMAGE_LINE_MD_RE.match(line):
		return ""  # remove entire line
	cleaned = IMAGE_INLINE_MD_RE.sub("", line)
	cleaned = IMAGE_TAG_HTML_RE.sub("", cleaned)
	return cleaned


def segment_blocks(md_text: str) -> List[str]:
	"""Segment markdown into blocks (paragraphs or tables)."""
	lines = md_text.splitlines()
	blocks: List[str] = []
	current: List[str] = []
	in_table = False

	def flush():
		nonlocal current, in_table
		if not current:
			return
		if in_table:
			# keep table with original line breaks
			blocks.append("\n".join(current).strip())
		else:
			# merge paragraph lines, preserving explicit breaks inside lists if any
			paragraph = []
			for l in current:
				if l.strip() == "":
					paragraph.append("")
				else:
					paragraph.append(l.rstrip())
			# collapse multiple blank lines inside paragraph
			cleaned_para: List[str] = []
			blank = False
			for l in paragraph:
				if l.strip() == "":
					if not blank:
						cleaned_para.append("")
					blank = True
				else:
					cleaned_para.append(l)
					blank = False
			blocks.append("\n".join(cleaned_para).strip())
		current = []
		in_table = False

	for raw_line in lines:
		line = strip_images(raw_line)
		if line == "":
			if in_table:
				# blank line terminates table
				flush()
			else:
				# blank inside paragraph triggers flush to keep separation
				flush()
			continue

		if TABLE_ROW_RE.match(line):
			if not in_table:
				# starting table; flush previous paragraph
				flush()
				in_table = True
			current.append(line)
		else:
			if in_table:
				# table ended; flush and start paragraph
				flush()
			current.append(line)

	flush()

	# Remove any empty blocks produced accidentally
	return [b for b in blocks if b.strip()]


def convert_markdown(md_text: str) -> str:
	blocks = segment_blocks(md_text)
	# Join with exactly one blank line
	return "\n\n".join(blocks).rstrip() + "\n"


def write_converted(md_path: Path, txt_path: Path, dry_run: bool) -> bool:
	try:
		if dry_run:
			print(f"[DRY RUN] convert {md_path} -> {txt_path}")
			return True
		txt_path.parent.mkdir(parents=True, exist_ok=True)
		md_text = md_path.read_text(encoding="utf-8", errors="ignore")
		converted = convert_markdown(md_text)
		txt_path.write_text(converted, encoding="utf-8")
		return True
	except Exception as e:
		print(f"✗ failed {md_path}: {e}")
		return False


def run(base_dir: Path, out_dir: Path, pattern: str, overwrite: bool, dry_run: bool) -> None:
	if not base_dir.exists():
		raise SystemExit(f"Base directory not found: {base_dir}")

	pages = sorted(find_markdown_pages(base_dir, pattern), key=lambda t: t[0])
	if not pages:
		print("No markdown pages found.")
		return
	print(f"Found {len(pages)} pages. Converting manually…")
	changed = 0
	skipped = 0
	for idx, md_path in pages:
		out_name = md_path.stem + ".txt"
		txt_path = out_dir / out_name
		if txt_path.exists() and not overwrite:
			print(f"[page {md_path}] skip (exists, no overwrite)")
			skipped += 1
			continue
		ok = write_converted(md_path, txt_path, dry_run=dry_run)
		if ok:
			print(f"[page {md_path}] -> {txt_path}")
			changed += 1
		else:
			print(f"[page {md_path}] failed")
	print("-" * 80)
	print(f"Done. Changed: {changed}, Skipped: {skipped}, Dry-run: {dry_run}")


def parse_args(argv=None):
	p = argparse.ArgumentParser(description="Manual markdown to text (preserve tables, remove images)")
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
