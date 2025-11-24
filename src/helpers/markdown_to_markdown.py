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

import re
from markdown_it import MarkdownIt
from bs4 import BeautifulSoup

def markdown_to_clean_text(md_text: str) -> str:
    """Keep lines starting with # (headings), remove code blocks, clean the rest."""
    # Precompile heading pattern: 1-6 # followed by space/tab
    HEADING_PATTERN = re.compile(r"^#{1,6}[\s]")

    # Bước 1: Loại bỏ code blocks (``` ... ```)
    md_no_code = re.sub(r"```[\s\S]*?```", "", md_text)

    # Bước 2: Tách thành dòng
    lines = md_no_code.splitlines()
    output_lines = []

    for line in lines:
        stripped = line.strip()
        # Kiểm tra heading hợp lệ: #, ##, ..., ###### + space/tab
        if HEADING_PATTERN.match(stripped):
            output_lines.append(stripped)
        else:
            output_lines.append(line)

    # Bước 3: Xử lý xen kẽ heading và nội dung
    result = []
    buffer = []

    for line in output_lines:
        stripped = line.strip()
        if HEADING_PATTERN.match(stripped):
            # Gửi buffer đi xử lý (nếu có)
            if buffer:
                md_part = "\n".join(buffer)
                html = MarkdownIt("commonmark").render(md_part)
                text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
                if text:
                    result.append(text)
                buffer = []
            result.append(stripped)
        else:
            buffer.append(line)

    # Xử lý phần cuối
    if buffer:
        md_part = "\n".join(buffer)
        html = MarkdownIt("commonmark").render(md_part)
        text = BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)
        if text:
            result.append(text)

    return "\n".join(result)


def clean_content(text: str, to_text: bool = False) -> str:
	"""Clean content with optional Markdown→text conversion.

	- Always replace '<br>' with a space.
	- If to_text=True, convert to plain text; otherwise keep Markdown as-is.
	"""
	step1 = text.replace("<br>", " ")
	step1 = step1.replace("<ul><li>", "- ")
	step1 = step1.replace("</li><li>", " - ")
	step1 = step1.replace("</li></ul>", "")
 
 
	while "  " in step1:
		step1 = step1.replace("  ", " ")
	while "** " in step1:
		step1 = step1.replace("** ", "**\n")
	while "---" in step1:
		step1 = step1.replace("---", "-")
		
	return step1

def process_file(src: Path, dst: Path, overwrite: bool, dry_run: bool, to_text: bool) -> str:

	content = src.read_text(encoding="utf-8")
	cleaned = clean_content(content, to_text=to_text)
	# dst.write_text(cleaned, encoding="utf-8")
	return cleaned


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
		dst = md_path.parent / f"file_{idx}.md"
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
	p.add_argument("--base_dir", default="src/data/pdfs", help="Base directory containing page_<n> folders")
	p.add_argument("--pattern", default="file_2.md", help="Glob pattern for page folders")
	p.add_argument("--overwrite", action="store_true", help="Overwrite existing cleaned files")
	p.add_argument("--dry_run", action="store_true", help="Show actions without writing files")
	p.add_argument("--to_text", action="store_true", help="After replacing <br>, convert Markdown to plain text (tables will be flattened)")
	return p.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	run(Path(args.base_dir), args.pattern, overwrite=args.overwrite, dry_run=args.dry_run, to_text=args.to_text)


if __name__ == "__main__":
	main()

