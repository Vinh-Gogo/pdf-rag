"""Post-process existing .txt content files to normalize heading spacing.

Task requirements (Vietnamese summary):
  - Mỗi dòng title bắt đầu bằng dấu '#'. Sau title hiện có một đoạn trống ("\n\n")
	cần xóa khoảng trống đó: heading nối tiếp ngay nội dung phía sau (chỉ một '\n').
  - Giữ nguyên bảng (table) và các dòng khác.
  - Ghi đè lên file cũ.

Features:
  - Detect markdown headings (lines starting with '#').
  - Remove any following blank lines until the next non-empty line or EOF.
  - Preserve internal blank lines between normal paragraphs.
  - Optionally restrict processing to a set of page indices via --include.
  - Dry-run mode to preview changes.

Usage:
  python -m src.helpers.text-to-text \
	--dir src/data/contents \
	--pattern "page_cleared_*.txt" \
	--include 158,159,160,161,162,163 \
	--overwrite

Exit codes:
  0 Success (even if no changes)
  1 Error (e.g. directory not found)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional

HEADING_RE = re.compile(r"^\s*#{1,6}\s")


def iter_files(base: Path, pattern: str) -> Iterable[Path]:
	yield from base.glob(pattern)


def should_process(path: Path, include_indices: Optional[set[int]]) -> bool:
	if include_indices is None:
		return True
	m = re.search(r"(\d+)", path.stem)
	if not m:
		return False
	return int(m.group(1)) in include_indices


def normalize_headings(text: str) -> tuple[str, bool]:
	lines = text.splitlines()
	out: List[str] = []
	i = 0
	changed = False
	n = len(lines)
	while i < n:
		line = lines[i]
		out.append(line)
		if HEADING_RE.match(line):
			# Skip following blank lines
			j = i + 1
			skipped_any = False
			while j < n and lines[j].strip() == "":
				skipped_any = True
				j += 1
			if skipped_any:
				changed = True
			i = j
			continue
		i += 1
	new_text = "\n".join(out).rstrip() + "\n"
	return new_text, changed


def process_file(path: Path, dry_run: bool) -> bool:
	try:
		original = path.read_text(encoding="utf-8", errors="ignore")
	except Exception as e:
		print(f"✗ read failed {path}: {e}")
		return False
	new_text, changed = normalize_headings(original)
	if not changed:
		print(f"[no-change] {path}")
		return True
	if dry_run:
		print(f"[DRY RUN] would update {path}")
		return True
	try:
		path.write_text(new_text, encoding="utf-8")
		print(f"[updated] {path}")
		return True
	except Exception as e:
		print(f"✗ write failed {path}: {e}")
		return False


def run(base_dir: Path, pattern: str, include: Optional[str], overwrite: bool, dry_run: bool) -> None:
	if not base_dir.exists():
		raise SystemExit(f"Directory not found: {base_dir}")
	include_set: Optional[set[int]] = None
	if include:
		try:
			include_set = {int(x.strip()) for x in include.split(',') if x.strip()}
		except ValueError:
			raise SystemExit("Invalid --include list; must be comma-separated integers")
	files = [p for p in iter_files(base_dir, pattern) if p.is_file() and should_process(p, include_set)]
	if not files:
		print("No files matched pattern/include.")
		return
	print(f"Processing {len(files)} files…")
	updated = 0
	for f in sorted(files):
		ok = process_file(f, dry_run=dry_run)
		if ok:
			updated += 1
	print("-" * 80)
	print(f"Done. Files processed: {len(files)}, Updated (or no-change ok): {updated}, Dry-run: {dry_run}")


def parse_args(argv=None):
	p = argparse.ArgumentParser(description="Normalize heading spacing in text files")
	p.add_argument("--dir", default="src/data/contents", help="Directory containing .txt files")
	p.add_argument("--pattern", default="page_cleared_*.txt", help="Glob pattern for files")
	p.add_argument("--include", help="Comma-separated page indices to restrict processing (e.g. 158,159,160)")
	p.add_argument("--overwrite", action="store_true", help="(Reserved) Compatibility flag; always overwrites")
	p.add_argument("--dry_run", action="store_true", help="Preview changes without writing")
	return p.parse_args(argv)


def main(argv=None):
	args = parse_args(argv)
	run(Path(args.dir), args.pattern, args.include, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
	main()
