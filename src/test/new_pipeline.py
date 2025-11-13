from __future__ import annotations
import argparse
import re
from pathlib import Path


def extract_page_index(name: str) -> int | None:
	"""Extract trailing integer from a string like 'page_123' or 'page_123.md'."""
	m = re.search(r"(\d+)(?:\.md)?$", name)
	return int(m.group(1)) if m else None


def list_markdown_files(base_dir: str | Path = "src/data/markdown") -> list[Path]:
	base = Path(base_dir)
	results: list[tuple[int, Path]] = []

	if not base.exists():
		return []

	# Expect structure: base/page_<n>/page_<n>.md
	for sub in base.iterdir():
		if not sub.is_dir():
			continue
		idx = extract_page_index(sub.name)
		if idx is None:
			continue
		md_path = sub / f"page_{idx}.md"
		if md_path.exists():
			results.append((idx, md_path))

	# Sort ascending by numeric index
	results.sort(key=lambda t: t[0])
	return [p for _, p in results]


def main(argv=None):
	parser = argparse.ArgumentParser(description="List all markdown files sorted by page index.")
	parser.add_argument(
		"--base_dir",
		default="src/data/markdown",
		help="Base directory containing page_<n> subfolders",
	)
	args = parser.parse_args(argv)

	files = list_markdown_files(args.base_dir)
	for p in files:
		print(str(p))


if __name__ == "__main__":
	main()

# pandoc -s src\data\markdown\page_161\page_161.md -o src\data\markdown\page_161\page_161.txt