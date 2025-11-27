#!/usr/bin/env python3
"""
Clean up markdown files generated from PDF processing.
- Replace <br> tags with spaces
- Replace -- with -
- Remove image reference lines like ![](_page_0_Picture_12.jpeg)
- Remove specific unwanted lines
"""

import os
import re
import glob
from pathlib import Path

# Lines to remove
UNWANTED_LINES = [
    "Thông tin về BIWASE",
    "Nền tảng phát triển bền vững",
    "Găn kết các bên liên quan",
    "Các chủ đề trọng yếu",
    "Kinh tế tuần hoàn Củng cố",
    "quản trị xanh",
    "Đảm bảo cho người lao động",
    "Lan tỏa giá trị cộng đồng và xã hội",
    "**CHƯƠNG 1 CHƯƠNG 2 CHƯƠNG 3 CHƯƠNG 4 CHƯƠNG 5 CHƯƠNG 6 CHƯƠNG 7 CHƯƠNG 8**",
    "Lan tỏa giá trị cộng đồng và xã hội",
    
    "Kinh tế tuần hoàn"
]

def clean_markdown_file(filepath):
    """Clean up a single markdown file according to the rules."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Replace <br> with space
        content = content.replace('<br>', ' ')

        # Replace all -- with -
        while '--' in content:
            content = content.replace('--', '-')

        # Split into lines for processing
        lines = content.split('\n')
        cleaned_lines = []

        for line in lines:
            line_stripped = line.strip()

            # Skip lines that are just image references
            if re.match(r'^!\[\]\(_page_\d+_Picture_\d+\.jpeg\)$', line_stripped):
                continue

            # Skip unwanted lines (check if any unwanted text is in the line)
            skip_line = False
            for unwanted in UNWANTED_LINES:
                if unwanted in line_stripped:
                    skip_line = True
                    break
            if skip_line:
                continue

            cleaned_lines.append(line)

        # Join lines back
        cleaned_content = '\n'.join(cleaned_lines)

        # Write back to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"Cleaned: {filepath}")
        return True

    except Exception as e:
        print(f"Error cleaning {filepath}: {e}")
        return False

def main():
    """Clean up all markdown files in src/data/markdown directory."""
    markdown_dir = Path("src/data/markdown")

    if not markdown_dir.exists():
        print(f"Directory {markdown_dir} does not exist!")
        return

    # Find all .md files
    md_files = list(markdown_dir.rglob("*.md"))

    if not md_files:
        print("No markdown files found!")
        return

    print(f"Found {len(md_files)} markdown files to clean up")

    cleaned_count = 0
    for md_file in md_files:
        if clean_markdown_file(md_file):
            cleaned_count += 1

    print(f"Successfully cleaned {cleaned_count} out of {len(md_files)} files")

if __name__ == "__main__":
    main()
