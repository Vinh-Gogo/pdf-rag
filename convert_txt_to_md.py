#!/usr/bin/env python3
"""
Convert all .txt files in src/data/markdown_clean to .md files
"""

import os
import shutil
from pathlib import Path

def convert_txt_to_md():
    """Convert all .txt files to .md files in the markdown_clean directory."""
    txt_dir = Path("src/data/markdown_clean")
    print(f"Looking for directory: {txt_dir}")
    print(f"Absolute path: {txt_dir.absolute()}")
    print(f"Directory exists: {txt_dir.exists()}")

    if not txt_dir.exists():
        print(f"Directory {txt_dir} does not exist!")
        return

    # Find all .txt files
    txt_files = list(txt_dir.glob("*.txt"))

    if not txt_files:
        print("No .txt files found!")
        return

    print(f"Found {len(txt_files)} .txt files to convert")

    converted_count = 0
    for txt_file in txt_files:
        # Create the corresponding .md file path
        md_file = txt_file.with_suffix('.md')

        try:
            # Copy the .txt file to .md file
            shutil.copy2(txt_file, md_file)
            print(f"Converted: {txt_file.name} -> {md_file.name}")
            converted_count += 1

        except Exception as e:
            print(f"Error converting {txt_file}: {e}")

    print(f"Successfully converted {converted_count} out of {len(txt_files)} files")

    # Automatically delete original .txt files
    if converted_count > 0:
        deleted_count = 0
        for txt_file in txt_files:
            try:
                txt_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {txt_file}: {e}")
        print(f"Deleted {deleted_count} original .txt files")

if __name__ == "__main__":
    convert_txt_to_md()
