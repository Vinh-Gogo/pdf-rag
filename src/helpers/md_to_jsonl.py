"""
Script to process markdown files and convert them to JSONL format.
Splits content by "\n\n\n" delimiter and outputs each chunk as a JSON object.
"""

import json
import argparse
from pathlib import Path


def process_md_to_jsonl(input_file: str, output_file: str = None) -> list[dict]:
    """
    Process a markdown file by splitting on "\n\n\n" and convert to JSONL format.
    
    Args:
        input_file: Path to the input markdown file
        output_file: Path to the output JSONL file (optional, defaults to input_file.jsonl)
    
    Returns:
        List of dictionaries with page and content keys
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = input_path.with_suffix('.jsonl')
    
    output_path = Path(output_file)
    
    # Read the markdown file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "\n\n\n" delimiter
    chunks = content.split("\n\n\n")
    
    # Create list of dictionaries
    results = []
    for i, chunk in enumerate(chunks, start=1):
        # Skip empty chunks
        chunk = chunk.strip()
        if chunk:
            # Replace newlines with spaces for single-line content
            content = chunk.replace('\n', ' ')
            # Remove multiple consecutive spaces
            while '  ' in content:
                content = content.replace('  ', ' ')
            results.append({
                "page": i,
                "content": content
            })
    
    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(results)} chunks from '{input_path}'")
    print(f"Output saved to '{output_path}'")
    
    return results


def process_directory(input_dir: str, output_dir: str = None) -> None:
    """
    Process all markdown files in a directory.
    
    Args:
        input_dir: Path to directory containing markdown files
        output_dir: Path to output directory (optional, defaults to same as input)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    md_files = list(input_path.glob("*.md"))
    
    if not md_files:
        print(f"No markdown files found in '{input_path}'")
        return
    
    print(f"Found {len(md_files)} markdown files")
    
    for md_file in md_files:
        output_file = output_path / md_file.with_suffix('.jsonl').name
        process_md_to_jsonl(str(md_file), str(output_file))


def main():
    parser = argparse.ArgumentParser(
        description="Convert markdown files to JSONL by splitting on '\\n\\n\\n'"
    )
    parser.add_argument(
        "input",
        help="Input markdown file or directory"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL file or directory (defaults to same location as input)",
        default=None
    )
    parser.add_argument(
        "-d", "--directory",
        action="store_true",
        help="Process all markdown files in a directory"
    )
    
    args = parser.parse_args()
    
    if args.directory:
        process_directory(args.input, args.output)
    else:
        process_md_to_jsonl(args.input, args.output)


if __name__ == "__main__":
    main()


"""
# Xử lý một file
python src/helpers/md_to_jsonl.py "src/data/pdfs/outputs_new/file.md"
# Xử lý một file với output cụ thể
python src/helpers/md_to_jsonl.py "src/data/pdfs/outputs_new/file.md" -o "src/data/pdfs/outputs_new/file.jsonl"
# Xử lý tất cả file .md trong thư mục
python src/helpers/md_to_jsonl.py "src/data/pdfs/outputs_new/" -d
"""