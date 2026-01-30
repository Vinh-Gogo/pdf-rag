import os
import re
from pathlib import Path

def load_stopwords(stopwords_path: str = 'src/data/stopwords/stopwords-vietnamese.txt') -> set:
    """Load stopwords from file into a set for fast lookup."""
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def remove_stopwords_from_text(text: str, stopwords: set) -> str:
    """Remove stopwords from Vietnamese text while preserving markdown structure."""
    # Split text into words, keeping punctuation attached
    words = re.findall(r'\b\w+\b', text)

    # Reconstruct text by replacing original words
    result = text
    for word in words:
        if word.lower() in stopwords:
            # Remove the word and any surrounding whitespace
            result = re.sub(r'\b' + re.escape(word) + r'\b', '', result)

    # Clean up extra whitespace but preserve newlines and paragraph breaks
    result = re.sub(r'[ \t]+', ' ', result)  # Replace multiple spaces/tabs with single space
    # result = re.sub(r'(?<!\n)\n\s+', '\n', result)  # Remove spaces after single newlines
    # result = re.sub(r'\s+\n(?!\n)', '\n', result)  # Remove spaces before single newlines
    result = result.strip()

    return result

def process_markdown_file(input_path: str, output_path: str, stopwords: set):
    """Process a single markdown file to remove stopwords."""
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            content = infile.read()
            processed_content = remove_stopwords_from_text(content, stopwords)
            processed_content = processed_content.replace('**', '')
            outfile.write(processed_content.strip())

    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    input_dir = Path('_pdf_md')
    output_dir = Path('_pdf_md_no_stopwords')

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stopwords
    stopwords_path = Path('src/data/stopwords/stopwords-vietnamese.txt')
    stopwords = load_stopwords(str(stopwords_path))
    print(f"Loaded {len(stopwords)} stopwords")

    # Process all markdown files
    md_files = list(input_dir.glob('*.md'))
    print(f"Found {len(md_files)} markdown files to process")

    for input_file in md_files:
        output_file = output_dir / input_file.name
        print(f"Processing {input_file.name} -> {output_file.name}")
        process_markdown_file(str(input_file), str(output_file), stopwords)

    print("Stopwords removal completed!")

if __name__ == '__main__':
    main()
