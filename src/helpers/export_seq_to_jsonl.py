"""
Export text sequences to JSONL format.

This script exports text sequences (paragraphs) from paired text files 
(raw and cleaned) to JSONL format, splitting by double newlines.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def extract_page_number(filename: str) -> int:
    """
    Extract page number from filename.
    
    Args:
        filename: Filename like 'page_1.txt' or 'page_1_clear.txt'
        
    Returns:
        Page number as integer
    """
    match = re.search(r'page_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0


def get_file_pairs(contents_dir: Path, grammar_dir: Path) -> List[Tuple[int, Path, Path]]:
    """
    Get pairs of raw and cleaned text files, sorted by page number.
    
    Args:
        contents_dir: Directory containing raw text files (page_X_clear.txt)
        grammar_dir: Directory containing cleaned text files (page_X.txt)
        
    Returns:
        List of tuples: (page_number, raw_file_path, cleaned_file_path)
    """
    pairs = []
    
    # Get all _clear.txt files from contents directory
    clear_files = list(contents_dir.glob("page_*_clear.txt"))
    
    for clear_file in clear_files:
        page_num = extract_page_number(clear_file.name)
        
        # Find corresponding cleaned file in grammar directory
        cleaned_file = grammar_dir / f"page_{page_num}.txt"
        
        if cleaned_file.exists():
            pairs.append((page_num, clear_file, cleaned_file))
        else:
            print(f"⚠️ Warning: No cleaned file found for page {page_num}")
    
    # Sort by page number
    pairs.sort(key=lambda x: x[0])
    
    return pairs


def read_text_file(file_path: Path) -> str:
    """
    Read text content from file.
    
    Args:
        file_path: Path to text file
        
    Returns:
        Text content as string
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""


def split_into_sequences(text: str) -> List[str]:
    """
    Split text into sequences by double newlines.
    
    Args:
        text: Text content to split
        
    Returns:
        List of text sequences (paragraphs)
    """
    # Split by double newlines
    sequences = text.split('\n\n')
    
    # Clean and filter empty sequences
    sequences = [seq.strip() for seq in sequences if seq.strip()]
    
    return sequences


def split_raw_text_intelligently(raw_text: str, cleaned_sequences: List[str]) -> List[str]:
    """
    Split raw text to match the number of cleaned sequences.
    Uses intelligent splitting based on sentences and line breaks.
    
    Args:
        raw_text: Raw text content
        cleaned_sequences: List of cleaned sequences to match count
        
    Returns:
        List of raw text sequences matching cleaned sequence count
    """
    target_count = len(cleaned_sequences)
    
    # If raw already has \n\n, use it
    if '\n\n' in raw_text:
        raw_sequences = split_into_sequences(raw_text)
        if len(raw_sequences) == target_count:
            return raw_sequences
    
    # Try splitting by single newline first
    lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
    
    if len(lines) >= target_count:
        # Distribute lines across target sequences
        sequences_per_group = len(lines) // target_count
        remainder = len(lines) % target_count
        
        result = []
        idx = 0
        for i in range(target_count):
            # Some groups get an extra line if there's remainder
            group_size = sequences_per_group + (1 if i < remainder else 0)
            group = lines[idx:idx + group_size]
            result.append('\n'.join(group))
            idx += group_size
        
        return result
    
    # If lines < target, try splitting by sentences
    import re
    # Split by sentence-ending punctuation
    sentences = re.split(r'([.!?。]+\s*)', raw_text)
    sentences = [''.join(sentences[i:i+2]).strip() for i in range(0, len(sentences)-1, 2)]
    sentences = [s for s in sentences if s]
    
    if len(sentences) >= target_count:
        # Distribute sentences
        sentences_per_group = len(sentences) // target_count
        remainder = len(sentences) % target_count
        
        result = []
        idx = 0
        for i in range(target_count):
            group_size = sentences_per_group + (1 if i < remainder else 0)
            group = sentences[idx:idx + group_size]
            result.append(' '.join(group))
            idx += group_size
        
        return result
    
    # Last resort: split by character count
    chars_per_seq = len(raw_text) // target_count
    result = []
    for i in range(target_count):
        start = i * chars_per_seq
        end = start + chars_per_seq if i < target_count - 1 else len(raw_text)
        result.append(raw_text[start:end].strip())
    
    return result


def export_to_jsonl(
    file_pairs: List[Tuple[int, Path, Path]],
    output_path: Path
) -> None:
    """
    Export file pairs to JSONL format with sequences.
    
    Args:
        file_pairs: List of (page_number, raw_file, cleaned_file) tuples
        output_path: Path to output JSONL file
    """
    print(f"\n📝 Exporting sequences from {len(file_pairs)} page pairs to JSONL...")
    print(f"📂 Output: {output_path}")
    
    total_sequences = 0
    skipped_pages = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for page_num, raw_file, cleaned_file in tqdm(file_pairs, desc="Processing pages"):
            # Read content from both files
            raw_text = read_text_file(raw_file)
            cleaned_text = read_text_file(cleaned_file)
            
            # Skip if either file is empty
            if not raw_text or not cleaned_text:
                print(f"⚠️ Skipping page {page_num}: empty content")
                skipped_pages.append(page_num)
                continue
            
            # Split cleaned text first
            cleaned_sequences = split_into_sequences(cleaned_text)
            
            # Split raw text intelligently to match cleaned sequence count
            raw_sequences = split_raw_text_intelligently(raw_text, cleaned_sequences)
            
            # Verify counts match
            if len(raw_sequences) != len(cleaned_sequences):
                print(f"⚠️ Warning: Page {page_num} still has mismatched counts after intelligent split:")
                print(f"   Raw: {len(raw_sequences)}, Cleaned: {len(cleaned_sequences)}")
            
            # Export sequences
            seq_count = min(len(raw_sequences), len(cleaned_sequences))
            
            for chap_idx in range(seq_count):
                json_obj = {
                    "page": page_num,
                    "chap_page": chap_idx + 1,  # 1-indexed
                    "raw_text": raw_sequences[chap_idx],
                    "cleaned_text": cleaned_sequences[chap_idx]
                }
                
                # Write as single line JSON
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                total_sequences += 1
    
    print(f"✅ Export completed: {output_path}")
    print(f"📊 Total sequences exported: {total_sequences}")
    
    if skipped_pages:
        print(f"⚠️ Skipped {len(skipped_pages)} pages: {skipped_pages}")


def main():
    """Main execution function."""
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    contents_dir = project_root / "src" / "data" / "contents"
    grammar_dir = project_root / "src" / "data" / "results" / "grammar"
    output_dir = project_root / "src" / "exports"
    output_file = output_dir / "sequences_comparison.jsonl"
    
    # Verify directories exist
    if not contents_dir.exists():
        print(f"❌ Error: Contents directory not found: {contents_dir}")
        return
    
    if not grammar_dir.exists():
        print(f"❌ Error: Grammar directory not found: {grammar_dir}")
        return
    
    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📊 SEQUENCES COMPARISON EXPORT TO JSONL")
    print("=" * 70)
    
    # Get file pairs
    print("\n🔍 Scanning for file pairs...")
    file_pairs = get_file_pairs(contents_dir, grammar_dir)
    print(f"✓ Found {len(file_pairs)} page pairs")
    
    if len(file_pairs) == 0:
        print("❌ No file pairs found. Exiting.")
        return
    
    # Show page range
    page_numbers = [p[0] for p in file_pairs]
    print(f"📄 Page range: {min(page_numbers)} - {max(page_numbers)}")
    
    # Export to JSONL
    export_to_jsonl(file_pairs, output_file)
    
    # Show statistics
    print("\n" + "=" * 70)
    print("📈 EXPORT STATISTICS")
    print("=" * 70)
    
    # Read and analyze the exported file
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Count sequences per page
    page_seq_counts = {}
    for line in lines:
        data = json.loads(line)
        page = data['page']
        if page not in page_seq_counts:
            page_seq_counts[page] = 0
        page_seq_counts[page] += 1
    
    print(f"Total pages: {len(page_seq_counts)}")
    print(f"Total sequences: {len(lines)}")
    print(f"Average sequences per page: {len(lines) / len(page_seq_counts):.2f}")
    print(f"Min sequences per page: {min(page_seq_counts.values())}")
    print(f"Max sequences per page: {max(page_seq_counts.values())}")
    
    # Show sample pages with most/least sequences
    sorted_pages = sorted(page_seq_counts.items(), key=lambda x: x[1])
    print(f"\nPages with fewest sequences:")
    for page, count in sorted_pages[:3]:
        print(f"  Page {page}: {count} sequences")
    
    print(f"\nPages with most sequences:")
    for page, count in sorted_pages[-3:]:
        print(f"  Page {page}: {count} sequences")
    
    print("\n✅ All operations completed successfully!")


if __name__ == "__main__":
    main()
