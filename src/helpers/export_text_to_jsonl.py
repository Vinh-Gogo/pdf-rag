"""
Export text data to JSONL format with similarity scoring.

This script exports paired text files (raw and cleaned) to JSONL format,
calculating similarity scores between them using embedding models.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.embedd import QwenEmbedding


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
            return f.read().strip()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""


def export_to_jsonl(
    file_pairs: List[Tuple[int, Path, Path]],
    output_path: Path,
    embedding_model: QwenEmbedding
) -> None:
    """
    Export file pairs to JSONL format with similarity scoring.
    
    Args:
        file_pairs: List of (page_number, raw_file, cleaned_file) tuples
        output_path: Path to output JSONL file
        embedding_model: Embedding model for similarity calculation
    """
    print(f"\n📝 Exporting {len(file_pairs)} page pairs to JSONL...")
    print(f"📂 Output: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for page_num, raw_file, cleaned_file in tqdm(file_pairs, desc="Processing pages"):
            # Read content from both files
            raw_text = read_text_file(raw_file)
            cleaned_text = read_text_file(cleaned_file)
            
            # Skip if either file is empty
            if not raw_text or not cleaned_text:
                print(f"⚠️ Skipping page {page_num}: empty content")
                continue
            
            # Calculate similarity score
            similarity_score = embedding_model.calculate_similarity(raw_text, cleaned_text)
            
            # Create JSON object
            json_obj = {
                "page": page_num,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "similar_score": round(float(similarity_score), 4)
            }
            
            # Write as single line JSON
            f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"✅ Export completed: {output_path}")


def main():
    """Main execution function."""
    
    # Define paths
    project_root = Path(__file__).parent.parent.parent
    contents_dir = project_root / "src" / "data" / "contents"
    grammar_dir = project_root / "src" / "data" / "results" / "grammar"
    output_dir = project_root / "src" / "exports"
    output_file = output_dir / "text_comparison.jsonl"
    
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
    print("📊 TEXT COMPARISON EXPORT TO JSONL")
    print("=" * 70)
    
    # Initialize embedding model
    print("\n🔧 Initializing embedding model...")
    embedding_model = QwenEmbedding()
    
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
    export_to_jsonl(file_pairs, output_file, embedding_model)
    
    # Show statistics
    print("\n" + "=" * 70)
    print("📈 EXPORT STATISTICS")
    print("=" * 70)
    
    # Read and analyze the exported file
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    similarities = []
    for line in lines:
        data = json.loads(line)
        similarities.append(data['similar_score'])
    
    print(f"Total records: {len(lines)}")
    print(f"Average similarity: {sum(similarities) / len(similarities):.4f}")
    print(f"Min similarity: {min(similarities):.4f}")
    print(f"Max similarity: {max(similarities):.4f}")
    
    print("\n✅ All operations completed successfully!")


if __name__ == "__main__":
    main()
