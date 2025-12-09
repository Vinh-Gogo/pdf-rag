#!/usr/bin/env python3
"""
Script to compare similarity between a question and text chunks from JSONL file
using QwenEmbedding model.
"""

import json
import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.models.halong_embedd import HalongEmbedding as QwenEmbedding
from src.models.embedd import QwenEmbedding
from src.models.dangvantuan_embedd import DangVanTuanEmbedding as QwenEmbedding

def load_chunks_from_jsonl(jsonl_path):
    """
    Load chunks from JSONL file and extract text fields.

    Args:
        jsonl_path (str): Path to JSONL file

    Returns:
        list[dict]: List of chunk dictionaries with text and metadata
    """
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                chunk = json.loads(line.strip())
                chunks.append(chunk)
    return chunks

def main():
    # File paths
    jsonl_file = "_chunks_dbg/BAN TIN BIWASE T1-2025 -A4 V2.chunks.jsonl"

    print("=" * 80)
    print("BIWASE Chunks Similarity Comparison using QwenEmbedding")
    print("=" * 80)

    # Check if JSONL file exists
    if not Path(jsonl_file).exists():
        print(f"Error: JSONL file '{jsonl_file}' not found!")
        return

    # Load chunks
    print(f"Loading chunks from {jsonl_file}...")
    chunks = load_chunks_from_jsonl(jsonl_file)
    texts = [chunk['text'] for chunk in chunks]
    print(f"Loaded {len(chunks)} chunks.")

    # Initialize embedding model
    print("Initializing QwenEmbedding model...")
    embedding_model = QwenEmbedding()

    # Get question from user
    print("\n" + "=" * 80)
    question = input("Enter your question: ").strip()
    if not question:
        print("Error: Question cannot be empty!")
        return

    print(f"\nQuestion: {question}")
    print("=" * 80)

    # Find most similar chunks
    print("Computing similarities...")
    top_results = embedding_model.find_most_similar(question, texts, top_k=5)

    # Display results
    print(f"\nTop 5 most similar chunks:")
    print("=" * 80)

    for rank, (text, score, idx) in enumerate(top_results, 1):
        chunk = chunks[idx]
        print(f"\n{rank}. Similarity Score: {score:.4f}")
        print(f"   Chunk ID: {chunk['chunk_id']}")
        print(f"   Page: {chunk['page_start']}-{chunk['page_end']}")
        print(f"   Text Preview: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
        print("-" * 80)

    print("\nDone!")

if __name__ == "__main__":
    main()
