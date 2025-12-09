#!/usr/bin/env python3
"""
Test script for HalongEmbedding model
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from src.models.halong_embedd import HalongEmbedding

def main():
    print("Testing HalongEmbedding model...")

    # Initialize model
    embedding_model = HalongEmbedding()

    # Test similarity
    text1 = "BIWASE là công ty nước sạch"
    text2 = "Công ty BIWASE cung cấp nước"

    similarity = embedding_model.calculate_similarity(text1, text2)
    print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")

    # Test with chunks
    docs = [
        "BIWASE quản lý hệ thống cấp nước tại Bình Dương",
        "Công ty TNHH MTV Sản xuất nước sạch BIWASE",
        "Doanh thu của BIWASE năm 2024"
    ]

    top_results = embedding_model.find_most_similar(text1, docs, top_k=3)
    print("\nTop similar results:")
    for i, (doc, score, idx) in enumerate(top_results, 1):
        print(f"{i}. Score: {score:.4f} - {doc}")

    print("Test completed successfully!")

if __name__ == "__main__":
    main()
