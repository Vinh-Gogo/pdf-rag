"""
Ví dụ sử dụng retrieval để tìm sequences tương đồng từ câu hỏi

Usage:
    python src/store/retrieval_example.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helpers.init_qdrant import create_qdrant_vectorstore
from src.helpers.pages_to_vec_store import display_page_retrieval_results

def main():
    print("🚀 DEMO RETRIEVAL SEQUENCES TỪ QDRANT")
    print("=" * 60)

    # Kết nối đến vector store đã tạo
    try:
        vectorstore = create_qdrant_vectorstore("esg_sequences")
        print("✅ Đã kết nối đến vector store 'esg_sequences'")
    except Exception as e:
        print(f"❌ Lỗi kết nối: {e}")
        return

    # Các câu hỏi test
    test_queries = [
        "phát triển bền vững là gì?",
        "báo cáo ESG bao gồm những nội dung gì?",
        "công ty quản lý rủi ro như thế nào?",
        "biện pháp phòng chống tham nhũng",
        "mục tiêu phát triển bền vững",
        "báo cáo tài chính và ESG"
    ]

    print(f"\n🧪 TEST RETRIEVAL VỚI {len(test_queries)} CÂU HỎI:")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}/{len(test_queries)}: '{query}'")
        print("-" * 40)

        # Retrieval
        # results = retrieve_similar_sequences(query, vectorstore, top_k=5)

        # Hiển thị kết quả
        # display_retrieval_results(results)

        print("\n" + "=" * 60)

    print("✅ HOÀN THÀNH DEMO RETRIEVAL!")

if __name__ == "__main__":
    main()