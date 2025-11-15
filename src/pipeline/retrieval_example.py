"""
Ví dụ sử dụng retrieval để tìm sequences và pages tương đồng từ câu hỏi

Usage:
    python src/pipeline/retrieval_example.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables
load_dotenv()

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helpers.vectorstore_from_sequences import (
    retrieve_similar_sequences,
    display_retrieval_results as display_sequences_results
)
from src.helpers.vectorstore_from_pages import (
    retrieve_similar_pages,
    display_page_retrieval_results
)


def get_unique_page_indices(sequences_results: List[Dict]) -> List[int]:
    """
    Trích xuất các page indices duy nhất từ kết quả sequences retrieval
    
    Args:
        sequences_results: Kết quả từ retrieve_similar_sequences
        
    Returns:
        List[int]: Danh sách page indices duy nhất, sắp xếp theo thứ tự
    """
    page_indices = set()
    for result in sequences_results:
        if result.get('page_index') != 'N/A':
            page_indices.add(result['page_index'])
    return sorted(list(page_indices))


def display_summary(pages_results: List[Dict], sequences_results: List[Dict]):
    """
    Hiển thị tóm tắt kết quả retrieval
    
    Args:
        pages_results: Kết quả từ retrieve_similar_pages
        sequences_results: Kết quả từ retrieve_similar_sequences
    """
    print(f"\n{'='*80}")
    print("� TÓM TẮT KẾT QUẢ RETRIEVAL")
    print("="*80)
    
    # Tóm tắt Pages
    if pages_results:
        print(f"\n📄 PAGES (Top-{len(pages_results)}):")
        page_indices = [r['page_index'] for r in pages_results if r['page_index'] != 'N/A']
        print(f"   Tìm thấy ở các trang: {page_indices}")
        avg_score = sum(r['similarity_score'] for r in pages_results) / len(pages_results)
        print(f"   Điểm trung bình: {avg_score:.4f}")
    
    # Tóm tắt Sequences
    if sequences_results:
        print(f"\n📝 SEQUENCES (Top-{len(sequences_results)}):")
        unique_pages = get_unique_page_indices(sequences_results)
        print(f"   Tìm thấy sequences ở {len(unique_pages)} trang: {unique_pages}")
        
        # Hiển thị chi tiết sequences theo page
        for page_idx in unique_pages:
            seqs = [r for r in sequences_results if r['page_index'] == page_idx]
            seq_indices = [r['seq_index'] for r in seqs if r['seq_index'] != 'N/A']
            print(f"   - Page {page_idx}: Sequences {seq_indices}")
        
        avg_score = sum(r['similarity_score'] for r in sequences_results) / len(sequences_results)
        print(f"   Điểm trung bình: {avg_score:.4f}")
    
    print("="*80)


def retrieve_both(query: str, top_k_pages: int = 5, top_k_sequences: int = 10):
    """
    Thực hiện retrieval trên cả hai collections: esg_pages và esg_sequences
    
    Args:
        query (str): Câu hỏi tìm kiếm
        top_k_pages (int): Số lượng pages trả về
        top_k_sequences (int): Số lượng sequences trả về
        
    Returns:
        tuple: (pages_results, sequences_results)
    """
    print(f"\n{'='*80}")
    print(f"🔍 QUERY: '{query}'")
    print("="*80)
    
    # Retrieval từ esg_pages
    print(f"\n1️⃣ Retrieval từ ESG_PAGES (Top-{top_k_pages}):")
    print("-" * 80)
    try:
        pages_results = retrieve_similar_pages(query, None, top_k=top_k_pages)
        print(f"✅ Tìm thấy {len(pages_results)} pages")
    except Exception as e:
        print(f"❌ Lỗi khi retrieval pages: {e}")
        pages_results = []
    
    # Retrieval từ esg_sequences
    print(f"\n2️⃣ Retrieval từ ESG_SEQUENCES (Top-{top_k_sequences}):")
    print("-" * 80)
    try:
        sequences_results = retrieve_similar_sequences(
            query, 
            collection_name="esg_sequences",
            top_k=top_k_sequences
        )
        print(f"✅ Tìm thấy {len(sequences_results)} sequences")
    except Exception as e:
        print(f"❌ Lỗi khi retrieval sequences: {e}")
        sequences_results = []
    
    return pages_results, sequences_results


def main():
    print("="*80)
    print("🚀 DEMO RETRIEVAL TỪ ESG_PAGES & ESG_SEQUENCES")
    print("="*80)
    print("📚 Collection 1: esg_pages - Toàn bộ nội dung từng trang")
    print("📝 Collection 2: esg_sequences - Các đoạn văn (sequences) nhỏ hơn")
    print("="*80)

    # Các câu hỏi test
    test_queries = [
        "Vốn điều lệ của công ty là bao nhiêu?",
        "Thể dục thể thao sức khỏe dồi dào, siêng năng mà luyện tập",
        "doanh thu của doanh nghiệp trong năm 2024",
        "Biện pháp phòng chống tham nhũng",
        "Hệ thống quản lý môi trường của công ty",
    ]

    print(f"\n🧪 TEST RETRIEVAL VỚI {len(test_queries)} CÂU HỎI")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"CÂU HỎI {i}/{len(test_queries)}")
        print('#'*80)
        
        # Thực hiện retrieval trên cả 2 collections
        pages_results, sequences_results = retrieve_both(
            query,
            top_k_pages=10,
            top_k_sequences=10
        )
        
        # Hiển thị kết quả chi tiết
        # if pages_results:
        #     print(f"\n{'='*80}")
        #     print("📄 CHI TIẾT KẾT QUẢ PAGES:")
        #     display_page_retrieval_results(pages_results)
        
        # if sequences_results:
        #     print(f"\n{'='*80}")
        #     print("📝 CHI TIẾT KẾT QUẢ SEQUENCES:")
        #     print("="*80)
        #     display_sequences_results(sequences_results)
        
        # Hiển thị tóm tắt
        display_summary(pages_results, sequences_results)
        
        # Hiển thị index để dễ sử dụng
        print(f"\n💡 HƯỚNG DẪN SỬ DỤNG KẾT QUẢ:")
        if pages_results:
            page_indices = [r['page_index'] for r in pages_results if r['page_index'] != 'N/A']
            print(f"   📄 Đọc toàn bộ nội dung ở các trang: {page_indices}")
        
        if sequences_results:
            unique_pages = get_unique_page_indices(sequences_results)
            print(f"   📝 Hoặc đọc các đoạn cụ thể ở {len(unique_pages)} trang: {unique_pages}")
            for page_idx in unique_pages[:3]:  # Hiển thị tối đa 3 pages
                seqs = [r for r in sequences_results if r['page_index'] == page_idx]
                seq_info = [(r['seq_index'], r['similarity_score']) for r in seqs if r['seq_index'] != 'N/A']
                print(f"      - Page {page_idx}: {len(seq_info)} sequences relevants")
    
    print(f"\n\n{'='*80}")
    print("✅ HOÀN THÀNH DEMO RETRIEVAL!")
    print("="*80)


if __name__ == "__main__":
    main()