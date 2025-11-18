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
    retrieve_bm25_sequences,
    retrieve_hybrid_sequences,
    display_sequences_results,
    read_sequences_from_directory,
    build_bm25_index
)
from src.helpers.vectorstore_from_pages import (
    retrieve_similar_pages,
    retrieve_bm25_pages,
    retrieve_hybrid_pages,
    display_page_results,
    read_pages_from_directory,
    build_bm25_pages_index
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


def display_summary(pages_all: Dict[str, List[Dict]], sequences_all: Dict[str, List[Dict]]):
    """
    Hiển thị tóm tắt kết quả retrieval
    
    Args:
        pages_results: Kết quả từ retrieve_similar_pages
        sequences_results: Kết quả từ retrieve_similar_sequences
    """
    print(f"\n{'='*80}")
    print("� TÓM TẮT KẾT QUẢ RETRIEVAL")
    print("="*80)
    
    # Helper để in tóm tắt cho từng mode
    def _summary_block(title: str, results: List[Dict]):
        if not results:
            print(f"   {title}: (không có kết quả)")
            return
        page_indices = [r.get('page_index') for r in results if r.get('page_index') != 'N/A']
        avg_score = sum(r['similarity_score'] for r in results) / len(results)
        print(f"   {title}: {len(results)} kết quả | Trang: {page_indices} | Avg score: {avg_score:.4f}")

    # Tóm tắt Pages theo từng chế độ
    if pages_all:
        print(f"\n📄 PAGES:")
        _summary_block('Dense', pages_all.get('dense', []))
        _summary_block('BM25', pages_all.get('bm25', []))
        _summary_block('Hybrid', pages_all.get('hybrid', []))
    
    # Tóm tắt Sequences
    if sequences_all:
        print(f"\n📝 SEQUENCES:")
        for mode, results in sequences_all.items():
            if not results:
                print(f"   {mode}: (không kết quả)")
                continue
            unique_pages = get_unique_page_indices(results)
            avg_score = sum(r['similarity_score'] for r in results) / len(results)
            print(f"   {mode}: {len(results)} seq | Pages {unique_pages} | Avg {avg_score:.4f}")
    
    print("="*80)


_PAGES_CORPUS: List[Dict] = []
_SEQUENCES_CORPUS: List[Dict] = []
_BM25_PAGES_READY = False
_BM25_SEQS_READY = False


def retrieve_both(query: str, top_k_pages: int = 5, top_k_sequences: int = 10, alpha_pages: float = 0.6, alpha_sequences: float = 0.6):
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
    
    # Retrieval từ esg_pages (3 modes)
    print(f"\n1️⃣ Retrieval PAGES (Top-{top_k_pages} mỗi mode):")
    print("-" * 80)
    pages_dense = []
    pages_bm25 = []
    pages_hybrid = []
    try:
        pages_dense = retrieve_similar_pages(query, None, top_k=top_k_pages, collection_name="esg_pages")
        # Pass corpus so BM25 can build if not built
        pages_bm25 = retrieve_bm25_pages(query, top_k=top_k_pages, pages=_PAGES_CORPUS)
        pages_hybrid = retrieve_hybrid_pages(query, top_k=top_k_pages, alpha=alpha_pages, pages=_PAGES_CORPUS, collection_name="esg_pages")
        print(f"✅ Pages: Dense={len(pages_dense)}, BM25={len(pages_bm25)}, Hybrid={len(pages_hybrid)}")
    except Exception as e:
        print(f"❌ Lỗi pages: {e}")

    # Retrieval từ esg_sequences (3 modes)
    print(f"\n2️⃣ Retrieval SEQUENCES (Top-{top_k_sequences} mỗi mode):")
    print("-" * 80)
    seq_dense = []
    seq_bm25 = []
    seq_hybrid = []
    try:
        seq_dense = retrieve_similar_sequences(query, collection_name="esg_sequences", top_k=top_k_sequences)
        seq_bm25 = retrieve_bm25_sequences(query, top_k=top_k_sequences, sequences=_SEQUENCES_CORPUS)
        seq_hybrid = retrieve_hybrid_sequences(query, collection_name="esg_sequences", top_k=top_k_sequences, alpha=alpha_sequences, sequences=_SEQUENCES_CORPUS)
        print(f"✅ Sequences: Dense={len(seq_dense)}, BM25={len(seq_bm25)}, Hybrid={len(seq_hybrid)}")
    except Exception as e:
        print(f"❌ Lỗi sequences: {e}")

    pages_all = {
        'dense': pages_dense,
        'bm25': pages_bm25,
        'hybrid': pages_hybrid
    }
    sequences_all = {
        'dense': seq_dense,
        'bm25': seq_bm25,
        'hybrid': seq_hybrid
    }
    return pages_all, sequences_all


def main():
    print("="*80)
    print("🚀 DEMO RETRIEVAL TỪ ESG_PAGES & ESG_SEQUENCES")
    print("="*80)
    print("📚 Collection 1: esg_pages - Toàn bộ nội dung từng trang")
    print("📝 Collection 2: esg_sequences - Các đoạn văn (sequences) nhỏ hơn")
    print("="*80)

    # Các câu hỏi test
    test_queries = [
        "Triển khai các cam kết chính sách",
        "Tiêu thụ năng lượng trong tổ chức",
        "Nguồn cung cấp nước",
        "Giảm phát thải khí nhà kính",
        "Số lượng nhân viên thuê mới và tỷ lệ thôi việc",
        "Sự tham gia của người lao động, tham vấn và truyền thông về an toàn và sức khỏe nghề nghiệp",
        "Số giờ đào tạo trung bình hàng năm của mỗi nhân viên"
    ]

    # Load corpora & build BM25 indices once
    global _PAGES_CORPUS, _SEQUENCES_CORPUS, _BM25_PAGES_READY, _BM25_SEQS_READY
    pages_dir = project_root / "src" / "data" / "contents"
    seqs_dir = project_root / "src" / "data" / "contents/md_to_plain_text"
    try:
        _PAGES_CORPUS = read_pages_from_directory(str(pages_dir))
        build_bm25_pages_index(_PAGES_CORPUS)
        _BM25_PAGES_READY = True
        print(f"🔎 BM25 Pages index built: {_BM25_PAGES_READY} | {len(_PAGES_CORPUS)} pages")
    except Exception as e:
        print(f"❌ Không build được BM25 pages index: {e}")
    try:
        _SEQUENCES_CORPUS = read_sequences_from_directory(str(seqs_dir), min_words=2)
        build_bm25_index(_SEQUENCES_CORPUS)
        _BM25_SEQS_READY = True
        print(f"🔎 BM25 Sequences index built: {_BM25_SEQS_READY} | {len(_SEQUENCES_CORPUS)} sequences")
    except Exception as e:
        print(f"❌ Không build được BM25 sequences index: {e}")

    print(f"\n🧪 TEST RETRIEVAL VỚI {len(test_queries)} CÂU HỎI")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'#'*80}")
        print(f"CÂU HỎI {i}/{len(test_queries)}")
        print('#'*80)
        
        # Thực hiện retrieval trên cả 2 collections
        pages_all, sequences_all = retrieve_both(
            query,
            top_k_pages=8,
            top_k_sequences=12,
            alpha_pages=0.6,
            alpha_sequences=0.6
        )

        # Hiển thị chi tiết: mỗi mode lấy top 3 để tránh quá dài
        def _show_mode(title: str, results: List[Dict], page: bool = False):
            if not results:
                print(f"\n{title}: (không có kết quả)")
                return
            print(f"\n{title} (top {min(3, len(results))}):")
            if page:
                display_page_results(results[:3])
            else:
                display_sequences_results(results[:3])

        # _show_mode("[PAGES Dense]", pages_all['dense'], page=True)
        # _show_mode("[PAGES BM25]", pages_all['bm25'], page=True)
        # _show_mode("[PAGES Hybrid]", pages_all['hybrid'], page=True)
        # _show_mode("[SEQUENCES Dense]", sequences_all['dense'])
        # _show_mode("[SEQUENCES BM25]", sequences_all['bm25'])
        # _show_mode("[SEQUENCES Hybrid]", sequences_all['hybrid'])

        # Tóm tắt tổng hợp
        display_summary(pages_all, sequences_all)
        
        # Hiển thị index để dễ sử dụng
        print(f"\n💡 HƯỚNG DẪN SỬ DỤNG KẾT QUẢ:")
        # Gợi ý đọc từ hybrid vì cân bằng nhất
        hybrid_pages = pages_all.get('hybrid', [])
        if hybrid_pages:
            page_indices = [r['page_index'] for r in hybrid_pages if r['page_index'] != 'N/A']
            print(f"   📄 Khuyến nghị đọc trước các trang (Hybrid): {page_indices[:10]}")
        hybrid_sequences = sequences_all.get('hybrid', [])
        if hybrid_sequences:
            unique_pages = get_unique_page_indices(hybrid_sequences)
            print(f"   📝 Các trang chứa sequences quan trọng (Hybrid): {unique_pages[:10]}")
            for page_idx in unique_pages[:3]:
                seqs = [r for r in hybrid_sequences if r['page_index'] == page_idx]
                seq_info = [(r['seq_index'], r['similarity_score']) for r in seqs if r['seq_index'] != 'N/A']
                print(f"      - Page {page_idx}: {len(seq_info)} sequences relevants")
    
    print(f"\n\n{'='*80}")
    print("✅ HOÀN THÀNH DEMO RETRIEVAL!")
    print("="*80)

if __name__ == "__main__":
    main()