"""
Pipeline hoàn chỉnh: PDF -> Split Pages -> Extract Text -> Vector Store

Quy trình:
1. Cắt PDF thành 168 trang đơn lẻ (hoặc số trang chỉ định)
2. Trích xuất text từ mỗi trang
3. Lưu text vào Qdrant Vector Store
4. Test retrieval
"""

import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.helpers.PDFText import PDFTextExtractor
from helpers.vectorstore_from_pages import (
    read_pages_from_directory,
    save_pages_to_json,
    store_pages_in_qdrant_direct,
    retrieve_similar_pages,
    display_page_retrieval_results
)


def run_complete_pipeline(
    pdf_path: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    max_pages: int = 168,
    collection_name: str = "esg_pages",
    skip_split: bool = False,
    skip_extract: bool = False,
    skip_vectorstore: bool = False
):
    """
    Chạy pipeline hoàn chỉnh từ PDF đến Vector Store
    
    Args:
        pdf_path (str): Đường dẫn đến file PDF
        start_page (int): Trang bắt đầu (1-based)
        end_page (Optional[int]): Trang kết thúc (1-based). None = tất cả
        max_pages (int): Số trang tối đa xử lý
        collection_name (str): Tên collection trong Qdrant
        skip_split (bool): Bỏ qua bước cắt PDF (nếu đã cắt rồi)
        skip_extract (bool): Bỏ qua bước trích xuất text (nếu đã có file text)
        skip_vectorstore (bool): Bỏ qua bước lưu vào vector store (chỉ xử lý PDF)
    
    Returns:
        bool: True nếu thành công
    """
    
    # Cấu hình đường dẫn
    output_text_dir = project_root / "src" / "data" / "raw"
    output_pdf_dir = project_root / "src" / "data" / "pdfs" / "pages"
    output_json_dir = project_root / "src" / "data" / "push"
    
    print("="*80)
    print("🚀 PDF TO VECTOR STORE PIPELINE")
    print("="*80)
    print(f"📖 PDF: {pdf_path}")
    print(f"📊 Range: Page {start_page} - {end_page if end_page else 'END'}")
    print(f"📏 Max pages: {max_pages}")
    print(f"🗄️ Collection: {collection_name}")
    print("="*80)
    
    # Initialize extractor
    extractor = PDFTextExtractor(
        pdf_path=pdf_path,
        output_dir=str(output_text_dir),
        split_output_dir=str(output_pdf_dir)
    )
    
    # Get total pages
    total_pages = extractor.get_page_count()
    if total_pages == 0:
        print("❌ Không thể mở PDF hoặc PDF rỗng")
        return False
    
    # Calculate actual end page
    if end_page is None:
        end_page = min(total_pages, max_pages)
    else:
        end_page = min(end_page, total_pages, max_pages)
    
    actual_pages = end_page - start_page + 1
    print(f"📄 Sẽ xử lý {actual_pages} trang (từ {start_page} đến {end_page})")
    
    # ============================================================================
    # STEP 1: Split PDF into individual pages
    # ============================================================================
    if not skip_split:
        print(f"\n{'='*80}")
        print("BƯỚC 1: CẮT PDF THÀNH CÁC TRANG ĐƠN LẺ")
        print("="*80)
        
        split_success = extractor.split_pdf_into_pages(
            start_page=start_page,
            end_page=end_page
        )
        
        if not split_success:
            print("❌ Lỗi khi cắt PDF")
            return False
        
        print(f"✅ Đã cắt {actual_pages} trang vào {output_pdf_dir}")
    else:
        print(f"\n⏭️ Bỏ qua BƯỚC 1: Cắt PDF (skip_split=True)")
    
    # ============================================================================
    # STEP 2: Extract text from each page
    # ============================================================================
    if not skip_extract:
        print(f"\n{'='*80}")
        print("BƯỚC 2: TRÍCH XUẤT TEXT TỪ MỖI TRANG")
        print("="*80)
        
        extract_success = extractor.extract_all_pages(
            start_page=start_page,
            end_page=end_page,
            clean_text=True
        )
        
        if not extract_success:
            print("❌ Lỗi khi trích xuất text")
            return False
        
        print(f"✅ Đã trích xuất text từ {actual_pages} trang vào {output_text_dir}")
    else:
        print(f"\n⏭️ Bỏ qua BƯỚC 2: Trích xuất text (skip_extract=True)")
    
    # ============================================================================
    # STEP 3: Read pages and prepare for vector store
    # ============================================================================
    if not skip_vectorstore:
        print(f"\n{'='*80}")
        print("BƯỚC 3: CHUẨN BỊ DỮ LIỆU CHO VECTOR STORE")
        print("="*80)
        
        pages = read_pages_from_directory(str(output_text_dir))
        
        if not pages:
            print("❌ Không tìm thấy pages để xử lý")
            return False
        
        # Filter pages by range
        pages = [p for p in pages if start_page <= int(p['page_index']) <= end_page]
        
        print(f"✅ Đã đọc {len(pages)} pages")
        
        # Display statistics
        total_words = sum(int(p['word_count']) for p in pages)
        print(f"\n📊 THỐNG KÊ:")
        print(f"   Tổng pages: {len(pages)}")
        print(f"   Tổng từ: {total_words:,}")
        print(f"   Trung bình: {total_words/len(pages):.1f} từ/page")
        
        # Save to JSON
        output_json_dir.mkdir(parents=True, exist_ok=True)
        output_json_file = output_json_dir / "pages_data.json"
        save_pages_to_json(pages, str(output_json_file))
        
        # ============================================================================
        # STEP 4: Store in Qdrant Vector Store
        # ============================================================================
        print(f"\n{'='*80}")
        print("BƯỚC 4: LƯU VÀO QDRANT VECTOR STORE")
        print("="*80)
        
        try:
            vectorstore = store_pages_in_qdrant_direct(pages, collection_name)
            print(f"✅ Đã lưu {len(pages)} pages vào collection '{collection_name}'")
            
            # ============================================================================
            # STEP 5: Test retrieval
            # ============================================================================
            print(f"\n{'='*80}")
            print("BƯỚC 5: TEST RETRIEVAL")
            print("="*80)
            
            test_queries = [
                "vốn điều lệ của công ty",
                "báo cáo tài chính",
                "hoạt động kinh doanh chính",
                "quản trị rủi ro"
            ]
            
            for query in test_queries:
                print(f"\n🔍 Query: '{query}'")
                results = retrieve_similar_pages(query, vectorstore, top_k=3)
                
                for i, result in enumerate(results, 1):
                    print(f"\n  {i}. Page {result['page_index']} (Score: {result['similarity_score']:.4f})")
                    print(f"     {result['content'][:150]}...")
                
                print("-" * 80)
            
        except Exception as e:
            print(f"❌ Lỗi khi lưu vào vector store: {e}")
            return False
    else:
        print(f"\n⏭️ Bỏ qua BƯỚC 3-5: Vector store (skip_vectorstore=True)")
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"\n{'='*80}")
    print("✅ PIPELINE HOÀN THÀNH!")
    print("="*80)
    print(f"📁 Text files: {output_text_dir}")
    print(f"📁 PDF pages: {output_pdf_dir}")
    if not skip_vectorstore:
        print(f"🗄️ Vector store collection: {collection_name}")
    print("="*80)
    
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Cấu hình
    PDF_PATH = project_root / "src" / "data" / "pdfs" / "file_2.pdf"
    START_PAGE = 1
    END_PAGE = 168  # None để xử lý tất cả
    MAX_PAGES = 168
    COLLECTION_NAME = "esg_pages"
    
    # Các flags để bỏ qua các bước (nếu đã chạy rồi)
    SKIP_SPLIT = False      # True nếu đã cắt PDF rồi
    SKIP_EXTRACT = False    # True nếu đã trích xuất text rồi
    SKIP_VECTORSTORE = False  # True nếu chỉ muốn xử lý PDF
    
    success = run_complete_pipeline(
        pdf_path=str(PDF_PATH),
        start_page=START_PAGE,
        end_page=END_PAGE,
        max_pages=MAX_PAGES,
        collection_name=COLLECTION_NAME,
        skip_split=SKIP_SPLIT,
        skip_extract=SKIP_EXTRACT,
        skip_vectorstore=SKIP_VECTORSTORE
    )
    
    if success:
        print("\n🎉 Tất cả các bước đã hoàn thành thành công!")
    else:
        print("\n❌ Pipeline gặp lỗi. Vui lòng kiểm tra log.")
