import os
import sys
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# PDF processing
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pdfminer.layout import LAParams
from io import StringIO
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import các components
from src.models.halong_embedd import HalongEmbedding
from src.models.embedd import QwenEmbedding
from src.models.dangvantuan_embedd import DangVanTuanEmbedding

from src.helpers.init_qdrant import qdrant_client
from qdrant_client.models import PointStruct, VectorParams, Distance


@dataclass
class PDFPage:
    """Cấu trúc dữ liệu cho một trang PDF"""
    page_index: int
    content: str
    word_count: int
    file_name: str
    sequence_id: str


@dataclass
class PDFDocument:
    """Cấu trúc dữ liệu cho toàn bộ document PDF"""
    file_path: str
    file_name: str
    total_pages: int
    pages: List[PDFPage]
    total_words: int

class PDFTextExtractor:
    """Trích xuất text từ PDF files"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Giới hạn độ dài text cho embedding
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: str) -> PDFDocument:
        """
        Trích xuất text từ PDF file sử dụng pdfminer

        Args:
            pdf_path (str): Đường dẫn đến file PDF

        Returns:
            PDFDocument: Document với thông tin chi tiết
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path_obj}")

        print(f"📖 Đang đọc PDF: {pdf_path_obj.name}")

        # Cấu hình pdfminer
        laparams = LAParams(
            line_margin=0.5,
            word_margin=0.1,
            char_margin=2.0,
            boxes_flow=0.5
        )

        # Trích xuất toàn bộ text từ PDF
        try:
            full_text = pdfminer_extract_text(pdf_path, laparams=laparams)
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {e}")

        # Làm sạch text
        full_text = self._clean_text(full_text)

        if not full_text.strip():
            raise ValueError("No text content found in PDF")

        # Chia text thành các chunks
        text_chunks = self.text_splitter.split_text(full_text)

        # Tạo pages từ chunks (pdfminer không phân biệt pages rõ ràng)
        pages = []
        for chunk_idx, chunk in enumerate(text_chunks):
            word_count = len(chunk.split())

            # Tạo sequence ID (không có thông tin page cụ thể từ pdfminer)
            sequence_id = f"{pdf_path_obj.stem}_seq_{chunk_idx + 1}"

            pdf_page = PDFPage(
                page_index=chunk_idx + 1,  # Sử dụng chunk index làm page index
                content=chunk,
                word_count=word_count,
                file_name=pdf_path_obj.name,
                sequence_id=sequence_id
            )
            pages.append(pdf_page)

        # Tính tổng số từ
        total_words = sum(page.word_count for page in pages)

        # Ước tính số trang (dựa trên số chunks)
        estimated_pages = max(1, len(text_chunks) // 3)  # Ước tính khoảng 3 chunks per page

        pdf_doc = PDFDocument(
            file_path=str(pdf_path_obj),
            file_name=pdf_path_obj.name,
            total_pages=estimated_pages,
            pages=pages,
            total_words=total_words
        )

        print(f"✅ Đã trích xuất {len(pages)} sequences, {total_words} từ từ PDF")
        return pdf_doc

    def _clean_text(self, text: str) -> str:
        """Làm sạch text đã trích xuất"""
        if not text:
            return ""

        # Loại bỏ các dòng trống liên tiếp
        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if line:  # Chỉ giữ lại dòng có nội dung
                cleaned_lines.append(line)

        # Ghép lại thành text
        cleaned_text = '\n'.join(cleaned_lines)

        # Loại bỏ khoảng trắng thừa
        cleaned_text = ' '.join(cleaned_text.split())

        return cleaned_text


class PDFVectorStorePipeline:
    """Pipeline hoàn chỉnh từ PDF đến Vector Store"""

    def __init__(self, embedding_model='qwen', collection_name: str = "esg_sequences"):
        """
        Khởi tạo pipeline

        Args:
            embedding_model (str): Model embedding ('halong', 'qwen', 'dangvantuan')
            collection_name (str): Tên collection trong Qdrant
        """
        self.collection_name = collection_name
        self.extractor = PDFTextExtractor()
        self.embedding = self._load_embedding_model(embedding_model)

        print(f"🔧 Pipeline initialized with {embedding_model} embedding and {collection_name} collection")

    def _load_embedding_model(self, model_name: str):
        """Load embedding model"""
        if model_name.lower() == 'qwen':
            return QwenEmbedding()
        elif model_name.lower() == 'dangvantuan':
            return DangVanTuanEmbedding()
        else:
            raise ValueError(f"Unknown embedding model: {model_name}")

    def process_single_pdf(self, pdf_path: str, use_text_correction: bool = False) -> PDFDocument:
        """
        Xử lý một file PDF

        Args:
            pdf_path (str): Đường dẫn đến PDF
            use_text_correction (bool): Có sử dụng LLM text correction không

        Returns:
            PDFDocument: Document đã xử lý
        """
        # Trích xuất text từ PDF
        pdf_doc = self.extractor.extract_text_from_pdf(pdf_path)

        # Áp dụng text correction nếu được yêu cầu
        if use_text_correction:
            pdf_doc = self._apply_text_correction(pdf_doc)

        return pdf_doc

    def _apply_text_correction(self, pdf_doc: PDFDocument) -> PDFDocument:
        """Áp dụng LLM text correction cho document"""
        print("🔧 Applying text correction...")

        try:
            # Import text correction module
            from src.helpers.llm_text_correction import correct_vietnamese

            corrected_pages = []
            for page in pdf_doc.pages:
                # Áp dụng correction cho từng page
                corrected_content = correct_vietnamese(
                    text=page.content,
                    repeat_reminder=1,
                    model=None,  # Sẽ được load trong function
                    tokenizer=None,
                    use_enhanced_prompt=True
                )

                # Tạo page mới với nội dung đã sửa
                corrected_page = PDFPage(
                    page_index=page.page_index,
                    content=corrected_content,
                    word_count=len(corrected_content.split()),
                    file_name=page.file_name,
                    sequence_id=page.sequence_id
                )
                corrected_pages.append(corrected_page)

            # Cập nhật document
            pdf_doc.pages = corrected_pages
            pdf_doc.total_words = sum(page.word_count for page in corrected_pages)

            print("✅ Text correction applied")

        except Exception as e:
            print(f"⚠️ Text correction failed: {e}")
            print("Continuing without text correction...")

        return pdf_doc

    def upload_to_vectorstore(self, pdf_doc: PDFDocument, batch_size: int = 50) -> Dict[str, Any]:
        """
        Upload document lên Qdrant vector store - xử lý từng page riêng biệt

        Args:
            pdf_doc (PDFDocument): Document cần upload
            batch_size (int): Kích thước batch (deprecated, giữ để tương thích)

        Returns:
            Dict[str, Any]: Thông tin upload
        """
        print(f"📤 Uploading {len(pdf_doc.pages)} sequences to {self.collection_name}...")

        # Chuẩn bị collection
        self._prepare_collection()

        uploaded_count = 0

        # Xử lý từng page riêng biệt
        for i, page in enumerate(pdf_doc.pages, 1):
            try:
                # Tạo embedding cho page này
                embedding = self.embedding.get_embedding(page.content)

                if embedding is not None:
                    # Tạo payload
                    payload = {
                        'sequence_id': page.sequence_id,
                        'content': page.content,
                        'file_name': page.file_name,
                        'page_index': page.page_index,
                        'word_count': page.word_count,
                        'total_sequences_in_file': len(pdf_doc.pages)
                    }

                    # Tạo point với ID là integer theo thứ tự
                    point_id = uploaded_count + 1  # ID từ 1, 2, 3...
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )

                    # Upload ngay lập tức
                    qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[point]  # Upload từng point một
                    )

                    uploaded_count += 1

                    if i % 10 == 0 or i == len(pdf_doc.pages):
                        print(f"  📦 Uploaded {i}/{len(pdf_doc.pages)} sequences")

            except Exception as e:
                print(f"  ⚠️ Failed to upload sequence {page.sequence_id}: {e}")
                continue

        result = {
            'collection_name': self.collection_name,
            'uploaded_sequences': uploaded_count,
            'total_pages': pdf_doc.total_pages,
            'total_words': pdf_doc.total_words,
            'file_name': pdf_doc.file_name
        }

        print(f"✅ Successfully uploaded {uploaded_count} sequences to {self.collection_name}")
        return result

    def _prepare_collection(self):
        """Chuẩn bị collection trong Qdrant"""
        try:
            # Lấy sample embedding để xác định vector size
            sample_text = "test embedding"
            sample_embedding = self.embedding.get_embedding(sample_text)

            if sample_embedding is None:
                raise ValueError("Cannot create embedding for sample text")

            vector_size = len(sample_embedding)
            print(f"📏 Embedding vector size: {vector_size}")

            # Kiểm tra collection có tồn tại không
            collections = qdrant_client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name in collection_names:
                print(f"📁 Collection '{self.collection_name}' already exists")
                # Thử upload sample để kiểm tra dimension
                try:
                    test_point = PointStruct(
                        id=999999,  # ID test
                        vector=sample_embedding.tolist(),
                        payload={'test': True}
                    )
                    qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=[test_point]
                    )
                    # Nếu thành công, xóa test point
                    qdrant_client.delete(
                        collection_name=self.collection_name,
                        points_selector=[999999]
                    )
                    print("✅ Collection dimension matches")
                except Exception as e:
                    if "Vector dimension error" in str(e):
                        print(f"⚠️ Collection dimension mismatch, deleting and recreating...")
                        # Xóa collection cũ
                        qdrant_client.delete_collection(self.collection_name)
                        print(f"✅ Deleted old collection '{self.collection_name}'")

                        # Tạo collection mới
                        qdrant_client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                        )
                        print(f"🆕 Created new collection '{self.collection_name}' with vector size {vector_size}")
                    else:
                        raise e
            else:
                # Tạo collection mới
                qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
                print(f"🆕 Created new collection '{self.collection_name}' with vector size {vector_size}")

        except Exception as e:
            print(f"❌ Error preparing collection: {e}")
            raise

    def process_directory(self, pdf_dir: str, use_text_correction: bool = False) -> List[Dict[str, Any]]:
        """
        Xử lý tất cả PDF files trong thư mục

        Args:
            pdf_dir (str): Thư mục chứa PDF files
            use_text_correction (bool): Có sử dụng text correction không

        Returns:
            List[Dict[str, Any]]: Kết quả xử lý cho từng file
        """
        pdf_dir_obj = Path(pdf_dir)
        if not pdf_dir_obj.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir_obj}")

        # Tìm tất cả file PDF
        pdf_files = list(pdf_dir_obj.glob("*.pdf"))
        print(f"📂 Found {len(pdf_files)} PDF files in {pdf_dir_obj}")

        results = []
        for pdf_file in pdf_files:
            try:
                print(f"\n🔄 Processing: {pdf_file.name}")

                # Xử lý PDF
                pdf_doc = self.process_single_pdf(str(pdf_file), use_text_correction)

                # Upload lên vector store
                upload_result = self.upload_to_vectorstore(pdf_doc)

                # Kết hợp kết quả
                result = {
                    'file_name': pdf_file.name,
                    'status': 'success',
                    'sequences_extracted': len(pdf_doc.pages),
                    'total_words': pdf_doc.total_words,
                    'upload_info': upload_result
                }
                results.append(result)

                print(f"✅ Completed: {pdf_file.name}")

            except Exception as e:
                print(f"❌ Failed to process {pdf_file.name}: {e}")
                results.append({
                    'file_name': pdf_file.name,
                    'status': 'error',
                    'error': str(e)
                })

        return results

    def save_results(self, results: List[Dict[str, Any]], output_file: str):
        """Lưu kết quả xử lý ra file JSON"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"💾 Results saved to {output_file}")


def main():
    """Main pipeline function"""
    print("🚀 PDF TO VECTOR STORE PIPELINE")
    print("=" * 50)

    # Cấu hình
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent

    # Thư mục input (PDF files)
    pdf_input_dir = project_root / "data" / "pdfs"  # Thư mục chứa PDF files

    # File output
    results_file = project_root / "data" / "pipeline_results.json"

    # Khởi tạo pipeline
    pipeline = PDFVectorStorePipeline(
        embedding_model='qwen',  # Có thể đổi thành 'qwen' hoặc 'dangvantuan'
        collection_name='esg_sequences'
    )

    # Kiểm tra thư mục input
    if not pdf_input_dir.exists():
        print(f"❌ PDF directory not found: {pdf_input_dir}")
        print("Please create the directory and add PDF files")
        return

    # Xử lý tất cả PDF trong thư mục
    print(f"📂 Processing PDFs from: {pdf_input_dir}")
    results = pipeline.process_directory(
        pdf_dir=str(pdf_input_dir),
        use_text_correction=False  # Có thể bật True nếu muốn dùng LLM correction
    )

    # Lưu kết quả
    pipeline.save_results(results, str(results_file))

    # Thống kê
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']

    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")

    if successful:
        total_sequences = sum(r['sequences_extracted'] for r in successful)
        total_words = sum(r['total_words'] for r in successful)
        print(f"   📄 Total sequences: {total_sequences}")
        print(f"   📝 Total words: {total_words}")

    if failed:
        print(f"   Failed files: {[r['file_name'] for r in failed]}")

    print(f"\n🎉 Pipeline completed! Results saved to {results_file}")


if __name__ == "__main__":
    main()