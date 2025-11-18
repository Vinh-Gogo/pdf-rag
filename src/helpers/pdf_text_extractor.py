"""
PDF Text Extractor - Trích xuất text từ PDF theo thứ tự đọc tự nhiên

Quy trình:
1. Load PDF và lấy danh sách từ (words) với tọa độ
2. Gom nhóm từ theo y0 (merge_words_by_y) → merge_rows
3. Tách từng hàng thành các cột theo x0 (split_rows_by_x) → merge_rows_columns
4. Trích xuất text: cột cách nhau bởi '|', hàng cách nhau bởi '\n\n'
5. Lưu file và ảnh (tùy chọn)

Author: AI Assistant
Date: 2025-11-11
"""

from PIL import Image
import io
import pymupdf
import os
from typing import List, Tuple, Optional


class PDFTextExtractor:
    """
    Trích xuất text từ PDF theo thứ tự đọc tự nhiên (trên → dưới, trái → phải)
    """
    
    def __init__(self, y_threshold: float = 2, x_threshold: float = 5):
        """
        Khởi tạo PDF Text Extractor
        
        Args:
            y_threshold (float): Ngưỡng khoảng cách y để gom hàng (pixel). Mặc định: 2
            x_threshold (float): Ngưỡng khoảng cách x để tách cột (pixel). Mặc định: 5
        """
        self.y_threshold = y_threshold
        self.x_threshold = x_threshold
    
    def merge_words_by_y(self, words: List[Tuple], y_threshold: float) -> List[List[Tuple]]:
        """
        Gom nhóm các từ có y0 gần nhau (cùng hàng ngang)
        
        Args:
            words: Danh sách từ từ page.get_text("words")
            y_threshold: Ngưỡng khoảng cách y0 để gộp (pixel)
        
        Returns:
            merge_rows: Danh sách các hàng, mỗi hàng chứa danh sách từ
        """
        if not words:
            return []
        
        # Sắp xếp theo y0, x0
        sorted_words = sorted(words, key=lambda w: (w[1], w[0]))
        
        merge_rows = []
        current_row = [sorted_words[0]]
        
        for word in sorted_words[1:]:
            x0, y0, x1, y1, text, block_no, line_no, word_no = word
            
            # Lấy y0 của từ cuối trong hàng hiện tại
            prev_y0 = current_row[-1][1]
            
            # Nếu y0 gần nhau thì cùng hàng
            if abs(y0 - prev_y0) <= y_threshold:
                current_row.append(word)
            else:
                # Khác hàng, lưu hàng cũ và bắt đầu hàng mới
                merge_rows.append(current_row)
                current_row = [word]
        
        # Thêm hàng cuối cùng
        if current_row:
            merge_rows.append(current_row)
        
        return merge_rows
    
    def split_rows_by_x(self, merge_rows: List[List[Tuple]], x_threshold: float) -> List[List[List[Tuple]]]:
        """
        Tách từng hàng thành các cột dựa trên khoảng cách x0
        
        Args:
            merge_rows: Danh sách hàng từ merge_words_by_y
            x_threshold: Ngưỡng khoảng cách x giữa 2 từ để tách cột (pixel)
        
        Returns:
            merge_rows_columns: Danh sách các hàng, mỗi hàng chứa danh sách cột, mỗi cột chứa danh sách từ
        """
        merge_rows_columns = []
        
        for row in merge_rows:
            if not row:
                continue
            
            # Sắp xếp từ trong hàng theo x0
            sorted_row = sorted(row, key=lambda w: w[0])
            
            columns = []
            current_column = [sorted_row[0]]
            
            for word in sorted_row[1:]:
                x0, y0, x1, y1, text, block_no, line_no, word_no = word
                
                # Lấy x1 của từ cuối trong cột hiện tại
                prev_x1 = current_column[-1][2]
                
                # Nếu khoảng cách x > threshold thì tách cột mới
                if x0 - prev_x1 > x_threshold:
                    columns.append(current_column)
                    current_column = [word]
                else:
                    current_column.append(word)
            
            # Thêm cột cuối cùng
            if current_column:
                columns.append(current_column)
            
            merge_rows_columns.append(columns)
        
        return merge_rows_columns
    
    def extract_text_from_pdf(
        self, 
        pdf_path: str, 
        page_number: int = 0,
        output_file: Optional[str] = None, 
        save_image: bool = False,
        verbose: bool = True
    ) -> str:
        """
        Trích xuất text từ PDF theo thứ tự đọc tự nhiên
        
        Args:
            pdf_path (str): Đường dẫn đến file PDF
            page_number (int): Số trang cần trích xuất (0-indexed). Mặc định: 0
            output_file (str): Đường dẫn file output. Nếu None, không lưu file
            save_image (bool): Có lưu ảnh render của PDF không. Mặc định: False
            verbose (bool): Hiển thị thông tin chi tiết. Mặc định: True
        
        Returns:
            str: Nội dung text đã trích xuất
        """
        
        if verbose:
            print(f"📄 Đang xử lý: {pdf_path}")
        
        # === BƯỚC 1: Load PDF và lấy words ===
        doc = pymupdf.open(pdf_path)
        
        if page_number >= len(doc):
            raise ValueError(f"Trang {page_number} không tồn tại. PDF chỉ có {len(doc)} trang.")
        
        page = doc[page_number]
        words = page.get_text("words")
        
        # Render ảnh nếu cần
        if save_image:
            pix = page.get_pixmap(matrix=pymupdf.Matrix(1, 1))
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            image_path = pdf_path.replace('.pdf', f'_page_{page_number + 1}_rendered.png')
            img.save(image_path)
            if verbose:
                print(f"✅ Đã lưu ảnh: {image_path}")
        
        # Sắp xếp words theo y0, x0
        sorted_words = sorted(words, key=lambda w: (w[1], w[0]))
        
        # === BƯỚC 2: Gom nhóm từ theo y0 (hàng ngang) ===
        merge_rows = self.merge_words_by_y(sorted_words, self.y_threshold)
        
        if verbose:
            print(f"📊 Bước 1: Gom nhóm theo y0 → Tìm được {len(merge_rows)} hàng")
        
        # === BƯỚC 3: Tách từng hàng thành các cột theo x0 ===
        merge_rows_columns = self.split_rows_by_x(merge_rows, self.x_threshold)
        total_cols = sum(len(cols) for cols in merge_rows_columns)
        
        if verbose:
            print(f"📊 Bước 2: Tách theo x0 → Tổng {total_cols} cột")
        
        # === BƯỚC 4: Trích xuất text ===
        full_text = []
        
        for row_idx, row_columns in enumerate(merge_rows_columns):
            row_parts = []
            
            for col_idx, column in enumerate(row_columns):
                col_text = " ".join([w[4] for w in column])
                row_parts.append(col_text)
            
            full_text.append(" | ".join(row_parts))
        
        final_text = "\n\n".join(full_text)
        
        doc.close()
        
        # === BƯỚC 5: Lưu file (nếu có) ===
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(final_text)
            if verbose:
                print(f"✅ Đã lưu text: {output_file}")
        
        if verbose:
            print(f"📖 Hoàn tất: {len(full_text)} hàng, {len(final_text)} ký tự\n")
        
        return final_text
    
    def extract_all_pages(
        self,
        pdf_path: str,
        output_dir: str,
        save_images: bool = False,
        verbose: bool = True
    ) -> List[str]:
        """
        Trích xuất text từ tất cả các trang trong PDF
        
        Args:
            pdf_path (str): Đường dẫn đến file PDF
            output_dir (str): Thư mục lưu các file text
            save_images (bool): Có lưu ảnh render không. Mặc định: False
            verbose (bool): Hiển thị thông tin chi tiết. Mặc định: True
        
        Returns:
            List[str]: Danh sách text của từng trang
        """
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for page_num in range(total_pages):
            # if verbose:
            #     print(f"\n{'='*100}")
            #     print(f"📄 Đang xử lý trang {page_num + 1}/{total_pages}")
            #     print(f"{'='*100}")
            
            output_file = os.path.join(output_dir, f"page_{page_num + 1}.txt")
            
            text = self.extract_text_from_pdf(
                pdf_path=pdf_path,
                page_number=page_num,
                output_file=output_file,
                save_image=save_images,
                verbose=verbose
            )
            
            results.append(text)
        
        if verbose:
            print(f"\n{'='*100}")
            print(f"✅ Đã hoàn tất trích xuất {total_pages} trang")
            print(f"📁 Kết quả lưu tại: {output_dir}")
            print(f"{'='*100}")
        
        return results


def main():
    """
    Hàm main để test
    """
    # Khởi tạo extractor
    extractor = PDFTextExtractor(y_threshold=20, x_threshold=10)
    
    # Ví dụ 1: Trích xuất 1 trang
    print("\n" + "="*100)
    print("VÍ DỤ 1: TRÍCH XUẤT 1 TRANG")
    print("="*100)
    
    for i in range(92, 168):
        text = extractor.extract_text_from_pdf(
            pdf_path=r"src\data\pdfs\file_2.pdf",
            page_number=i,
            output_file=fr"src\data\contents\page_{i+1}.txt",
            save_image=False,
            verbose=True
        )
    
    # # In preview
    # print("\n📖 PREVIEW (500 ký tự đầu):")
    # print("-" * 100)
    # print(text[:500])
    # print("...")
    # print("-" * 100)
    
    # all_texts = extractor.extract_all_pages(
    #     pdf_path="data/pdfs/file_2.pdf",
    #     output_dir="data/extracted_texts",
    #     save_images=False,
    #     verbose=True
    # )


if __name__ == "__main__":
    main()
