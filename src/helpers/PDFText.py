import fitz  # PyMuPDF
import os
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


class PDFTextExtractor:
    """
    A class for extracting and processing text from PDF files using PyMuPDF.
    Supports splitting PDF into individual pages and extracting text.
    """
    
    def __init__(self, pdf_path: str, output_dir: str = "src/data/contents", split_output_dir: str = "src/data/pdfs/pages"):
        """
        Initialize the PDF text extractor.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save extracted text files
            split_output_dir (str): Directory to save split PDF pages
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.split_output_dir = split_output_dir
        self.pdf_file = None
        self.blocks = []  # Lưu trữ tất cả các block văn bản theo mỗi page
        self.total_pages = 0
        
        # Create output directories if they don't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.split_output_dir, exist_ok=True)
    
    def open_pdf(self) -> bool:
        """
        Open the PDF file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.pdf_file = fitz.open(self.pdf_path)
            self.total_pages = len(self.pdf_file)
            print(f"📖 Đã mở PDF: {self.pdf_path}")
            print(f"📄 Tổng số trang: {self.total_pages}")
            return True
        except Exception as e:
            print(f"❌ Error opening PDF: {e}")
            return False
    
    def split_pdf_into_pages(self, start_page: int = 1, end_page: Optional[int] = None) -> bool:
        """
        Split PDF into individual page files.
        
        Args:
            start_page (int): Starting page number (1-based indexing)
            end_page (Optional[int]): Ending page number (1-based indexing). If None, process all pages.
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.pdf_file:
            if not self.open_pdf():
                return False
        
        try:
            if end_page is None:
                end_page = self.total_pages
            
            # Validate page range
            if start_page < 1 or end_page > self.total_pages or start_page > end_page:
                print(f"❌ Invalid page range: {start_page}-{end_page}. Total pages: {self.total_pages}")
                return False
            
            print(f"\n🔪 Đang cắt PDF thành các trang từ {start_page} đến {end_page}...")
            
            for page_num in range(start_page, end_page + 1):
                # Create a new PDF with single page
                output_pdf = fitz.open()
                output_pdf.insert_pdf(self.pdf_file, from_page=page_num-1, to_page=page_num-1)
                
                # Save the single page PDF
                output_path = os.path.join(self.split_output_dir, f"page_{page_num}.pdf")
                output_pdf.save(output_path)
                output_pdf.close()
                
                if page_num % 20 == 0 or page_num == end_page:
                    print(f"   ✅ Đã cắt {page_num}/{end_page} trang")
            
            print(f"✅ Hoàn thành! Đã cắt {end_page - start_page + 1} trang vào {self.split_output_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Error splitting PDF: {e}")
            return False
    
    def extract_text_from_page(self, page_number: int, clean_text: bool = True) -> Optional[str]:
        """
        Extract text from a specific page using PyMuPDF.
        
        Args:
            page_number (int): Page number (1-based indexing)
            clean_text (bool): Whether to clean the extracted text
            
        Returns:
            Optional[str]: Extracted text or None if error
        """
        if not self.pdf_file:
            print("❌ PDF file not opened")
            return None
            
        try:
            page = self.pdf_file[page_number - 1]
            
            # Extract text blocks with position info
            text_blocks = page.get_text("blocks")
            
            # Sort blocks by vertical position (top to bottom), then horizontal (left to right)
            sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))
            
            text_content = ""
            for block in sorted_blocks:
                if len(block) >= 5:
                    block_text = block[4]  # Text content of each block
                    
                    if clean_text:
                        # Clean extra whitespace and newlines
                        block_text = ' '.join(block_text.split())
                    
                    text_content += block_text + "\n\n"
                    self.blocks.append(np.array([block]))  # Lưu block vào danh sách
            
            # Remove trailing whitespace
            text_content = text_content.strip()
                
            return text_content
        except Exception as e:
            print(f"❌ Error extracting text from page {page_number}: {e}")
            return None
    
    def save_page_text(self, page_number: int, text: str) -> bool:
        """
        Save text content to a file.
        
        Args:
            page_number (int): Page number
            text (str): Text content to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            text_path = os.path.join(self.output_dir, f"page_{page_number}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Saved {text_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving page {page_number}: {e}")
            return False
    
    def clean_text_file(self, page_number: int) -> bool:
        """
        Clean whitespace from a text file.
        
        Args:
            page_number (int): Page number
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            text_path = os.path.join(self.output_dir, f"page_{page_number}.txt")
            with open(text_path, "r+", encoding="utf-8") as f:
                lines = f.readlines()
                f.seek(0)
                for line in lines:
                    f.write(line.strip() + "\n")
                f.truncate()
            print(f"✅ Cleaned {text_path}")
            return True
        except Exception as e:
            print(f"❌ Error cleaning page {page_number}: {e}")
            return False
    
    def extract_all_pages(self, start_page: int = 1, end_page: Optional[int] = None, clean_text: bool = True) -> bool:
        """
        Extract text from all pages in the PDF.
        
        Args:
            start_page (int): Starting page number (1-based indexing)
            end_page (Optional[int]): Ending page number (1-based indexing). If None, process all pages.
            clean_text (bool): Whether to clean the extracted text
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.open_pdf():
            return False
            
        try:
            if not self.pdf_file:
                print("❌ PDF file not available")
                return False
            
            if end_page is None:
                end_page = self.total_pages
            
            # Validate page range
            if start_page < 1 or end_page > self.total_pages or start_page > end_page:
                print(f"❌ Invalid page range: {start_page}-{end_page}. Total pages: {self.total_pages}")
                return False
            
            print(f"\n📝 Đang trích xuất text từ trang {start_page} đến {end_page}...")
            
            # Extract and save text from each page
            for page_number in range(start_page, end_page + 1):
                text = self.extract_text_from_page(page_number, clean_text=clean_text)
                if text is not None:
                    self.save_page_text(page_number, text)
                    
                    if page_number % 20 == 0 or page_number == end_page:
                        print(f"   ✅ Đã xử lý {page_number}/{end_page} trang")
                else:
                    print(f"   ⚠️ Không thể trích xuất text từ trang {page_number}")
            
            print(f"✅ Hoàn thành trích xuất {end_page - start_page + 1} trang!")
            return True
            
        except Exception as e:
            print(f"❌ Error during extraction: {e}")
            return False
        finally:
            self.close_pdf()
    
    def get_page_count(self) -> int:
        """
        Get total number of pages in the PDF.
        
        Returns:
            int: Total number of pages
        """
        if not self.pdf_file:
            if not self.open_pdf():
                return 0
            should_close = True
        else:
            should_close = False
        
        page_count = self.total_pages
        
        if should_close:
            self.close_pdf()
        
        return page_count
    
    def close_pdf(self):
        """Close the PDF file."""
        if self.pdf_file:
            self.pdf_file.close()
            self.pdf_file = None
    
    def get_blocks(self) -> List:
        """
        Get all extracted text blocks.
        
        Returns:
            List: List of text blocks
        """
        return self.blocks
    
    def clear_blocks(self):
        """Clear the stored text blocks."""
        self.blocks.clear()


# Example usage
if __name__ == "__main__":
    # Initialize the extractor
    pdf_path = r"src/data/pdfs/file_2.pdf"
    output_text_dir = r"src/data/raw"
    output_pdf_dir = r"src/data/pdfs/pages"
    
    print("="*80)
    print("🚀 PDF TEXT EXTRACTOR - PyMuPDF")
    print("="*80)
    
    extractor = PDFTextExtractor(
        pdf_path=pdf_path,
        output_dir=output_text_dir,
        split_output_dir=output_pdf_dir
    )
    
    # Get total page count
    total_pages = extractor.get_page_count()
    print(f"\n📊 Tổng số trang trong PDF: {total_pages}")
    
    # Define page range (change these values as needed)
    START_PAGE = 1
    END_PAGE = min(168, total_pages)  # Maximum 168 pages or total pages
    
    print(f"🎯 Sẽ xử lý từ trang {START_PAGE} đến {END_PAGE}")
    
    # Step 1: Split PDF into individual pages
    print(f"\n{'='*80}")
    print("BƯỚC 1: CẮT PDF THÀNH CÁC TRANG ĐƠN LẺ")
    print("="*80)
    split_success = extractor.split_pdf_into_pages(start_page=START_PAGE, end_page=END_PAGE)
    
    if not split_success:
        print("❌ Failed to split PDF")
        exit(1)
    
    # Step 2: Extract text from all pages
    print(f"\n{'='*80}")
    print("BƯỚC 2: TRÍCH XUẤT TEXT TỪ MỖI TRANG")
    print("="*80)
    extract_success = extractor.extract_all_pages(start_page=START_PAGE, end_page=END_PAGE)
    
    if extract_success:
        print(f"\n✅ HOÀN THÀNH!")
        print(f"   📊 Total blocks extracted: {len(extractor.get_blocks())}")
        print(f"   📁 Text files saved to: {output_text_dir}")
        print(f"   📁 PDF pages saved to: {output_pdf_dir}")
    else:
        print("\n❌ Extraction failed")