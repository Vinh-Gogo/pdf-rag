import fitz  # PyMuPDF
import os
import numpy as np
from typing import List, Optional


class PDFTextExtractor:
    """
    A class for extracting and processing text from PDF files.
    """
    
    def __init__(self, pdf_path: str, output_dir: str = "src/data/contents"):
        """
        Initialize the PDF text extractor.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str): Directory to save extracted text files
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.pdf_file = None
        self.blocks = []  # Lưu trữ tất cả các block văn bản theo mỗi page
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def open_pdf(self) -> bool:
        """
        Open the PDF file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.pdf_file = fitz.open(self.pdf_path)
            return True
        except Exception as e:
            print(f"❌ Error opening PDF: {e}")
            return False
    
    def extract_text_from_page(self, page_number: int) -> Optional[str]:
        """
        Extract text from a specific page.
        
        Args:
            page_number (int): Page number (1-based indexing)
            
        Returns:
            Optional[str]: Extracted text or None if error
        """
        if not self.pdf_file:
            print("❌ PDF file not opened")
            return None
            
        try:
            page = self.pdf_file[page_number - 1]
            text_blocks = page.get_text_blocks()
            
            text_content = ""
            for block in text_blocks:
                text_content += block[4]  # Text content of each block
                self.blocks.append(np.array([block]))  # Lưu block vào danh sách
                
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
    
    def extract_all_pages(self) -> bool:
        """
        Extract text from all pages in the PDF.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.open_pdf():
            return False
            
        try:
            if not self.pdf_file:
                print("❌ PDF file not available")
                return False
                
            total_pages = len(self.pdf_file)
            
            # Extract and save text from each page
            for page_number in range(1, total_pages + 1):
                text = self.extract_text_from_page(page_number)
                if text is not None:
                    self.save_page_text(page_number, text)
                else:
                    return False
            
            # Clean whitespace from all saved files
            for page_number in range(1, total_pages + 1):
                if not self.clean_text_file(page_number):
                    return False
            
            print("🎉 Hoàn thành trích xuất và lưu văn bản từ PDF.")
            return True
            
        except Exception as e:
            print(f"❌ Error during extraction: {e}")
            return False
        finally:
            self.close_pdf()
    
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
    output_dir = r"src/data/raw"
    
    extractor = PDFTextExtractor(pdf_path, output_dir)
    
    # Extract all pages
    success = extractor.extract_all_pages()
    
    if success:
        print(f"📊 Total blocks extracted: {len(extractor.get_blocks())}")
    else:
        print("❌ Extraction failed")