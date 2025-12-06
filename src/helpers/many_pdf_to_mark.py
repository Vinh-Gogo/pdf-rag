import os
import gc
import torch
import cv2
import numpy as np
from pathlib import Path
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import time
import psutil
import logging
import platform
import fitz  # PyMuPDF
from PIL import Image
import io

# Thiết lập logging chi tiết hơn
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_pdf_for_conversion(pdf_path):
    """
    Preprocess PDF để cải thiện khả năng nhận dạng văn bản
    """
    try:
        # Mở PDF bằng PyMuPDF
        doc = fitz.open(pdf_path)
        processed_pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Kiểm tra xem page có chứa văn bản không
            text = page.get_text()
            if len(text.strip()) < 50:  # Nếu văn bản quá ít
                # Chuyển page thành hình ảnh và xử lý
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Chuyển đổi sang numpy array cho xử lý OpenCV
                img_array = np.array(img)
                
                # Xử lý hình ảnh để cải thiện chất lượng
                img_array = preprocess_image(img_array)
                
                # Chuyển lại thành PIL Image
                processed_img = Image.fromarray(img_array)
                processed_pages.append(processed_img)
            else:
                # Nếu có đủ văn bản, giữ nguyên page
                processed_pages.append(None)  # None để chỉ định không cần xử lý đặc biệt
        
        doc.close()
        return processed_pages
    
    except Exception as e:
        logger.error(f"Error preprocessing PDF {pdf_path}: {str(e)}")
        return []

def preprocess_image(img_array):
    """
    Xử lý hình ảnh để cải thiện chất lượng OCR
    """
    try:
        # Chuyển sang grayscale nếu là ảnh màu
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array.copy()
        
        # Áp dụng thresholding thích nghi
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Khử nhiễu
        denoised = cv2.fastNlMeansDenoising(thresh, h=10)
        
        # Cân bằng độ sáng
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return img_array  # Trả về ảnh gốc nếu có lỗi

def fallback_conversion_method(pdf_path, output_dir):
    """
    Phương pháp fallback khi conversion chính thất bại
    Sử dụng PyMuPDF để trích xuất văn bản cơ bản
    """
    try:
        pdf_file = Path(pdf_path)
        output_path = Path(output_dir)
        output_file_path = output_path / (pdf_file.stem + "_fallback.md")
        
        doc = fitz.open(pdf_path)
        text_content = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_content.append(f"## Page {page_num + 1}\n")
                text_content.append(text.strip() + "\n\n")
        
        doc.close()
        
        # Lưu kết quả fallback
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("# Fallback Conversion - Basic Text Extraction\n\n")
            f.write(f"**Source file:** {pdf_file.name}\n\n")
            f.write(f"**Conversion method:** PyMuPDF basic text extraction\n\n")
            f.write(f"**Warning:** This is a fallback conversion. Some formatting and images may be lost.\n\n")
            f.write("---\n\n")
            f.write("".join(text_content))
        
        return {
            "status": "fallback",
            "message": f"⚠ Fallback conversion for {pdf_file.name} -> {output_file_path.name}",
            "file": pdf_file.name,
            "success": True,
            "output_exists": False
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"✗ Fallback conversion failed for {pdf_file.name}: {str(e)}",
            "file": pdf_file.name,
            "success": False,
            "error": str(e),
            "output_exists": False
        }

def process_single_pdf(pdf_file_path, output_dir, config_dict, model_dict):
    """
    Xử lý một file PDF đơn lẻ với xử lý lỗi nâng cao
    """
    try:
        pdf_file = Path(pdf_file_path)
        output_path = Path(output_dir)
        output_file_path = output_path / (pdf_file.stem + ".md")
        
        # Kiểm tra nếu file đã tồn tại
        if output_file_path.exists():
            return {
                "status": "skipped",
                "message": f"✓ Skipping {pdf_file.name} - already converted",
                "file": pdf_file.name,
                "success": True,
                "output_exists": True
            }
        
        # Preprocess PDF trước khi conversion
        logger.info(f"Preprocessing PDF: {pdf_file.name}")
        processed_pages = preprocess_pdf_for_conversion(pdf_file_path)
        
        # Khởi tạo converter
        config_parser = ConfigParser(config_dict)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=model_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        
        # Xử lý PDF với xử lý lỗi chi tiết
        start_time = time.time()
        
        try:
            # Thử conversion chính
            rendered = converter(str(pdf_file))
            processing_time = time.time() - start_time
            
            # Trích xuất text
            text, _, images = text_from_rendered(rendered)
            
            # Kiểm tra chất lượng kết quả
            if len(text.strip()) < 100:  # Nếu kết quả quá ngắn
                logger.warning(f"Low quality conversion for {pdf_file.name}, trying fallback")
                raise Exception("Low quality conversion result")
            
            # Lưu kết quả
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            result = {
                "status": "success",
                "message": f"✓ Converted {pdf_file.name} -> {output_file_path.name} ({processing_time:.2f}s)",
                "file": pdf_file.name,
                "success": True,
                "time": processing_time,
                "output_exists": False
            }
            
        except Exception as e:
            logger.error(f"Main conversion failed for {pdf_file.name}: {str(e)}")
            logger.info(f"Trying fallback method for {pdf_file.name}")
            
            # Thử phương pháp fallback
            fallback_result = fallback_conversion_method(pdf_file_path, output_dir)
            
            if fallback_result["success"]:
                result = fallback_result
                logger.info(f"Fallback conversion succeeded for {pdf_file.name}")
            else:
                logger.error(f"Both main and fallback conversions failed for {pdf_file.name}")
                result = {
                    "status": "error",
                    "message": f"✗ Both conversions failed for {pdf_file.name}: {str(e)}",
                    "file": pdf_file.name,
                    "success": False,
                    "error": str(e),
                    "output_exists": False
                }
        
        # Giải phóng bộ nhớ
        del converter
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        logger.exception(f"Critical error processing {pdf_file_path}: {str(e)}")
        return {
            "status": "error",
            "message": f"✗ Critical error for {Path(pdf_file_path).name}: {str(e)}",
            "file": Path(pdf_file_path).name,
            "success": False,
            "error": str(e),
            "output_exists": False
        }

def get_optimal_config(use_llm=False, force_ocr=False, languages=None, enhanced_accuracy=True):
    """
    Get optimal configuration for high accuracy PDF conversion with error handling
    """
    config = {
        "output_format": "markdown",
        "batch_size": 1,  # Giảm batch size để tăng độ ổn định
        "max_pages": None,
        "enable_caching": True,
        "use_gpu": torch.cuda.is_available(),
        
        # Cấu hình OCR nâng cao
        "ocr_all_pages": force_ocr or enhanced_accuracy,
        "ocr_languages": languages or "vi",
        "ocr_engine": "ocrmypdf",
        
        # Cấu hình xử lý văn bản
        "text_detection_threshold": 0.7 if enhanced_accuracy else 0.5,
        "table_detection_threshold": 0.6 if enhanced_accuracy else 0.4,
        
        # Tối ưu cho tài liệu phức tạp
        "preserve_layout": True,
        "detect_columns": True,
        "detect_tables": True,
        "detect_equations": True,
        
        # Cấu hình xử lý hình ảnh
        "image_processing": True,
        "image_quality_threshold": 0.5,
        
        # Xử lý lỗi và fallback
        "enable_fallback": True,
        "min_text_length": 50,
        
        # Cấu hình LLM
        "use_llm": use_llm,
        "llm_confidence_threshold": 0.8,
    }
    
    if use_llm:
        config.update({
            "llm_service": "openai",
            "llm_model": "gpt-4o-mini",
            "llm_api_key": os.getenv("OPENAI_API_KEY", ""),
            "llm_max_tokens": 4096,
            "llm_temperature": 0.1,
        })
    
    return config

def convert_pdf_to_markdown(input_dir: str, output_dir: str, max_workers=None, 
                          use_llm=False, force_ocr=False, languages="eng+vie",
                          force_reconvert=False, enhanced_accuracy=True):
    """
    Convert PDF files to Markdown with high accuracy and robust error handling
    
    Args:
        enhanced_accuracy: Enable enhanced accuracy mode with preprocessing and fallbacks
    """
    start_time = time.time()
    
    # Tạo thư mục output nếu chưa tồn tại
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tìm tất cả file PDF
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory {input_dir} does not exist!")
        return
    
    # Lấy tất cả file PDF
    pdf_files = list(input_path.glob("*.pdf"))
    pdf_files = [f for f in pdf_files if f.is_file()]
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files total")
    
    # Lọc các file cần xử lý
    pdf_files_to_process = []
    for pdf_file in pdf_files:
        output_file_path = output_path / (pdf_file.stem + ".md")
        fallback_file_path = output_path / (pdf_file.stem + "_fallback.md")
        
        if force_reconvert:
            pdf_files_to_process.append(pdf_file)
            logger.info(f"Force reconvert: {pdf_file.name}")
        else:
            if not output_file_path.exists() and not fallback_file_path.exists():
                pdf_files_to_process.append(pdf_file)
    
    logger.info(f"{len(pdf_files_to_process)} files need to be processed")
    
    if not pdf_files_to_process and not force_reconvert:
        logger.info("All files already converted. Nothing to do.")
        return
    
    # Get optimal configuration
    config = get_optimal_config(
        use_llm=use_llm, 
        force_ocr=force_ocr, 
        languages=languages,
        enhanced_accuracy=enhanced_accuracy
    )
    
    logger.info("=== Configuration Summary ===")
    logger.info(f"Enhanced Accuracy Mode: {enhanced_accuracy}")
    logger.info(f"Use LLM for accuracy improvement: {use_llm}")
    logger.info(f"Force OCR processing: {force_ocr}")
    logger.info(f"OCR Languages: {languages}")
    logger.info(f"Batch Size: {config['batch_size']}")
    logger.info(f"GPU Acceleration: {config['use_gpu']}")
    logger.info(f"Enable Fallback Methods: {config.get('enable_fallback', True)}")
    logger.info("==============================")
    
    # Kiểm tra dependencies
    check_dependencies()
    
    # Khởi tạo model một lần
    logger.info("Initializing models...")
    model_dict = create_model_dict()
    
    # Tự động xác định số worker tối ưu
    if max_workers is None:
        cpu_cores = multiprocessing.cpu_count()
        available_memory = psutil.virtual_memory().available / (1024 ** 3)  # GB
        max_workers = min(cpu_cores, max(1, int(available_memory / 4)))  # 4GB per worker for accuracy
        logger.info(f"System has {cpu_cores} CPU cores and {available_memory:.1f}GB available memory")
    
    max_workers = min(max_workers, len(pdf_files_to_process))
    logger.info(f"Using {max_workers} parallel workers")
    
    # Xử lý song song với progress bar
    results = []
    successes = 0
    total_processing_time = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_pdf, str(pdf_file), output_dir, config, model_dict)
            for pdf_file in pdf_files_to_process
        ]
        
        for future in tqdm(as_completed(futures), total=len(pdf_files_to_process), desc="Processing PDFs"):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                successes += 1
                if result["status"] in ["success", "fallback"]:
                    total_processing_time += result.get("time", 0)
    
    # Hiển thị kết quả chi tiết
    print("\n=== Conversion Results ===")
    success_count = 0
    fallback_count = 0
    error_count = 0
    
    for result in results:
        print(result["message"])
        if result["status"] == "success":
            success_count += 1
        elif result["status"] == "fallback":
            fallback_count += 1
        elif result["status"] == "error":
            error_count += 1
    
    # Tính toán thống kê
    total_time = time.time() - start_time
    avg_processing_time = total_processing_time / max(1, (success_count + fallback_count))
    
    print(f"\n=== Final Results Summary ===")
    print(f"✅ Successfully converted: {success_count}")
    print(f"⚠️ Fallback conversions: {fallback_count}")
    print(f"❌ Failed conversions: {error_count}")
    print(f"📊 Total files processed: {len(pdf_files_to_process)}")
    print(f"⏱️ Total execution time: {total_time:.2f} seconds")
    print(f"⏱️ Average processing time: {avg_processing_time:.2f} seconds/file")
    print(f"⚡ Throughput: {len(pdf_files_to_process)/total_time:.2f} files/second" if total_time > 0 else "⚡ Throughput: N/A")
    
    # Giải phóng bộ nhớ model
    del model_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    dependencies = {
        "PyMuPDF (fitz)": False,
        "OpenCV (cv2)": False,
        "Pillow (PIL)": False,
        "numpy": False
    }
    
    try:
        import fitz
        dependencies["PyMuPDF (fitz)"] = True
    except ImportError as e:
        logger.warning(f"PyMuPDF not installed: {str(e)}")
    
    try:
        import cv2
        dependencies["OpenCV (cv2)"] = True
    except ImportError as e:
        logger.warning(f"OpenCV not installed: {str(e)}")
    
    try:
        from PIL import Image
        dependencies["Pillow (PIL)"] = True
    except ImportError as e:
        logger.warning(f"Pillow not installed: {str(e)}")
    
    try:
        import numpy as np
        dependencies["numpy"] = True
    except ImportError as e:
        logger.warning(f"numpy not installed: {str(e)}")
    
    logger.info("=== Dependency Check ===")
    for dep, status in dependencies.items():
        status_str = "✅" if status else "❌"
        logger.info(f"{status_str} {dep}")
    logger.info("=======================")

if __name__ == "__main__":
    # Đường dẫn tuyệt đối
    current_dir = Path(__file__).parent.parent.parent
    input_dir = str(current_dir / "src" / "data" / "pdfs" / "inputs")
    output_dir = str(current_dir / "src" / "data" / "pdfs" / "outputs")
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Kiểm tra thư mục tồn tại
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        print("Please create the directory or update the path")
        exit(1)
    
    # === CẤU HÌNH CHO ĐỘ CHÍNH XÁC CAO VỚI XỬ LÝ LỖI ===
    convert_pdf_to_markdown(
        input_dir=input_dir,
        output_dir=output_dir,
        max_workers=None,    # Tự động điều chỉnh
        use_llm=False,       # Không dùng LLM để tránh thêm phức tạp
        force_ocr=True,      # Bắt buộc OCR để xử lý PDF quét
        languages="vi", # Hỗ trợ tiếng Việt + tiếng Anh
        force_reconvert=False,  # Đặt True để convert lại tất cả files
        enhanced_accuracy=True  # Bật chế độ độ chính xác cao với preprocessing
    )