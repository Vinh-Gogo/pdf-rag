#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoModelForVision2Seq
from qwen_vl_utils import process_vision_info
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
import time
import re
import warnings
import sys
import os

# Add project root to path for imports when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from src.models.embedd import QwenEmbedding

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class OCRLLMProcessor:
    
    def __init__(self, model_id: str = "prithivMLmods/Qwen2-VL-OCR-2B-Instruct"):
        """
        Initialize OCR model with optimized GPU settings
        
        Args:
            model_id: HuggingFace model identifier
        """
        self.model_id = model_id
        self.device = self._setup_device()
        self.model = None
        self.processor = None
        
        print(f"🚀 Initializing OCR model: {model_id}")
        self._load_model()
        print("✅ OCR model loaded and ready!\n")
    
    def _setup_device(self):
        """Setup and configure device (GPU/CPU)"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            device = "cuda"
            print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"✓ CUDA: {torch.version.cuda}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            device = "cpu"
            print("⚠ No GPU found, using CPU (slower)")
        return device
    
    def _load_model(self):
        """Load and optimize the OCR model"""
        try:
            # self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            #     self.model_id,
            #     trust_remote_code=True,
            #     dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            #     device_map="auto",
            #     low_cpu_mem_usage=True
            # ).eval()
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, 
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).eval()
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.device == "cuda":
                print(f"✓ Mô hình đã tải lên GPU")
                print(f"✓ Bộ nhớ đã sử dụng: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            else:
                print(f"✓ Mô hình đã tải lên CPU")
            
            return self.model  
        except Exception as e:
            print(f"\n✗ Lỗi khi tải mô hình: {e}")
            raise e
    
    def process_image(self, image_path: str, additional_content: str = "", max_tokens: int = 1024 * 4) -> str:
        """Process a single image and extract OCR text"""
        image_path_obj = Path(image_path)
        if not image_path_obj.exists():
            raise FileNotFoundError(f"Image not found: {image_path_obj}")
        
        # Auto-load grammar reference if not provided
        if not additional_content:
            # Extract page number from image filename (e.g., page_1.png -> page_1)
            page_name = image_path_obj.stem
            grammar_file = Path("src/data/contents") / f"{page_name}.txt"
            
            if grammar_file.exists():
                additional_content = grammar_file.read_text(encoding="utf-8")
                print(f"📄 Loaded grammar reference: {grammar_file}")
            else:
                print(f"⚠ No grammar reference found: {grammar_file}")
        
        print(f"🔍 Đang xử lý: {image_path_obj}")
        start_time = time.time()
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(image_path_obj)},
                        {"type": "text", "text":  f"""
**Task**
You are an **OCR engine**. Your entire output must be the exact text that appears in the image, with no extra text added. Do not add, subtract, or change any characters. Keep all line breaks, spacing, punctuation, and case as shown.

**Context**
* Input: a clear photo or scan of a printed/handwritten page.
* Language: Vietnamese (may contain diacritics).
* Output format: plain UTF-8 text, no captions, no explanations, no credits.
* Segments:

<paragraph 1>
<paragraph 2>
<paragraph 3>

**All jumbled words from this image **

{additional_content}

"""},
                    ],
                }
            ]
            
            print(f"{additional_content[:50]}...")
            
            # Xử lý ảnh
            if self.processor and self.model:
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                vision_info = process_vision_info(messages)
                image_inputs = vision_info[0] if isinstance(vision_info, tuple) else vision_info
            
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            
                # Sinh kết quả OCR với tăng max_tokens
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        temperature=0.0,  
                        top_p=1.0,
                        repetition_penalty=1.05,
                    )
            
                # Giải mã kết quả
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
            
                # Loại bỏ các token đặc biệt và box coordinates
                output_text = output_text.replace("<|im_end|>", "").strip()
                
                # Loại bỏ box coordinates nếu có
                output_text = re.sub(r'<\|box_start\|>.*?<\|box_end\|>', '', output_text)
                output_text = re.sub(r'<\|.*?\|>', '', output_text)  # Loại bỏ các special tokens khác
                output_text = output_text.strip()
                
                # Tính thời gian xử lý
                processing_time = time.time() - start_time
                
                print(f"  ✓ Hoàn thành trong {processing_time:.2f}s")
                print(f"  ✓ Trích xuất: {len(output_text)} ký tự")
                
                # Dọn dẹp bộ nhớ GPU
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
               
                return output_text
            else:
                return ""
            
        except Exception as e:
            print(f"  ✗ Lỗi xử lý {image_path_obj.name}: {e}")
            raise e
    
    def _clean_output(self, text: str) -> str:
        """Clean and format OCR output text"""
        # Remove special tokens
        text = text.replace("<|im_end|>", "").strip()
        
        # Remove box coordinates if present
        text = re.sub(r'<\|box_start\|>.*?<\|box_end\|>', '', text)
        text = re.sub(r'<\|.*?\|>', '', text)  # Remove other special tokens
        
        return text.strip()
    
    def process_batch(self, image_dir: str, output_dir: Optional[str] = None, 
                     additional_content: str = "", start_index: int = 0, 
                     end_index: Optional[int] = None) -> dict:

        image_dir_obj = Path(image_dir)
        if output_dir is None:
            output_dir_obj = Path("src/data/results/ocr")
        else:
            output_dir_obj = Path(output_dir)
        
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        all_images = list(image_dir_obj.glob("*.png")) + \
                    list(image_dir_obj.glob("*.jpg")) + \
                    list(image_dir_obj.glob("*.jpeg"))
        
        # Sort by number in filename
        def extract_number(path):
            # Look for patterns like: page_1, page_01, page_001, etc.
            match = re.search(r'page_?(\d+)', path.stem, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # Fallback: look for any number in filename
            match = re.search(r'(\d+)', path.stem)
            return int(match.group(1)) if match else 0
        
        image_files = sorted(all_images, key=extract_number)
        
        # Apply index filtering
        if end_index is None:
            image_files = image_files[start_index:]
        else:
            image_files = image_files[start_index:end_index]
        
        if not image_files:
            print(f"❌ No images found in {image_dir_obj}")
            return {}
        
        print(f"\n📊 Found {len(image_files)} images to process")
        print(f"📂 Output directory: {output_dir_obj}")
        
        # Process images
        results_summary = []
        total_start_time = time.time()
        successful = 0
        
        for idx, image_path in enumerate(image_files, 1):
            
            print(f"\n[{idx}/{len(image_files)}] Đang xử lý: {image_path}")
            
            try:
                # Process image (additional_content will auto-load if empty)
                output_text = self.process_image(
                    str(image_path), 
                    additional_content=additional_content
                )
                
                # Lưu kết quả text
                output_filename = image_path.stem + "_ocr.txt"
                output_path = output_dir_obj / output_filename
                output_path.write_text(output_text, encoding="utf-8")
                
                print(f"  ✓ Đã lưu: {output_path}")
                successful += 1
                
                results_summary.append({
                    "image_file": image_path.name,
                    "output_file": output_filename,
                    "chars": len(output_text),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"  ✗ Lỗi: {e}")
                results_summary.append({
                    "image_file": image_path.name,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Summary
        total_time = time.time() - total_start_time
        failed = len(image_files) - successful
        
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 70)
        print(f"✓ Total images: {len(image_files)}")
        print(f"✓ Successful: {successful}")
        if failed > 0:
            print(f"❌ Failed: {failed}")
        print(f"✓ Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"✓ Average: {total_time/len(image_files):.2f}s/image")
        
        # Save summary
        summary_data = {
            "total_images": len(image_files),
            "successful": successful,
            "failed": failed,
            "total_time": round(total_time, 2),
            "average_time": round(total_time/len(image_files), 2),
            "device": self.device,
            "model": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "results": results_summary
        }
        
        summary_path = output_dir_obj / "batch_summary.json"
        summary_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print(f"📋 Summary saved: {summary_path}")
        
        if self.device == "cuda":
            print(f"🔧 Max GPU memory used: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
        
        return summary_data

if __name__ == "__main__":
    # Example usage
    try:
        # Initialize OCR processor
        ocr = OCRLLMProcessor("Qwen/Qwen3-VL-2B-Instruct")
        
        # Process batch of images
        results = ocr.process_batch(
            image_dir="src/data/ground_struct/pages_images_processed",
            output_dir="src/data/results/ocr_30",
            additional_content="",  # Add your context here
            start_index=85 # Start from image 129
        )
        
        print("\n✅ Batch processing completed!")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Critical error: {e}")