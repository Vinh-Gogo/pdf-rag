from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re
from pathlib import Path
import sys

# GPU Configuration
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Model Configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"  # Optimized for speed/accuracy balance

def extract_page_number(filename: str) -> int:
    """Extract page number from filename (e.g., 'page_8.txt' -> 8)"""
    match = re.search(r"page_(\d+)", filename)
    return int(match.group(1)) if match else None

def generate_meta_block(
    text: str,
    model,
    tokenizer,
    current_page: int,
    prev_page_text: str = "",
    summary_prev_pages: str = "",
    next_page_text: str = "",
    summary_next_pages: str = "",
    has_flowchart: bool = False,
    max_retries: int = 2
) -> str:

    base_prompt = f"""You are a metadata extraction expert for RAG systems. Your ONLY task is to output EXACTLY ONE block in this format:

[
META:
current_page={current_page};

topic_prev_pages=<concise Vietnamese topic>;
summary_prev_pages = ["<short description 1>", "<short description 2>", ...];
topic_current_pages=<concise Vietnamese topic>;
summary_current_pages = ["<short description 1>", "<short description 2>", ...];
topic_next_pages=<concise Vietnamese topic>;
summary_next_pages = ["<short description 1>", "<short description 2>", ...];
]

RULES (VIOLATION = FAILURE):
1. OUTPUT ONLY THE [META: ...] BLOCK - NO OTHER TEXT.
2. TOPIC AND RELATED PAGES MUST BE IN VIETNAMESE.
3. If the current page represents the content of the previous page, then markdown takes the title of the previous page and marks (continued).
4. ATTENTION TO THE (*), (.), (,), (;), (-) MARKS TO UNDERSTAND THE CONTEXT BETTER.
5. SUMMARY PAGE: 2-8 short phrases EXTRACTED. If there are 8 lines (#) then there are 8 phrases.
6. Be honest, DO NOT include any content that is not provided. Do not add any questions to the summaries of each page
7. USE DOUBLE QUOTES FOR LIST ITEMS.
8. START WITH [META: AND END WITH ].

INPUT EXAMPLE:
```
Trang 6:

## Title 1
[Content]

## Title 2
[Content ..]

### Title 3
[Content ..]

## Title 4
[Content ..]

#### Title 5
[Content ..]

# Title 6
[Content ..]

#### Title 7
[Content ..]
```

OUTPUT EXAMPLE:
```
[META:
current_page=6;

topic_prev_pages=Title;
summary_prev_pages = ["Sub-Title 1", "Sub-Title 2", "Sub-Title 3", "Sub-Title 4", "Sub-Title 5"];
topic_current_pages=Title (Continued);
summary_current_pages = ["Sub-Title 1", "Sub-Title 2", "Sub-Title 3", "Sub-Title 4", "Sub-Title 5", "Sub-Title 6"];
topic_next_pages=Title (Continued);
summary_next_pages = ["Sub-Title 1", "Sub-Title 2", "Sub-Title 3", "Sub-Title 4", "Sub-Title 5", "Sub-Title 6"];
]
```

CURRENT PAGE CONTENT (page {current_page}):
{text.strip()}

PREVIOUS PAGE CONTENT (page {current_page-1}):
{prev_page_text.strip() or 'NO CONTENT'}

NEXT PAGE CONTENT (page {current_page+1}):
{next_page_text.strip() or 'NO CONTENT'}
"""

    for attempt in range(max_retries):
        messages = [
            {"role": "system", "content": "You are a precision metadata extraction system. Output ONLY the requested block."},
            {"role": "user", "content": base_prompt}
        ]

        try:
            # Tokenize and generate
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=1024*8,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    top_k=30,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            # Decode and clean
            generated = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            return generated.strip().split("</think>")[1].strip()

        except Exception as e:
            print(f"❌ Generation error on attempt {attempt+1}: {str(e)}")
            if attempt == max_retries - 1:
                # Only print final error on last attempt
                print(f"⚠️ All {max_retries} attempts failed for page {current_page}")

    # Fallback block (strict format compliance)
    return f"""[META:
current_page={current_page}
topic=chưa xác định
fragment_level=2
prev_page= (continuation=false, similarity=0.0)
next_page= (continuation=false, has_flowchart={str(has_flowchart).lower()})
related_pages= ["nội dung chưa xác định"]
cross_ref= ["unknown"]
]"""

def main():
    # Setup paths - More robust approach
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Try multiple possible directory structures
    possible_content_dirs = [
        project_root / "src" / "data" / "contents" / "pages",
        project_root / "src" / "data" / "contenrs" / "pages",  # Original typo
        project_root / "data" / "contents" / "pages",
        project_root / "data" / "pages",
        project_root / "pages"
    ]
    
    contents_dir = None
    for dir_path in possible_content_dirs:
        if dir_path.exists():
            contents_dir = dir_path
            print(f"✅ Found content directory: {dir_path.absolute()}")
            break
    
    if contents_dir is None:
        # Create a sample directory structure for testing
        contents_dir = project_root / "src" / "data" / "contents" / "pages"
        print(f"⚠️ No content directory found. Creating sample structure at: {contents_dir.absolute()}")
        contents_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample files for testing
        for i in range(1, 4):
            sample_file = contents_dir / f"page_{i}.txt"
            if not sample_file.exists():
                with open(sample_file, 'w', encoding='utf-8') as f:
                    f.write(f"Đây là nội dung mẫu cho trang {i}. Đây là một đoạn văn bản để kiểm tra hệ thống metadata.")
                print(f"📝 Created sample file: {sample_file.name}")
    
    output_dir = contents_dir.parent / "corrected_texts"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"🚀 Loading model: {MODEL_NAME}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).eval()
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Falling back to standard loading: {str(e)}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            ).eval()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        except Exception as inner_e:
            print(f"❌ Critical error loading model: {str(inner_e)}")
            sys.exit(1)

    # Collect pages
    content_files = {}
    for file in contents_dir.glob("*.txt"):
        if "_ocr" not in file.stem and file.is_file():
            page_num = extract_page_number(file.name)
            if page_num and page_num >= 1:  # Start from page 1
                content_files[page_num] = file

    if not content_files:
        print("❌ No valid content files found. Check directory structure.")
        return

    print(f"✅ Found {len(content_files)} pages to process (pages {min(content_files)}-{max(content_files)})")

    # Process pages sequentially
    processed_pages = []
    for page_num in sorted(content_files.keys()):
        print(f"\n{'='*50}")
        print(f"📄 Processing page {page_num}")
        print(f"{'='*50}")

        # Read current page
        with open(content_files[page_num], 'r', encoding='utf-8') as f:
            current_text = f.read()

        # Read context pages
        prev_text = ""
        next_text = ""
        has_flowchart = False
        
        if (page_num - 1) in content_files:
            with open(content_files[page_num-1], 'r', encoding='utf-8') as f:
                prev_text = f.read()
        
        if (page_num + 1) in content_files:
            next_file = content_files[page_num+1]
            with open(next_file, 'r', encoding='utf-8') as f:
                next_text = f.read()
            # Detect flowcharts in next page
            has_flowchart = any(term in next_text.lower() 
                              for term in ["flowchart", "biểu đồ", "sơ đồ", "chart", "graph", "diagram"])

        # Generate META block
        meta_block = generate_meta_block(
            text=current_text,
            model=model,
            tokenizer=tokenizer,
            current_page=page_num,
            prev_page_text=prev_text,
            summary_prev_pages=prev_text.split,
            next_page_text=next_text,
            has_flowchart=has_flowchart
        )

        # Save result
        output_path = output_dir / f"meta_page_{page_num}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(meta_block)
        
        print(f"✅ Generated META block for page {page_num}:")
        print(meta_block)
        processed_pages.append(page_num)

    # Final summary
    print(f"\n{'='*50}")
    print(f"🎉 PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"Pages processed: {len(processed_pages)} (pages {min(processed_pages)}-{max(processed_pages)})")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()