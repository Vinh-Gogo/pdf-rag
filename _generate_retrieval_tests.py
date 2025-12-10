import os
import json
import glob
import re
import random
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configuration
MD_DIR = "_pdf_md"
OUTPUT_FILE = "tests/data/retrieval_test_cases.jsonl"
# MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" # Using a lighter model for generation speed, or match env
# If you want to use the one from llm_text_correction.py:
# MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" 
MODEL_NAME = "Qwen/Qwen3-1.7B"

def load_model():
    print(f"Loading model: {MODEL_NAME}")
    try:
        # Check GPU availability
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("No GPU detected. Using CPU inference (slower).")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            # attn_implementation="flash_attention_2"
        ).eval()
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
        print(f"✓ Model loaded successfully: {MODEL_NAME}\n")
        return model, tokenizer
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n✗ Error loading model {MODEL_NAME}: {e}")
        print("  Falling back to mock question generation for all files.\n")
        return None, None

def generate_questions(text_chunk, model, tokenizer):
    if not model:
        return []

    prompt = f"""Based on the following text, generate 7 specific, diverse questions in Vietnamese that can be answered using the information in the text.
Include questions about:
1. Specific dates or time periods mentioned
2. Quantitative data (numbers, percentages, amounts)
3. Key entities (companies, people, organizations)
4. Main topics or subjects
5. Events or activities described
6. Locations or places mentioned
7. Conclusions or outcomes

Format the output as a JSON list of strings.
    
Text:
"{text_chunk[:2000]}"
    
Output JSON:
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates diverse questions for retrieval testing. Always output strictly valid JSON as an array of strings."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Log GPU memory usage if available
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            if peak_memory > 0.1:  # Only log if significant
                pass  # Silent GPU tracking
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Try to parse JSON
        try:
            # Find JSON list in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                if isinstance(questions, list) and len(questions) > 0:
                    return questions[:7]  # Limit to 7 questions
                else:
                    return []
            else:
                return []
        except Exception as e:
            return []
    except Exception as e:
        return []


def generate_smart_mock_questions(chunk):
    """
    Generate 7 realistic Vietnamese questions based on content analysis.
    Extracts entities and uses natural language templates.
    """
    questions = []
    
    # Extract potential entities
    # Dates: ngày X/Y/Z, tháng X, năm YYYY
    dates = re.findall(r'(?:ngày\s+)?\d{1,2}/\d{1,2}/\d{4}|\btháng\s+\d{1,2}(?:/\d{4})?|\bnăm\s+\d{4}', chunk, re.IGNORECASE)
    
    # Numbers with units (revenue, percentage, etc.)
    numbers = re.findall(r'\d+(?:[.,]\d+)?\s*(?:tỷ|triệu|%|m³|MW|km|tấn|đồng|USD|EUR)', chunk, re.IGNORECASE)
    
    # Bold text (often important entities) - markdown format
    bold_entities = re.findall(r'\*\*([^*]+)\*\*', chunk)
    
    # Company/Organization names (patterns like "Công ty", "BIWASE", etc.)
    companies = re.findall(r'(?:Công ty|Tổng công ty|Chi nhánh|BIWASE|BIWELCO|GIWACO)[^,.;:]*', chunk)
    
    # Person names with titles
    persons = re.findall(r'(?:ông|bà|Chủ tịch|Giám đốc|Phó)\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]*)*', chunk)
    
    # Question templates
    templates_date = [
        "Sự kiện này diễn ra vào thời gian nào?",
        "Ngày nào được đề cập trong văn bản?",
        "{date} có sự kiện gì xảy ra?",
    ]
    
    templates_number = [
        "Số liệu {number} liên quan đến nội dung gì?",
        "Kết quả đạt được là bao nhiêu?",
        "Con số {number} thể hiện điều gì?",
    ]
    
    templates_company = [
        "{company} đã thực hiện hoạt động gì?",
        "Thông tin về {company} trong văn bản là gì?",
        "{company} có vai trò gì trong nội dung này?",
    ]
    
    templates_person = [
        "{person} đã phát biểu hoặc làm gì?",
        "Vai trò của {person} là gì?",
        "{person} có liên quan đến sự kiện nào?",
    ]
    
    templates_bold = [
        "{entity} là gì?",
        "Thông tin chi tiết về {entity}?",
        "Nội dung liên quan đến {entity} là gì?",
    ]
    
    templates_general = [
        "Nội dung chính của đoạn văn này là gì?",
        "Thông tin quan trọng nào được đề cập?",
        "Văn bản này nói về chủ đề gì?",
        "Chủ đề chính của đoạn này là gì?",
        "Điều gì được nhấn mạnh trong văn bản này?",
    ]
    
    # Generate questions based on extracted entities
    if dates and random.random() > 0.3:
        template = random.choice(templates_date)
        if "{date}" in template:
            q = template.format(date=dates[0])
        else:
            q = template
        if q not in questions:
            questions.append(q)
    
    if numbers and random.random() > 0.3:
        template = random.choice(templates_number)
        if "{number}" in template:
            q = template.format(number=numbers[0])
        else:
            q = template
        if q not in questions:
            questions.append(q)
    
    if companies and random.random() > 0.4:
        company = companies[0].strip()[:50]  # Limit length
        template = random.choice(templates_company)
        q = template.format(company=company)
        if q not in questions:
            questions.append(q)
    
    if persons and random.random() > 0.5:
        person = persons[0].strip()
        template = random.choice(templates_person)
        q = template.format(person=person)
        if q not in questions:
            questions.append(q)
    
    if bold_entities and random.random() > 0.3:
        entity = bold_entities[0].strip()[:80]  # Limit length
        template = random.choice(templates_bold)
        q = template.format(entity=entity)
        if q not in questions:
            questions.append(q)

    # Fill up to 7 questions with general questions if needed
    while len(questions) < 7:
        general_q = random.choice(templates_general)
        if general_q not in questions:
            questions.append(general_q)
        else:
            break
    
    # If still not enough, add context-based questions
    if len(questions) < 7:
        # Extract first meaningful sentence
        sentences = re.split(r'[.!?]', chunk)
        first_sentence = next((s.strip() for s in sentences if len(s.strip()) > 20), None)
        if first_sentence:
            # Create question from first sentence topic
            topic_words = first_sentence.split()[:8]
            topic = " ".join(topic_words)
            q = f"Thông tin về \"{topic}...\" là gì?"
            if q not in questions:
                questions.append(q)
    
    # Remove duplicates and limit to 7 questions max
    return list(dict.fromkeys(questions[:7]))

def main():
    # Create output directory
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Clear previous output file (overwrite instead of append)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass  # Clear file
    
    # Load model
    model, tokenizer = load_model()
    use_mock = model is None
    
    md_files = sorted(glob.glob(os.path.join(MD_DIR, "*.md")))
    print(f"Found {len(md_files)} MD files.")
    print(f"Question generation mode: {'MOCK (entity-based)' if use_mock else 'MODEL-BASED (Qwen-1.7B)'}")
    print(f"Target: ~7 questions per chunk, ~100 questions per file\n")
    
    test_cases = []
    total_questions_generated = 0
    
    for md_file in tqdm(md_files, desc="Processing files"):
        filename = os.path.basename(md_file)
        
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Split content into chunks (e.g., by headers or paragraphs)
        # Simple splitting by triple newlines for now, grouping paragraphs
        paragraphs = content.split('\n\n\n')
        chunks = []
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) < 1000:
                current_chunk += "\n\n" + p
            else:
                if len(current_chunk.strip()) > 100:
                    chunks.append(current_chunk.strip())
                current_chunk = p
        if len(current_chunk.strip()) > 100:
            chunks.append(current_chunk.strip())
        
        # Process ALL chunks from the file (not just 2 random)
        file_questions_count = 0
        for chunk in chunks:
            if model:
                questions = generate_questions(chunk, model, tokenizer)
            else:
                # Smart mock questions based on content analysis
                questions = generate_smart_mock_questions(chunk)

            for q in questions:
                if q and len(q.strip()) > 5:  # Basic validation
                    test_case = {
                        "question": q,
                        "expected_file": filename,
                        "expected_text_snippet": chunk
                    }
                    test_cases.append(test_case)
                    file_questions_count += 1
                    total_questions_generated += 1
                    
                    # Write immediately to file (append mode)
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as elem:
                        elem.write(json.dumps(test_case, ensure_ascii=False) + '\n')
        
        # Log file-level statistics
        tqdm.write(f"  {filename}: {file_questions_count} questions ({len(chunks)} chunks)")

    print(f"\n" + "="*60)
    print(f"✓ Generated {len(test_cases)} test cases total")
    print(f"✓ Average per file: {len(test_cases) / len(md_files):.1f} questions")
    print(f"✓ Output saved to: {OUTPUT_FILE}")
    print("="*60)

if __name__ == "__main__":
    main()
