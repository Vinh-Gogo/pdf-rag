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
MODEL_NAME = "Qwen/Qwen3-1.7B"

def load_model():
    print(f"Loading model: {MODEL_NAME}")
    try:
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
        ).eval()
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
        print(f"[OK] Model loaded successfully: {MODEL_NAME}\n")
        return model, tokenizer
    except Exception as e:
        print(f"[ERROR] Error loading model {MODEL_NAME}: {e}")
        print("  Falling back to mock question generation for all files.\n")
        return None, None

def extract_key_topics(full_document):
    """Trich xuat cac chu de chinh tu toan bo document, loai bo bang va metadata."""
    clean_text = re.sub(r'\|.*?\|.*?\|', '', full_document)
    clean_text = re.sub(r'!\[.*?\]\(.*?\)', '', clean_text)
    clean_text = re.sub(r'\*\*(.*?)\*\*', r'\1', clean_text)
    clean_text = re.sub(r'\*(.*?)\*', r'\1', clean_text)
    
    sentences = [s.strip() for s in re.split(r'[.!?]', clean_text) if len(s.strip()) > 30]
    
    important_keywords = ['muc tieu', 'ket luan', 'de xuat', 'phan tich', 'danh gia', 'giai phap', 'van de', 'thach thuc']
    key_sentences = []
    
    for sent in sentences[:30]:
        if any(keyword in sent.lower() for keyword in important_keywords):
            key_sentences.append(sent)
            if len(key_sentences) >= 5:
                break
    
    return "\n".join(key_sentences[:5]) if key_sentences else " ".join(sentences[:3])

def smart_chunk_document(content):
    """
    Chia document theo cau truc:
    1. Theo section headers (## hoac ###) neu co
    2. Theo bold text sections (**...**)
    3. Theo long paragraphs neu khong co sections
    """
    chunks = []
    
    # Thu 1: Chia theo section headers (##)
    sections = re.split(r'\n##\s+', content)
    
    for section in sections:
        if len(section.strip()) < 100:
            continue
            
        if len(section) < 2000:
            chunks.append(section.strip())
        else:
            # Chia section lon theo subsection headers (###)
            subsections = re.split(r'\n###\s+', section)
            current_chunk = ""
            
            for subsection in subsections:
                if len(current_chunk) + len(subsection) <900:
                    if current_chunk:
                        current_chunk += "\n### " + subsection
                    else:
                        current_chunk = subsection
                else:
                    if len(current_chunk.strip()) > 300:
                        chunks.append(current_chunk.strip())
                    current_chunk = subsection
            
            if len(current_chunk.strip()) > 300:
                chunks.append(current_chunk.strip())
    
    # Neu khong co header, thu chia theo bold sections
    if not chunks:
        # Tach theo **bold** pattern
        bold_sections = re.split(r'(\*\*[^*]+\*\*)', content)
        
        current_chunk = ""
        for i, section in enumerate(bold_sections):
            if not section.strip():
                continue
            
            # Neu la bold text (starts with **), them vao chunk
            if section.startswith('**'):
                if current_chunk and len(current_chunk) > 300:
                    chunks.append(current_chunk.strip())
                current_chunk = section
            else:
                if len(current_chunk) + len(section) <900:
                    current_chunk += "\n" + section
                else:
                    if len(current_chunk.strip()) > 300:
                        chunks.append(current_chunk.strip())
                    current_chunk = section
        
        if len(current_chunk.strip()) > 300:
            chunks.append(current_chunk.strip())
    
    # Neu van khong co chunks, chia theo paragraphs
    if not chunks:
        content_normalized = re.sub(r'\n\n\n+', '\n\n', content)
        paragraphs = content_normalized.split('\n\n')
        
        current_chunk = ""
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            
            if len(current_chunk) + len(p) <900:
                if current_chunk:
                    current_chunk += "\n\n" + p
                else:
                    current_chunk = p
            else:
                if len(current_chunk.strip()) > 300:
                    chunks.append(current_chunk.strip())
                current_chunk = p
        
        if len(current_chunk.strip()) > 300:
            chunks.append(current_chunk.strip())
    
    # Loc chunks
    chunks = [chunk for chunk in chunks if 300 < len(chunk) < 3000]
    
    # Neu van khong co chunks, tra ve toan bo content nhung tach thanh 2-3 parts
    if not chunks and len(content.strip()) > 600:
        # Chia file thanh 2-3 parts
        content_clean = content.strip()
        part_size = len(content_clean) // 3
        
        chunk1 = content_clean[:part_size].strip()
        chunk2 = content_clean[part_size:2*part_size].strip()
        chunk3 = content_clean[2*part_size:].strip()
        
        for chunk in [chunk1, chunk2, chunk3]:
            if len(chunk) > 300:
                chunks.append(chunk)
    elif not chunks:
        chunks = [content.strip()]
    
    return chunks

def generate_questions(text_chunk, full_document_context, model, tokenizer):
    """Sinh cau hoi tu chunk voi ngu canh toan cuc, tranh table data."""
    if not model:
        return []
    
    has_table = bool(re.search(r'\|.*?\|.*?\|', text_chunk))
    doc_summary = extract_key_topics(full_document_context)
    
    if has_table:
        prompt = f"""Ban la chuyen gia tao cau hoi danh gia he thong truy xuat thong tin.

NGU CANH TOAN BO TAI LIEU:
{doc_summary}

BANG DU LIEU:
{text_chunk.strip()[:1200]}

YEU CAU:
Tao 4-5 cau hoi tieng Viet ve:
- Muc dich cua bang nay la gi?
- Xu huong chinh duoc the hien?
- Ket luan quan trong tu du lieu?
- Y nghia cua bang trong context cua tai lieu?

HAY HOI VE: Y nghia, muc dich, ket luan, xu huong
KHONG HOI VE: So lieu cu the, con so, o bang, dong cot

DINH DANG: JSON array cua chuoi tieng Viet, khong co giai thich."""
    else:
        prompt = f"""Ban la chuyen gia tao cau hoi danh gia he thong truy xuat thong tin.

NGU CANH TOAN BO TAI LIEU:
{doc_summary}

DOAN VAN CU THE:
{text_chunk.strip()[:1500]}

YEU CAU CAU HOI:
Tao 7 cau hoi tieng Viet chat luong cao ve:
1. Khai niem chinh, y nghia
2. Muc dich, ly do
3. Moi quan he giua cac y
4. Nguyen nhan - he qua
5. Tam quan trong, tac dong
6. Cac yeu to lien quan
7. Ket luan hoac danh gia

TAP TRUNG: Khai niem, y nghia, muc dich, moi quan he
DUNG NGON NGU: Tu nhien, khong trich dan truc tiep
TRANH:
   - Hoi so lieu cu the (doanh thu, ty le %, so luong, gia)
   - Hoi ve du lieu bang bieu
   - Hoi ten file, metadata ky thuat
   - Lap lai cau hoi trong cac bang

VI DU TOT:
- "Muc tieu chinh cua du an nay la gi?"
- "Tai sao giai phap nay duoc coi la quan trong?"
- "Moi quan he giua cac thanh phan duoc mo ta nhu the nao?"

VI DU XAU (tranh):
- "Doanh thu nam 2023 la bao nhieu?"
- "Bang 2 co bao nhieu dong?"

DINH DANG DAU RA: JSON array cua chuoi tieng Viet, khong co giai thich."""
    
    messages = [
        {"role": "system", "content": "Ban la tro ly tao cau hoi chat luong cao. Luon xuat ra JSON hop le la array chuoi."},
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
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.3,
                top_p=0.85,
                repetition_penalty=1.2
            )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        try:
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                questions = json.loads(match.group(0))
                if isinstance(questions, list) and len(questions) > 0:
                    filtered_questions = filter_invalid_questions(questions)
                    return filtered_questions[:7]
                else:
                    return []
            else:
                return []
        except Exception as e:
            return []
    except Exception as e:
        return []

def filter_invalid_questions(questions):
    """Loc cau hoi ve table data va thong tin khong phu hop."""
    table_keywords = [
        'bang', 'dong', 'cot', 'o', 'hang', 'cell', 'row', 'column',
        'bao nhieu', 'may', 'so luong', 'tong', 'chi tiet', 'danh sach',
        'so lieu', 'con so', 'ty le', '%', 'doanh thu', 'loi nhuan', 'gia',
        'nam', 'thang', 'quy', 'file', 'trang', 'metadata'
    ]
    
    filtered = []
    for q in questions:
        q_lower = q.lower()
        if not any(keyword in q_lower for keyword in table_keywords):
            filtered.append(q)
    
    return filtered

def generate_smart_mock_questions(chunk):
    """Sinh 7 cau hoi mock dua tren phan tich noi dung, tranh table data."""
    
    if re.search(r'\|.*?\|.*?\|', chunk):
        return [
            "Bang nay the hien thong tin gi?",
            "Muc dich chinh cua bang du lieu nay la gi?",
            "Ket luan quan trong tu bang nay la gi?",
            "Xu huong chinh duoc the hien trong bang?",
            "Bang nay ho tro cho luan diem nao trong tai lieu?",
            "Thong tin nao trong bang duoc coi la quan trong nhat?",
            "Moi lien he giua bang nay va cac phan khac cua tai lieu?"
        ]
    
    questions = []
    concept_questions = [
        "Y tuong chinh cua doan nay la gi?",
        "Khai niem quan trong duoc gioi thieu o day la gi?",
        "Muc dich chinh cua phan nay la gi?",
        "Tai sao thong tin nay lai quan trong?",
        "Ket luan rut ra tu doan van nay la gi?",
        "Moi quan he giua cac yeu to duoc mo ta nhu the nao?",
        "Dieu gi duoc nhan manh trong doan nay?"
    ]
    
    questions.extend(random.sample(concept_questions, min(5, len(concept_questions))))
    
    bold_entities = re.findall(r'\*\*([^*]+)\*\*', chunk)
    if bold_entities:
        entity = bold_entities[0].strip()[:60]
        questions.append(f'Thong tin ve "{entity}" co y nghia gi trong ngu canh nay?')
    
    companies = re.findall(r'(?:Cong ty|Tong cong ty|Chi nhanh|BIWASE|BIWELCO|GIWACO)[^,.;:]*', chunk)
    if companies:
        company = companies[0].strip()[:40]
        questions.append(f"Vai tro cua {company} duoc mo ta nhu the nao?")
    
    questions = list(dict.fromkeys(questions[:7]))
    
    while len(questions) < 7:
        questions.append(random.choice(concept_questions))
    
    return questions[:7]

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        pass
    
    model, tokenizer = load_model()
    use_mock = model is None
    
    md_files = sorted(glob.glob(os.path.join(MD_DIR, "*.md")))
    print(f"Found {len(md_files)} MD files.")
    print(f"Generation mode: {'MOCK (entity-based)' if use_mock else 'MODEL-BASED (Qwen-1.7B)'}")
    print(f"Improvements: Global context, smart chunking, table data filtering")
    print(f"Target: ~7 questions per chunk, ~100+ questions per file\n")
    
    test_cases = []
    total_questions_generated = 0
    
    for md_file in tqdm(md_files, desc="Processing files"):
        filename = os.path.basename(md_file)
        
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                full_content = f.read()
        except:
            try:
                with open(md_file, 'r', encoding='latin-1') as f:
                    full_content = f.read()
            except:
                continue
        
        chunks = smart_chunk_document(full_content)
        
        file_questions_count = 0
        for chunk in chunks:
            if model:
                questions = generate_questions(chunk, full_content, model, tokenizer)
            else:
                questions = generate_smart_mock_questions(chunk)
            
            for q in questions:
                if q and len(q.strip()) > 5:
                    test_case = {
                        "question": q,
                        "expected_file": filename,
                        "expected_text_snippet": chunk[:500]
                    }
                    test_cases.append(test_case)
                    file_questions_count += 1
                    total_questions_generated += 1
                    
                    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(test_case, ensure_ascii=False) + '\n')
        
        tqdm.write(f"  {filename}: {file_questions_count} questions ({len(chunks)} chunks)")
    
    print(f"\n" + "="*70)
    print(f"[OK] Generated {len(test_cases)} test cases")
    print(f"[OK] Average per file: {len(test_cases) / len(md_files):.1f} questions")
    print(f"[OK] Output saved to: {OUTPUT_FILE}")
    print("="*70)

if __name__ == "__main__":
    main()
