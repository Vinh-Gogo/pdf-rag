import re

file_path = "_generate_retrieval_tests.py"

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find the generate_questions function
func_start = content.find("def generate_questions(text_chunk, model, tokenizer):")
func_end = content.find("\ndef generate_smart_mock_questions(", func_start)

new_function = '''def generate_questions(text_chunk, model, tokenizer):
    """Generate specific, factual Vietnamese questions from document text."""
    if not model:
        return []

    # Extract specific information for concrete questions
    dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', text_chunk)
    months_years = re.findall(r'tháng\s+\d{1,2}/\d{4}', text_chunk)
    numbers_with_units = re.findall(r'\d+(?:[.,]\d+)?\s*(?:tỷ|triệu|%|m³|MW|km|tấn|đồng|USD|EUR)', text_chunk)
    companies = re.findall(r'(?:Công ty|Tổng công ty|Chi nhánh|BIWASE|BIWELCO|GIWACO|Ngân hàng)[^,.;\\n]*', text_chunk)
    persons = re.findall(r'(?:ông|bà|Chủ tịch|Giám đốc|Phó|Thứ trưởng)\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+(?:\s+[A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]*)*', text_chunk)
    projects = re.findall(r'\*\*([^*]{10,100})\*\*', text_chunk)

    user_prompt = f"""Generate 7 highly specific, factual questions in Vietnamese based on the following document excerpt.

EXTRACTED INFORMATION:
- Dates/Months: {', '.join(set(dates + months_years))[:100] if dates or months_years else 'None'}
- Numbers/Metrics: {', '.join(set(numbers_with_units))[:100] if numbers_with_units else 'None'}
- Organizations: {', '.join(set([o.strip()[:50] for o in companies]))[:100] if companies else 'None'}
- People/Persons: {', '.join(set(persons[:3]))[:80] if persons else 'None'}
- Projects/Initiatives: {', '.join(set(projects[:2]))[:100] if projects else 'None'}

DOCUMENT TEXT:
{text_chunk[:2000]}

REQUIREMENTS:
Generate 7 questions in Vietnamese that:
1. Are highly SPECIFIC and CONCRETE - use exact names, dates, numbers from the text
2. Ask about ACTUAL EVENTS, ACTIVITIES, ROLES, and RESULTS mentioned in the document
3. Reference SPECIFIC ORGANIZATIONS, PEOPLE, PROJECTS, and DATES
4. Are answerable directly from the provided text

GOOD EXAMPLES (specific and factual):
- "Doanh thu năm tháng 4/2025 là bao nhiêu?" (revenue in April 2025)
- "Ngày 15/03/2025 xảy ra điều gì?" (what happened on 15/03/2025)
- "BIWASE có vai trò gì trong việc phát triển hạ tầng nước?" (BIWASE's role in water infrastructure)
- "Các điểm nổi bật trong bản tin tháng 12/2023 là gì?" (highlights in December 2023 bulletin)
- "Vì sao ông X nhấn mạnh tầm quan trọng của quản lý nước?" (why person X emphasized water management)
- "Chi nhánh Xử lý chất thải Bình Dương đã thực hiện hoạt động gì?" (what did the waste treatment branch do)
- "Thông tin chi tiết về Tổ hợp lò hơi - turbine – phát điện công suất 5MW là gì?" (details about specific equipment)

BAD EXAMPLES (generic, abstract - AVOID):
- "Khái niệm và ý nghĩa của dự án này là gì?" (too abstract)
- "Các yếu tố liên quan ảnh hưởng đến hiệu suất?" (too vague)
- "Nguyên nhân - hậu quả của dự án?" (too generic)

OUTPUT FORMAT:
Return a valid JSON array of exactly 7 strings (questions in Vietnamese), with no numbering, no explanations, and no additional text."""
    
    messages = [
        {"role": "system", "content": "You are an expert question generator for Vietnamese information retrieval systems. Generate highly specific, factual questions based on concrete details from documents. Always output valid JSON as an array of strings with no numbering or explanations."},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=600,
                do_sample=True,
                temperature=0.25,
                top_p=0.80,
                repetition_penalty=1.3
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
                    return questions[:7]
                else:
                    return []
            else:
                return []
        except Exception as e:
            return []
    except Exception as e:
        return []

'''

if func_start != -1 and func_end != -1:
    new_content = content[:func_start] + new_function + "\n" + content[func_end:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✓ Successfully updated generate_questions function")
    print("  - Added specific information extraction (dates, numbers, organizations, people)")
    print("  - Implemented English system prompt with clear instructions")
    print("  - Implemented English user prompt with examples and requirements")
    print("  - Lowered temperature to 0.25 for more focused output")
    print("  - Increased max_new_tokens to 600 for complete responses")
else:
    print("✗ Could not find function to replace")
