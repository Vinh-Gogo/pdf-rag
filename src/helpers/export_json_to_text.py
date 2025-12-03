import json
import os
import re
import unicodedata
# VietnameseToneNormalization.md
# https://github.com/VinAIResearch/BARTpho/blob/main/VietnameseToneNormalization.md

TONE_NORM_VI = {
    'òa': 'oà', 'Òa': 'Oà', 'ÒA': 'OÀ',\
    'óa': 'oá', 'Óa': 'Oá', 'ÓA': 'OÁ',\
    'ỏa': 'oả', 'Ỏa': 'Oả', 'ỎA': 'OẢ',\
    'õa': 'oã', 'Õa': 'Oã', 'ÕA': 'OÃ',\
    'ọa': 'oạ', 'Ọa': 'Oạ', 'ỌA': 'OẠ',\
    'òe': 'oè', 'Òe': 'Oè', 'ÒE': 'OÈ',\
    'óe': 'oé', 'Óe': 'Oé', 'ÓE': 'OÉ',\
    'ỏe': 'oẻ', 'Ỏe': 'Oẻ', 'ỎE': 'OẺ',\
    'õe': 'oẽ', 'Õe': 'Oẽ', 'ÕE': 'OẼ',\
    'ọe': 'oẹ', 'Ọe': 'Oẹ', 'ỌE': 'OẸ',\
    'ùy': 'uỳ', 'Ùy': 'Uỳ', 'ÙY': 'UỲ',\
    'úy': 'uý', 'Úy': 'Uý', 'ÚY': 'UÝ',\
    'ủy': 'uỷ', 'Ủy': 'Uỷ', 'ỦY': 'UỶ',\
    'ũy': 'uỹ', 'Ũy': 'Uỹ', 'ŨY': 'UỸ',\
    'ụy': 'uỵ', 'Ụy': 'Uỵ', 'ỤY': 'UỴ'
    }

def normalize_vnese(text):
    for i, j in TONE_NORM_VI.items():
        text = text.replace(i, j)
    # Remove control characters (ASCII 0–31, plus DEL 127)
    # text = re.sub(r'[\x00-\x1F\x7F]', '', text)
    # normalize spacing
    text = text.replace('\xa0', ' ')
    # Normalize input text to NFC
    text = unicodedata.normalize("NFC", text)
    return text

def export_json_to_text(input_file, output_dir):
    # Load JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract all content from pages with page headers
    all_content = []
    for page in data['pages']:
        page_num = page['page'].split('_')[1]  # Extract number from "page_X"
        content = str(page['content'])
        content = content.replace('\n\n|', ' \n\n|')
        content = content.replace('\n\n-', ' \n\n-')
        content = content.replace('**', ' ')
        content = content.replace('*', ' * ')
        content = content.replace(' # ', '# ')
        content = content.replace(' ## ', '## ')
        content = content.replace(' ### ', '### ')
        content = content.replace(' #### ', '#### ')
        
        content = content.replace('####', '\t+')
        content = content.replace('###', '\t+')
        content = content.replace('#', '')

        while '  ' in content:
            content = content.replace('  ', ' ')
        
        content = normalize_vnese(content)    
        
        content = repr(content)

        formatted_page = f"[Page {page_num}]\n\n{content}"
        all_content.append(formatted_page)

    # Join all formatted pages with newlines
    full_text = '\n\n'.join(all_content)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Write to output file
    output_file = os.path.join(output_dir, 'all_plaintext_final.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)

    print(f"Exported JSON to text file: {output_file}")

if __name__ == "__main__":
    input_file = r"D:\pdf-rag\src\data\json\all_plaintext.json"
    output_dir = r"D:\pdf-rag\src\data\push"
    export_json_to_text(input_file, output_dir)
