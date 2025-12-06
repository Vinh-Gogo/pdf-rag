import os
import re

def clean_markdown(content):
    lines = content.split('\n')
    cleaned = [line for line in lines if not line.strip().startswith('![]')]
    content = '\n'.join(cleaned).replace("<br>", " ").replace("## •", "##").replace("# •", "#")
    # content = repr(content)
    content = content.replace("<sup>\\*</sup>", "*")
    while '--' in content:
        content = content.replace("--", "-")
    while '  ' in content:
        content = content.replace("  ", " ")
    
    # Remove superscript references
    content = re.sub(r'<sup>*+</sup>', ' ', content)
    return content

def main():
    outputs_dir = 'src/data/pdfs/outputs/'
    process_dir = 'src/data/pdfs/process/'
    os.makedirs(process_dir, exist_ok=True)

    for subdir in os.listdir(outputs_dir):
        subdir_path = os.path.join(outputs_dir, subdir)
        if os.path.isdir(subdir_path):
            md_file = os.path.join(subdir_path, subdir + '.md')
            if os.path.exists(md_file):
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                cleaned = clean_markdown(content)
                output_file = os.path.join(process_dir, subdir + '.md')
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned)
                print(f"Processed {md_file} -> {output_file}")

if __name__ == "__main__":
    main()
