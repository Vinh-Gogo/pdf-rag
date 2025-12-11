import json
from pathlib import Path

class Helper:

    def read_json_content(self, file_path: str) -> list[str]:
        """
        Read JSON file and extract all 'content' values from the array of objects.

        Args:
            file_path (str): Path to the JSON file
        Returns:
            list[str]: List of all content values
        """
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        contents = []
        for item in data:
            if 'content' in item:
                contents.append(item['content'])

        return contents

    def get_data_for_benchmark(self, path: str, debug: bool=True) -> list[str]:
        
        page_content = self.read_json_content(path)
        seq_for_benchmark = []

        for i in range(len(page_content)):

            if debug:
                print(f"\n\n============= Trang thứ {i + 1}")

            content = page_content[i].replace("<ul><li>", '- ').replace("</li><li>", "- ").replace("</li></ul>", "")
            sequence = ""

            for seq in content.split('-'):
                if len(seq.split()) > 8:
                    sequence = seq
                else:
                    sequence += seq

                process_sequence = ' '.join(sequence.split()).replace('\n\n', ' ').replace('\n', '').replace("|", ",")

                seq_for_benchmark.append(process_sequence)

            if debug:
                print(f"\n== Độ dài đoạn: {len(sequence.split())}\n {seq}")

        return seq_for_benchmark

# help = Helper()

# path = r'src\data\push\pages\pages_data.json'
# data = help.get_data_for_benchmark(path, debug=True)

import json
from pathlib import Path


def process_md_to_jsonl(input_file: str, output_file: str = None) -> list[dict]:
    """
    Process a markdown file by splitting on "\n\n\n" and convert to JSONL format.
    
    Args:
        input_file: Path to the input markdown file
        output_file: Path to the output JSONL file (optional, defaults to input_file.jsonl)
    
    Returns:
        List of dictionaries with page and content keys
    """
    input_path = Path(input_file)
    
    if output_file is None:
        output_file = input_path.with_suffix('.jsonl')
    
    output_path = Path(output_file)
    
    # Read the markdown file
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by "\n\n\n" delimiter
    chunks = content.split("\n\n\n")
    
    # Create list of dictionaries
    results = []
    for i, chunk in enumerate(chunks, start=1):
        # Skip empty chunks
        chunk = chunk.strip()
        if chunk:
            results.append({
                "page": i,
                "content": chunk
            })
    
    # Write to JSONL file
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(results)} chunks from '{input_path}'")
    print(f"Output saved to '{output_path}'")
    
    return results
