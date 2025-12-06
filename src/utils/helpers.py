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

help = Helper()

path = r'src\data\push\pages\pages_data.json'
data = help.get_data_for_benchmark(path, debug=True)
