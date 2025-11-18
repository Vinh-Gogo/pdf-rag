import os
import glob
from typing import List


def _numeric_suffix(path: str) -> int:
    base = os.path.basename(path)
    num_part = base.replace('page_cleared_', '').replace('.txt', '')
    try:
        return int(num_part)
    except ValueError:
        return 10**12


def preprocess_text() -> None:
    contents_glob = os.path.join('src', 'data', 'contents', 'page_cleared_*.txt')
    files: List[str] = glob.glob(contents_glob)
    files.sort(key=_numeric_suffix)

    drop_dir = os.path.join('src', 'data', 'contents', 'drop')
    os.makedirs(drop_dir, exist_ok=True)

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # print(len(files))
        
        lines = content.splitlines()
        start_idx = None
        for idx, line in enumerate(lines):
            if ('#' in line) or ('＃' in line):
                start_idx = idx
                break

        if start_idx is None:
            output = content
            action = "Copied unchanged (no '#')"
        else:
            output = '\n'.join(lines[start_idx:])
            # Preserve trailing newline if existed
            if content.endswith(('\n', '\r\n')):
                output += '\n'
            action = 'Processed and saved'

        output_path = os.path.join(drop_dir, os.path.basename(filepath))
        with open(output_path, 'w', encoding='utf-8') as out:
            out.write(output)
        print(f"{action}: {output_path}")


if __name__ == '__main__':
    preprocess_text()
