import json
import re

def parse_output_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('[ T ]') or line.startswith('[ F ]'):
            # Start new entry
            status = 'T' if '[ T ]' in line else 'F'
            entry_lines = []
            i += 1
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('[ T ]') or line.startswith('[ F ]'):
                    # Next entry starts
                    i -= 1
                    break
                if line:
                    entry_lines.append(line)
                i += 1
            # Parse entry_lines into dict
            entry = {'status': status}
            for eline in entry_lines:
                match = re.match(r'^\[ ([^]]+) \](.*)$', eline)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    entry[key] = value
            entries.append(entry)
        else:
            i += 1

    # Clean the keys
    key_map = {
        'idx': 'idx',
        'T': 'type',
        'Q': 'question',
        'A': 'answer',
        'Raw': 'raw',
        'Ref': 'ref'
    }

    cleaned_entries = []
    for entry in entries:
        cleaned = {}
        for k, v in entry.items():
            cleaned_key = key_map.get(k, k)
            cleaned[cleaned_key] = v
        cleaned_entries.append(cleaned)

    return cleaned_entries

if __name__ == "__main__":
    data = parse_output_txt('output.txt')
    with open('output.txt', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
