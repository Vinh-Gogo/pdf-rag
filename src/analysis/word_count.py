import os
import json

directory = 'src/data/pdfs/jsonl_stopwords'
total_words = 0

for filename in os.listdir(directory):
    if filename.endswith('.jsonl'):
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                text = data['text']
                words = text.split()
                total_words += len(words)
                print(len(words))

print(f"Total words: {total_words}")