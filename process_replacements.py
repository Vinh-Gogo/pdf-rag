import json

# Load the JSON file
with open('src/data/json/all_plaintext.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# # Process each page's content
# for page in data['pages']:
#     content = page['content']
#     # Replace to ""
#     content = content.replace('|', '')
#     content = content.replace('-', '')
#     content = content.replace('#', '')
#     content = content.replace('**', '')
#     content = content.replace('*', '')
#     # Replace \n\n to " | "
#     content = content.replace('\n\n', ' | ')
#     # Replace \n to ", "
#     content = content.replace('\n', ', ')
#     page['content'] = content

# Save the modified JSON
with open('src/data/jsonl/all_plaintext.jsonl', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Processing complete.")
