with open('public/stopwords-vietnamese.txt', 'r', encoding='utf-8') as f:
    words = [line.strip() for line in f if line.strip()]

unique_words = list(set(words))

with open('public/stopwords-vietnamese.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sorted(unique_words)))