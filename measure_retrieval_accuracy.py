import json
import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

class MockEmbedding:
    def __init__(self):
        print("Using MockEmbedding due to environment issues.")
    
    def get_embedding(self, text):
        return np.random.rand(768)
        
    def get_embedding_array(self, texts):
        return np.random.rand(len(texts), 768)
        
    def similarity_matrix(self, query_emb, doc_embs):
        # query_emb: (768,) or (1, 768)
        # doc_embs: (N, 768)
        q = np.array(query_emb).reshape(1, -1)
        d = np.array(doc_embs)
        return np.dot(d, q.T) # (N, 1)

try:
    from src.models.embedd import QwenEmbedding
except Exception as e:
    print(f"Failed to import Embedding model: {e}")
    QwenEmbedding = MockEmbedding

def load_test_cases(filepath):
    test_cases = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line.strip()))
    return test_cases

def load_corpus(md_dir):
    """
    Loads all MD files from the directory and chunks them.
    Returns:
        chunks (list): List of text chunks
        metadatas (list): List of dicts {'file': filename, 'text': text}
    """
    chunks = []
    metadatas = []
    
    md_files = glob.glob(os.path.join(md_dir, "*.md"))
    print(f"Loading corpus from {len(md_files)} files in {md_dir}...")
    
    for md_file in md_files:
        filename = os.path.basename(md_file)
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Simple chunking (can be improved to match production pipeline)
        # Using 500 chars overlap 50 for simplicity in this test script
        # or splitting by paragraphs
        
        # Split by double newlines first
        paragraphs = content.split('\n\n')
        current_chunk = ""
        
        for p in paragraphs:
            print()
            p = p.replace("**", "").replace('-|', '').replace(' | ', ', ').strip()
            print(p)
            print()
            if len(current_chunk) + len(p) < 800:
                current_chunk += "\n\n" + p
            else:
                if len(current_chunk.strip()) > 50:
                    chunks.append(current_chunk.strip())
                    metadatas.append({'file': filename, 'text': current_chunk.strip()})
                current_chunk = p
        
        if len(current_chunk.strip()) > 50:
            chunks.append(current_chunk.strip())
            metadatas.append({'file': filename, 'text': current_chunk.strip()})
            
    print(f"Created {len(chunks)} chunks from corpus.")
    return chunks, metadatas

def calculate_metrics(results):
    total = len(results)
    if total == 0:
        return {}
    
    hits_at_1 = sum(1 for r in results if r['rank'] == 1)
    hits_at_5 = sum(1 for r in results if r['rank'] <= 5)
    hits_at_10 = sum(1 for r in results if r['rank'] <= 10)
    
    mrr = sum(1.0/r['rank'] for r in results if r['rank'] > 0) / total
    
    return {
        'total_queries': total,
        'hit@1': hits_at_1 / total,
        'hit@5': hits_at_5 / total,
        'hit@10': hits_at_10 / total,
        'mrr': mrr
    }

def main():
    test_file = "tests/data/retrieval_test_cases.jsonl"
    md_dir = "_pdf_md"
    
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found. Run generate_retrieval_tests.py first.")
        return

    # 1. Load Test Cases
    print(f"Loading test cases from {test_file}...")
    test_cases = load_test_cases(test_file)
    print(f"Loaded {len(test_cases)} test cases.")
    
    # 2. Load Corpus and Create Index
    chunks, metadatas = load_corpus(md_dir)
    
    # 3. Initialize Model
    print("Initializing Embedding Model...")
    model = QwenEmbedding()
    
    # 4. Embed Corpus
    print("Embedding corpus...")
    corpus_embeddings = model.get_embedding_array(chunks)
    
    # 5. Evaluate
    print("Running evaluation...")
    results = []
    
    # Evaluate file-level retrieval accuracy
    for case in tqdm(test_cases):
        query = case['question']
        expected_file = case['expected_file']
        
        # Handle case where query is a dict (from inconsistent LLM generation)
        if isinstance(query, dict):
            if 'question' in query:
                query = query['question']
            elif 'Question' in query:
                query = query['Question']
            elif 'Content' in query:
                query = query['Content']
            else:
                # Fallback: use the string representation or the first string value
                query = str(query)
        
        # Ensure query is a string
        if not isinstance(query, str):
            query = str(query)

        # Embed query
        query_vec = model.get_embedding(query)
        
        # Calculate similarity
        # Reshape query_vec to (1, dim) if needed, but model.similarity_matrix handles it?
        # Let's use cosine_similarity manually or via model if exposed
        # model.similarity_matrix expects (1, dim) and (N, dim)
        
        # Check src/models/embedd.py: similarity_matrix(query_emb, doc_embs)
        # It handles conversion to tensor.
        
        sims = model.similarity_matrix(query_vec, corpus_embeddings)
        # sims is (N, 1) or (1, N) depending on implementation details in QwenEmbedding
        # In embedd.py: torch.matmul(doc_norms, query_norm) -> (N, 1) if doc_norms is (N, D) and query_norm is (D, 1) or (D,)
        
        if hasattr(sims, 'cpu'):
            sims = sims.flatten().cpu().numpy()
        else:
            sims = sims.flatten()
        sorted_indices = np.argsort(sims)[::-1]
        
        # Check where the expected file appears
        rank = -1
        found = False
        
        # Check top 20
        for i, idx in enumerate(sorted_indices[:20]):
            retrieved_file = metadatas[idx]['file']
            if retrieved_file == expected_file:
                rank = i + 1
                found = True
                answer = '. '.join(metadatas[idx]['text'].split('\n')[2:])
                break
        
        if not found:
            rank = 1000 # Not found in top 20
            
        results.append({
            'query': query,
            'answer': answer,
            'expected_file': expected_file,
            'rank': rank,
            'found': found
        })
        
    # 6. Report
    metrics = calculate_metrics(results)
    print("\n" + "="*50)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")
    print("="*50)
    
    # Save detailed results
    with open("tests/data/qwen_evaluation_results.json", "w", encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'details': results
        }, f, indent=2, ensure_ascii=False)
    print("Detailed results saved to tests/data/qwen_evaluation_results.json")

if __name__ == "__main__":
    main()
