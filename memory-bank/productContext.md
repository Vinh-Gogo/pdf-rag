# Product Context

## Why This Project Exists
BIWASE is a Vietnamese water and environment corporation that publishes monthly bulletins containing:
- Production and business activities
- Key metrics and performance data
- Announcements and news
- Personnel activities and achievements

These bulletins need to be:
1. **Searchable**: Allow users to query and find relevant information
2. **Testable**: Have ground truth test cases for retrieval system evaluation
3. **Reliable**: Ensure the RAG system can find the correct source document for any query

## Problems It Solves
1. **Manual Search Challenge**: Finding specific information in 34+ monthly bulletins is tedious
2. **Retrieval Quality Assessment**: Without test cases, can't measure if the system finds correct documents
3. **Language-Specific Retrieval**: Vietnamese documents need appropriate embeddings and evaluation

## How It Should Work

### End-to-End Flow
```
PDF Bulletins (34 files)
    ↓
Convert to Markdown (_pdf_md/)
    ↓
Generate Test Questions (generate_retrieval_tests.py)
    - Extract all chunks from each file
    - Generate 7 questions per chunk using Qwen-1.7B
    - Output: JSONL with {question, expected_file, snippet}
    ↓
Evaluate Retrieval (measure_retrieval_accuracy.py)
    - Embed full markdown files
    - For each question: embed query, rank files by similarity
    - Check if correct file appears in top 1, 3, or 10
    - Calculate metrics: accuracy@1, @3, @10, MRR
    ↓
Results (evaluation_results.json)
    - Metrics summary
    - Detailed results per query
```

## User Experience Goals
1. **Question Generation**: Automatic, diverse, Vietnamese language
2. **Evaluation**: Fast, accurate, comprehensive metrics
3. **Results**: Clear reporting of retrieval quality
4. **Reliability**: Fallback mechanisms when model unavailable

## Current Results (Baseline)
- **273 test questions** generated across 34 files
- **Accuracy@1**: 42.1% (good first-rank accuracy)
- **Accuracy@3**: 64.5% (top 3 covers ~2/3 of queries)
- **Accuracy@10**: 89.4% (top 10 covers most queries)
- **MRR**: 0.561 (solid average reciprocal rank)

**Interpretation**: System performs well for file-level retrieval. Most queries find the correct document in top 3, and almost all find it in top 10.
