# Common Tasks & Patterns

## Retrieval Testing Flow

**Last performed**: 2025-12-10

### Steps
1. **Generate Test Cases**:
   Run `python generate_retrieval_tests.py` to create `tests/data/retrieval_test_cases.jsonl`.
   *(Requires Qwen model or fallback to Mock)*

2. **Run Evaluation**:
   Run `python measure_retrieval_accuracy.py` to calculate metrics.
   - Outputs: Console summary + `tests/data/evaluation_results.json`.

### Key Metrics to Watch
- **Accuracy@5**: Primary success metric (Aim for > 85%).
- **MRR**: Mean Reciprocal Rank (Aim for > 0.7).

---

## Adding New Embedding Model

### Steps
1. Create new class in `src/models/` implementing `get_embedding` interface.
2. Ensure `sentence_transformers` loading handles CUDA/CPU correctly.
3. Update `src/pipeline/pipeline_pdf_vec.py` to support new model name.
4. Update `measure_retrieval_accuracy.py` to import new model for testing.

---

## Markdown to JSONL Processing

**Last performed**: 2025-12-10

### Purpose
Convert processed markdown files (from PDFs) into JSONL format for RAG ingestion.

### Steps
1. **Process Single File**:
   ```bash
   python src/helpers/md_to_jsonl.py "path/to/file.md" -o "path/to/output.jsonl"
   ```

2. **Process Directory** (all `.md` files):
   ```bash
   python src/helpers/md_to_jsonl.py "path/to/directory/" -d
   ```

### Output Format
```jsonl
{"page": 1, "content": "Content of first section..."}
{"page": 2, "content": "Content of second section..."}
```

### Notes
- Splits markdown content by `\n\n\n` delimiter.
- Normalizes content by replacing newlines with single spaces.
- Page numbers are 1-indexed based on chunk position.
