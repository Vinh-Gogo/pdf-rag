# Technical Context

## Technology Stack

### Core Dependencies
- **Python**: 3.12.9
- **PyTorch**: For model inference and GPU support
- **Transformers**: For Qwen model loading
- **NumPy**: For embedding operations
- **tqdm**: For progress bars

### AI/ML Models
- **Question Generation**: Qwen/Qwen3-1.7B (AutoModelForCausalLM)
- **Embeddings**: QwenEmbedding (from src.models.embedd)
- **Tokenizer**: Qwen's fast tokenizer with chat template support

### Development Environment
- **OS**: Windows with bash shell (git bash)
- **GPU Support**: CUDA-capable GPU (auto-detected)
- **Virtual Environment**: `.venv/` (Python 3.12.9)
- **Package Manager**: `uv` (modern Python package manager)

## Directory Structure

```
pdf-rag/
├── _pdf_md/                    # Converted markdown files (34 files)
├── _chunks_dbg/                # Debug chunks (for reference)
├── _pdf_table/                 # Extracted tables from PDFs
├── src/                        # Source code
│   └── models/
│       └── embedd.py           # QwenEmbedding implementation
├── tests/                      # Test infrastructure
│   └── data/
│       ├── retrieval_test_cases.jsonl    # Generated test questions
│       └── evaluation_results.json       # Evaluation metrics
├── neo4j/                      # Neo4j config (not currently used)
├── generate_retrieval_tests.py # Main: Question generation
├── measure_retrieval_accuracy.py # Main: Evaluation script
├── pyproject.toml              # Project metadata
└── memory-bank/                # This documentation
```

## File Formats

### Input: Markdown Files (_pdf_md/)
- 34 files total
- Naming: `BAN TIN BIWASE T*.md` (Vietnamese bulletin names)
- Content: Converted from PDF bulletins
- Sample files:
  - `BAN TIN BIWASE T1-2025 -A4 V2.md`
  - `ban-tin-biwase-t1-2023-a4.md`

### Output 1: Test Cases (tests/data/retrieval_test_cases.jsonl)
```json
{
  "question": "Vietnamese question text",
  "expected_file": "filename.md",
  "expected_text_snippet": "chunk text from the file"
}
```
- One JSON object per line
- Streaming write for memory efficiency
- ~273 questions total (~8 per file avg)

### Output 2: Evaluation Results (tests/data/evaluation_results.json)
```json
{
  "metrics": {
    "total_queries": 273,
    "accuracy@1": 0.4212,
    "accuracy@3": 0.6447,
    "accuracy@10": 0.8938,
    "mrr": 0.5609
  },
  "details": [
    {
      "query": "question text",
      "expected_file": "filename.md",
      "rank": rank_position,
      "found": true/false
    }
  ]
}
```

## Key Configuration Parameters

### Question Generation (generate_retrieval_tests.py)
```python
MODEL_NAME = "Qwen/Qwen3-1.7B"
MD_DIR = "_pdf_md"
OUTPUT_FILE = "tests/data/retrieval_test_cases.jsonl"

# Generation parameters
max_new_tokens = 1500
temperature = 0.6
top_p = 0.9
questions_per_chunk = 7
chunk_size_limit = 1000  # characters

# Fallback strategy
use_mock_when_model_fails = True
entity_extraction_patterns = {
    "dates": r'(?:ngày\s+)?\d{1,2}/\d{1,2}/\d{4}',
    "numbers": r'\d+(?:[.,]\d+)?\s*(?:tỷ|triệu|%)',
    "companies": r'(?:Công ty|BIWASE|BIWELCO)',
}
```

### Evaluation (measure_retrieval_accuracy.py)
```python
TEST_FILE = "tests/data/retrieval_test_cases.jsonl"
MD_DIR = "_pdf_md"
OUTPUT_FILE = "tests/data/evaluation_results.json"

# Evaluation parameters
top_k_check = 10  # Check top 1, 3, 10
metrics_calculated = ['accuracy@1', 'accuracy@3', 'accuracy@10', 'mrr']
```

## GPU Considerations

### Detection
```python
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
```

### Optimization
- `dtype=torch.bfloat16`: Mixed precision for memory efficiency
- `device_map="auto"`: Automatic GPU placement
- `torch.compile()`: Graph optimization (reduce-overhead mode)
- Silent memory tracking: Avoid verbose output

## API/Service Integration

### QwenEmbedding Class (src/models/embedd.py)
- `get_embedding(text)`: Single text → embedding vector
- `get_embedding_array(texts)`: Multiple texts → embedding matrix
- `similarity_matrix(query_emb, doc_embs)`: Cosine similarity computation

### Fallback: MockEmbedding
- Used when real embedding model unavailable
- Returns random vectors (for testing)
- Implements same interface as QwenEmbedding

## Development Commands

```bash
# Activate environment
source .venv/Scripts/activate

# Generate test questions
python generate_retrieval_tests.py

# Run evaluation
python measure_retrieval_accuracy.py

# View results
cat tests/data/evaluation_results.json | python -m json.tool
```

## Known Dependencies & Constraints
- **Python 3.12.9 required** (specified in pyproject.toml)
- **GPU memory**: Qwen-1.7B needs ~4-8GB VRAM
- **Model loading**: First run downloads from HuggingFace (~3.5GB)
- **Tokenizer**: Requires `trust_remote_code=True` for Qwen
- **Embedding model**: Must be available in src/models/embedd.py
