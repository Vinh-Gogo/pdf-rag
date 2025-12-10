# System Patterns & Architecture

## System Overview
```
┌─────────────────────────────────────────────────────────────┐
│                    PDF-RAG Pipeline                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PDF → Markdown → Chunks → Questions → Embeddings → Eval  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Question Generation (generate_retrieval_tests.py)
- **Approach**: Dual-mode question generation
  - **Primary**: Qwen-1.7B model (semantic question generation)
  - **Fallback**: Entity extraction + template-based (when model unavailable)
- **Chunking Strategy**: 
  - Split by `\n\n\n` (triple newlines) into paragraphs
  - Accumulate paragraphs until chunk reaches ~1000 characters
  - Minimum chunk size: 100 characters
- **Questions Per Chunk**: 7 diverse questions covering:
  1. Dates/time periods
  2. Quantitative data (numbers, %)
  3. Key entities (companies, people)
  4. Main topics/subjects
  5. Events/activities
  6. Locations/places
  7. Conclusions/outcomes
- **Output Format**: JSONL lines with `{question, expected_file, expected_text_snippet}`

### 2. Evaluation (measure_retrieval_accuracy.py)
- **Corpus**: Load full markdown files (no chunking)
  - 34 files = 34 documents to rank
- **Query Processing**:
  - Embed each test question
  - Calculate similarity with all 34 file embeddings
  - Rank files by similarity score
  - Check if correct file appears in top 1, 3, or 10
- **Metrics Calculated**:
  - `accuracy@1`: Queries where correct file ranks #1
  - `accuracy@3`: Queries where correct file in top 3
  - `accuracy@10`: Queries where correct file in top 10
  - `mrr`: Mean Reciprocal Rank (1/avg_rank)

## Design Patterns

### Pattern 1: Graceful Degradation
- **Primary**: Use Qwen-1.7B model for question generation
- **Fallback**: Use entity extraction + templates if model fails
- **Result**: System always produces test cases (quality may vary)

### Pattern 2: File-Level Abstraction
- **Test Generation**: Work with chunks (for diverse questions)
- **Evaluation**: Work with full files (for retrieval accuracy)
- **Rationale**: Questions come from chunks, but retrieval targets files

### Pattern 3: GPU-Aware Inference
- **Auto-detect**: Check `torch.cuda.is_available()`
- **Precision**: Use bfloat16 for memory efficiency
- **Device**: Use `device_map="auto"` for flexible GPU placement
- **Logging**: Silent GPU memory tracking (avoid verbose output)

### Pattern 4: Vietnamese Language Processing
- **Question Generation**: Request 7 Vietnamese questions in prompt
- **Entity Extraction**: Vietnamese-specific patterns (dates, companies, names)
- **Evaluation**: No special processing (embeddings handle it)

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Qwen-1.7B (not 0.5B or 4B) | Balance between speed and quality |
| 7 questions/chunk (not 3) | More comprehensive test coverage |
| temperature=0.6 (not 0.7 or 0.5) | Balanced diversity vs consistency |
| max_new_tokens=1500 (up from 1024) | Accommodate 7 questions |
| File-level evaluation (not chunk) | Real-world retrieval scenario |
| top@10 check (not top@20) | Practical ranking threshold |
| JSONL output format | Streaming write, efficient parsing |

## Data Flow Architecture

```
QUESTION GENERATION
  Input: 34 MD files, ~14-15 chunks per file
    ↓
  Process: Qwen-1.7B generates 7Q/chunk
    ↓
  Fallback: Entity extraction + templates
    ↓
  Output: JSONL (tests/data/retrieval_test_cases.jsonl)
    ↓
  Result: ~273 test questions (8 per file avg)

EVALUATION
  Input: JSONL test cases + 34 MD files
    ↓
  Load: Embed all 34 files as complete documents
    ↓
  Process: For each question, rank files by similarity
    ↓
  Rank: Find position of correct file in top 10
    ↓
  Output: JSON (tests/data/evaluation_results.json)
    ↓
  Report: Metrics (accuracy@1, @3, @10, MRR)
```
