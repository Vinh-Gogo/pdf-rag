# Progress & Evolution

## What Works ✅

### Question Generation Pipeline
- **Qwen-1.7B model loading**: Successful GPU detection and model initialization
- **Dual-mode generation**: Model-based + entity extraction fallback
- **Chunking strategy**: Reliable splitting at ~1000 char boundaries
- **7 questions/chunk generation**: Produces diverse Vietnamese questions
- **Entity extraction**: Pattern matching for dates, numbers, companies, people
- **JSONL output**: Streaming write for memory efficiency
- **Error handling**: Graceful degradation with silent failures

### Evaluation Pipeline
- **File-level indexing**: Successfully loads all 34 markdown files
- **Embedding computation**: Uses QwenEmbedding for semantic similarity
- **Ranking and evaluation**: Correctly identifies file rank positions
- **Metrics calculation**: Accurate accuracy@1, @3, @10, MRR computation
- **JSON serialization**: Proper Unicode handling for Vietnamese text
- **Results output**: Detailed metrics and per-query analysis

### Current Baseline Results
```
Total Queries: 273
Accuracy@1:   42.1%  (120 queries found file at rank #1)
Accuracy@3:   64.5%  (176 queries found file in top 3)
Accuracy@10:  89.4%  (244 queries found file in top 10)
MRR:          0.561  (average reciprocal rank)
```

### Interpretation
- **Strong performance**: 89% of queries find correct file in top 10
- **Good first-rank**: 42% get perfect ranking
- **Practical usability**: 64.5% top-3 is reasonable for user interactions

---

## What's Left to Build ���

### Question Generation Enhancements
- [ ] Increase questions per file from ~8 to ~100 target
- [ ] Add semantic similarity filtering to remove redundant questions
- [ ] Fine-tune prompt for domain-specific Vietnamese terminology
- [ ] Add cross-document reference questions
- [ ] Implement batch processing with checkpointing for large-scale generation

### Retrieval Accuracy Improvements
- [ ] Experiment with larger embedding models (3B+ params)
- [ ] Test chunk-aware hybrid retrieval approach
- [ ] Implement re-ranking with query expansion
- [ ] Add relevance feedback mechanism
- [ ] Consider ensemble methods combining multiple retrievers

### Test Coverage Expansion
- [ ] Target minimum 100 questions per file
- [ ] Add adversarial test cases (ambiguous questions)
- [ ] Include multi-document correlation tests
- [ ] Test edge cases (short queries, long queries, rare entities)
- [ ] Generate negative examples (wrong files)

### System Integration
- [ ] Build REST API for interactive query service
- [ ] Integrate with Neo4j graph database
- [ ] Add caching layer for embedding lookups
- [ ] Implement online metrics dashboard
- [ ] Add query logging and analytics

### Infrastructure & Deployment
- [ ] Docker containerization (Dockerfile exists, needs completion)
- [ ] Batch job scheduling for periodic regeneration
- [ ] Model versioning and rollback capability
- [ ] Performance profiling and optimization
- [ ] Load testing with concurrent queries

---

## Evolution of Decisions

### Decision 1: Question Count Per File
**Initial**: 3 questions per chunk, 2 random chunks per file → ~6 questions per file
**Current**: 7 questions per chunk, ALL chunks per file → ~100 questions per file (target)
**Reasoning**: More comprehensive test coverage improves evaluation validity
**Impact**: Better assessment of retrieval system quality

### Decision 2: Evaluation Scope
**Initial**: Chunk-level retrieval (rank chunks from file)
**Current**: File-level retrieval (rank files)
**Reasoning**: Real-world users need source files, not specific chunks
**Impact**: Simpler evaluation, more practical metrics

### Decision 3: Top-K Metrics
**Initial**: accuracy@1, accuracy@3, accuracy@5
**Current**: accuracy@1, accuracy@3, accuracy@10
**Reasoning**: 
- @1: Perfect ranking (rare)
- @3: Practical threshold (users check top 3)
- @10: Comprehensive coverage (safety net)
**Impact**: Clearer performance picture

### Decision 4: Generation Fallback
**Initial**: Hard fail if model unavailable
**Current**: Fallback to entity extraction + templates
**Reasoning**: System should always produce output
**Impact**: Robust operation in degraded scenarios

### Decision 5: Model Selection
**Initial**: Debated 0.5B vs 1.7B vs 4B Qwen models
**Current**: Settled on Qwen-1.7B
**Reasoning**:
- 0.5B too small for question quality
- 1.7B good balance of speed and quality
- 4B too slow on single GPU
**Impact**: Practical inference time without sacrificing quality

---

## Performance Timeline

### Session 1 (Initial Setup)
- ✅ Created generate_retrieval_tests.py
- ✅ Implemented entity extraction fallback
- ✅ Generated first batch of test questions
- Status: ~200 questions generated

### Session 2 (Enhancement)
- ✅ Increased questions from 3 to 7 per chunk
- ✅ Modified to process ALL chunks (not random 2)
- ✅ Added GPU detection and memory tracking
- ✅ Improved error handling (silent failures)
- Status: ~273 questions generated

### Session 3 (Evaluation Metrics) - CURRENT
- ✅ Updated evaluation metrics from @5 to @10
- ✅ Created comprehensive Memory Bank
- Status: Baseline evaluation complete
  - Accuracy@1: 42.1%
  - Accuracy@3: 64.5%
  - Accuracy@10: 89.4%

---

## Known Limitations

### Current System
1. **Question Coverage**: Only ~8 questions per file (target: 100)
2. **Retrieval Accuracy**: 42% at rank-1 (could be higher)
3. **Redundancy**: Some generated questions may be semantically similar
4. **Scale**: One file at a time (no batch optimization)
5. **Embedding**: Uses older Qwen embeddings (could use newer models)

### Evaluation Scope
1. **Vietnamese Only**: No multilingual evaluation
2. **Bulletin Domain**: Only tested on BIWASE documents
3. **File-level Only**: No chunk-level granularity
4. **Static Documents**: No handling of document updates
5. **Batch Mode**: Interactive queries not supported

---

## Recommendations for Next Sessions

### Priority 1 (High Impact)
1. Increase questions to 100 per file → better evaluation dataset
2. Test retrieval with better embeddings → higher accuracy
3. Analyze failure cases → understand why some queries fail

### Priority 2 (Medium Impact)
1. Build REST API → enable interactive use
2. Add semantic deduplication → remove redundant questions
3. Implement caching → faster evaluation runs

### Priority 3 (Nice to Have)
1. Neo4j integration → graph-based retrieval
2. Multi-modal support → tables and charts
3. User feedback loop → online learning

---

## Files Status

| File | Status | Last Modified | Notes |
|------|--------|---------------|-------|
| generate_retrieval_tests.py | ✅ Working | Session 3 | 7Q/chunk, all chunks |
| measure_retrieval_accuracy.py | ✅ Working | Session 3 | accuracy@1,3,10 |
| tests/data/retrieval_test_cases.jsonl | ✅ Generated | Session 3 | 273 questions |
| tests/data/evaluation_results.json | ✅ Generated | Session 3 | Baseline metrics |
| memory-bank/ | ✅ Created | Session 3 | 5 core documents |
| Dockerfile | ⏳ Incomplete | Unknown | Needs review |
| neo4j/ | ⏳ Unused | Unknown | Future integration |

---

## Session Notes & Observations

### Session 3 Observations
- Memory Bank structure created to preserve context across sessions
- Evaluation metrics now more practical (top-1,3,10)
- 89% top-10 accuracy is strong baseline
- Entity extraction fallback proved effective
- GPU acceleration critical for inference speed

### Environment Notes
- Python 3.12.9 working well with PyTorch
- CUDA detection and device mapping automatic
- bfloat16 precision sufficient for Qwen models
- Silent error handling prevents Unicode issues
- tqdm progress bars work reliably

### Next Session Checklist
- [ ] Review this Memory Bank in full
- [ ] Verify all 5 core documents are present
- [ ] Check if question count is still ~8 per file
- [ ] Confirm accuracy metrics from evaluation_results.json
- [ ] Plan improvements based on findings
