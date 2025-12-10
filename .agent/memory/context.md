# Project Context

## Current Status
- **Active Phase**: Data Processing & Optimization
- **Recent Focus**: Processing markdown files for RAG pipeline.
- **Current Task**: Markdown to JSONL conversion pipeline.

## Recent Achievements
- ✅ Created `generate_retrieval_tests.py` for automated test case generation.
- ✅ Created `measure_retrieval_accuracy.py` for evaluating retrieval performance.
- ✅ Fixed critical Triton/CUDA compatibility issues on Windows.
- ✅ Verified retrieval pipeline with 490 test cases (Accuracy@5: ~67%).
- ✅ Created `src/helpers/md_to_jsonl.py` for converting markdown to JSONL format.
  - Splits content by `\n\n\n` delimiter (page separator).
  - Outputs `{"page": N, "content": "..."}` format.
  - Normalizes content by replacing newlines with spaces.
  - Successfully processed 42 chunks from BIWASE annual report.

## Next Steps
1. **Refine Question Generation**: Improve LLM prompts to generate more diverse and challenging questions.
2. **Optimize Chunking**: Align testing chunking strategy with production pipeline (`PDFTextExtractor`).
3. **Expand Evaluation**: Add generation quality metrics (e.g., faithfulness, answer relevance).
4. **Integration**: Integrate evaluation scripts into CI/CD or mainline workflow.
5. **Process More PDFs**: Use `md_to_jsonl.py` to process additional markdown files.

## Operational Notes
- **Environment**: Running locally on Windows with CUDA GPU (1 device).
- **Key Issues**: `triton` library is incompatible with Windows; masked via `sys.modules` hack in `embedd.py`.
- **Data Location**: Processed JSONL files stored in `src/data/pdfs/outputs_new/`.
