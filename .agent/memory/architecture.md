# System Architecture

## High-Level Flow

1. **Ingestion Pipeline**:
   - **Input**: PDF Documents (`_pdf_md` / raw PDFs).
   - **Processing**: Text extraction, Optical Character Recognition (OCR) if needed.
   - **Chunking**: `RecursiveCharacterTextSplitter` or semantic chunking.
   - **Embedding**: `QwenEmbedding` model vectorizes text chunks.
   - **Storage**: Vectors stored in Qdrant; Metadata in Neo4j (hybrid approach).

2. **Retrieval Pipeline**:
   - **Input**: User Query.
   - **Embedding**: Query vectorized using same embedding model.
   - **Search**: K-Nearest Neighbors (KNN) search in Vector Store.
   - **Ranking**: Cosine similarity ranking.

3. **Generation Pipeline**:
   - **Context**: Top-K retrieved chunks.
   - **Augmentation**: Context appended to system prompt.
   - **Inference**: LLM (Qwen) generates response based on context + query.

## Directory Structure
- `src/`: Core source code.
    - `models/`: Embedding and LLM wrappers (`embedd.py`, `halong_embedd.py`).
    - `data/`: Data storage and processing.
        - `pdfs/outputs_new/`: Processed markdown and JSONL files.
    - `pipeline/`: Pipeline orchestration logic.
    - `helpers/`: Utility functions.
        - `md_to_jsonl.py`: Converts markdown to JSONL format (splits by `\n\n\n`).
        - `pdfs_to_markdown.py`: PDF to Markdown conversion.
- `tests/`: Test scripts and data.
    - `data/`: Generated test cases and results.
- `_pdf_md/`: Processed Markdown files from PDFs.

## Data Flow
`PDF` -> `Text/Markdown` -> `Chunks` -> `Vectors` -> `Vector DB`
`Query` -> `Vector` -> `Search` -> `Retrieved Chunks` -> `LLM` -> `Answer`
