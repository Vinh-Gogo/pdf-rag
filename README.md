# PDF-RAG

A complete pipeline for extracting text from PDF files, splitting into pages and sequences, and storing in Qdrant vector database for semantic search and retrieval.

## 🎯 Features

- **PDF Scan**: Extract text from PDF files with high accuracy
- **Dual-Level Indexing**:
  - **Pages**: Store entire page content for broad context
  - **Sequences**: Store individual paragraphs for precise retrieval
- **Vector Store**: Powered by Qdrant for fast semantic search
- **Embeddings**: Using local Qwen3-Embedding-0.6B model
- **Retrieval System**: Search across both pages and sequences simultaneously

## 📁 Project Structure

```
rag/
...src/
  api/                     # FastAPI backend for RAG service
  app/                     # Next.js frontend chat interface
  data/                    # Sample PDFs and processed content
  helpers/                 # Utility scripts for PDF processing and vector store
  pipeline/                # Complete pipelines for PDF to vector store
  tests/                   # Unit tests for various components
...README.md               # This file
...CHAT_README.md          # Chat interface documentation
```

## 🚀 Quick Start

### 1. Installation

```bash
python src/api/rag_service.py
```

```bash
lea26@home-6-8-2025 MINGW64 /d/WETEC/rag (main)
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" && nvm use node && cd src/app && npm run dev
```

### 1.1. Pre-Process scan PDFs (if needed)

```ps
# for debugging: process specific PDF files
python -m src.helpers.pdfs_to_markdown --input_dir src/data/pdfs/pages --output_dir src/data/pdfs/markdown --pattern "page_1.pdf" --overwrite
```

### 2. Environment Setup

Create `.env` file:

### 3. Run Complete Pipeline

```bash
# Process PDF: Split pages + Extract text + Store in Qdrant
python src/pipeline/pdf_to_vectorstore_pipeline.py

# Or process sequences from raw text files
python src/helpers/sequences_to_vectorstore.py
```

### 4. Test Retrieval

```bash
# Retrieve from both esg_pages and esg_sequences
python src/pipeline/retrieval_example.py
```

## 📚 Usage Examples

### Store Pages in Vector Store

```python
from src.helpers.pages_to_vec_store import (
    read_pages_from_directory,
    store_pages_in_qdrant_direct
)

# Read pages
pages = read_pages_from_directory("src/data/contents")

# Store in Qdrant
vectorstore = store_pages_in_qdrant_direct(pages, "esg_pages")
```

### Store Sequences in Vector Store

```python
from src.helpers.sequences_to_vectorstore import run_sequences_pipeline

# Run complete pipeline
run_sequences_pipeline(
    input_dir="src/data/raw",
    output_json="src/data/push/sequences_data.json",
    collection_name="esg_sequences",
    min_words=10
)
```

### Retrieval

```python
from src.helpers.pages_to_vec_store import retrieve_similar_pages
from src.helpers.sequences_to_vectorstore import retrieve_similar_sequences

# Retrieve pages
pages_results = retrieve_similar_pages(
    "Vốn điều lệ của công ty", 
    vectorstore, 
    top_k=5
)

# Retrieve sequences
sequences_results = retrieve_similar_sequences(
    "Vốn điều lệ của công ty",
    collection_name="esg_sequences",
    top_k=10
)
```

## 🗄️ Vector Store Collections

### esg_pages
- **Content**: Entire page text
- **Use case**: Broad context, full page understanding
- **Metadata**: `page_index`, `content`, `seq`, `word_count`

### esg_sequences
- **Content**: Individual paragraphs/sections
- **Use case**: Precise retrieval, specific information
- **Metadata**: `page_index`, `seq_index`, `seq_id`, `content`, `word_count`

## 🔧 Configuration

### Pipeline Settings

Edit in `pdf_to_vectorstore_pipeline.py`:

```python
START_PAGE = 1
END_PAGE = 168
MAX_PAGES = 168
COLLECTION_NAME = "esg_pages"
```

### Sequences Settings

Edit in `sequences_to_vectorstore.py`:

```python
MIN_WORDS = 10      # Minimum words per sequence
BATCH_SIZE = 50     # Upload batch size
```

## 📊 Performance

- **PDF Splitting**: ~168 pages in seconds
- **Text Extraction**: PyMuPDF (fast and accurate)
- **Embedding**: Local model via Docker (no API costs)
- **Vector Store**: Qdrant (optimized for similarity search)

## 🛠️ Technologies

- **PDF Processing**: PyMuPDF (fitz)
- **Embeddings**: Qwen3-Embedding-0.6B
- **Vector Database**: Qdrant
- **Framework**: LangChain
- **Language**: Python 3.12+

## 📝 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👤 Author

Vinh-Gogo

## 🔗 Links

- [GitHub Repository](https://github.com/Vinh-Gogo/pdf-rag)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
