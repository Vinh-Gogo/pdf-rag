# PDF RAG - PDF Retrieval-Augmented Generation

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
├── src/
│   ├── data/
│   │   ├── pdfs/              # Input PDF files
│   │   │   └── pages/         # Split PDF pages
│   │   ├── contents/          # Extracted text from pages
│   │   ├── raw/               # Raw text files
│   │   └── push/              # JSON data for vector store
│   ├── helpers/
│   │   ├── PDFText.py         # PDF text extraction using PyMuPDF
│   │   ├── init_qdrant.py     # Qdrant client initialization
│   │   ├── pages_to_vec_store.py      # Pages to vector store
│   │   └── sequences_to_vectorstore.py # Sequences to vector store
│   └── pipeline/
│       ├── pdf_to_vectorstore_pipeline.py  # Complete pipeline
│       └── retrieval_example.py            # Retrieval examples
├── .env                       # Environment variables
├── .gitignore
├── requirments.txt
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Vinh-Gogo/pdf-rag.git
cd pdf-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirments.txt
```

### 1.1. Pre-Process scan PDFs (if needed)

```ps
# PDF >> markdown ( pip install marker-pdf )
(.venv) PS D:\rag> marker_single path\page_6.pdf --output_dir path\pages
Recognizing Layout: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:12<00:00, 12.58s/it]
Running OCR Error Detection: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  6.59it/s]
Detecting bboxes: 0it [00:00, ?it/s]
Detecting bboxes: 0it [00:00, ?it/s]
2025-11-12 09:48:35,050 [INFO] marker: Saved markdown to path\pages\page_6
2025-11-12 09:48:35,050 [INFO] marker: Total time: 15.751145124435425

# python -m src.helpers.pdfs_to_markdown --input_dir src/data/pdfs/pages --output_dir src/data/pdfs/markdown --pattern "page_1.pdf" --overwrite
```

### 2. Environment Setup

Create `.env` file:

```env
# Qdrant Configuration
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_url

# Local Embedding Model (via Docker)
OPENAI_API_MODEL_NAME_EMBED=
OPENAI_BASE_URL_EMBED=
OPENAI_API_KEY_EMBED=
```

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

### Extract Text from PDF

```python
from src.helpers.PDFText import PDFTextExtractor

# Initialize extractor
extractor = PDFTextExtractor(
    pdf_path="src/data/pdfs/file.pdf",
    output_dir="src/data/contents"
)

# Split PDF into individual pages
extractor.split_pdf_into_pages(start_page=1, end_page=168)

# Extract text from all pages
extractor.extract_all_pages(start_page=1, end_page=168)
```

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
- [PyMuPDF Documentation](https://pymupdf.readthedocs.io/)
