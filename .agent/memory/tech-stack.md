# Technology Stack

## Core
- **Language**: Python 3.12+
- **Frameworks**: LangChain, SentenceTransformers
- **Vector Database**: Qdrant (inferred from codebase structure)
- **Database**: Neo4j (graph database support present)

## AI / ML
- **LLM**: Qwen (Qwen2.5-Instruct, Qwen3)
- **Embeddings**: 
    - `Qwen/Qwen3-Embedding-0.6B` (Start-of-the-art multilingual)
    - `dangvantuan/vietnamese-embedding` (Fallback/Alternative)
- **Libraries**: `torch`, `transformers`, `sentence-transformers`

## Testing & Evaluation
- **Framework**: Custom python scripts
- **Metrics**: Accuracy@K, MRR

## Infrastructure / Tools
- **Containerization**: Docker, Docker Compose
- **Environment Management**: `venv` / `pip`
- **OS**: Windows (Development)

## Key Dependencies
- `torch` (with CUDA support)
- `transformers`
- `sentence-transformers`
- `numpy`
- `tqdm`
- `pymupdf` (PDF processing - inferred)
