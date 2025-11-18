import os
import sys
import json
import uuid
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
from pydantic import SecretStr

# Add project root to path
script_dir = Path(__file__).resolve().parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import existing components
from src.helpers.init_qdrant import qdrant_client

# LangChain imports
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_openai import OpenAIEmbeddings

# Import page-level retrieval functions
try:
    from src.helpers.vectorstore_from_pages import retrieve_similar_pages as retrieve_similar_pages_fn
except ImportError:
    print("Warning: Could not import page retrieval functions. Page-level search will be unavailable.")
    retrieve_similar_pages_fn = None

# Environment variables
from dotenv import load_dotenv
load_dotenv()

# ======================================
# OpenAI Embeddings Setup
# ======================================

# ======================================
# FastAPI Models
# ======================================

class QueryRequest(BaseModel):
    message: str
    session_id: str = "default"
    top_k: int = 4
    temperature: float = 0.0

class TokenResponse(BaseModel):
    type: str  # "token", "done", "error"
    token: Optional[str] = None
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None

# ======================================
# Global Components
# ======================================

# Initialize OpenAI embeddings for vLLM service (Qwen model)
embeddings = OpenAIEmbeddings(
    model=os.getenv("OPENAI_API_MODEL_NAME_EMBED", "Qwen/Qwen3-Embedding-0.6B"),
    base_url=os.getenv("OPENAI_BASE_URL_EMBED"),
    api_key=SecretStr(os.getenv("OPENAI_API_KEY_EMBED", "text")),
    tiktoken_enabled=False,
)

# RAG Configuration
DEFAULT_COLLECTION = "esg_sequences"
PAGE_COLLECTION = "esg_pages"

# ======================================
# RAG Service Functions
# ======================================

def get_vectorstore(collection_name: str = DEFAULT_COLLECTION):
    """Get existing Qdrant vectorstore with OpenAI embeddings"""
    try:
        vectorstore = Qdrant.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            prefer_grpc=True
        )
        return vectorstore
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorstore error: {e}")

async def retrieve_similar_chunks(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Find top pages by chunk similarity, then return ALL chunks from those pages"""
    try:
        # Embed query
        query_vector = embeddings.embed_query(query)

        # Step 1: Find top similar chunks to identify relevant pages
        search_results = qdrant_client.search(
            collection_name=DEFAULT_COLLECTION,
            query_vector=query_vector,
            limit=top_k * 2,  # Get more initially to find diverse pages
            with_payload=True,
            with_vectors=False
        )

        # Step 2: Extract unique page indices from top results
        page_scores = {}
        for result in search_results:
            payload = result.payload or {}
            page_idx = payload.get("page_index")
            if page_idx is not None:
                if page_idx not in page_scores:
                    page_scores[page_idx] = float(result.score)
                else:
                    # Keep the highest score for this page
                    page_scores[page_idx] = max(page_scores[page_idx], float(result.score))

        # Get top pages by their highest chunk scores
        top_pages = sorted(page_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        if not top_pages:
            return []

        # Step 3: Retrieve ALL chunks from each top page
        all_chunks = []
        rank = 1

        for page_idx, page_score in top_pages:
            # Get all chunks from this page
            page_chunks = await retrieve_page_content(page_idx)
            if page_chunks:
                for chunk in page_chunks:
                    chunk["page_score"] = page_score
                    chunk["page_rank"] = rank
                all_chunks.extend(page_chunks)
                rank += 1

        return all_chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

async def retrieve_page_content(page_index: int) -> List[Dict[str, Any]]:
    """Retrieve all content chunks for a specific page_index"""
    try:
        # For collections without index on page_index, scroll through all records and filter in Python
        search_results = qdrant_client.scroll(
            collection_name=DEFAULT_COLLECTION,
            scroll_filter=None,  # No filter to avoid index requirement
            limit=10000,  # Large limit to get all chunks, might need adjustment based on data size
            with_payload=True,
            with_vectors=False
        )

        # Process results and filter by page_index
        chunks = []
        for point in search_results[0]:  # results is (points, next_offset)
            payload = point.payload
            if payload and payload.get("page_index") == page_index:
                chunks.append({
                    "content": payload.get("content", ""),
                    "metadata": payload,  # Put the whole payload in metadata for consistency
                    "id": payload.get("seq_id", f"page_{page_index}_unknown"),
                    "page_index": page_index
                })

        # Sort chunks by seq_index if available for proper ordering
        chunks.sort(key=lambda x: x["metadata"].get("seq_index", 0))

        return chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Page retrieval error: {e}")

async def stream_rag_response(query: str, top_k: int = 4) -> AsyncGenerator[str, None]:
    """Stream similarity search response"""
    try:
        # Get similar chunks
        sources = await retrieve_similar_chunks(query, top_k)

        # Group chunks by page and stream them
        chunks_by_page = {}
        for source in sources:
            page_idx = source['page_index']
            if page_idx not in chunks_by_page:
                chunks_by_page[page_idx] = []
            chunks_by_page[page_idx].append(source)

        # Stream complete pages
        for page_idx in sorted(chunks_by_page.keys()):
            page_chunks = chunks_by_page[page_idx]
            page_rank = page_chunks[0]['page_rank']

            # First chunk: indicate page start
            first_chunk = page_chunks[0]
            yield f"data: {TokenResponse(type='page_start', token=f'Page {page_rank}: {page_idx} ({len(page_chunks)} chunks)', sources=[first_chunk]).json()}\n\n"

            # Stream all chunks from this page
            for chunk in page_chunks:
                chunk_data = TokenResponse(
                    type="chunk",
                    token=f"Content: {chunk['content'][:150]}..." if len(chunk['content']) > 150 else f"Content: {chunk['content']}",
                    sources=[chunk]
                )
                yield f"data: {chunk_data.json()}\n\n"

            # End of page
            yield f"data: {TokenResponse(type='page_end', token=f'--- End of Page {page_idx} ---', sources=[]).json()}\n\n"

        # Final response with all sources
        final_data = TokenResponse(
            type="done",
            answer=f"Retrieved {len(sources)} chunks from {top_k} most relevant pages for query: '{query}'",
            sources=sources
        )
        yield f"data: {final_data.json()}\n\n"

    except Exception as e:
        error_data = TokenResponse(type="error", token=f"Error: {str(e)}")
        yield f"data: {error_data.json()}\n\n"

# ======================================
# FastAPI App
# ======================================

app = FastAPI(title="PDF RAG Service", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "PDF RAG Service is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    """
    Page-level search: Find and return complete similar pages.
    Uses esg_pages collection for page-level retrieval.
    """
    try:
        if retrieve_similar_pages_fn is None:
            raise HTTPException(status_code=500, detail="Page-level search not available. Check vectorstore_from_pages.py import.")

        # Use page-level search from vectorstore_from_pages.py
        pages = retrieve_similar_pages_fn(
            query=request.message,
            top_k=request.top_k,
            collection_name="esg_pages"
        )

        # Stream page results
        async def generate_pages():
            for i, page in enumerate(pages, 1):
                # Stream each page as a complete chunk
                page_index = page.get("page_index", "N/A")
                token_text = f'[PAGE LEVEL] Page {page_index} (Rank {i})'
                yield f"data: {TokenResponse(type='page_start', token=token_text, sources=[page]).json()}\n\n"

                yield f"data: {TokenResponse(type='chunk', token=page.get('content', ''), sources=[page]).json()}\n\n"

            # Final response
            yield f"data: {TokenResponse(type='done', answer=f'Found {len(pages)} similar pages for query: \"{request.message}\"', sources=pages).json()}\n\n"

        return StreamingResponse(
            generate_pages(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        error_resp = TokenResponse(type="error", token=f"Page-level search error: {str(e)}")
        data = f"data: {error_resp.json()}\n\n"
        return StreamingResponse(
            iter([data]),
            media_type="text/event-stream"
        )

@app.post("/api/query_seq")
async def query_seq_endpoint(request: QueryRequest):
    """
    Sequence-level search: Find top pages by chunk similarity, return ALL chunks from those pages.
    Uses esg_sequences collection for semantic chunk retrieval.
    """
    async def generate():
        async for chunk in stream_rag_response(
            query=request.message,
            top_k=request.top_k
        ):
            yield chunk

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# ======================================
# Main
# ======================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "src.api.rag_service:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
