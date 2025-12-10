# Project Brief

## Overview
PDF RAG System is a retrieval-augmented generation application designed to process Vietnamese PDF documents, extract content, and provide accurate answers to user queries based on the document knowledge base.

## Core Features
1. **PDF Processing**: Extract text and tables from PDF files.
2. **Indexing**: Chunking, embedding, and storing vectors in Qdrant (or similar).
3. **Retrieval**: Semantic search using multilingual embedding models (Qwen, Halong).
4. **Generation**: Answering questions using LLMs (Qwen) based on retrieved context.
5. **Evaluation**: Comprehensive framework for measuring retrieval accuracy.

## Goals
- High accuracy in Vietnamese document retrieval.
- Robust handling of PDF formatting and layout.
- Scalable vector search.
- Clean and maintainable codebase.

## Constraints
- **Environment**: Windows, Python 3.12, CUDA GPU integration (requires specific handling for Triton).
- **Language**: Primary focus on Vietnamese language support.
