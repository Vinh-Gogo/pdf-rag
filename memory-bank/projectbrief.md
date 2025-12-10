# PDF-RAG Project Brief

## Project Overview
**PDF-RAG** is a Retrieval-Augmented Generation (RAG) system for Vietnamese business documents. It extracts text from PDF files (primarily BIWASE company bulletins), creates test cases with questions, and evaluates semantic retrieval accuracy using embeddings.

## Core Problem Statement
BIWASE (Binh Duong Water & Environment Corporation) needs a system to:
1. Convert PDF bulletins into searchable text
2. Generate comprehensive test questions from document content
3. Evaluate retrieval system's ability to find the correct source document for any query

## Primary Objectives
1. **Generate Retrieval Test Cases**: Create ~100 questions per markdown file from all documents
2. **Evaluate Retrieval Accuracy**: Measure file-level retrieval (not chunk-level) using semantic embeddings
3. **Assess System Performance**: Track accuracy@1, accuracy@3, accuracy@10, and MRR metrics

## Key Success Metrics
- **Accuracy@1**: % of queries where correct file ranks #1 (target: >40%)
- **Accuracy@3**: % of queries where correct file in top 3 (target: >60%)
- **Accuracy@10**: % of queries where correct file in top 10 (target: >80%)
- **MRR (Mean Reciprocal Rank)**: Average 1/rank across all queries (target: >0.55)

## Scope
- **34 markdown files** from BIWASE bulletins (T1-2023 through T12-2025)
- **~3,400+ total test questions** (~100 per file)
- **File-level retrieval evaluation** (34 files total, not individual chunks)
- **Vietnamese language focus** with entity-based and model-based question generation

## Constraints & Decisions
- Using **Qwen-1.7B model** for question generation (7 questions per chunk, 1500 max tokens)
- **GPU-accelerated inference** with bfloat16 precision
- **Fallback to mock generation** (entity extraction + templates) if model fails
- **File-level accuracy only** - don't need chunk-level precision
- Temperature=0.6 for balanced diversity without sacrificing consistency
