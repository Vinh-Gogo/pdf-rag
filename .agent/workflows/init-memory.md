---
description: Initialize the Memory Bank system for the project
---

# Initialize Memory Bank

1. **Analyze Project Structure**
   - Review file structure and key directories
   - Identify main configuration files (package.json, requirements.txt, etc.)
   - Understand the core purpose and technology stack

2. **Create Core Memory Files**
   - Create `.agent/memory/project-brief.md`: Basic project info (manual input required later)
   - Create `.agent/memory/product-vision.md`: Infer product goals
   - Create `.agent/memory/context.md`: Current development state
   - Create `.agent/memory/tech-stack.md`: Document detected technologies
   - Create `.agent/memory/architecture.md`: High-level system design
   - Create `.agent/memory/patterns/common-tasks.md`: Placeholder for future patterns

3. **Populate Initial Content**
   - **Context**: "Project appears to be a PDF RAG system. Recently implemented retrieval testing."
   - **Tech Stack**: Python, LangChain, Qdrant (inferred), SentenceTransformers, Torch, etc.
   - **Architecture**: PDF extraction -> Chunking -> Embedding -> Vector Store -> Retrieval -> Generation.

4. **Verify Creation**
   - Check if all files exist in `.agent/memory/`
   - Notify user of successful initialization
