# PDF RAG Chat Interface

A modern chat interface for testing the PDF RAG system, built with Next.js and integrated with a Python FastAPI backend.

## Architecture

```
Frontend (Next.js) <--> API Bridge <--> Python RAG Service <--> Qdrant Vector DB
- Streaming chat UI
- Server-sent events (SSE)
- Custom embeddings (Qwen)
- OpenAI GPT generation
```

## Features

- **Two Search Modes**:
  - **Page-level search** (`/api/query`): Direct semantic search on complete pages
  - **Sequence-level search** (`/api/query_seq`): Chunk-based search returning all chunks from similar pages
- **Streaming Response**: Real-time display of retrieved content
- **Source Display**: Shows retrieved content with similarity scores and rankings
- **Page Attribution**: Each result shows which page it came from
- **Responsive UI**: Modern chat interface with text input and results panel

## Setup Instructions

### 1. Prerequisites

- Node.js (installed via system)
- Python 3.8+ with required dependencies
- Qdrant vector database running
- OpenAI API key

### 2. Python Backend Setup

1. Ensure your Qdrant database contains vectors in collection "esg_sequences"
2. Set up environment variables in `.env`:
   ```bash
   # Copy from .env.example or root .env.example
   cp .env.example .env
   # Fill in the required keys
   ```

3. Start the Python RAG service:
   ```bash
   # From project root
   cd src
   python -m api.rag_service
   # Or
   python src/api/rag_service.py
   ```

   The API will run on `http://localhost:8000`

### 3. Next.js Frontend Setup

1. Navigate to the app directory:
   ```bash
   cd src/app
   ```

2. Install dependencies (if needed):
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env.local
   # PYTHON_API_URL should point to http://localhost:8000
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open `http://localhost:3000` in your browser

### 4. Testing the System

1. Upload PDF documents using the existing pipeline:
   ```bash
   python src/pipeline/pipeline_pdf_vec.py
   ```

2. Ask questions in the chat interface about your uploaded documents (e.g., "company policies", "environmental reports", "risk management")

3. The system will:
   - Use Qwen embeddings to find the most relevant pages
   - Return ALL chunks from top-5 most similar pages
   - Stream complete page content with proper organization
   - Show page rankings and complete document context
   - Allow PDF upload directly through the UI for immediate processing

## API Endpoints

### Python Service (`http://localhost:8000`)

- `GET /`: Health check
- `GET /health`: Health status
- `POST /api/query`: **Page-level search** - Direct semantic search on complete pages using `esg_pages` collection
  - **Strategy**: Find and return complete similar pages
  - **Input**: `{"message": "query text", "top_k": 5}`
  - **Output**: Complete page content and similarity scores
- `POST /api/query_seq`: **Sequence-level search** - Chunk-based search returning all chunks from similar pages using `esg_sequences` collection
  - **Strategy**: Find top pages by chunk similarity, return ALL chunks from those pages
  - **Input**: `{"message": "query text", "top_k": 5}`
  - **Output**: All chunk content from most relevant pages

### Next.js Proxy (`http://localhost:3000`)

- `POST /api/chat`: Frontend proxy to Python service (currently calls `/api/query_seq`)
- `POST /api/upload-pdf`: PDF file upload and processing endpoint
  - **Input**: Multipart form data with 'pdf' field (max 20MB)
  - **Processing**: Triggers PDF pipeline processing
  - **Output**: Success/error status with processing details

## Environment Variables

### Python (.env)
```bash
OPENAI_API_KEY=sk-...          # Required for generation
QDRANT_URL=http://localhost:6333  # Your Qdrant URL
QDRANT_API_KEY=                 # Optional, for secured Qdrant
```

### Next.js (.env.local)
```bash
PYTHON_API_URL=http://localhost:8000  # Python service URL
NEXT_PUBLIC_APP_NAME="PDF RAG Chat"   # App display name
```

## Troubleshooting

### Common Issues

1. **Embeddings vector size mismatch**:
   - Ensure Qdrant collection was created with the same embedding model
   - The service auto-detects vector size but may fail if incompatible

2. **Streaming not working**:
   - Check browser console for SSE errors
   - Verify Python service is running and accessible

3. **No sources displayed**:
   - Retrieval might be failing
   - Check Qdrant connectivity and collection existence

4. **OpenAI API errors**:
   - Verify API key is set correctly
   - Check API quota and billing

### Logs

- Python service logs to console
- Next.js logs to browser console and terminal
- Check network tab for API call failures

## Future Enhancements

- [ ] Payload CMS integration for chat logging
- [ ] Session persistence across browser refreshes
- [ ] Chat history management
- [ ] Multi-model support (different OpenAI models)
- [ ] Document upload via UI
- [ ] Advanced prompt templating
- [ ] User authentication
- [ ] Rate limiting and security

## Development

### Adding New Features

1. Modify `ChatWindow.tsx` for UI changes
2. Update `route.ts` for API changes
3. Extend `rag_service.py` for backend features

### Building for Production

```bash
# Python
pip install -r requirements.txt
python src/api/rag_service.py  # In production, use gunicorn or similar

# Next.js
cd src/app
npm run build
npm start  # For production
```

The system provides a modern, streaming chat interface for testing your PDF RAG system, maintaining compatibility with existing custom embeddings while enabling real-time user interaction.
