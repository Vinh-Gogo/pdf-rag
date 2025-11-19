# PDF RAG Chat Interface v1.0

A comprehensive, modern chat interface for testing the PDF RAG system, featuring advanced UI/UX, beautiful Vietnamese typography, dual-query search, and seamless document interaction.

## Architecture

```
Frontend (Next.js 14 + TypeScript) <--> API Bridge <--> Python RAG Service <--> Qdrant Vector DB
├── Stunning UI with Vietnamese typography
├── Streaming chat with Server-sent events (SSE)
├── Dual-query search (Page + Sequence level)
├── Real-time content rendering with Markdown
├── Chat history persistence (localStorage)
├── PDF upload and processing
├── Resizable results panel
└── Custom embeddings (Qwen) + OpenAI GPT generation
```

## ✨ New Features & Improvements

### 🎨 **Beautiful Vietnamese UI/UX**
- **Vietnamese Typography**: Professional fonts specifically for Vietnamese text with diacritical marks (á, é, ì, ó, ú, ỳ, ả, ẻ, ỉ, ỏ, ủ, ỷ, etc.)
- **Stunning Welcome Screen**: Diffuse color animated background with floating icons, gradient text, and feature highlights
- **Responsive Design**: Mobile-first approach with smooth animations and transitions
- **Auto-scroll Intelligence**: Scrolls on message send (immediate feedback) and on response arrival (show results)
- **Resizable Results Panel**: Drag-to-resize sources display with constraints (256px-800px)

### 📝 **Advanced Markdown Rendering**
- **Header Support**: `## Beautiful Vietnamese Headers` with styled typography
- **Professional Tables**: Automatic HTML table rendering from markdown `|` syntax
- **Bold Text Formatting**: `**highlighted text**` processing
- **Content Deduplication**: Prevents repetition when consolidating page sources
- **Clean Display**: Removes raw markdown syntax from user interface

### 🚀 **Enhanced Chat Experience**
- **Dual-Query System**:
  - **Page-level Search**: Direct semantic search returning complete pages
  - **Sequence-level Search**: High-precision chunks from top-retrieved pages
  - **Cross-Filtered Results**: Sequence chunks only from top-k page results (>15 words, sorted by score)
- **Smart PDF Upload**: Direct UI upload (20MB limit) with processing pipeline integration
- **Chat History Persistence**: Automatic localStorage saving/loading with timestamps
- **Streaming Responses**: Real-time token-by-token display without blocking UI
- **Loading States**: Beautiful animated indicators and progress feedback

### ⚡ **Technical Enhancements**
- **TypeScript**: Full type safety with interfaces for Message, Source, and API responses
- **Error Handling**: Comprehensive error boundaries and user-friendly error messages
- **Performance**: Optimized rendering with useEffect, useRef, and memoization
- **Accessibility**: ARIA labels, keyboard navigation, and screen reader support
- **Memory Management**: Efficient state management and cleanup

### 🎯 **User Experience Features**
- **Input Box Positioning**: Fixed overlapping issues (proper spacing from messages)
- **Smooth Animations**: Bouncing icons, fade transitions, and hover effects
- **Visual Feedback**: Immediate response on button click/press Enter
- **Clear History**: One-click conversation reset with localStorage clearance
- **Session Management**: Unique session IDs for tracking conversations

## Core Features

### 🎯 **Advanced Search & Retrieval**
- **Dual-Query Architecture**:
  - **Page-Level Search** (`/api/query`): Direct semantic search returning complete pages from `esg_pages` collection
  - **Sequence-Level Search** (`/api/query_seq`): High-precision chunks from top-retrieved pages using `esg_sequences` collection
  - **Cross-Filtered Results**: Sequence chunks only from top-k page results (≥15 words, sorted by similarity score)
- **Smart Result Classification**: Page-level vs sequence-level results with distinct visual markers
- **Relevance Filtering**: Top-5 most relevant sequences from the most relevant pages

### 💬 **Interactive Chat Interface**
- **Real-time Streaming**: Server-sent events for token-by-token response display
- **Auto-Scroll Intelligence**:
  - Immediate scroll on message send (visual feedback)
  - Delayed scroll on response arrival (show results)
- **Chat History**: LocalStorage persistence with timestamps and session management
- **PDF Upload**: Direct UI upload (20MB limit) with pipeline integration

### 🎨 **Beautiful Visualization**
- **Vietnamese Typography**: Custom `.vietnamese-header` class for perfect accent rendering
- **Advanced Markdown Rendering**: Headers (`##`), bold (`**`), tables (`|`), with syntax cleanup
- **Welcome Screen**: Animated diffuse colors, floating icons, gradient text, feature cards
- **Responsive Design**: Mobile-first with smooth animations and transitions
- **Resizable Sources Panel**: Drag-to-resize with smooth UI feedback

### ⚡ **Technical Excellence**
- **TypeScript**: Full type safety with comprehensive interfaces
- **Performance Optimized**: Efficient state management, memoization, and cleanup
- **Error Resilient**: Comprehensive error handling with user-friendly messages
- **Memory Efficient**: Optimized content deduplication and streaming

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
