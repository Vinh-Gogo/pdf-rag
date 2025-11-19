'use client';
import { useState, useRef, useEffect } from 'react';

// Function to render comprehensive markdown content
const renderMarkdown = (text: string) => {
  let content = text;

  // Process bold text first
  content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Process headers (## ) - convert and collect for separate rendering
  const headers: { html: string, text: string }[] = [];
  content = content.replace(/^## (.*)$/gm, (match, headerText) => {
    const html = `<h2 class="vietnamese-header">${headerText}</h2>`;
    headers.push({ html, text: match }); // Store original and html
    return `__HEADER_${headers.length - 1}__`; // Placeholder
  });

  // Check for tables
  const lines = content.split('\n');
  let headerCells: string[] = [];
  const rows: string[][] = [];
  let tableFound = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    // Check for table header separator
    if (line.includes('|') && lines[i + 1] && lines[i + 1].includes('|-') && !tableFound) {
      tableFound = true;
      headerCells = line.split('|').map(cell => cell.trim()).filter(cell => cell);
      i++; // Skip separator line
      // Collect rows until empty line
      while (i + 1 < lines.length) {
        i++;
        const rowLine = lines[i].trim();
        if (rowLine.includes('|')) {
          const rowCells = rowLine.split('|').map(cell => cell.trim()).filter(cell => cell);
          if (rowCells.length > 0) {
            rows.push(rowCells);
          }
        } else if (rowLine === '') {
          break;
        }
      }
      break;
    }
  }

  // Handle both tables and headers
  const components: React.ReactNode[] = [];

  // Add headers as separate components
  headers.forEach(header => {
    components.push(<div key={`header-${components.length}`} dangerouslySetInnerHTML={{ __html: header.html }} />);
  });

  // Add table if found
  if (tableFound && headerCells.length > 0) {
    const tableComponent = (
      <div key="table" className="my-4">
        <table className="border-collapse border border-gray-300 w-full">
          <thead>
            <tr className="bg-gray-100">
              {headerCells.map((cell, index) => (
                <th key={index} className="border border-gray-300 px-3 py-2 text-left font-semibold">
                  <span dangerouslySetInnerHTML={{ __html: cell }} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="border border-gray-300 px-3 py-2">
                    <span dangerouslySetInnerHTML={{ __html: cell }} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
    components.push(tableComponent);
  }

  // Process remaining content by removing markdown syntax
  let remainingContent = content;

  // Remove table markdown
  if (tableFound) {
    const tableRegex = /\|[^\n]*\|\n\|[\s:-]*\|\n(?:\|[^\n]*\|\n)+/g;
    remainingContent = remainingContent.replace(tableRegex, '').trim();
    remainingContent = remainingContent
      .split('\n')
      .filter(line => !line.includes('|') || (!line.match(/^\|/) && !line.match(/^[\s:-]*$/)))
      .join('\n')
      .trim();
  }

  // Remove header markdown
  headers.forEach((_, idx) => {
    remainingContent = remainingContent.replace(`__HEADER_${idx}__`, '').trim();
  });

  // Add remaining content
  if (remainingContent.trim()) {
    const paragraphs = remainingContent.split('\n').filter(p => p.trim());
    paragraphs.forEach((paragraph, idx) => {
      if (paragraph.trim()) {
        components.push(<p key={`content-${idx}`} className="mb-2" dangerouslySetInnerHTML={{ __html: paragraph.replace(/\n/g, '<br />') }} />);
      }
    });
  }

  if (components.length === 0) {
    return <span>{text}</span>;
  }

  return <div>{components}</div>;
};

interface Message {
  role: 'user' | 'assistant';
  text: string;
}

interface Source {
  content: string;
  metadata: Record<string, unknown>;
  id: string;
  rank?: number;
  score?: number | null;
  page_index?: number | null;
  resultType?: 'page-level' | 'sequence-level';
  marker?: string;
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [uploadMessage, setUploadMessage] = useState<string>('');
  const [panelWidth, setPanelWidth] = useState(384); // Default 384px (w-96 = 24rem = 384px)
  const [isResizing, setIsResizing] = useState(false);
  const sessionId = 'session-' + (Math.random() * 1e9 | 0);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Scroll to bottom when sources/response arrives
  useEffect(() => {
    if (sources.length > 0 && messagesEndRef.current) {
      setTimeout(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
      }, 100); // Small delay to ensure DOM is updated
    }
  }, [sources]);

  // Load messages from localStorage on component mount
  useEffect(() => {
    const savedMessages = localStorage.getItem('chatMessages');
    if (savedMessages) {
      try {
        const parsedMessages = JSON.parse(savedMessages);
        setMessages(parsedMessages);
      } catch (error) {
        console.error('Failed to load saved messages:', error);
      }
    }
  }, []);

  // Save messages to localStorage whenever messages change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chatMessages', JSON.stringify(messages));
      localStorage.setItem('chatLastSaved', new Date().toISOString());
    }
  }, [messages]);

  // Function to clear chat history
  const clearHistory = () => {
    setMessages([]);
    setSources([]);
    localStorage.removeItem('chatMessages');
    localStorage.removeItem('chatLastSaved');
  };

  // Resize functionality
  const handleResizeStart = (e: React.MouseEvent) => {
    setIsResizing(true);
    e.preventDefault();
  };

  const handleResizeEnd = () => {
    setIsResizing(false);
  };

  const handleResize = (e: MouseEvent) => {
    if (!isResizing) return;

    const newWidth = window.innerWidth - e.clientX;
    // Constraints: min 256px, max 800px
    const constrainedWidth = Math.max(256, Math.min(800, newWidth));
    setPanelWidth(constrainedWidth);
  };

  // Add resize listeners
  useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleResize);
      document.addEventListener('mouseup', handleResizeEnd);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    } else {
      document.removeEventListener('mousemove', handleResize);
      document.removeEventListener('mouseup', handleResizeEnd);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    }

    return () => {
      document.removeEventListener('mousemove', handleResize);
      document.removeEventListener('mouseup', handleResizeEnd);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isResizing]);

  // Prevent default drag behavior
  const handleDragStart = (e: React.DragEvent) => {
    e.preventDefault();
  };

  // Function to handle PDF upload
  const uploadPdf = async (file: File) => {
    setIsUploading(true);
    setUploadMessage('');

    try {
      const formData = new FormData();
      formData.append('pdf', file);

      const response = await fetch('/api/upload-pdf', {
        method: 'POST',
        body: formData,
      });

      const result = await response.json();

      if (response.ok) {
        setUploadMessage(`✅ Successfully uploaded and processed: ${file.name}`);
        setMessages(prev => [...prev, {
          role: 'assistant',
          text: `Document "${file.name}" has been uploaded and processed. You can now ask questions about it!`
        }]);
      } else {
        setUploadMessage(`❌ Upload failed: ${result.error || 'Unknown error'}`);
        setMessages(prev => [...prev, {
          role: 'assistant',
          text: `Failed to upload document: ${result.error || 'Unknown error'}`
        }]);
      }
    } catch (error) {
      const errorMessage = `Upload error: ${error instanceof Error ? error.message : 'Unknown error'}`;
      setUploadMessage(`❌ ${errorMessage}`);
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: errorMessage
      }]);
    } finally {
      setIsUploading(false);
    }
  };

  // Function to handle file selection
  const handleFileSelect = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setUploadMessage('❌ Please select a PDF file only');
      return;
    }

    // Validate file size (20MB limit)
    if (file.size > 20 * 1024 * 1024) {
      setUploadMessage('❌ File size too large. Maximum 20MB allowed.');
      return;
    }

    await uploadPdf(file);

    // Clear the file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Function to trigger file selection
  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    // Clear previous results
    setSources([]);

    // Add user message
    setMessages(prev => [...prev, { role: 'user', text: userMessage }]);

    // Auto-scroll to show the message
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, 50);

    try {
      // Execute both APIs simultaneously
      const [pageResponse, seqResponse] = await Promise.all([
        fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMessage, sessionId, endpoint: 'query' }), // Page-level
        }),
        fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: userMessage, sessionId, endpoint: 'query_seq' }), // Sequence-level
        }),
      ]);

      if (!pageResponse.body || !seqResponse.body) {
        throw new Error('No response body');
      }

      // Process page-level results
      const pageReader = pageResponse.body.getReader();
      const seqReader = seqResponse.body.getReader();
      const decoder = new TextDecoder();

      let pageSources: Source[] = [];
      let seqSources: Source[] = [];
      const pageIndices = new Set<number>();
      let allChunksLoaded = false;

      interface ChunkData {
        type: string;
        sources?: Source[];
        token?: string;
        answer?: string;
      }

      // Function to process streaming chunks
      const processChunk = (chunkData: ChunkData, isPageLevel: boolean) => {
        if (chunkData.type === 'page_start' && chunkData.sources) {
          if (isPageLevel) {
            pageSources = [...pageSources, ...chunkData.sources];
            // Collect page indices from page-level results
            chunkData.sources.forEach((source) => {
              if (source.page_index !== null && source.page_index !== undefined) {
                pageIndices.add(source.page_index);
              }
            });
          }
        } else if (chunkData.type === 'chunk' && chunkData.sources) {
          if (isPageLevel) {
            pageSources = [...pageSources, ...chunkData.sources];
          } else {
            // For sequence-level, collect all chunks for now
            seqSources = [...seqSources, ...chunkData.sources];
          }
        } else if (chunkData.type === 'done') {
          allChunksLoaded = true;
        }
      };

      // Read from both streams
      const readStreams = async () => {
        let pageDone = false;
        let seqDone = false;

        while (!pageDone || !seqDone) {
          if (!pageDone) {
            try {
              const { done, value } = await pageReader.read();
              if (!done) {
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim());
                lines.forEach(line => {
                  if (line.startsWith('data: ')) {
                    try {
                      const data = JSON.parse(line.substring(6));
                      processChunk(data, true);
                    } catch (jsonError) {
                      console.error('Page stream parse error:', jsonError, 'Line content:', line);
                      // Skip malformed JSON chunks instead of crashing
                    }
                  }
                });
              } else {
                pageDone = true;
              }
            } catch (e) {
              console.error('Page read error:', e);
              pageDone = true;
            }
          }

          if (!seqDone) {
            try {
              const { done, value } = await seqReader.read();
              if (!done) {
                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n').filter(line => line.trim());
                lines.forEach(line => {
                  if (line.startsWith('data: ')) {
                    try {
                      const data = JSON.parse(line.substring(6));
                      processChunk(data, false);
                    } catch (jsonError) {
                      console.error('Seq stream parse error:', jsonError, 'Line content:', line);
                      // Skip malformed JSON chunks instead of crashing
                    }
                  }
                });
              } else {
                seqDone = true;
              }
            } catch (e) {
              console.error('Seq read error:', e);
              seqDone = true;
            }
          }
        }
      };

      // Process streams
      await readStreams();

      // Cross-filter sequence results: only chunks from top-k pages, >15 words, sorted by score
      const relevantSeqChunks = seqSources
        .filter(chunk => {
          const wordCount = chunk.content.split(' ').length;
          return chunk.page_index !== null && chunk.page_index !== undefined &&
                 pageIndices.has(chunk.page_index) && wordCount > 15;
        })
        .sort((a, b) => (b.score || 0) - (a.score || 0))
        .slice(0, 5); // Top 5 chunks

      // Group page sources by page_index and consolidate
      const groupedPageSources = pageSources.reduce((groups, source) => {
        const pageIndex = source.page_index || 0;
        if (!groups[pageIndex]) {
          groups[pageIndex] = [];
        }
        groups[pageIndex].push(source);
        return groups;
      }, {} as Record<number, Source[]>);

      // Combine content for each page (with deduplication)
      const consolidatedPageSources = Object.entries(groupedPageSources).map(([pageIndex, sources]) => {
        const firstSource = sources[0]; // Use first source as base

        // Deduplicate content to avoid repetition
        const uniqueContent: string[] = [];
        const seenContent = new Set<string>();

        sources.forEach(s => {
          const trimmedContent = s.content.trim();
          if (trimmedContent && !seenContent.has(trimmedContent)) {
            uniqueContent.push(trimmedContent);
            seenContent.add(trimmedContent);
          }
        });

        const combinedContent = uniqueContent.join('\n\n---\n\n');

        return {
          ...firstSource,
          content: combinedContent,
          score: Math.max(...sources.map(s => s.score || 0)) // Use highest score
        };
      });

      // Add type markers to distinguish results
      const markedPageSources = consolidatedPageSources.map(source => ({
        ...source,
        resultType: 'page-level' as const,
        marker: '📄 PAGE LEVEL'
      }));

      const markedSeqSources = relevantSeqChunks.map(source => ({
        ...source,
        resultType: 'sequence-level' as const,
        marker: '📝 HIGH PRECISION'
      }));

      // Combine results: high-precision segments first, then pages
      const combinedSources = [
        ...markedSeqSources,
        ...markedPageSources
      ];

      // Update UI with progress
      setMessages(prev => {
        const newMessages = [...prev];
        const lastMessage = newMessages[newMessages.length - 1];

        if (lastMessage?.role === 'assistant') {
          newMessages[newMessages.length - 1] = {
            ...lastMessage,
            text: `Found ${pageSources.length} pages and ${relevantSeqChunks.length} high-precision segments`
          };
        } else {
          newMessages.push({
            role: 'assistant',
            text: `Found ${pageSources.length} pages and ${relevantSeqChunks.length} high-precision segments`
          });
        }

        return newMessages;
      });

      // Set final sources
      setSources(combinedSources);

    } catch (error) {
      console.error('Fetch error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        text: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center min-h-[60vh] p-8">
              {/* Animated Background Elements */}
              <div className="absolute inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-40 -left-40 w-80 h-80 bg-blue-400/20 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute -bottom-40 -right-40 w-80 h-80 bg-purple-400/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-pink-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
              </div>

              {/* Main Welcome Card */}
              <div className="relative bg-white/90 backdrop-blur-sm border border-gray-200/50 shadow-2xl rounded-3xl p-10 max-w-2xl w-full">
                {/* Floating Icons */}
                <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 flex space-x-4">
                  <div className="bg-blue-500 p-4 rounded-full shadow-lg animate-bounce">
                    <span className="text-2xl">📄</span>
                  </div>
                  <div className="bg-green-500 p-4 rounded-full shadow-lg animate-bounce" style={{ animationDelay: '0.2s' }}>
                    <span className="text-2xl">🤖</span>
                  </div>
                  <div className="bg-purple-500 p-4 rounded-full shadow-lg animate-bounce" style={{ animationDelay: '0.4s' }}>
                    <span className="text-2xl">💬</span>
                  </div>
                </div>

                {/* Title */}
                <div className="text-center mt-8">
                  <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
                    PDF RAG Chat
                  </h1>

                  {/* Subtitle */}
                  <div className="mb-8">
                    <p className="text-xl md:text-2xl font-bold text-gray-700 mb-2">
                      Welcome to the Future! 🚀
                    </p>
                    <div className="flex items-center justify-center space-x-2 text-gray-600">
                      <span className="text-sm">✨</span>
                      <p className="text-lg">
                        Ask intelligent questions about your documents
                      </p>
                      <span className="text-sm">✨</span>
                    </div>
                  </div>

                  {/* Feature Highlights */}
                  <div className="grid md:grid-cols-3 gap-6 mt-8 text-center">
                    <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                      <div className="text-3xl mb-3">🔍</div>
                      <h3 className="font-semibold text-blue-800 mb-2">Smart Retrieval</h3>
                      <p className="text-sm text-gray-600">AI-powered chunk extraction</p>
                    </div>

                    <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                      <div className="text-3xl mb-3">💡</div>
                      <h3 className="font-semibold text-purple-800 mb-2">AI Answers</h3>
                      <p className="text-sm text-gray-600">Context-aware responses</p>
                    </div>

                    <div className="bg-gradient-to-br from-pink-50 to-pink-100 p-6 rounded-xl shadow-sm hover:shadow-md transition-shadow">
                      <div className="text-3xl mb-3">📊</div>
                      <h3 className="font-semibold text-pink-800 mb-2">Rich Results</h3>
                      <p className="text-sm text-gray-600">Tables, headers & more</p>
                    </div>
                  </div>

                  {/* Call to Action */}
                  <div className="mt-8">
                    <div className="inline-flex items-center space-x-3 bg-gradient-to-r from-green-400 to-blue-500 text-white px-6 py-3 rounded-full shadow-lg hover:shadow-xl transition-shadow">
                      <span className="text-xl">👆</span>
                      <span className="font-semibold">Click the green button to upload your PDF!</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-800 shadow'
                }`}
              >
                {message.text}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white px-4 py-2 rounded-lg shadow">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          )}

          {/* Invisible scroll target */}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="border-t p-4 mt-2">
          {/* Upload Status */}
          {(isUploading || uploadMessage) && (
            <div className="mb-2 text-sm text-gray-600">
              {isUploading ? (
                <span className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500 mr-2"></div>
                  Processing PDF upload...
                </span>
              ) : (
                <span>{uploadMessage}</span>
              )}
            </div>
          )}

          <div className="flex space-x-2">
            {/* Hidden File Input */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              accept=".pdf"
              className="hidden"
              aria-label="Upload PDF file"
            />

            {/* Upload Button */}
            <button
              onClick={triggerFileSelect}
              disabled={isUploading || isLoading}
              className="px-4 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-sm font-medium"
              title="Upload a PDF document"
            >
              📄 Upload PDF
            </button>

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask questions about your documents... (e.g., 'company policies', 'environmental reports')"
              className="flex-1 p-3 border rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 text-black"
              rows={2}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Ask
            </button>
            <button
              onClick={clearHistory}
              disabled={isLoading}
              className="px-3 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center justify-center"
              title="New conversation"
            >
              <img
                src="/new-direct-message.svg"
                alt="New conversation"
                className="w-5 h-5 fill-current"
              />
            </button>
          </div>
        </div>
      </div>

      {/* Resize Handle */}
      {sources.length > 0 && (
        <div
          className="w-1 bg-gray-300 hover:bg-blue-400 cursor-col-resize flex-shrink-0 select-none"
          onMouseDown={handleResizeStart}
          onDragStart={handleDragStart}
          title="Drag to resize panel"
        />
      )}

      {/* Sources Panel */}
      {sources.length > 0 && (
        <div
          className="border-l bg-white p-4 overflow-y-auto flex-shrink-0"
          style={{ width: `${panelWidth}px` }}
        >
          <h3 className="font-semibold text-lg mb-3 text-black">Retrieved Pages</h3>
          <div className="space-y-2">
            {sources.map((source, index) => (
              <div key={index} className={`p-3 rounded border ${
                source.resultType === 'page-level'
                  ? 'bg-blue-50 border-blue-200'
                  : 'bg-purple-50 border-purple-200'
              }`}>
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      source.resultType === 'page-level'
                        ? 'bg-blue-100 text-blue-800'
                        : 'bg-purple-100 text-purple-800'
                    }`}>
                      {source.marker || `Rank ${source.rank || index + 1}`}
                    </span>
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs font-medium">
                      Page {source.page_index || 'N/A'}
                    </span>
                    <span className="text-xs text-gray-500">ID: {source.id}</span>
                  </div>
                  {source.score && (
                    <span className="text-xs text-gray-400">
                      Score: {source.score.toFixed(3)}
                    </span>
                  )}
                </div>
                <div className={`text-sm leading-relaxed max-h-96 overflow-y-auto p-3 rounded border ${
                  source.resultType === 'page-level'
                    ? 'bg-blue-50 border-blue-100'
                    : 'bg-purple-50 border-purple-100'
                }`}>
                  <div className="font-medium text-xs mb-2 text-black">
                    {source.resultType === 'page-level' ? '📄 PAGE LEVEL RESULT' : '📝 SEQUENCE-LEVEL RESULT'}
                  </div>
<div className="text-black whitespace-pre-wrap text-sm leading-relaxed">
  {renderMarkdown(source.content)}
</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
