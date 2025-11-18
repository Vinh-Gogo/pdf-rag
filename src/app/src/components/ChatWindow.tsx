'use client';
import { useState, useRef, useEffect } from 'react';

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
}

export default function ChatWindow() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [sources, setSources] = useState<Source[]>([]);
  const [uploadMessage, setUploadMessage] = useState<string>('');
  const sessionId = 'session-' + (Math.random() * 1e9 | 0);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage, sessionId }),
      });

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let allSources: Source[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(line.substring(6));

            if (data.type === 'page_start' && data.sources) {
              allSources = [...allSources, ...data.sources];
              setSources([...allSources]);

              // Show which page is being loaded
              setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];

                if (lastMessage?.role === 'assistant') {
                  newMessages[newMessages.length - 1] = {
                    ...lastMessage,
                    text: data.token || 'Loading page content...'
                  };
                } else {
                  newMessages.push({
                    role: 'assistant',
                    text: data.token || 'Loading page content...'
                  });
                }

                return newMessages;
              });
            } else if (data.type === 'chunk' && data.sources) {
              // Add each retrieved chunk to sources
              allSources = [...allSources, ...data.sources];
              setSources([...allSources]);

              // Update progress
              setMessages(prev => {
                if (prev.length === 0) return prev;
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];

                if (lastMessage?.role === 'assistant') {
                  newMessages[newMessages.length - 1] = {
                    ...lastMessage,
                    text: `Loaded ${allSources.length} chunks from ${new Set(allSources.map(s => s.page_index)).size} pages...`
                  };
                }

                return newMessages;
              });
            } else if (data.type === 'done') {
              // Final response
              setSources(data.sources || allSources);
              setMessages(prev => {
                const newMessages = [...prev];
                const lastMessage = newMessages[newMessages.length - 1];

                if (lastMessage?.role === 'assistant') {
                  newMessages[newMessages.length - 1] = {
                    ...lastMessage,
                    text: data.answer || `Found ${allSources.length} similar chunks`
                  };
                } else {
                  newMessages.push({
                    role: 'assistant',
                    text: data.answer || `Found ${allSources.length} similar chunks`
                  });
                }

                return newMessages;
              });
            } else if (data.type === 'error') {
              console.error('Stream error:', data.token);
              setMessages(prev => [...prev, {
                role: 'assistant',
                text: `Error: ${data.token}`
              }]);
            }
          } catch (parseError) {
            console.error('Parse error:', parseError, 'Line:', line);
          }
        }
      }
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
            <div className="text-center text-gray-500 mt-8">
              <p className="text-lg">Welcome to PDF RAG Chat</p>
              <p className="text-sm mt-2">Ask questions about your documents and see the retrieved chunks.</p>
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
        </div>

        {/* Input */}
        <div className="border-t p-4 -mt-12.5">
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
              className="px-4 py-3 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
              title="Clear chat history"
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      {/* Sources Panel */}
      {sources.length > 0 && (
        <div className="w-96 border-l bg-white p-4 overflow-y-auto">
          <h3 className="font-semibold text-lg mb-3 text-black">Retrieved Pages</h3>
          <div className="space-y-2">
            {sources.map((source, index) => (
              <div key={index} className="p-3 bg-gray-50 rounded border">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs font-medium">
                      Rank {source.rank || index + 1}
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
                <div className="text-sm text-gray-800 leading-relaxed max-h-96 overflow-y-auto bg-gray-50 p-3 rounded border">
                  {source.content}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
