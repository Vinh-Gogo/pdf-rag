import { NextRequest, NextResponse } from 'next/server';

export async function POST(request: NextRequest) {
  try {
    const { message, sessionId, endpoint } = await request.json();

    if (!message || !message.trim()) {
      return NextResponse.json({ error: 'message is required' }, { status: 400 });
    }

    // Get Python API URL from environment or default
    const pythonApiUrl = process.env.PYTHON_API_URL || 'http://localhost:8000';

    // Determine which endpoint to call based on request parameter
    const apiEndpoint = endpoint === 'query_seq' ? '/api/query_seq' : '/api/query';

    // Call Python RAG API
    const response = await fetch(`${pythonApiUrl}${apiEndpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message: message.trim(),
        session_id: sessionId || 'default',
        top_k: 5,
        temperature: 0.0
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Python API error:', response.status, errorText);
      return NextResponse.json(
        { error: 'RAG service error' },
        { status: response.status }
      );
    }

    // Return the streaming response from Python API
    return new Response(response.body, {
      status: response.status,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

  } catch (error) {
    console.error('Chat API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}

// Handle OPTIONS for CORS if needed
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    },
  });
}
