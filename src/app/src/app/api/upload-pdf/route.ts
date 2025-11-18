import { NextRequest, NextResponse } from 'next/server';
import { spawn } from 'child_process';
import { writeFile, unlink } from 'fs/promises';
import { join } from 'path';
import { tmpdir } from 'os';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('pdf') as File;

    if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
      return NextResponse.json(
        { error: 'No PDF file provided or invalid file type' },
        { status: 400 }
      );
    }

    // Check file size (20MB limit)
    if (file.size > 20 * 1024 * 1024) {
      return NextResponse.json(
        { error: 'File size too large. Maximum 20MB allowed.' },
        { status: 400 }
      );
    }

    // Create temporary file path
    const tempDir = tmpdir();
    const tempFileName = `upload_${Date.now()}_${file.name}`;
    const tempFilePath = join(tempDir, tempFileName);

    // Save uploaded file to temporary location
    const buffer = await file.arrayBuffer();
    await writeFile(tempFilePath, Buffer.from(buffer));

    try {
      // Run the Python pipeline script
      const pythonScript = 'src/pipeline/pipeline_pdf_vec.py';

      // Get Python executable path and environment
      const pythonExecutable = process.platform === 'win32' ? 'python' : 'python3';

      return new Promise((resolve, reject) => {
        const pythonProcess = spawn(pythonExecutable, [pythonScript, tempFilePath], {
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: process.cwd(),
          env: { ...process.env, PYTHONPATH: process.cwd() }
        });

        let stdout = '';
        let stderr = '';

        pythonProcess.stdout.on('data', (data) => {
          stdout += data.toString();
          console.log('Pipeline output:', data.toString());
        });

        pythonProcess.stderr.on('data', (data) => {
          stderr += data.toString();
          console.error('Pipeline error:', data.toString());
        });

        pythonProcess.on('close', async (code) => {
          // Clean up temp file
          try {
            await unlink(tempFilePath);
          } catch (cleanupError) {
            console.error('Error cleaning up temp file:', cleanupError);
          }

          if (code === 0) {
            // Success
            resolve(NextResponse.json({
              success: true,
              message: `Successfully processed PDF: ${file.name}`,
              output: stdout
            }));
          } else {
            // Pipeline failed
            const errorMessage = stderr || stdout || 'Pipeline execution failed';
            console.error('Pipeline failed with code:', code, 'Error:', errorMessage);
            resolve(NextResponse.json(
              {
                error: `Pipeline failed: ${errorMessage}`,
                code: code
              },
              { status: 500 }
            ));
          }
        });

        pythonProcess.on('error', (error) => {
          console.error('Failed to start pipeline:', error);

          // Clean up temp file on error
          try {
            unlink(tempFilePath).catch(() => {});
          } catch (cleanupError) {
            console.error('Error cleaning up temp file on process error:', cleanupError);
          }

          resolve(NextResponse.json(
            { error: `Failed to execute pipeline: ${error.message}` },
            { status: 500 }
          ));
        });

        // Add timeout (5 minutes for large PDFs)
        const timeout = setTimeout(() => {
          console.error('Pipeline timeout - killing process');
          pythonProcess.kill();

          try {
            unlink(tempFilePath).catch(() => {});
          } catch (cleanupError) {
            console.error('Error cleaning up temp file on timeout:', cleanupError);
          }

          resolve(NextResponse.json(
            { error: 'Pipeline execution timed out (5 minutes)' },
            { status: 408 }
          ));
        }, 5 * 60 * 1000); // 5 minutes

        pythonProcess.on('close', (code) => {
          clearTimeout(timeout);
        });
      });

    } catch (error) {
      // Clean up temp file on error
      try {
        await unlink(tempFilePath);
      } catch (cleanupError) {
        console.error('Error cleaning up temp file:', cleanupError);
      }

      console.error('Upload processing error:', error);
      return NextResponse.json(
        { error: 'Failed to process uploaded file' },
        { status: 500 }
      );
    }

  } catch (error) {
    console.error('Upload request error:', error);
    return NextResponse.json(
      { error: 'Invalid request or file upload failed' },
      { status: 400 }
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
