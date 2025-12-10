"""Batch convert PDFs to Markdown using marker_single.

Usage (from project root):
    python -m src.helpers.pdfs_to_markdown \
        --input_dir src/data/pdfs/pages \
        --output_dir src/data/markdown \
        --start_index 1

"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import os
from datetime import datetime

def run_marker(pdf_path: Path, output_dir: Path, use_gpu: bool = False, gpu_id: int = 0):
    """Run marker on a single PDF using the Python API.
    Returns an object with returncode, stdout, stderr attributes.
    """
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered
        
        # Set GPU device if requested
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            try:
                import torch
                print(f"[gpu] torch.cuda.is_available() = {torch.cuda.is_available()}")
            except Exception:
                print("[gpu] torch not available")
        
        # Initialize converter with models
        artifact_dict = create_model_dict()
        converter = PdfConverter(artifact_dict=artifact_dict)
        
        # Convert PDF
        rendered = converter(str(pdf_path))
        
        # Extract markdown text - rendered is a RenderedOutput object
        if hasattr(rendered, 'markdown'):
            markdown_text = rendered.markdown
        elif hasattr(rendered, 'text'):
            markdown_text = rendered.text
        else:
            # Try text_from_rendered helper
            markdown_text = text_from_rendered(rendered)
        
        # Write output
        output_file = output_dir / f"{pdf_path.stem}.md"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        # Return success result compatible with old interface
        class MockProc:
            returncode = 0
            stdout = f"Success: {output_file}"
            stderr = ""
        
        return MockProc()
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        
        class MockProc:
            returncode = 1
            stdout = ""
            stderr = error_msg
        
        return MockProc()

def convert_batch(input_dir: Path, output_dir: Path, overwrite: bool, log_dir: Path, start_index: int = 1, use_gpu: bool = False, gpu_id: int = 0) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Sort by numeric component if filenames have a pattern like 'page_<number>.pdf'
    def numeric_key(p: Path):
        stem = p.stem  # e.g. 'page_10'
        # Extract last continuous digits
        import re
        m = re.search(r'(\d+)$', stem)
        if m:
            try:
                num = int(m.group(1))
                return (0, num)  # Numeric files first, then by number
            except ValueError:
                return (1, stem)  # Non-numeric after numeric
        return (1, stem)  # Non-numeric files
    pdf_files = sorted(input_dir.glob("*.pdf"), key=numeric_key)
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return

    total = len(pdf_files)
    if start_index < 1:
        print(f"--start_index must be >= 1 (got {start_index}). Defaulting to 1.")
        start_index = 1
    if start_index > total:
        print(f"--start_index ({start_index}) is greater than total files ({total}). Nothing to do.")
        return

    print(f"Found {total} PDF files. Starting conversion…")

    # Slice to start from the specified file (1-based index)
    files_to_process = pdf_files[start_index - 1:]

    for idx, pdf_path in enumerate(files_to_process, start=start_index):
        base_name = pdf_path.stem  # e.g., page_1
        expected_md = output_dir / f"{base_name}.md"
        if expected_md.exists() and not overwrite:
            print(f"[{idx}/{total}] Skip {pdf_path.name} (already converted)")
            continue
        proc = run_marker(pdf_path, output_dir, use_gpu=use_gpu, gpu_id=gpu_id)
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"

        # Write log
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{base_name}_{ts}.log"
        with open(log_file, "w", encoding="utf-8") as lf:
            lf.write(f"Command return code: {proc.returncode}\n")
            lf.write("=== STDOUT ===\n")
            lf.write(proc.stdout or "(empty)\n")
            lf.write("\n=== STDERR ===\n")
            lf.write(proc.stderr or "(empty)\n")

        print(f"[{idx}/{total}] {pdf_path} -> {status} at: {expected_md}")

    print("Conversion batch finished.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch convert PDFs to Markdown using marker_single.")
    p.add_argument("--input_dir", default="src/data/pdfs/inputs_new", help="Directory containing PDF files")
    p.add_argument("--output_dir", default="src/data/pdfs/outputs_new", help="Directory to write markdown files")
    p.add_argument("--overwrite", help="Overwrite existing markdown files")
    p.add_argument("--log_dir", default="src/data/markdown/markdown_logs", help="Directory to write conversion logs")
    p.add_argument("--start_index", type=int, default=1, help="1-based index of the first file to process after sorting") # Default to 93 for resuming large batches
    p.add_argument("--use_gpu", action="store_true", help="Use GPU for processing if available")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    convert_batch(
        input_dir,
        output_dir,
        args.overwrite,
        log_dir,
        start_index=args.start_index,
        use_gpu=args.use_gpu,
    )

if __name__ == "__main__":
    main()
