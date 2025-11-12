"""Batch convert PDFs to Markdown using marker_single.

Prerequisites:
    - marker_single command available in current virtual environment (e.g. `pip install pdf-marker` or the appropriate package providing marker_single).
    - Activate venv before running.

Usage (from project root):
    python -m src.helpers.pdfs_to_markdown \
        --input_dir src/data/pdfs/pages \
        --output_dir src/data/pdfs/markdown \
        --pattern page_*.pdf \
        --overwrite \
        --start_index 1

Notes:
    - Creates output_dir if missing.
    - Skips files already converted unless --overwrite specified.
    - Captures stderr/stdout per file to a log folder.
    - Files are sorted numerically by trailing digits (e.g., page_2 before page_10).
    - Use --start_index (1-based) to resume from a specific file in the sorted list.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
import os
from datetime import datetime

DEFAULT_PATTERN = "*.pdf"

def run_marker(pdf_path: Path, output_dir: Path, use_gpu: bool = False, gpu_id: int = 0) -> subprocess.CompletedProcess:
    """Run marker_single on a single PDF and return the CompletedProcess.
    Tries the executable first, then falls back to `python -m marker_single` if needed.
    If use_gpu is True, sets common CUDA env hints (no CLI flags) and falls back to CPU if the executable is missing.
    """
    def device_env() -> dict[str, str] | None:
        if not use_gpu:
            return None
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu_id))
        return env

    # We do not pass a --device flag; marker_single doesn't support it. Rely on CUDA_VISIBLE_DEVICES and library auto-detection.
    # Optional: log basic GPU detection to stdout for user visibility.
    if use_gpu:
        try:
            import torch  # type: ignore
            print(f"[gpu] torch.cuda.is_available() = {torch.cuda.is_available()}")
        except Exception:
            print("[gpu] torch not available; proceeding to set CUDA_VISIBLE_DEVICES only.")

    # 1) Try direct executable
    exec_cmd = ["marker_single", str(pdf_path), "--output_dir", str(output_dir)]
    try:
        proc = subprocess.run(
            exec_cmd,
            capture_output=True,
            text=True,
            check=False,
            env=device_env(),
        )
        # If the executable isn't found or clearly not available, fall back to module
        not_found_markers = ("not found", "not recognized", "No such file or directory")
        if proc.returncode != 0 and (
            any(m in (proc.stderr or "") for m in not_found_markers) or proc.returncode == 127
        ):
            raise FileNotFoundError("marker_single executable not available")
        return proc
    except FileNotFoundError:
        # 2) Try as a module with current Python
        module_cmd = [
            sys.executable,
            "-m",
            "marker_single",
            str(pdf_path),
            "--output_dir",
            str(output_dir),
        ]
        proc2 = subprocess.run(
            module_cmd,
            capture_output=True,
            text=True,
            check=False,
            env=device_env(),
        )
        return proc2

def convert_batch(input_dir: Path, output_dir: Path, pattern: str, overwrite: bool, log_dir: Path, start_index: int = 1, use_gpu: bool = False, gpu_id: int = 0) -> None:
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
                return int(m.group(1))
            except ValueError:
                return stem
        return stem
    pdf_files = sorted(input_dir.glob(pattern), key=numeric_key)
    if not pdf_files:
        print(f"No PDF files found matching pattern '{pattern}' in {input_dir}")
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

        print(f"[{idx}/{total}] {pdf_path} -> {status}")

    print("Conversion batch finished.")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Batch convert PDFs to Markdown using marker_single.")
    p.add_argument("--gpu", action="store_true", help="Attempt to use CUDA (adds --device cuda if supported; falls back to CPU if not)")
    p.add_argument("--gpu_id", type=int, default=0, help="GPU id to expose via CUDA_VISIBLE_DEVICES when --gpu is set")
    p.add_argument("--input_dir", default="src/data/pdfs/pages", help="Directory containing PDF files")
    p.add_argument("--output_dir", default="src/data/markdown", help="Directory to write markdown files")
    p.add_argument("--pattern", default=DEFAULT_PATTERN, help="Glob pattern to match PDF files (default *.pdf)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing markdown files")
    p.add_argument("--log_dir", default="src/data/markdown/markdown_logs", help="Directory to write conversion logs")
    p.add_argument("--start_index", type=int, default=25, help="1-based index of the first file to process after sorting")
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
        args.pattern,
        args.overwrite,
        log_dir,
        start_index=args.start_index,
        use_gpu=args.gpu,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    main()
