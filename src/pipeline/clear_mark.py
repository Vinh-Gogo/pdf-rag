from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent  # goes from /src/pipeline -> /rag
sys.path.insert(0, str(project_root))

from src.helpers.markdown_to_markdown import process_file, markdown_to_clean_text

def clear_mark(
    src: Path,
    base_dir: Path,
    pattern: str = "page_cleared_*.txt",
    overwrite: bool = False,
    dry_run: bool = True,
    to_text: bool = False
) -> str:

    return process_file(src, base_dir, overwrite=overwrite, dry_run=dry_run, to_text=to_text)

if __name__ == "__main__":

    src_directory = Path(fr'src\data\pdfs\source.md')
    base_directory = Path(fr'src\data\pipeline.txt')

    contents = clear_mark(src_directory, base_directory)

    print(contents)
    # contents = base_directory.read_text(encoding='utf-8')

    new_contents = []
    for row in contents.split("\n\n"):
        row = markdown_to_clean_text(row)
        row = row.replace('<br>', " ")
        new_contents.append(row)
        

    path_new_file = base_directory.parent / "cleaned_source.txt"
    path_new_file.write_text("\n\n".join(new_contents), encoding='utf-8')