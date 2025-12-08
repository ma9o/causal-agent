from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"


def load_text_chunks(path: Path, separator: str = "\n\n---\n\n") -> list[str]:
    """Load text chunks from a preprocessed file."""
    content = path.read_text()
    return [chunk.strip() for chunk in content.split(separator) if chunk.strip()]


def get_latest_preprocessed_file(directory: Path | None = None) -> Path | None:
    """
    Find the most recently modified .txt file in the preprocessed directory.

    Args:
        directory: Directory to search (default: data/preprocessed/)

    Returns:
        Path to latest file, or None if no files found
    """
    search_dir = directory or PREPROCESSED_DIR
    txt_files = list(search_dir.glob("*.txt"))

    if not txt_files:
        return None

    # Sort by modification time, newest first
    return max(txt_files, key=lambda p: p.stat().st_mtime)


def resolve_input_path(filename: str | None = None) -> Path:
    """
    Resolve input path from filename or find latest preprocessed file.

    Args:
        filename: Optional filename (just name, not full path) in preprocessed dir.
                  If None, returns the latest preprocessed file.

    Returns:
        Full path to the input file

    Raises:
        FileNotFoundError: If no matching file found
    """
    if filename:
        path = PREPROCESSED_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return path

    latest = get_latest_preprocessed_file()
    if not latest:
        raise FileNotFoundError(f"No preprocessed files found in {PREPROCESSED_DIR}")

    return latest
