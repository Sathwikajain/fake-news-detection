"""
data_loading.py
---------------
Handles loading, validating, and initial inspection of the dataset.
Supports CSV and XLSX formats with columns: 'title', 'text', 'label'.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV or XLSX dataset from disk.

    Args:
        filepath: Path to the .csv or .xlsx file.

    Returns:
        Raw DataFrame with at least columns ['title', 'text', 'label'].

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or format is unsupported.
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .xlsx")

    print(f"[DataLoader] Loaded {len(df):,} rows from '{filepath}'")
    _validate_columns(df)
    _print_summary(df)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Ensure required columns exist (case-insensitive)."""
    # Normalize column names to lowercase for comparison
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"title", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )


def _print_summary(df: pd.DataFrame) -> None:
    """Print a quick summary of the dataset."""
    print("\n[DataLoader] Dataset Summary:")
    print(f"  Shape        : {df.shape}")
    print(f"  Null values  :\n{df[['title', 'text', 'label']].isnull().sum().to_string()}")

    # Label distribution
    label_counts = df["label"].value_counts()
    print(f"\n  Label distribution:")
    for lbl, cnt in label_counts.items():
        pct = cnt / len(df) * 100
        print(f"    {lbl}: {cnt:,} ({pct:.1f}%)")


def get_label_mapping(df: pd.DataFrame) -> dict:
    """
    Infer a consistent label -> integer mapping.

    Supports common conventions:
      - 'FAKE' / 'REAL'
      - 'fake' / 'real'
      - 0 / 1
      - 1 / 0 (inverted)

    Returns:
        dict mapping raw label -> int  (0 = REAL, 1 = FAKE)
    """
    unique = df["label"].dropna().unique()
    unique_lower = [str(u).strip().lower() for u in unique]

    if set(unique_lower) <= {"fake", "real"}:
        return {"real": 0, "fake": 1,
                "REAL": 0, "FAKE": 1,
                "Real": 0, "Fake": 1}

    if set(unique_lower) <= {"0", "1"}:
        # Assume 1 = FAKE, 0 = REAL
        return {0: 0, 1: 1, "0": 0, "1": 1}

    # Fallback: sort and assign 0, 1 alphabetically
    sorted_labels = sorted(str(u) for u in unique)
    mapping = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
    print(f"[DataLoader] Warning: Unknown label scheme. Assigned mapping: {mapping}")
    return mapping


def merge_title_text(df: pd.DataFrame, sep: str = " [SEP] ") -> pd.DataFrame:
    """
    Combine 'title' and 'text' into a single 'content' column.

    Args:
        df: DataFrame with 'title' and 'text' columns.
        sep: Separator string inserted between title and body.

    Returns:
        DataFrame with a new 'content' column.
    """
    df = df.copy()
    df["title"] = df["title"].fillna("").astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["content"] = df["title"].str.strip() + sep + df["text"].str.strip()
    print(f"[DataLoader] Merged 'title' + 'text' -> 'content'  (sep='{sep}')")
    return df


if __name__ == "__main__":
    # Quick smoke test with a synthetic dataset
    import tempfile

    sample = pd.DataFrame({
        "title": ["Breaking: Aliens land on Earth!", "Congress passes new budget"],
        "text": ["Scientists confirm extraterrestrial life discovered.", "The bill was approved by a majority vote."],
        "label": ["FAKE", "REAL"],
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample.to_csv(f.name, index=False)
        tmp_path = f.name

    df = load_dataset(tmp_path)
    mapping = get_label_mapping(df)
    df = merge_title_text(df)
    print(df[["content", "label"]].to_string())
    os.unlink(tmp_path)
