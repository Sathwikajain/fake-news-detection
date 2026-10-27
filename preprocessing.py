"""
preprocessing.py
----------------
Minimal text preprocessing pipeline tailored for BERT-based models.
BERT handles its own tokenization internally, so we only apply light
normalization: lowercasing, stripping HTML/URLs/special chars, de-duplication,
and null removal.
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Text Cleaning
# ---------------------------------------------------------------------------

def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS URLs and bare www. links."""
    return re.sub(r"https?://\S+|www\.\S+", " ", text)


def remove_special_characters(text: str) -> str:
    """
    Keep only alphanumeric characters, whitespace, and common punctuation.
    Removes emojis, control chars, and exotic Unicode symbols.
    """
    # Normalize unicode to closest ASCII equivalent where possible
    text = unicodedata.normalize("NFKD", text)
    # Keep letters, digits, spaces, and basic punctuation
    text = re.sub(r"[^a-zA-Z0-9\s.,!?'\"-]", " ", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines/tabs into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str, lowercase: bool = True) -> str:
    """
    Full lightweight cleaning pipeline:
      1. Remove HTML tags
      2. Remove URLs
      3. Remove special characters
      4. Normalize whitespace
      5. Optionally lowercase

    Args:
        text: Raw input string.
        lowercase: Whether to convert to lowercase (recommended for bert-base-uncased).

    Returns:
        Cleaned string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = normalize_whitespace(text)

    if lowercase:
        text = text.lower()

    return text


# ---------------------------------------------------------------------------
# Dataset-level preprocessing
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "content",
    label_col: str = "label",
    label_mapping: dict = None,
    lowercase: bool = True,
    max_length: int = 512,
) -> pd.DataFrame:
    """
    Clean and prepare the full DataFrame for training.

    Steps:
      1. Drop rows with null text or label.
      2. Drop exact-duplicate rows.
      3. Apply text cleaning to the text column.
      4. Drop rows where cleaned text is empty or too short.
      5. Map labels to integers using label_mapping.
      6. Truncate very long texts to avoid extreme token counts.

    Args:
        df: Input DataFrame.
        text_col: Name of the text column to clean.
        label_col: Name of the label column.
        label_mapping: Dict mapping raw labels -> ints. If None, inferred.
        lowercase: Whether to lowercase during cleaning.
        max_length: Approximate character limit for truncation (rough proxy
                    for token count; BERT tokenizer will apply precise limit later).

    Returns:
        Cleaned DataFrame with integer 'label' column.
    """
    df = df.copy()
    n_start = len(df)

    # Step 1: Drop nulls
    df = df.dropna(subset=[text_col, label_col])
    print(f"[Preprocessor] Dropped {n_start - len(df)} rows with null values.")

    # Step 2: Drop duplicates
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[text_col])
    print(f"[Preprocessor] Dropped {before_dedup - len(df)} duplicate rows.")

    # Step 3: Clean text
    df[text_col] = df[text_col].apply(lambda t: clean_text(t, lowercase=lowercase))

    # Step 4: Drop empty texts
    before_empty = len(df)
    df = df[df[text_col].str.len() > 10]
    print(f"[Preprocessor] Dropped {before_empty - len(df)} near-empty text rows.")

    # Step 5: Map labels
    if label_mapping is None:
        label_mapping = _infer_mapping(df[label_col])

    df["label_int"] = df[label_col].map(
        lambda x: label_mapping.get(x, label_mapping.get(str(x).strip().lower()))
    )

    # Drop rows with unmappable labels
    before_label = len(df)
    df = df.dropna(subset=["label_int"])
    df["label_int"] = df["label_int"].astype(int)
    if before_label - len(df):
        print(f"[Preprocessor] Dropped {before_label - len(df)} rows with unmappable labels.")

    # Step 6: Rough character truncation (real truncation is done by tokenizer)
    # We cap at max_length * 6 chars (~6 chars/token on average)
    char_cap = max_length * 6
    df[text_col] = df[text_col].apply(lambda t: t[:char_cap])

    print(f"[Preprocessor] Final dataset size: {len(df):,} samples")
    return df.reset_index(drop=True)


def _infer_mapping(label_series: pd.Series) -> dict:
    """Infer a label -> int mapping from unique values in the series."""
    unique = label_series.dropna().unique()
    unique_str = sorted(str(u).strip().lower() for u in unique)

    if set(unique_str) <= {"fake", "real"}:
        return {"fake": 1, "real": 0, "FAKE": 1, "REAL": 0,
                "Fake": 1, "Real": 0, 1: 1, 0: 0}

    # Fallback: alphabetical assignment
    return {lbl: idx for idx, lbl in enumerate(sorted(str(u) for u in unique))}


# ---------------------------------------------------------------------------
# Train / Validation / Test Split
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    text_col: str = "content",
    label_col: str = "label_int",
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train / validation / test sets.

    Args:
        df: Cleaned DataFrame.
        text_col: Text column name.
        label_col: Integer label column name.
        test_size: Fraction reserved for test set (e.g. 0.20).
        val_size: Fraction of remaining data reserved for validation.
        random_state: Reproducibility seed.
        stratify: If True, maintain label distribution across splits.

    Returns:
        (train_df, val_df, test_df)
    """
    stratify_col = df[label_col] if stratify else None

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col,
    )

    stratify_val = train_val_df[label_col] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1.0 - test_size),
        random_state=random_state,
        stratify=stratify_val,
    )

    print(f"\n[Preprocessor] Dataset splits:")
    print(f"  Train : {len(train_df):,}  ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val   : {len(val_df):,}  ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test  : {len(test_df):,}  ({len(test_df)/len(df)*100:.1f}%)")

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_class_weights(labels: List[int]) -> dict:
    """
    Compute inverse-frequency class weights to handle imbalance.

    Args:
        labels: List/array of integer class labels.

    Returns:
        dict {class_int: weight_float}
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels_arr = np.array(labels)
    classes = np.unique(labels_arr)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels_arr
    )
    mapping = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[Preprocessor] Class weights: {mapping}")
    return mapping


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample = pd.DataFrame({
        "content": [
            "Breaking news! <b>Aliens</b> have landed in https://example.com New York!!!",
            "The senate passed the new infrastructure bill by a wide margin.",
            None,
            "Fake article with lots of   extra   spaces and... weird chars €£",
            "Real news about economy growth in Q3 2024.",
        ],
        "label": ["FAKE", "REAL", "FAKE", "FAKE", "REAL"],
    })

    cleaned = clean_dataframe(sample)
    print("\nCleaned sample:")
    print(cleaned[["content", "label", "label_int"]].to_string())

    train, val, test = split_dataset(cleaned)
    print(f"\nSplit complete. Train={len(train)}, Val={len(val)}, Test={len(test)}")
