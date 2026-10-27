"""
train_ensemble.py
-----------------
Hybrid ensemble combining:
  1. FakEdiT-BERT (deep contextual features + soft probabilities)
  2. TF-IDF + Logistic Regression (lexical / surface-level features)
  3. TF-IDF + Random Forest (non-linear lexical patterns)

Ensemble strategy: **Soft voting** (average predicted probabilities)
  - Optionally: stacking with a meta-learner (LogisticRegression on top of
    the three base classifiers' probability outputs)

The ensemble is more robust than BERT alone because:
  - Classical models capture vocabulary patterns that BERT may over-smooth.
  - Ensemble reduces variance and guards against BERT overconfidence.
  - CPU-friendly: once BERT embeddings are extracted, ML inference is fast.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional, List

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from train_bert import FakEditBertClassifier, FakeNewsDataset, DEFAULT_CONFIG

try:
    from transformers import BertTokenizerFast
except ImportError:
    BertTokenizerFast = None


# ---------------------------------------------------------------------------
# Feature extraction: BERT soft probabilities
# ---------------------------------------------------------------------------

def extract_bert_probabilities(
    model: FakEditBertClassifier,
    texts: List[str],
    tokenizer,
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 256,
) -> np.ndarray:
    """
    Extract soft probability vectors from the BERT model for each text.

    Args:
        model: Trained FakEditBertClassifier in eval mode.
        texts: List of cleaned text strings.
        tokenizer: Matching BertTokenizerFast.
        device: CPU or CUDA device.
        batch_size: Inference batch size.
        max_length: Max token length.

    Returns:
        np.ndarray of shape (N, 2) — [P(REAL), P(FAKE)] per sample.
    """
    model.eval()
    dummy_labels = [0] * len(texts)
    dataset = FakeNewsDataset(texts, dummy_labels, tokenizer, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs = []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)  # (N, 2)


# ---------------------------------------------------------------------------
# Build classical ML pipelines
# ---------------------------------------------------------------------------

def build_logreg_pipeline() -> Pipeline:
    """TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="saga",
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def build_rf_pipeline() -> Pipeline:
    """TF-IDF + Random Forest pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.90,
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])


# ---------------------------------------------------------------------------
# Ensemble: soft voting
# ---------------------------------------------------------------------------

class HybridEnsemble:
    """
    Soft-voting ensemble over:
      - BERT classifier (probability output)
      - TF-IDF + Logistic Regression
      - TF-IDF + Random Forest

    Weights are tunable; default gives BERT a higher weight.
    """

    def __init__(
        self,
        bert_weight: float = 0.6,
        lr_weight: float = 0.25,
        rf_weight: float = 0.15,
    ):
        self.bert_weight = bert_weight
        self.lr_weight = lr_weight
        self.rf_weight = rf_weight

        self.lr_pipeline = build_logreg_pipeline()
        self.rf_pipeline = build_rf_pipeline()

        self._bert_model = None
        self._tokenizer = None
        self._device = None
        self._max_length = 256
        self._batch_size = 32

    def set_bert(
        self,
        model: FakEditBertClassifier,
        tokenizer,
        device: torch.device,
        max_length: int = 256,
        batch_size: int = 32,
    ):
        self._bert_model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size

    def _bert_probs(self, texts: List[str]) -> np.ndarray:
        return extract_bert_probabilities(
            self._bert_model, texts, self._tokenizer,
            self._device, self._batch_size, self._max_length,
        )

    def fit(self, train_texts: List[str], train_labels: List[int]):
        """Train the classical ML components (BERT is already fine-tuned)."""
        print("[Ensemble] Training TF-IDF + Logistic Regression...")
        self.lr_pipeline.fit(train_texts, train_labels)
        print("[Ensemble] Training TF-IDF + Random Forest...")
        self.rf_pipeline.fit(train_texts, train_labels)
        print("[Ensemble] Classical models trained.")

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Soft-vote: weighted average of the three classifiers' probabilities.

        Returns:
            np.ndarray of shape (N, 2)  [P(REAL), P(FAKE)]
        """
        bert_p = self._bert_probs(texts)             # (N, 2)
        lr_p   = self.lr_pipeline.predict_proba(texts)  # (N, 2)
        rf_p   = self.rf_pipeline.predict_proba(texts)  # (N, 2)

        combined = (
            self.bert_weight * bert_p
            + self.lr_weight  * lr_p
            + self.rf_weight  * rf_p
        )
        return combined

    def predict(self, texts: List[str]) -> np.ndarray:
        """Return hard class predictions (0=REAL, 1=FAKE)."""
        proba = self.predict_proba(texts)
        return np.argmax(proba, axis=1)

    def save(self, save_dir: str):
        """Persist classical models and weights."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "lr_pipeline.pkl", "wb") as f:
            pickle.dump(self.lr_pipeline, f)
        with open(path / "rf_pipeline.pkl", "wb") as f:
            pickle.dump(self.rf_pipeline, f)

        meta = {
            "bert_weight": self.bert_weight,
            "lr_weight": self.lr_weight,
            "rf_weight": self.rf_weight,
        }
        with open(path / "ensemble_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Ensemble] Models saved to: {save_dir}")

    @classmethod
    def load(cls, save_dir: str) -> "HybridEnsemble":
        """Load classical models and meta from disk."""
        path = Path(save_dir)

        with open(path / "ensemble_meta.json") as f:
            meta = json.load(f)

        ensemble = cls(
            bert_weight=meta["bert_weight"],
            lr_weight=meta["lr_weight"],
            rf_weight=meta["rf_weight"],
        )
        with open(path / "lr_pipeline.pkl", "rb") as f:
            ensemble.lr_pipeline = pickle.load(f)
        with open(path / "rf_pipeline.pkl", "rb") as f:
            ensemble.rf_pipeline = pickle.load(f)

        print(f"[Ensemble] Models loaded from: {save_dir}")
        return ensemble


# ---------------------------------------------------------------------------
# Training + evaluation pipeline
# ---------------------------------------------------------------------------

def train_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    bert_model: FakEditBertClassifier,
    tokenizer,
    text_col: str = "content",
    label_col: str = "label_int",
    save_dir: str = "models/ensemble",
    bert_weight: float = 0.6,
    lr_weight: float = 0.25,
    rf_weight: float = 0.15,
    max_length: int = 256,
) -> Tuple[HybridEnsemble, Dict]:
    """
    Train the full hybrid ensemble and evaluate it.

    Args:
        train_df / val_df / test_df: Preprocessed data splits.
        bert_model: Fine-tuned FakEditBertClassifier.
        tokenizer: Corresponding BertTokenizerFast.
        text_col: Column with cleaned text.
        label_col: Column with integer labels.
        save_dir: Where to save ensemble models.
        bert/lr/rf_weight: Voting weights.
        max_length: BERT tokenizer limit.

    Returns:
        (ensemble, metrics_dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_texts  = train_df[text_col].tolist()
    val_texts    = val_df[text_col].tolist()
    test_texts   = test_df[text_col].tolist()

    train_labels = train_df[label_col].tolist()
    val_labels   = val_df[label_col].tolist()
    test_labels  = test_df[label_col].tolist()

    # Build and configure ensemble
    ensemble = HybridEnsemble(
        bert_weight=bert_weight,
        lr_weight=lr_weight,
        rf_weight=rf_weight,
    )
    ensemble.set_bert(bert_model, tokenizer, device, max_length=max_length)

    # Train classical components
    combined_train_texts  = train_texts + val_texts
    combined_train_labels = train_labels + val_labels
    ensemble.fit(combined_train_texts, combined_train_labels)

    # Evaluate on test set
    print("\n[Ensemble] Running test set evaluation...")
    test_probs = ensemble.predict_proba(test_texts)
    test_preds = np.argmax(test_probs, axis=1)

    acc = accuracy_score(test_labels, test_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average="weighted", zero_division=0
    )
    cm = confusion_matrix(test_labels, test_preds)

    print("\n" + "=" * 60)
    print("ENSEMBLE TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(test_labels, test_preds, target_names=["REAL", "FAKE"]))
    print(f"Confusion Matrix:\n{cm}")

    # Per-classifier breakdown
    print("\n[Ensemble] Individual classifier metrics:")
    for name, pipe in [("LogReg", ensemble.lr_pipeline), ("RandForest", ensemble.rf_pipeline)]:
        preds_i = pipe.predict(test_texts)
        acc_i = accuracy_score(test_labels, preds_i)
        _, _, f1_i, _ = precision_recall_fscore_support(
            test_labels, preds_i, average="weighted", zero_division=0
        )
        print(f"  {name:12s}  Acc={acc_i:.4f}  F1={f1_i:.4f}")

    bert_preds = np.argmax(extract_bert_probabilities(
        bert_model, test_texts, tokenizer, device, max_length=max_length
    ), axis=1)
    acc_bert = accuracy_score(test_labels, bert_preds)
    _, _, f1_bert, _ = precision_recall_fscore_support(
        test_labels, bert_preds, average="weighted", zero_division=0
    )
    print(f"  {'BERT':12s}  Acc={acc_bert:.4f}  F1={f1_bert:.4f}")
    print(f"  {'Ensemble':12s}  Acc={acc:.4f}  F1={f1:.4f}  ← Combined")

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "bert_accuracy": float(acc_bert),
        "bert_f1": float(f1_bert),
    }

    ensemble.save(save_dir)

    with open(Path(save_dir) / "ensemble_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return ensemble, metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage:
        python train_ensemble.py

    Run this AFTER train_bert.py has saved a model to models/bert/.
    """
    from data_loading import load_dataset, merge_title_text, get_label_mapping
    from preprocessing import clean_dataframe, split_dataset, compute_class_weights
    from train_bert import FakEditBertClassifier, DEFAULT_CONFIG, train_bert_model
    from transformers import BertTokenizerFast

    DATASET_PATH = "data/fake_news.csv"
    BERT_MODEL_DIR = "models/bert"
    ENSEMBLE_SAVE_DIR = "models/ensemble"

    # Load + preprocess
    df = load_dataset(DATASET_PATH)
    df = merge_title_text(df)
    label_mapping = get_label_mapping(df)
    df = clean_dataframe(df, label_mapping=label_mapping)
    train_df, val_df, test_df = split_dataset(df)
    class_weights = compute_class_weights(train_df["label_int"].tolist())

    # Load or train BERT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_DIR)
    bert_model = FakEditBertClassifier(
        model_name="bert-base-uncased",
        dropout=DEFAULT_CONFIG["dropout"],
    ).to(device)
    bert_model.load_state_dict(
        torch.load(f"{BERT_MODEL_DIR}/best_model.pt", map_location=device)
    )

    # Train ensemble
    ensemble, metrics = train_ensemble(
        train_df, val_df, test_df,
        bert_model=bert_model,
        tokenizer=tokenizer,
        save_dir=ENSEMBLE_SAVE_DIR,
    )
    print(f"\nFinal ensemble accuracy: {metrics['accuracy']:.4f}")
