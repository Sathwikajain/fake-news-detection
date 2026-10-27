"""
predict.py
----------
Inference module for the fake news detection system.

Loads the saved ensemble (BERT + classical ML) and exposes:
  - predict_single(text) -> (label, confidence, breakdown)
  - predict_batch(texts) -> list of (label, confidence) tuples

Handles:
  - Model loading from disk (BERT + tokenizer + classical ML pipelines)
  - Text preprocessing consistent with training
  - Confidence calibration
  - Human-readable output with emoji indicators
"""

import os
import json
import pickle
import warnings
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union

from preprocessing import clean_text

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

LABEL_REAL = 0
LABEL_FAKE = 1

LABEL_DISPLAY = {
    LABEL_REAL: "Real News ✅",
    LABEL_FAKE: "Fake News ❌",
}

LABEL_NAME = {
    LABEL_REAL: "REAL",
    LABEL_FAKE: "FAKE",
}


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class FakeNewsPredictor:
    """
    End-to-end predictor that wraps the trained ensemble model.

    Usage:
        predictor = FakeNewsPredictor.load(
            bert_dir="models/bert",
            ensemble_dir="models/ensemble",
        )
        result = predictor.predict_single(
            title="Shocking discovery by scientists",
            text="Researchers have found a cure for all diseases using tap water."
        )
        print(result["display"])
        # → "Fake News ❌  (confidence: 94.2%)"
    """

    def __init__(
        self,
        bert_model,
        tokenizer,
        lr_pipeline,
        rf_pipeline,
        ensemble_meta: dict,
        device: torch.device,
        max_length: int = 256,
        batch_size: int = 32,
    ):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.lr_pipeline = lr_pipeline
        self.rf_pipeline = rf_pipeline
        self.meta = ensemble_meta
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        self.bert_model.eval()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        bert_dir: str = "models/bert",
        ensemble_dir: str = "models/ensemble",
        device: Optional[torch.device] = None,
        max_length: int = 256,
    ) -> "FakeNewsPredictor":
        """
        Load the full ensemble from saved checkpoints.

        Args:
            bert_dir: Directory containing best_model.pt and tokenizer files.
            ensemble_dir: Directory containing lr_pipeline.pkl, rf_pipeline.pkl,
                          and ensemble_meta.json.
            device: Optional torch device. Defaults to CPU.
            max_length: Max token length (must match training).

        Returns:
            Initialized FakeNewsPredictor.
        """
        from transformers import BertTokenizerFast
        from train_bert import FakEditBertClassifier, DEFAULT_CONFIG

        if device is None:
            device = torch.device("cpu")  # Prefer CPU for deployment

        bert_path = Path(bert_dir)
        ens_path  = Path(ensemble_dir)

        print(f"[Predictor] Loading BERT tokenizer from: {bert_dir}")
        tokenizer = BertTokenizerFast.from_pretrained(bert_dir)

        print(f"[Predictor] Loading FakEdiT-BERT model from: {bert_dir}/best_model.pt")
        bert_model = FakEditBertClassifier(
            model_name="bert-base-uncased",
            dropout=0.0,  # No dropout at inference
        ).to(device)
        state_dict = torch.load(
            bert_path / "best_model.pt",
            map_location=device,
        )
        bert_model.load_state_dict(state_dict)
        bert_model.eval()

        print(f"[Predictor] Loading classical ML pipelines from: {ensemble_dir}")
        with open(ens_path / "lr_pipeline.pkl", "rb") as f:
            lr_pipeline = pickle.load(f)
        with open(ens_path / "rf_pipeline.pkl", "rb") as f:
            rf_pipeline = pickle.load(f)

        with open(ens_path / "ensemble_meta.json") as f:
            meta = json.load(f)

        print("[Predictor] All models loaded. Ready for inference.")
        return cls(
            bert_model=bert_model,
            tokenizer=tokenizer,
            lr_pipeline=lr_pipeline,
            rf_pipeline=rf_pipeline,
            ensemble_meta=meta,
            device=device,
            max_length=max_length,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _preprocess(self, title: str, text: str) -> str:
        """Merge and clean title + text exactly as done during training."""
        merged = f"{title.strip()} [SEP] {text.strip()}"
        return clean_text(merged, lowercase=True)

    def _bert_proba(self, texts: List[str]) -> np.ndarray:
        """Get BERT soft probabilities for a list of cleaned texts."""
        from train_bert import FakeNewsDataset
        from torch.utils.data import DataLoader

        dummy_labels = [0] * len(texts)
        dataset = FakeNewsDataset(texts, dummy_labels, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)

                logits = self.bert_model(input_ids, attention_mask, token_type_ids)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)  # (N, 2)

    def _ensemble_proba(self, texts: List[str]) -> np.ndarray:
        """Soft-vote across BERT, LR, and RF."""
        bert_p = self._bert_proba(texts)
        lr_p   = self.lr_pipeline.predict_proba(texts)
        rf_p   = self.rf_pipeline.predict_proba(texts)

        w_bert = self.meta.get("bert_weight", 0.6)
        w_lr   = self.meta.get("lr_weight",   0.25)
        w_rf   = self.meta.get("rf_weight",   0.15)

        return w_bert * bert_p + w_lr * lr_p + w_rf * rf_p

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(
        self,
        title: str = "",
        text: str = "",
        return_breakdown: bool = True,
    ) -> Dict:
        """
        Predict whether a news article is FAKE or REAL.

        Args:
            title: News headline / title.
            text: Article body / content.
            return_breakdown: If True, include per-classifier probabilities.

        Returns:
            dict with keys:
              - label       : int  (0=REAL, 1=FAKE)
              - label_name  : str  ("REAL" / "FAKE")
              - display     : str  ("Real News ✅ ..." / "Fake News ❌ ...")
              - confidence  : float  (0-1)
              - proba       : [P(REAL), P(FAKE)]
              - breakdown   : dict (only if return_breakdown=True)
        """
        cleaned = self._preprocess(title, text)
        if not cleaned:
            return _empty_result("Input text is empty after cleaning.")

        proba = self._ensemble_proba([cleaned])[0]  # (2,)
        label = int(np.argmax(proba))
        confidence = float(proba[label])

        result = {
            "label": label,
            "label_name": LABEL_NAME[label],
            "display": f"{LABEL_DISPLAY[label]}  (confidence: {confidence*100:.1f}%)",
            "confidence": confidence,
            "proba": {"REAL": float(proba[0]), "FAKE": float(proba[1])},
        }

        if return_breakdown:
            bert_p = self._bert_proba([cleaned])[0]
            lr_p   = self.lr_pipeline.predict_proba([cleaned])[0]
            rf_p   = self.rf_pipeline.predict_proba([cleaned])[0]
            result["breakdown"] = {
                "BERT":         {"REAL": float(bert_p[0]), "FAKE": float(bert_p[1])},
                "LogisticReg":  {"REAL": float(lr_p[0]),   "FAKE": float(lr_p[1])},
                "RandomForest": {"REAL": float(rf_p[0]),   "FAKE": float(rf_p[1])},
            }

        return result

    def predict_batch(
        self,
        items: List[Dict[str, str]],
    ) -> List[Dict]:
        """
        Batch prediction for efficiency.

        Args:
            items: List of dicts with 'title' and 'text' keys.

        Returns:
            List of result dicts (same structure as predict_single, no breakdown).
        """
        cleaned_texts = [
            self._preprocess(item.get("title", ""), item.get("text", ""))
            for item in items
        ]

        probas = self._ensemble_proba(cleaned_texts)  # (N, 2)

        results = []
        for i, proba in enumerate(probas):
            label = int(np.argmax(proba))
            confidence = float(proba[label])
            results.append({
                "label": label,
                "label_name": LABEL_NAME[label],
                "display": f"{LABEL_DISPLAY[label]}  (confidence: {confidence*100:.1f}%)",
                "confidence": confidence,
                "proba": {"REAL": float(proba[0]), "FAKE": float(proba[1])},
            })

        return results


def _empty_result(reason: str) -> Dict:
    return {
        "label": -1,
        "label_name": "UNKNOWN",
        "display": f"⚠️  Cannot classify: {reason}",
        "confidence": 0.0,
        "proba": {"REAL": 0.5, "FAKE": 0.5},
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Interactive command-line demo.

    Usage:
        python predict.py
    """
    import sys

    predictor = FakeNewsPredictor.load(
        bert_dir="models/bert",
        ensemble_dir="models/ensemble",
    )

    print("\n" + "=" * 60)
    print("  Fake News Detection System — Interactive Demo")
    print("=" * 60)
    print("Type 'quit' to exit.\n")

    while True:
        print("Enter the news TITLE (or press Enter to skip):")
        title = input("  Title > ").strip()
        if title.lower() == "quit":
            break

        print("Enter the news CONTENT:")
        text = input("  Content > ").strip()
        if text.lower() == "quit":
            break

        if not title and not text:
            print("⚠️  Please provide at least some text.\n")
            continue

        result = predictor.predict_single(title=title, text=text)
        print(f"\n  → {result['display']}")

        if "breakdown" in result:
            print("\n  Per-classifier breakdown:")
            for clf_name, scores in result["breakdown"].items():
                fake_pct = scores["FAKE"] * 100
                real_pct = scores["REAL"] * 100
                print(f"    {clf_name:14s}  FAKE={fake_pct:.1f}%  REAL={real_pct:.1f}%")
        print()
