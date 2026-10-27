"""
train_bert.py
-------------
Fine-tunes a BERT-based classifier (bert-base-uncased) for fake news detection.

Architecture: FakEdiT-inspired dual-attention BERT
  - Pre-trained BERT encoder (bert-base-uncased)
  - Multi-head self-attention over CLS token features (FakEdiT-style refinement)
  - Dropout regularization
  - Binary classification head

Training features:
  - AdamW optimizer with linear warm-up + cosine decay schedule
  - Class-weighted loss to handle imbalance
  - Early stopping
  - Mixed precision (if CUDA available)
  - Saves best checkpoint based on validation F1
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from transformers import (
    BertTokenizerFast,
    BertModel,
    get_linear_schedule_with_warmup,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model_name": "bert-base-uncased",
    "max_length": 256,          # Token limit (256 balances speed vs coverage)
    "batch_size": 16,
    "epochs": 5,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "dropout": 0.3,
    "weight_decay": 0.01,
    "patience": 2,              # Early stopping patience (epochs)
    "save_dir": "models/bert",
    "seed": 42,
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FakeNewsDataset(Dataset):
    """PyTorch Dataset wrapping tokenized text + labels."""

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: BertTokenizerFast,
        max_length: int = 256,
    ):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get(
                "token_type_ids", torch.zeros_like(self.encodings["input_ids"])
            )[idx],
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
        return item


# ---------------------------------------------------------------------------
# Model: FakEdiT-inspired BERT classifier
# ---------------------------------------------------------------------------

class FakEditBertClassifier(nn.Module):
    """
    BERT-based fake news classifier inspired by the FakEdiT architecture.

    FakEdiT (Fake News Detection Transformer) augments the standard BERT
    classifier by adding a refinement layer over the pooled representation:
      1. BERT encoder extracts contextual embeddings.
      2. A lightweight multi-head self-attention over the CLS token hidden
         state and a learned 'edit' query vector refines the representation.
      3. Layer norm + residual connection for stable training.
      4. Classification head produces logits.

    This simulates the key FakEdiT insight: treating the classification
    decision as an 'editing' step that aligns evidence with the CLS token.
    """

    def __init__(self, model_name: str = "bert-base-uncased", dropout: float = 0.3):
        super().__init__()

        # Core BERT encoder
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base

        # ---- FakEdiT refinement block ----
        # Learned 'edit query' vector (acts as attention anchor)
        self.edit_query = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Multi-head attention: edit query attends to BERT sequence output
        self.edit_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )
        self.edit_norm = nn.LayerNorm(hidden_size)

        # ---- Classification head ----
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # concat CLS + edit output
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Returns:
            logits: (batch_size, 2)
        """
        # 1. BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state  # (B, T, H)
        cls_output = outputs.pooler_output           # (B, H)  — [CLS] token

        # 2. FakEdiT refinement
        batch_size = sequence_output.size(0)
        edit_q = self.edit_query.expand(batch_size, -1, -1)  # (B, 1, H)

        # Build key padding mask: True where attention_mask == 0 (pad tokens)
        key_padding_mask = (attention_mask == 0)  # (B, T)

        edit_out, _ = self.edit_attention(
            query=edit_q,
            key=sequence_output,
            value=sequence_output,
            key_padding_mask=key_padding_mask,
        )  # (B, 1, H)
        edit_out = edit_out.squeeze(1)  # (B, H)
        edit_out = self.edit_norm(edit_out + cls_output)  # residual

        # 3. Concatenate CLS and edit representations
        combined = torch.cat([cls_output, edit_out], dim=-1)  # (B, 2H)
        combined = self.dropout(combined)

        # 4. Classification
        logits = self.classifier(combined)  # (B, 2)
        return logits


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Seed everything for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict:
    """Run evaluation over a DataLoader and return metrics dict."""
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return {
        "loss": total_loss / len(loader),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_bert_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "content",
    label_col: str = "label_int",
    config: dict = None,
    class_weights: dict = None,
) -> Tuple[FakEditBertClassifier, BertTokenizerFast, Dict]:
    """
    Fine-tune the FakEdiT-BERT model on the provided data splits.

    Args:
        train_df, val_df, test_df: DataFrames with text and label columns.
        text_col: Column containing cleaned text.
        label_col: Column containing integer labels.
        config: Training hyperparameters (see DEFAULT_CONFIG).
        class_weights: Optional dict {class: weight} for imbalanced datasets.

    Returns:
        (trained_model, tokenizer, metrics_dict)
    """
    if config is None:
        config = DEFAULT_CONFIG

    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[BERT Trainer] Using device: {device}")

    # ---- Tokenizer ----
    print(f"[BERT Trainer] Loading tokenizer: {config['model_name']}")
    tokenizer = BertTokenizerFast.from_pretrained(config["model_name"])

    # ---- Datasets & Loaders ----
    train_texts = train_df[text_col].tolist()
    val_texts   = val_df[text_col].tolist()
    test_texts  = test_df[text_col].tolist()

    train_labels = train_df[label_col].tolist()
    val_labels   = val_df[label_col].tolist()
    test_labels  = test_df[label_col].tolist()

    print("[BERT Trainer] Tokenizing datasets (this may take a moment)...")
    train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer, config["max_length"])
    val_dataset   = FakeNewsDataset(val_texts,   val_labels,   tokenizer, config["max_length"])
    test_dataset  = FakeNewsDataset(test_texts,  test_labels,  tokenizer, config["max_length"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # ---- Model ----
    print(f"[BERT Trainer] Initializing FakEdiT-BERT model...")
    model = FakEditBertClassifier(
        model_name=config["model_name"],
        dropout=config["dropout"],
    ).to(device)

    # ---- Loss function (weighted for imbalance) ----
    if class_weights:
        weight_tensor = torch.tensor(
            [class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
            dtype=torch.float,
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print(f"[BERT Trainer] Using weighted loss: {class_weights}")
    else:
        criterion = nn.CrossEntropyLoss()

    # ---- Optimizer ----
    # Separate weight decay for biases/LayerNorm vs weights
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped, lr=config["learning_rate"])

    # ---- LR Scheduler: linear warmup + linear decay ----
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # ---- Mixed precision scaler ----
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None

    # ---- Training loop ----
    best_val_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    save_dir = Path(config["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[BERT Trainer] Starting training for {config['epochs']} epochs")
    print(f"  Total steps: {total_steps}  |  Warmup steps: {warmup_steps}")
    print("=" * 60)

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with autocast():
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()

            if (step + 1) % max(1, len(train_loader) // 5) == 0:
                print(f"  Epoch {epoch}/{config['epochs']} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_metrics = evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - t0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_acc"].append(val_metrics["accuracy"])

        print(f"\nEpoch {epoch}/{config['epochs']}  [{elapsed:.0f}s]")
        print(f"  Train Loss : {avg_train_loss:.4f}")
        print(f"  Val Loss   : {val_metrics['loss']:.4f}")
        print(f"  Val Acc    : {val_metrics['accuracy']:.4f}")
        print(f"  Val F1     : {val_metrics['f1']:.4f}")

        # Save best model
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), save_dir / "best_model.pt")
            tokenizer.save_pretrained(save_dir)
            print(f"  ✅ New best model saved (Val F1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config['patience']}")

        if patience_counter >= config["patience"]:
            print(f"\n[BERT Trainer] Early stopping triggered at epoch {epoch}.")
            break

        print("-" * 60)

    # ---- Load best weights and evaluate on test set ----
    model.load_state_dict(torch.load(save_dir / "best_model.pt", map_location=device))
    print("\n[BERT Trainer] Evaluating best model on test set...")
    test_metrics = evaluate(model, test_loader, device, criterion)

    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(classification_report(
        test_metrics["labels"],
        test_metrics["predictions"],
        target_names=["REAL", "FAKE"],
    ))
    cm = confusion_matrix(test_metrics["labels"], test_metrics["predictions"])
    print(f"Confusion Matrix:\n{cm}")

    # Save config + metrics
    results = {
        "config": config,
        "history": history,
        "test_metrics": {k: float(v) if not isinstance(v, list) else v
                         for k, v in test_metrics.items()
                         if k not in ("predictions", "labels")},
        "confusion_matrix": cm.tolist(),
    }
    with open(save_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[BERT Trainer] Model and results saved to: {save_dir}")
    return model, tokenizer, results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage:
        python train_bert.py

    Expects preprocessed splits stored as CSVs (generated by preprocessing.py).
    Edit paths below to match your dataset location.
    """
    from data_loading import load_dataset, merge_title_text, get_label_mapping
    from preprocessing import clean_dataframe, split_dataset, compute_class_weights

    DATASET_PATH = "data/fake_news.csv"  # <-- Update this path

    df = load_dataset(DATASET_PATH)
    df = merge_title_text(df)
    label_mapping = get_label_mapping(df)
    df = clean_dataframe(df, label_mapping=label_mapping)

    train_df, val_df, test_df = split_dataset(df)

    class_weights = compute_class_weights(train_df["label_int"].tolist())

    config = {
        **DEFAULT_CONFIG,
        "epochs": 3,
        "batch_size": 16,
        "save_dir": "models/bert",
    }

    model, tokenizer, results = train_bert_model(
        train_df, val_df, test_df,
        config=config,
        class_weights=class_weights,
    )
    print(f"\nFinal test accuracy: {results['test_metrics']['accuracy']:.4f}")
