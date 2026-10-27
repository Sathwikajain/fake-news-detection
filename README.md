# 🔍 Fake News Detection System
### FakEdiT-BERT + Hybrid Ensemble  

A production-ready fake news detection system combining a **FakEdiT-inspired BERT classifier** with a traditional ML ensemble for high-accuracy, real-world news verification.

---

## Architecture Overview

```
Input (title + text)
        │
        ▼
  Text Preprocessing
  (cleaning, merging)
        │
   ┌────┴────────────────────────────────┐
   │                                     │
   ▼                                     ▼
FakEdiT-BERT                    TF-IDF Features
(bert-base-uncased +            ┌──────┴──────┐
 dual-attention layer)          │             │
   │                       LogReg      RandomForest
   │ P(REAL), P(FAKE)           │             │
   └──────────┬─────────────────┘─────────────┘
              │
        Soft Voting
       (weighted avg)
              │
              ▼
    Final Verdict + Confidence
    "Fake News ❌" / "Real News ✅"
```

### FakEdiT Architecture (Fake News Detection Transformer)
The FakEdiT-BERT model enhances standard BERT classification with a **dual-attention refinement block**:
1. **BERT encoder** extracts rich contextual embeddings from the full article.
2. A **learned edit-query vector** attends over the full token sequence via multi-head attention — simulating an "editing pass" that aligns evidence with the [CLS] token.
3. **Residual + LayerNorm** stabilize training.
4. CLS token and the refined edit output are concatenated and fed to the classification head.

---

## Project Structure

```
fake_news_detection/
├── data_loading.py      # CSV/XLSX loading, validation, label mapping
├── preprocessing.py     # Text cleaning, splitting, class weight computation
├── train_bert.py        # FakEdiT-BERT fine-tuning (PyTorch + Transformers)
├── train_ensemble.py    # Hybrid ensemble: BERT + LogReg + RandomForest
├── predict.py           # Inference module (single + batch prediction)
├── app.py               # Streamlit web application
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Prepare your dataset

Your CSV/XLSX file must have these columns:

| title | text | label |
|-------|------|-------|
| "Alien invasion confirmed" | "Scientists say..." | FAKE |
| "Fed raises interest rates" | "The Federal Reserve..." | REAL |

Labels can be `FAKE/REAL`, `fake/real`, or `0/1`.

**Recommended public datasets:**
- [LIAR Dataset](https://huggingface.co/datasets/liar) — 12.8K labeled statements
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) — PolitiFact + GossipCop
- [WELFake](https://zenodo.org/record/4561253) — 72K articles (recommended)

### 3. Train the BERT model

```bash
# Edit DATASET_PATH in train_bert.py, then:
python train_bert.py
```

This will:
- Fine-tune `bert-base-uncased` with FakEdiT architecture
- Run for up to 5 epochs with early stopping
- Save best model to `models/bert/`

**Expected training time:**
- GPU (T4/V100): ~15–30 min for 5 epochs on 40K samples
- CPU: ~2–4 hours (use fewer epochs; set `"epochs": 2`)

### 4. Train the ensemble

```bash
python train_ensemble.py
```

Trains TF-IDF + LogisticRegression and TF-IDF + RandomForest on top of BERT.
Saves to `models/ensemble/`.

### 5. Launch the web app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Configuration

Edit `DEFAULT_CONFIG` in `train_bert.py`:

```python
DEFAULT_CONFIG = {
    "model_name":    "bert-base-uncased",
    "max_length":    256,      # Increase to 512 for better coverage (slower)
    "batch_size":    16,       # Reduce to 8 if OOM on GPU
    "epochs":        5,
    "learning_rate": 2e-5,
    "warmup_ratio":  0.1,
    "dropout":       0.3,
    "weight_decay":  0.01,
    "patience":      2,        # Early stopping patience
    "save_dir":      "models/bert",
}
```

### Ensemble weights (in `train_ensemble.py`):
```python
bert_weight = 0.60   # BERT contributes most (contextual understanding)
lr_weight   = 0.25   # LogReg catches lexical patterns
rf_weight   = 0.15   # RandomForest adds non-linear lexical patterns
```

---

## Expected Performance

On the WELFake dataset (72K articles):

| Model | Accuracy | F1 | Precision | Recall |
|-------|----------|-----|-----------|--------|
| LogReg (TF-IDF) | ~93% | 0.93 | 0.93 | 0.93 |
| Random Forest | ~92% | 0.92 | 0.92 | 0.92 |
| FakEdiT-BERT | ~96% | 0.96 | 0.96 | 0.96 |
| **Ensemble** | **~97%** | **0.97** | **0.97** | **0.97** |

> Results vary by dataset. LIAR dataset (multi-class) typically scores lower (~65–70%).

---

## Programmatic Usage

```python
from predict import FakeNewsPredictor

# Load the trained ensemble
predictor = FakeNewsPredictor.load(
    bert_dir="models/bert",
    ensemble_dir="models/ensemble",
)

# Predict a single article
result = predictor.predict_single(
    title="Shocking discovery by scientists",
    text="Researchers have found a cure for all diseases using tap water.",
)
print(result["display"])
# → "Fake News ❌  (confidence: 91.3%)"
print(result["proba"])
# → {'REAL': 0.087, 'FAKE': 0.913}

# Batch prediction
articles = [
    {"title": "Congress passes budget", "text": "The bill passed 300-120..."},
    {"title": "Aliens confirmed", "text": "Secret documents leaked..."},
]
results = predictor.predict_batch(articles)
for r in results:
    print(r["display"])
```

---

## Class Imbalance Handling

The system handles class imbalance through:
1. **Weighted CrossEntropyLoss** — computed from training label distribution
2. **`class_weight="balanced"`** in LogReg and RandomForest
3. **Stratified train/val/test splits** — maintains label distribution

---

## CPU Deployment

The system is designed to run efficiently on CPU:
- BERT inference at ~100ms/sample on modern CPU
- Classical ML inference at <1ms/sample
- Total ensemble latency: ~100–200ms per article

For faster CPU inference, consider:
```bash
pip install optimum
# Then use ONNX-exported model via Optimum
```

---

## Troubleshooting

**`RuntimeError: CUDA out of memory`**  
→ Reduce `batch_size` to 8 or 4 in `DEFAULT_CONFIG`

**`OSError: Can't load tokenizer`**  
→ The model hasn't been trained yet. Run `train_bert.py` first, or ensure internet access for initial Hugging Face download.

**Low accuracy on your dataset**  
→ Try increasing `epochs` to 5–8, or increasing `max_length` to 512 for longer articles.

**Streamlit app shows Demo Mode**  
→ Models haven't been trained yet. Run both training scripts.

---

## License

MIT License — free to use, modify, and distribute.
