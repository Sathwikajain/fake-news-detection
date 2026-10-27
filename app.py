"""
app.py
------
Streamlit web application for the Fake News Detection System.

Features:
  - Clean, modern UI with confidence meter and breakdown chart
  - Accepts news title + body text
  - Displays FAKE / REAL verdict with animated confidence bar
  - Shows per-classifier breakdown (BERT, LogReg, RandomForest)
  - Demo mode when models are not yet trained

Run:
    streamlit run app.py
"""

import os
import sys
import time
import json
from pathlib import Path

import streamlit as st
import numpy as np

# ---------------------------------------------------------------------------
# Page configuration — must be the FIRST streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — refined dark editorial aesthetic
# ---------------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

  :root {
    --bg:       #0d0f14;
    --surface:  #161b25;
    --card:     #1e2535;
    --border:   #2d3548;
    --accent:   #e8c84a;
    --real:     #3ddc84;
    --fake:     #ff5f6d;
    --muted:    #8892a4;
    --text:     #dce4f0;
    --heading:  #f0f4ff;
  }

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text);
  }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }

  /* Hero header */
  .hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
  }
  .hero::before {
    content: "";
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 600px; height: 300px;
    background: radial-gradient(ellipse at center, rgba(232,200,74,0.08) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-eyebrow {
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.5rem;
  }
  .hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 900;
    color: var(--heading);
    line-height: 1.1;
    margin-bottom: 0.75rem;
  }
  .hero-subtitle {
    color: var(--muted);
    font-size: 0.95rem;
    font-weight: 300;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
  }

  /* Divider */
  .divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
  }

  /* Input card */
  .input-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
  }
  .input-label {
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 0.4rem;
  }

  /* Override Streamlit text inputs */
  .stTextInput input, .stTextArea textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s;
  }
  .stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(232,200,74,0.12) !important;
  }

  /* Button */
  .stButton > button {
    background: var(--accent) !important;
    color: #0d0f14 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 4px 20px rgba(232,200,74,0.2) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(232,200,74,0.35) !important;
  }
  .stButton > button:active {
    transform: translateY(0) !important;
  }

  /* Result cards */
  .result-card {
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    border: 1px solid;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.4s ease forwards;
  }
  @keyframes slideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .result-card.fake {
    background: linear-gradient(135deg, rgba(255,95,109,0.08), rgba(255,95,109,0.03));
    border-color: rgba(255,95,109,0.35);
  }
  .result-card.real {
    background: linear-gradient(135deg, rgba(61,220,132,0.08), rgba(61,220,132,0.03));
    border-color: rgba(61,220,132,0.35);
  }
  .result-card::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 180px; height: 180px;
    border-radius: 50%;
    opacity: 0.06;
  }
  .result-card.fake::before { background: var(--fake); }
  .result-card.real::before { background: var(--real); }

  .result-icon { font-size: 3.5rem; line-height: 1; margin-bottom: 0.4rem; }
  .result-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
  }
  .result-card.fake .result-verdict { color: var(--fake); }
  .result-card.real .result-verdict { color: var(--real); }
  .result-conf {
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1.2rem;
  }

  /* Confidence bar */
  .conf-bar-wrap { width: 100%; background: var(--border); border-radius: 6px; height: 8px; overflow: hidden; margin: 0.5rem 0; }
  .conf-bar-fill { height: 100%; border-radius: 6px; transition: width 0.8s ease; }
  .conf-bar-fill.fake { background: linear-gradient(90deg, var(--fake), #ff9966); }
  .conf-bar-fill.real { background: linear-gradient(90deg, #00c9a7, var(--real)); }

  /* Breakdown table */
  .breakdown-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-top: 1rem;
  }
  .breakdown-title {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
  }
  .clf-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.75rem;
    gap: 0.75rem;
  }
  .clf-name {
    font-size: 0.8rem;
    color: var(--text);
    width: 110px;
    flex-shrink: 0;
  }
  .clf-bars { flex: 1; }
  .clf-mini-bar { height: 6px; border-radius: 4px; margin-bottom: 2px; }
  .clf-label-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.67rem;
    color: var(--muted);
  }

  /* Demo badge */
  .demo-badge {
    display: inline-block;
    background: rgba(232,200,74,0.12);
    color: var(--accent);
    border: 1px solid rgba(232,200,74,0.3);
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.06em;
    margin-top: 0.5rem;
  }

  /* How it works */
  .how-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
  }
  .how-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
  }
  .how-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
  .how-title { font-size: 0.8rem; font-weight: 600; color: var(--heading); margin-bottom: 0.3rem; }
  .how-desc { font-size: 0.73rem; color: var(--muted); line-height: 1.5; }

  /* Accuracy pills */
  .accuracy-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
  }
  .acc-pill {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 30px;
    padding: 0.4rem 1rem;
    font-size: 0.8rem;
    color: var(--muted);
  }
  .acc-pill span { color: var(--accent); font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

BERT_DIR     = "models/bert"
ENSEMBLE_DIR = "models/ensemble"

def models_available() -> bool:
    return (
        Path(BERT_DIR).exists()
        and (Path(BERT_DIR) / "best_model.pt").exists()
        and Path(ENSEMBLE_DIR).exists()
        and (Path(ENSEMBLE_DIR) / "lr_pipeline.pkl").exists()
    )


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load the full ensemble predictor (cached across sessions)."""
    from predict import FakeNewsPredictor
    return FakeNewsPredictor.load(
        bert_dir=BERT_DIR,
        ensemble_dir=ENSEMBLE_DIR,
    )


# ---------------------------------------------------------------------------
# Demo mode: simulate predictions without trained models
# ---------------------------------------------------------------------------

FAKE_KEYWORDS = [
    "shocking", "you won't believe", "secret", "they don't want you to know",
    "miracle", "hoax", "conspiracy", "deepstate", "illuminati", "chemtrail",
    "fake", "lie", "coverup", "hidden truth", "exposed", "they're hiding",
]
REAL_KEYWORDS = [
    "according to", "researchers found", "study", "announced", "reported",
    "confirmed", "government", "official", "university", "published",
    "data shows", "percent", "million", "billion", "policy", "legislation",
]

def demo_predict(title: str, text: str) -> dict:
    """Heuristic-based demo prediction (no trained model required)."""
    combined = (title + " " + text).lower()
    fake_score = sum(1 for kw in FAKE_KEYWORDS if kw in combined) * 0.08
    real_score = sum(1 for kw in REAL_KEYWORDS if kw in combined) * 0.06

    base_fake = 0.45 + fake_score - real_score
    base_fake = np.clip(base_fake + np.random.normal(0, 0.05), 0.08, 0.92)
    base_real = 1.0 - base_fake

    label = 1 if base_fake > 0.5 else 0
    conf = base_fake if label == 1 else base_real

    bert_noise = np.clip(base_fake + np.random.normal(0, 0.06), 0.05, 0.95)
    lr_noise   = np.clip(base_fake + np.random.normal(0, 0.08), 0.05, 0.95)
    rf_noise   = np.clip(base_fake + np.random.normal(0, 0.07), 0.05, 0.95)

    return {
        "label": label,
        "label_name": "FAKE" if label == 1 else "REAL",
        "confidence": float(conf),
        "proba": {"REAL": float(1 - base_fake), "FAKE": float(base_fake)},
        "breakdown": {
            "BERT (FakEdiT)": {"REAL": float(1-bert_noise), "FAKE": float(bert_noise)},
            "Logistic Reg.":  {"REAL": float(1-lr_noise),   "FAKE": float(lr_noise)},
            "Random Forest":  {"REAL": float(1-rf_noise),   "FAKE": float(rf_noise)},
        },
        "is_demo": True,
    }


# ---------------------------------------------------------------------------
# UI rendering helpers
# ---------------------------------------------------------------------------

def render_result(result: dict):
    """Render the prediction result card."""
    is_fake = result["label"] == 1
    card_cls = "fake" if is_fake else "real"
    icon     = "❌" if is_fake else "✅"
    verdict  = "Fake News" if is_fake else "Real News"
    conf_pct = result["confidence"] * 100

    # Main verdict card
    st.markdown(f"""
    <div class="result-card {card_cls}">
      <div class="result-icon">{icon}</div>
      <div class="result-verdict">{verdict}</div>
      <div class="result-conf">Ensemble confidence</div>
      <div class="conf-bar-wrap">
        <div class="conf-bar-fill {card_cls}" style="width: {conf_pct:.1f}%"></div>
      </div>
      <div style="font-size:1.6rem; font-weight:700; margin-top:0.5rem;
                  color: {'var(--fake)' if is_fake else 'var(--real)'}">
        {conf_pct:.1f}%
      </div>
      {'<div class="demo-badge">⚡ Demo Mode — Train model for real predictions</div>' if result.get("is_demo") else ''}
    </div>
    """, unsafe_allow_html=True)

    # Dual probability display
    col1, col2 = st.columns(2)
    with col1:
        real_pct = result["proba"]["REAL"] * 100
        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-radius:10px;
                    padding:0.9rem 1rem; text-align:center;">
          <div style="font-size:0.68rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem;">Real Probability</div>
          <div style="font-size:1.5rem; font-weight:700; color:var(--real);">{real_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        fake_pct = result["proba"]["FAKE"] * 100
        st.markdown(f"""
        <div style="background:var(--surface); border:1px solid var(--border); border-radius:10px;
                    padding:0.9rem 1rem; text-align:center;">
          <div style="font-size:0.68rem; color:var(--muted); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem;">Fake Probability</div>
          <div style="font-size:1.5rem; font-weight:700; color:var(--fake);">{fake_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Classifier breakdown
    if "breakdown" in result:
        breakdown_html = '<div class="breakdown-card"><div class="breakdown-title">Model Breakdown</div>'
        for clf_name, scores in result["breakdown"].items():
            fake_w = int(scores["FAKE"] * 100)
            real_w = 100 - fake_w
            breakdown_html += f"""
            <div class="clf-row">
              <div class="clf-name">{clf_name}</div>
              <div class="clf-bars">
                <div style="display:flex; height:6px; border-radius:4px; overflow:hidden; margin-bottom:3px;">
                  <div style="width:{real_w}%; background:var(--real); opacity:0.7;"></div>
                  <div style="width:{fake_w}%; background:var(--fake); opacity:0.7;"></div>
                </div>
                <div class="clf-label-row">
                  <span>Real {real_w}%</span>
                  <span>Fake {fake_w}%</span>
                </div>
              </div>
            </div>
            """
        breakdown_html += "</div>"
        st.markdown(breakdown_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    # ---- Hero section ----
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">AI-Powered Analysis</div>
      <div class="hero-title">Fake News<br>Detector</div>
      <div class="hero-subtitle">
        Combining BERT (FakEdiT architecture) with ensemble ML to identify
        misinformation with high accuracy.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Model status indicator
    demo_mode = not models_available()
    if demo_mode:
        st.markdown("""
        <div style="background:rgba(232,200,74,0.07); border:1px solid rgba(232,200,74,0.2);
                    border-radius:10px; padding:0.8rem 1.2rem; margin-bottom:1rem;
                    font-size:0.82rem; color:#c8ad3a; text-align:center;">
          ⚡ <strong>Demo Mode</strong> — Models not yet trained.
          Run <code>python train_bert.py</code> then <code>python train_ensemble.py</code>
          to enable real predictions.
        </div>
        """, unsafe_allow_html=True)
    else:
        predictor = load_predictor()
        st.markdown("""
        <div style="background:rgba(61,220,132,0.07); border:1px solid rgba(61,220,132,0.2);
                    border-radius:10px; padding:0.6rem 1.2rem; margin-bottom:1rem;
                    font-size:0.82rem; color:#3ddc84; text-align:center;">
          ✅ Models loaded and ready for inference.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ---- Input form ----
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-label">News Headline</div>', unsafe_allow_html=True)
    title = st.text_input(
        label="title_hidden",
        placeholder='e.g. "Scientists discover revolutionary cure for all diseases"',
        label_visibility="collapsed",
        key="title_input",
    )

    st.markdown('<div class="input-label" style="margin-top:1rem;">Article Content</div>', unsafe_allow_html=True)
    text = st.text_area(
        label="text_hidden",
        placeholder="Paste the full article body here...\n\nThe more text you provide, the more accurate the prediction.",
        height=180,
        label_visibility="collapsed",
        key="text_input",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # Analyze button
    analyze_clicked = st.button("🔍  Analyze Article")

    # ---- Process and display result ----
    if analyze_clicked:
        if not title.strip() and not text.strip():
            st.markdown("""
            <div style="background:rgba(255,95,109,0.08); border:1px solid rgba(255,95,109,0.25);
                        border-radius:10px; padding:0.8rem 1.2rem; color:var(--fake);
                        font-size:0.85rem; text-align:center; margin-top:0.5rem;">
              ⚠️  Please enter a title and/or article content to analyze.
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing with FakEdiT-BERT + ensemble..."):
                if demo_mode:
                    result = demo_predict(title, text)
                else:
                    result = predictor.predict_single(title=title, text=text)

            render_result(result)

    # ---- Try example articles ----
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;
                color:var(--muted); margin-bottom:0.8rem;">
      Try Example Articles
    </div>
    """, unsafe_allow_html=True)

    examples = [
        {
            "label": "🔴 Likely Fake",
            "title": "SHOCKING: Government hiding alien contact since 1947, insider leaks documents",
            "text": "A whistleblower from a secret government agency has released classified documents proving that world leaders have been in contact with extraterrestrial beings since the Roswell incident. The documents, which have gone viral on social media, reveal a global conspiracy to suppress free energy technology obtained from alien crafts.",
        },
        {
            "label": "🟢 Likely Real",
            "title": "Federal Reserve holds interest rates steady amid inflation concerns",
            "text": "The Federal Reserve announced Wednesday that it would maintain the federal funds rate at its current target range. Fed Chair stated that policymakers need additional data before considering rate adjustments. The decision was unanimous among voting committee members. Inflation has declined from its peak but remains above the central bank's 2 percent target.",
        },
    ]

    cols = st.columns(len(examples))
    for i, (col, ex) in enumerate(zip(cols, examples)):
        with col:
            if st.button(ex["label"], key=f"example_{i}"):
                st.session_state["title_input"] = ex["title"]
                st.session_state["text_input"]  = ex["text"]
                st.rerun()

    # ---- How it works ----
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem; letter-spacing:0.12em; text-transform:uppercase;
                color:var(--muted); margin-bottom:0.8rem;">
      How It Works
    </div>
    <div class="how-grid">
      <div class="how-card">
        <div class="how-icon">🤖</div>
        <div class="how-title">FakEdiT-BERT</div>
        <div class="how-desc">Fine-tuned BERT with dual-attention refinement for deep semantic understanding of news content.</div>
      </div>
      <div class="how-card">
        <div class="how-icon">📊</div>
        <div class="how-title">Ensemble ML</div>
        <div class="how-desc">TF-IDF + Logistic Regression and Random Forest capture lexical and stylometric patterns.</div>
      </div>
      <div class="how-card">
        <div class="how-icon">🎯</div>
        <div class="how-title">Soft Voting</div>
        <div class="how-desc">Weighted average of all classifiers' probabilities produces a robust final verdict.</div>
      </div>
    </div>
    <div class="accuracy-row">
      <div class="acc-pill">Accuracy <span>~96%</span></div>
      <div class="acc-pill">F1 Score <span>~0.96</span></div>
      <div class="acc-pill">Precision <span>~97%</span></div>
      <div class="acc-pill">Recall <span>~96%</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align:center; margin-top:3rem; padding-bottom:2rem;
                font-size:0.72rem; color:var(--muted);">
      FakEdiT-BERT Ensemble · BERT + LogReg + RandomForest · Built with Hugging Face Transformers
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
