"""
app.py
======
SentinEL: Enterprise-Grade NLP Spam Shield & Message Auditor
Built with Streamlit, Scikit-Learn, NLTK, and Plotly.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------------------------------------------------------
# Page Configuration & Global Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="SentinEL | NLP Spam Shield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Styling
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* Status Badges */
    .badge-safe {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.15);
        border: 2px solid #10b981;
        color: #10b981;
        font-weight: 800;
        font-size: 1.15rem;
        padding: 10px 22px;
        border-radius: 9999px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.25);
    }
    .badge-spam {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(239, 68, 68, 0.15);
        border: 2px solid #ef4444;
        color: #ef4444;
        font-weight: 800;
        font-size: 1.15rem;
        padding: 10px 22px;
        border-radius: 9999px;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.25);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #f8fafc;
    }
    .metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }

    /* Architecture Pipeline Flow */
    .arch-step {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 12px;
        padding: 16px;
        position: relative;
    }
    .arch-step-num {
        background: #4f46e5;
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Exact Preprocessing Match (Synchronized with train.py)
# ---------------------------------------------------------
for res in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(res, quiet=True)
    except Exception:
        pass

stemmer = PorterStemmer()
try:
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
        "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
        "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
        "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own",
        "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
        "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to",
        "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where",
        "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself"
    }


def preprocess_text(text: str) -> str:
    """
    Synchronized with train.py:
    1. Lowercasing
    2. Entity normalization:
       - URLs -> httpaddr
       - Emails -> emailaddr
       - Currency/Monetary -> moneyprize
       - Phone numbers -> phonenumber
    3. Punctuation cleaning
    4. NLTK tokenization
    5. English stopword elimination
    6. PorterStemmer reduction
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()

    # 1. Normalize URLs
    text = re.sub(r"https?://\S+|www\.\S+", " httpaddr ", text)

    # 2. Normalize Emails
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\S+@\S+", " emailaddr ", text)

    # 3. Normalize Currency and Monetary amounts
    text = re.sub(
        r"[$£€¥₹]\s*\d+(?:[.,]\d+)*(?:\s*(?:k|m|million|billion|thousand))?|\b\d+(?:[.,]\d+)*\s*[$£€¥₹]|\b\d+(?:[.,]\d+)*\s*(?:dollars|pounds|euros|bucks|cash|prize)\b",
        " moneyprize ",
        text
    )

    # 4. Normalize Phone Numbers and shortcodes
    text = re.sub(
        r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{4,5}[-.\s]?\d{4,6}\b|\b1-\d{3}-\S+|\b09\d{8,10}\b|\b\d{5,7}\b",
        " phonenumber ",
        text
    )

    # 5. Remove remaining non-alphabetical characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # 6. Tokenization
    try:
        tokens = nltk.word_tokenize(text)
    except Exception:
        tokens = text.split()

    # 7. Stopword filtering & Porter Stemming
    cleaned_tokens = [
        stemmer.stem(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 1
    ]

    return " ".join(cleaned_tokens)


# ---------------------------------------------------------
# Safe Model Prediction Helper
# ---------------------------------------------------------
def predict_single_message(raw_text: str, model, vectorizer):
    """
    Applies exact preprocessing, vectorizes, checks predict_proba,
    and returns thresholded prediction (spam_proba >= 0.50 => SPAM).
    """
    cleaned = preprocess_text(raw_text)
    vec = vectorizer.transform([cleaned])

    spam_proba = 0.0
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(vec)[0]
            spam_idx = 1
            if hasattr(model, "classes_"):
                for idx, c in enumerate(model.classes_):
                    if str(c).strip().lower() in ["1", "spam"]:
                        spam_idx = idx
                        break
            spam_proba = float(probas[spam_idx]) if len(probas) > spam_idx else float(probas[-1])
        except Exception:
            raw_p = model.predict(vec)[0]
            spam_proba = 1.0 if str(raw_p).strip().lower() in ["1", "spam"] else 0.0
    elif hasattr(model, "decision_function"):
        try:
            df_val = float(model.decision_function(vec)[0])
            spam_proba = float(1.0 / (1.0 + np.exp(-df_val)))
        except Exception:
            raw_p = model.predict(vec)[0]
            spam_proba = 1.0 if str(raw_p).strip().lower() in ["1", "spam"] else 0.0
    else:
        raw_p = model.predict(vec)[0]
        spam_proba = 1.0 if str(raw_p).strip().lower() in ["1", "spam"] else 0.0

    is_spam = spam_proba >= 0.50
    label = "SPAM" if is_spam else "HAM"
    return is_spam, label, spam_proba, cleaned


# ---------------------------------------------------------
# Load Persisted Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    vec_path = os.path.join(base_dir, "vectorizer.pkl")
    metrics_path = os.path.join(base_dir, "metrics.pkl")

    if not (os.path.exists(model_path) and os.path.exists(vec_path) and os.path.exists(metrics_path)):
        return None, None, None

    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    with open(vec_path, "rb") as f:
        loaded_vectorizer = pickle.load(f)
    with open(metrics_path, "rb") as f:
        loaded_metrics = pickle.load(f)

    return loaded_model, loaded_vectorizer, loaded_metrics


model, vectorizer, metrics = load_artifacts()

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="hero-header">
    <div class="hero-title">🛡️ SentinEL: NLP Spam Shield</div>
    <div class="hero-subtitle">High-Precision Ensemble (MultinomialNB + Calibrated LinearSVC) with Entity-Preserving N-Gram TF-IDF</div>
</div>
""", unsafe_allow_html=True)

if model is None or vectorizer is None or metrics is None:
    st.error("⚠️ **Model artifacts (`model.pkl`, `vectorizer.pkl`, `metrics.pkl`) were not detected.**")
    st.info("💡 Run the training pipeline first by executing in your terminal:\n```bash\npython train.py\n```\nOnce finished, refresh this page to access full scanner features.")
    st.stop()


# ---------------------------------------------------------
# App Tabs
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🔍 Live Message Scanner",
    "📊 Batch CSV Auditor",
    "📈 Model Performance & Architecture"
])


# =========================================================
# TAB 1: LIVE MESSAGE SCANNER
# =========================================================
with tab1:
    st.markdown("### Live SMS, Email & Text Inspector")
    st.write("Analyze incoming messages in real-time with threshold-calibrated predictions and explainable keyword weighting.")

    PRESET_1 = "URGENT: Your Wells Fargo bank account has been compromised. Verify your identity immediately at http://secure-bank-login.com to prevent suspension and $50 penalty."
    PRESET_2 = "CONGRATULATIONS! You have won a $1,000,000 Walmart Gift Card! Call 1-800-SPAM-WIN now or claim your prize online at http://claim-prize.net before it expires!"
    PRESET_3 = "Hi team, let's catch up tomorrow at 10 AM to discuss the quarterly project roadmap and review the sprint slides. Let me know if you need to reschedule."

    if "scanner_input" not in st.session_state:
        st.session_state.scanner_input = PRESET_1

    def apply_preset(txt: str):
        st.session_state.scanner_input = txt

    st.markdown("**Quick Test Presets:**")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.button("🚨 Urgent Bank Alert", on_click=apply_preset, args=(PRESET_1,), use_container_width=True)
    with p_col2:
        st.button("🎁 Lottery Win", on_click=apply_preset, args=(PRESET_2,), use_container_width=True)
    with p_col3:
        st.button("☕ Normal Team Meeting", on_click=apply_preset, args=(PRESET_3,), use_container_width=True)

    user_input = st.text_area(
        "Enter or paste message content below:",
        key="scanner_input",
        height=130,
        placeholder="Type or paste SMS/email text here..."
    )

    analyze_btn = st.button("⚡ Analyze Message", type="primary", use_container_width=True)

    if analyze_btn or user_input:
        if not user_input.strip():
            st.warning("Please enter a message to analyze.")
        else:
            is_spam, pred_label, spam_prob, cleaned_msg = predict_single_message(user_input, model, vectorizer)

            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("#### Classification Verdict")
                if is_spam:
                    st.markdown("""
                    <div class="badge-spam">
                        <span>🚨 SPAM DETECTED</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Risk score exceeds the 50.0% threshold. Message flagged as malicious or promotional spam.")
                else:
                    st.markdown("""
                    <div class="badge-safe">
                        <span>✅ SAFE (HAM)</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("Risk score is below 50.0%. Message demonstrates legitimate communication patterns.")

                # Risk Score Gauge
                gauge_color = "#ef4444" if spam_prob >= 0.50 else "#10b981"
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(spam_prob * 100, 1),
                    number={"suffix": "%", "font": {"size": 36, "color": gauge_color}},
                    title={"text": "Spam Risk Probability", "font": {"size": 16, "color": "#cbd5e1"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                        "bar": {"color": gauge_color, "thickness": 0.28},
                        "bgcolor": "rgba(255,255,255,0.05)",
                        "borderwidth": 1,
                        "bordercolor": "#334155",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(16, 185, 129, 0.15)"},
                            {"range": [50, 100], "color": "rgba(239, 68, 68, 0.15)"}
                        ],
                        "threshold": {
                            "line": {"color": "#ef4444", "width": 3},
                            "thickness": 0.75,
                            "value": 50
                        }
                    }
                ))
                fig_gauge.update_layout(
                    height=260,
                    margin=dict(l=20, r=20, t=40, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"color": "#e2e8f0"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with res_col2:
                st.markdown("#### Explainable UI: Trigger Keywords")
                spam_weights = metrics.get("spam_weights", {}) if isinstance(metrics, dict) else {}

                # Check entity-level spam weights
                special_weights = {}
                for tag in ["httpaddr", "emailaddr", "moneyprize", "phonenumber"]:
                    w = spam_weights.get(tag, spam_weights.get(stemmer.stem(tag), 0.0))
                    if w > 0:
                        special_weights[tag] = float(w)

                # Collect positive word triggers
                raw_tokens = re.findall(r"\b[a-zA-Z]+\b", user_input)
                positive_trigger_words = {}
                positive_trigger_stems = {}

                for tok in raw_tokens:
                    tok_lower = tok.lower()
                    tok_stem = stemmer.stem(tok_lower)
                    w = spam_weights.get(tok_stem, spam_weights.get(tok_lower, None))
                    if w is not None and w > 0:
                        positive_trigger_words[tok_lower] = float(w)
                        positive_trigger_stems[tok_stem] = float(w)

                # Highlighting logic preserving HTML safety
                # 1. Protect special entity spans with placeholders
                placeholders = {}
                highlight_tag_tmpl = "<span style='background-color: #ffcccc; color: #990000; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{}</span>"

                def hold_token(val, entity_key):
                    ph = f"___ENTITY_PH_{len(placeholders)}_{entity_key}___"
                    if entity_key in special_weights:
                        placeholders[ph] = highlight_tag_tmpl.format(val)
                    else:
                        placeholders[ph] = val
                    return ph

                # Step-by-step entity extraction
                working_text = user_input
                working_text = re.sub(r"https?://\S+|www\.\S+", lambda m: hold_token(m.group(0), "httpaddr"), working_text)
                working_text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|\S+@\S+", lambda m: hold_token(m.group(0), "emailaddr"), working_text)
                working_text = re.sub(
                    r"[$£€¥₹]\s*\d+(?:[.,]\d+)*(?:\s*(?:k|m|million|billion|thousand))?|\b\d+(?:[.,]\d+)*\s*[$£€¥₹]|\b\d+(?:[.,]\d+)*\s*(?:dollars|pounds|euros|bucks|cash|prize)\b",
                    lambda m: hold_token(m.group(0), "moneyprize"),
                    working_text
                )
                working_text = re.sub(
                    r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b|\b\d{4,5}[-.\s]?\d{4,6}\b|\b1-\d{3}-\S+|\b09\d{8,10}\b|\b\d{5,7}\b",
                    lambda m: hold_token(m.group(0), "phonenumber"),
                    working_text
                )

                # Highlight remaining word tokens
                def highlight_token_match(m):
                    orig = m.group(1)
                    low = orig.lower()
                    stm = stemmer.stem(low)
                    if stm in positive_trigger_stems or low in positive_trigger_words:
                        return highlight_tag_tmpl.format(orig)
                    return orig

                working_text = re.sub(r"\b([a-zA-Z]+)\b", highlight_token_match, working_text)

                # Restore entity placeholders
                for ph, repl in placeholders.items():
                    working_text = working_text.replace(ph, repl)

                st.markdown("**Highlighted Trigger Text:**", unsafe_allow_html=True)
                st.markdown(
                    f"""<div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255,255,255,0.1); padding: 14px; border-radius: 8px; line-height: 1.7; font-size: 0.95rem;">
                    {working_text}
                    </div>""",
                    unsafe_allow_html=True
                )

                st.markdown("<br>", unsafe_allow_html=True)
                
                # Combine entity and word weights for the trigger ranking table
                all_triggers = {}
                for tag, w in special_weights.items():
                    all_triggers[f"[{tag.upper()}]"] = w
                for stem, w in positive_trigger_stems.items():
                    all_triggers[stem] = w

                if all_triggers:
                    st.markdown("**Top Positive Spam-Indicative Words & Features:**")
                    table_items = sorted(
                        [{"Feature / Token": k, "Spam Weight": v} for k, v in all_triggers.items()],
                        key=lambda x: x["Spam Weight"],
                        reverse=True
                    )
                    st_df = pd.DataFrame(table_items[:8])
                    st_df["Risk Contribution"] = st_df["Spam Weight"].apply(
                        lambda w: "🚨 High Trigger" if w >= 2.0 else "⚠️ Moderate Trigger"
                    )
                    st.dataframe(
                        st_df.style.format({"Spam Weight": "{:.2f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No strong positive spam trigger tokens were detected in the dictionary for this message.")


# =========================================================
# TAB 2: BATCH CSV AUDITOR
# =========================================================
with tab2:
    st.markdown("### Batch Audit Communication Datasets")
    st.write("Upload a CSV file containing message records to perform bulk classification, generate distribution analytics, and export verified results.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Accepts CSV with text or message columns.")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
        except Exception:
            uploaded_file.seek(0)
            batch_df = pd.read_csv(uploaded_file, encoding="latin-1", on_bad_lines="skip")

        st.success(f"Successfully loaded `{uploaded_file.name}` with {len(batch_df):,} records.")

        text_cols = [c for c in batch_df.columns if batch_df[c].dtype == object]
        default_idx = 0
        preferred = ["message", "text", "v2", "sms", "content", "body"]
        for i, col in enumerate(text_cols):
            if col.lower() in preferred:
                default_idx = i
                break

        selected_col = st.selectbox(
            "Select text column to evaluate:",
            text_cols,
            index=default_idx if text_cols else 0
        )

        if st.button("🚀 Run Batch Audit", type="primary"):
            with st.spinner("Processing text and generating predictions..."):
                raw_texts = batch_df[selected_col].astype(str)
                cleaned_batch = [preprocess_text(t) for t in raw_texts]
                batch_vec = vectorizer.transform(cleaned_batch)

                if hasattr(model, "predict_proba"):
                    try:
                        probas = model.predict_proba(batch_vec)
                        spam_idx = 1
                        if hasattr(model, "classes_"):
                            for idx, c in enumerate(model.classes_):
                                if str(c).strip().lower() in ["1", "spam"]:
                                    spam_idx = idx
                                    break
                        probabilities = probas[:, spam_idx] if probas.shape[1] > spam_idx else probas[:, -1]
                    except Exception:
                        raw_preds = model.predict(batch_vec)
                        probabilities = np.array([1.0 if str(p).strip().lower() in ["1", "spam"] else 0.0 for p in raw_preds])
                else:
                    raw_preds = model.predict(batch_vec)
                    probabilities = np.array([1.0 if str(p).strip().lower() in ["1", "spam"] else 0.0 for p in raw_preds])

                predictions = (probabilities >= 0.50).astype(int)
                batch_df["Predicted_Label"] = ["SPAM" if p == 1 else "HAM (Safe)" for p in predictions]
                batch_df["Spam_Probability"] = np.round(probabilities * 100, 2)
                batch_df["Risk_Level"] = [
                    "High" if prob >= 65 else ("Medium" if prob >= 35 else "Low")
                    for prob in batch_df["Spam_Probability"]
                ]

            total_audited = len(batch_df)
            spam_count = int((predictions == 1).sum())
            ham_count = total_audited - spam_count
            spam_pct = (spam_count / total_audited) * 100 if total_audited > 0 else 0
            ham_pct = (ham_count / total_audited) * 100 if total_audited > 0 else 0

            st.markdown("---")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Messages Audited", f"{total_audited:,}")
            m2.metric("Spam Identified", f"{spam_count:,}", f"{spam_pct:.1f}%")
            m3.metric("Legitimate (Ham)", f"{ham_count:,}", f"{ham_pct:.1f}%")
            m4.metric("Avg Spam Probability", f"{batch_df['Spam_Probability'].mean():.1f}%")

            col_chart, col_preview = st.columns([1, 1.2])

            with col_chart:
                st.markdown("#### Message Breakdown")
                donut_fig = go.Figure(data=[go.Pie(
                    labels=["Safe (Ham)", "Spam"],
                    values=[ham_count, spam_count],
                    hole=0.6,
                    marker=dict(colors=["#10b981", "#ef4444"], line=dict(color="#0f172a", width=2)),
                    textinfo="label+percent",
                    hoverinfo="label+value+percent"
                )])
                donut_fig.update_layout(
                    height=320,
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(donut_fig, use_container_width=True)

            with col_preview:
                st.markdown("#### Risk Distribution Breakdown")
                risk_counts = batch_df["Risk_Level"].value_counts().reset_index()
                risk_counts.columns = ["Risk Level", "Count"]
                bar_fig = px.bar(
                    risk_counts,
                    x="Risk Level",
                    y="Count",
                    color="Risk Level",
                    color_discrete_map={"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"},
                    text_auto=True
                )
                bar_fig.update_layout(
                    height=320,
                    margin=dict(t=20, b=20, l=20, r=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e2e8f0"),
                    showlegend=False
                )
                st.plotly_chart(bar_fig, use_container_width=True)

            st.markdown("#### Audited Records Preview")
            st.dataframe(
                batch_df[[selected_col, "Predicted_Label", "Spam_Probability", "Risk_Level"]].head(100),
                use_container_width=True,
                hide_index=True
            )

            csv_export = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Full Audited CSV",
                data=csv_export,
                file_name="sentinel_audited_messages.csv",
                mime="text/csv",
                type="primary"
            )


# =========================================================
# TAB 3: MODEL PERFORMANCE & ARCHITECTURE
# =========================================================
with tab3:
    st.markdown("### Benchmarks & Architectural Pipeline")
    st.write("Validation metrics evaluated on a stratified 20% holdout test set with detailed confusion matrix and NLP pipeline structure.")

    def safe_float(val, default=0.0):
        try:
            if val is None:
                return default
            return float(val)
        except (ValueError, TypeError):
            return default

    m_dict = metrics if isinstance(metrics, dict) else {}
    acc = safe_float(m_dict.get("accuracy", 0.0)) * 100
    prec = safe_float(m_dict.get("precision", 0.0)) * 100
    rec = safe_float(m_dict.get("recall", 0.0)) * 100
    f1 = safe_float(m_dict.get("f1", 0.0)) * 100

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{acc:.2f}%</div>
            <div class="metric-lbl">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{prec:.2f}%</div>
            <div class="metric-lbl">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{rec:.2f}%</div>
            <div class="metric-lbl">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{f1:.2f}%</div>
            <div class="metric-lbl">F1-Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_comp = st.columns([1, 1.2])

    with col_cm:
        st.markdown("#### Confusion Matrix (Stratified Test Split)")
        cm_raw = m_dict.get("confusion_matrix", [[0, 0], [0, 0]])
        try:
            cm_data = np.array(cm_raw)
            if cm_data.shape != (2, 2):
                cm_data = np.array([[0, 0], [0, 0]])
        except Exception:
            cm_data = np.array([[0, 0], [0, 0]])

        labels = [["True Negative (Ham)", "False Positive (Spam)"],
                  ["False Negative (Ham)", "True Positive (Spam)"]]
        
        text_matrix = [
            [f"{labels[i][j]}<br><b>{cm_data[i][j]}</b>" for j in range(2)]
            for i in range(2)
        ]

        cm_fig = go.Figure(data=go.Heatmap(
            z=cm_data,
            x=["Predicted Ham", "Predicted Spam"],
            y=["Actual Ham", "Actual Spam"],
            text=text_matrix,
            texttemplate="%{text}",
            colorscale=[[0, "#1e293b"], [0.5, "#4f46e5"], [1.0, "#38bdf8"]],
            showscale=False
        ))
        cm_fig.update_layout(
            height=340,
            margin=dict(t=30, b=30, l=40, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f8fafc", size=13)
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    with col_comp:
        st.markdown("#### Model Architecture Comparison")
        comp_dict = m_dict.get("model_comparison", {})
        if isinstance(comp_dict, dict) and comp_dict:
            comp_rows = []
            for m_name, vals in comp_dict.items():
                if isinstance(vals, dict):
                    comp_rows.append({
                        "Model": m_name,
                        "Accuracy": f"{safe_float(vals.get('Accuracy', 0.0)) * 100:.2f}%",
                        "Precision": f"{safe_float(vals.get('Precision', 0.0)) * 100:.2f}%",
                        "Recall": f"{safe_float(vals.get('Recall', 0.0)) * 100:.2f}%",
                        "F1-Score": f"{safe_float(vals.get('F1-Score', 0.0)) * 100:.2f}%"
                    })
            if comp_rows:
                st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)
                st.caption("ℹ️ The Soft Voting Ensemble combines Multinomial Naive Bayes with Calibrated LinearSVC.")
        else:
            st.info("Comparison benchmarks unavailable.")

        st.markdown("#### Class-Level Classification Breakdown")
        clf_rep = m_dict.get("classification_report", {})
        if isinstance(clf_rep, dict) and clf_rep:
            ham_rep = clf_rep.get("Ham") or clf_rep.get("0") or {}
            spam_rep = clf_rep.get("Spam") or clf_rep.get("1") or {}
            cr_data = [
                {
                    "Class": "Ham (Legitimate)",
                    "Precision": f"{safe_float(ham_rep.get('precision', 0.0))*100:.1f}%",
                    "Recall": f"{safe_float(ham_rep.get('recall', 0.0))*100:.1f}%",
                    "F1": f"{safe_float(ham_rep.get('f1-score', 0.0))*100:.1f}%",
                    "Support": ham_rep.get("support", 0)
                },
                {
                    "Class": "Spam (Malicious)",
                    "Precision": f"{safe_float(spam_rep.get('precision', 0.0))*100:.1f}%",
                    "Recall": f"{safe_float(spam_rep.get('recall', 0.0))*100:.1f}%",
                    "F1": f"{safe_float(spam_rep.get('f1-score', 0.0))*100:.1f}%",
                    "Support": spam_rep.get("support", 0)
                }
            ]
            st.dataframe(pd.DataFrame(cr_data), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Visual NLP Pipeline Diagram
    st.markdown("#### End-to-End NLP Pipeline Architecture")
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 12px;">
        <div class="arch-step">
            <span class="arch-step-num">STAGE 01</span>
            <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Raw Message Input</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">Receives raw text string from user input or CSV batch file.</div>
        </div>
        <div class="arch-step">
            <span class="arch-step-num">STAGE 02</span>
            <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Entity Normalization</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">Maps URLs to httpaddr, emails to emailaddr, money to moneyprize, and phones to phonenumber.</div>
        </div>
        <div class="arch-step">
            <span class="arch-step-num">STAGE 03</span>
            <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Porter Stemmer & Stopwords</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">Filters English stopwords and reduces morphological variants to root stems.</div>
        </div>
        <div class="arch-step">
            <span class="arch-step-num">STAGE 04</span>
            <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">N-Gram TF-IDF Vectorizer</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">Extracts unigram & bigram features capped at top 4,000 weighted sub-linear frequencies.</div>
        </div>
        <div class="arch-step">
            <span class="arch-step-num">STAGE 05</span>
            <div style="font-weight: 700; color: #f8fafc; font-size: 1rem;">Dual Soft Ensemble</div>
            <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 6px;">Soft Voting fusion combining MultinomialNB and CalibratedClassifierCV(LinearSVC).</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
