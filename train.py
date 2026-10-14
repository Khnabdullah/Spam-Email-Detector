"""
train.py
========
Production-grade training script for SMS & Email Spam Classification.
- Auto-detects and loads spam.csv handling various encodings and schemas
  (handles Subject + Message merging, v1/v2 legacy schemas, and drops unnamed columns).
- Applies advanced NLTK preprocessing with entity normalization:
  URLs -> httpaddr, Emails -> emailaddr, Currency -> moneyprize, Phone -> phonenumber.
- Extracts features with TfidfVectorizer(ngram_range=(1, 2), max_features=4000).
- Trains a Soft Voting Ensemble of MultinomialNB and CalibratedClassifierCV(LinearSVC)
  on an 80/20 stratified train/test split.
- Exports model.pkl, vectorizer.pkl, and metrics.pkl for Streamlit serving.
"""

import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------------
# NLTK Setup
# ---------------------------------------------------------
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception as exc:
        print(f"[!] Warning downloading NLTK resource '{resource}': {exc}")

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
    Cleans, lowercases, normalizes entities into dedicated tokens:
      - URLs -> httpaddr
      - Emails -> emailaddr
      - Currency/Monetary -> moneyprize
      - Phone numbers -> phonenumber
    Applies punctuation cleanup, NLTK stopword removal, and Porter stemming.
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
# Dynamic Dataset Ingestion & Schema Standardization
# ---------------------------------------------------------
def load_and_standardize_dataset(filepath: str = "spam.csv") -> pd.DataFrame:
    """
    Inspects and loads CSV dynamically detecting encodings:
    latin-1, utf-8, cp1252, iso-8859-1 with on_bad_lines='skip'.
    Handles legacy SMS schemas (v1/v2, discarding unnamed columns),
    email schemas (Subject + Message, or text/label),
    and standardizes target labels into binary 0 (Ham) and 1 (Spam).
    """
    if not os.path.exists(filepath):
        alt_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath),
            os.path.join(os.getcwd(), filepath)
        ]
        found = False
        for p in alt_paths:
            if os.path.exists(p):
                filepath = p
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Could not locate '{filepath}' in current directory or project path.")

    encodings_to_try = ["latin-1", "utf-8", "cp1252", "iso-8859-1", "utf-8-sig"]
    df = None
    used_encoding = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, encoding=enc, on_bad_lines="skip")
            used_encoding = enc
            print(f"[+] Successfully loaded '{filepath}' using encoding: {enc}")
            break
        except (UnicodeDecodeError, pd.errors.ParserError, Exception):
            continue

    if df is None:
        raise ValueError(f"Failed to read '{filepath}' with candidate encodings: {encodings_to_try}")

    # Drop unnamed columns created by trailing commas (e.g., 'Unnamed: 2', 'Unnamed: 3')
    unnamed_cols = [c for c in df.columns if str(c).strip().lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Clean whitespace in column names
    df.columns = [str(c).strip() for c in df.columns]
    col_map_lower = {str(c).lower(): c for c in df.columns}

    clean_df = pd.DataFrame()

    # Case A: Email schema with both Subject and Message columns
    if "subject" in col_map_lower and "message" in col_map_lower:
        subj_col = col_map_lower["subject"]
        msg_col = col_map_lower["message"]
        print(f"[i] Merging email columns '{subj_col}' and '{msg_col}' into raw_text...")
        clean_df["raw_text"] = df[subj_col].fillna("").astype(str) + " " + df[msg_col].fillna("").astype(str)

    # Case B: Standard / Legacy SMS schema (v1/v2, label/text, Category/Message, etc.)
    else:
        text_candidates = ["v2", "message", "text", "sms", "content", "body", "email", "sms_text"]
        detected_text_col = None
        for cand in text_candidates:
            if cand in col_map_lower:
                detected_text_col = col_map_lower[cand]
                break

        if detected_text_col is None:
            # Fallback to the first string/object column with high cardinality / text
            for col in df.columns:
                if df[col].dtype == object and df[col].nunique() > 10:
                    detected_text_col = col
                    break

        if detected_text_col is None:
            cols = list(df.columns)
            detected_text_col = cols[1] if len(cols) > 1 else cols[0]

        print(f"[i] Using text column: '{detected_text_col}'")
        clean_df["raw_text"] = df[detected_text_col].fillna("").astype(str)

    # Detect Label column
    label_candidates = ["v1", "spam/ham", "category", "label", "target", "class", "spam", "type"]
    detected_label_col = None
    for cand in label_candidates:
        if cand in col_map_lower:
            detected_label_col = col_map_lower[cand]
            break

    if detected_label_col is None:
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 5:
                detected_label_col = col
                break

    if detected_label_col is None:
        cols = list(df.columns)
        detected_label_col = cols[0]

    print(f"[i] Using label column: '{detected_label_col}'")

    raw_labels = df[detected_label_col].astype(str).str.strip().str.lower()

    def parse_label(val):
        if val in ["spam", "1", "1.0", "true", "yes", "positive"]:
            return 1
        return 0

    clean_df["label"] = raw_labels.apply(parse_label)
    clean_df = clean_df.dropna(subset=["raw_text"])
    clean_df = clean_df[clean_df["raw_text"].str.strip() != ""]

    print(f"[i] Dataset loaded: {len(clean_df)} records (Spam: {clean_df['label'].sum()}, Ham: {len(clean_df) - clean_df['label'].sum()})")
    return clean_df


# ---------------------------------------------------------
# Main Training Routine
# ---------------------------------------------------------
def train_pipeline():
    print("=" * 60)
    print("      SENTINEL NLP SPAM CLASSIFIER - TRAINING PIPELINE      ")
    print("=" * 60)

    # 1. Load data
    df = load_and_standardize_dataset("spam.csv")

    # 2. Text Preprocessing
    print("[*] Running NLTK preprocessing (entity normalization, stopwords, PorterStemmer)...")
    df["cleaned_text"] = df["raw_text"].apply(preprocess_text)

    X_texts = df["cleaned_text"].values
    y = df["label"].values

    # 3. Stratified 80/20 Split
    print("[*] Splitting dataset: 80% Train, 20% Test (Stratified)...")
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        X_texts, y, test_size=0.20, random_state=42, stratify=y
    )

    # 4. Feature Extraction: TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
    print("[*] Fitting TfidfVectorizer (ngram_range=(1, 2), max_features=4000)...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=4000)
    X_train_vec = vectorizer.fit_transform(X_train_texts)
    X_test_vec = vectorizer.transform(X_test_texts)

    # 5. Model Architecture & Ensemble
    print("[*] Initializing MultinomialNB and CalibratedClassifierCV(LinearSVC)...")
    nb_clf = MultinomialNB(alpha=0.1)
    
    base_svc = LinearSVC(random_state=42, max_iter=3000, dual="auto")
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=5)

    # Train individual models
    nb_clf.fit(X_train_vec, y_train)
    calibrated_svc.fit(X_train_vec, y_train)

    # Soft Voting Ensemble
    print("[*] Assembling Soft Voting Ensemble...")
    ensemble_clf = VotingClassifier(
        estimators=[("nb", nb_clf), ("svc", calibrated_svc)],
        voting="soft",
        weights=[1.0, 1.2]
    )
    ensemble_clf.fit(X_train_vec, y_train)

    # 6. Evaluation on Test Split
    print("\n[*] Evaluating Ensemble Model on Unseen 20% Test Set...")
    y_pred = ensemble_clf.predict(X_test_vec)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    clf_report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"], output_dict=True)

    nb_pred = nb_clf.predict(X_test_vec)
    svc_pred = calibrated_svc.predict(X_test_vec)

    comparison_data = {
        "Ensemble (Voting)": {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        },
        "MultinomialNB": {
            "Accuracy": float(accuracy_score(y_test, nb_pred)),
            "Precision": float(precision_score(y_test, nb_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, nb_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, nb_pred, zero_division=0))
        },
        "Calibrated LinearSVC": {
            "Accuracy": float(accuracy_score(y_test, svc_pred)),
            "Precision": float(precision_score(y_test, svc_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, svc_pred, zero_division=0)),
            "F1-Score": float(f1_score(y_test, svc_pred, zero_division=0))
        }
    }

    feature_names = vectorizer.get_feature_names_out()
    log_prob_diff = nb_clf.feature_log_prob_[1] - nb_clf.feature_log_prob_[0]
    spam_weights = {
        feature_names[i]: float(log_prob_diff[i])
        for i in range(len(feature_names))
    }

    metrics_payload = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "model_comparison": comparison_data,
        "test_size": len(y_test),
        "train_size": len(y_train),
        "total_samples": len(df),
        "spam_samples": int(df["label"].sum()),
        "ham_samples": int(len(df) - df["label"].sum()),
        "spam_weights": spam_weights,
        "feature_count": len(feature_names)
    }

    print("-" * 60)
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {prec * 100:.2f}%")
    print(f"Recall    : {rec * 100:.2f}%")
    print(f"F1-Score  : {f1 * 100:.2f}%")
    print(f"Confusion Matrix:\n{np.array(cm)}")
    print("-" * 60)

    artifacts = {
        "model.pkl": ensemble_clf,
        "vectorizer.pkl": vectorizer,
        "metrics.pkl": metrics_payload
    }

    for fname, obj in artifacts.items():
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
        with open(out_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"[OK] Saved artifact: {out_path}")

    print("\n[SUCCESS] Pipeline training and artifact generation completed successfully!")


if __name__ == "__main__":
    train_pipeline()