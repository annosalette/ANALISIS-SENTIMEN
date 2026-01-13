# ===========================================
# streamlit_sentiment_app.py
# ===========================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random

# -------------------------
# Preprocessing Text
# -------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"@[\w_]+|#", " ", s)
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------
# Lexicon & Negation
# -------------------------
POSITIF = {
    "bagus","seru","keren","hebat","mantap","menang","gg","top",
    "lancar","suka","jago","menarik","terbaik","asik","puas","oke"
}
NEGATIF = {
    "buruk","jelek","lag","noob","toxic","kalah","lemot","kecewa",
    "marah","bete","ngehang","error","ampas","parah"
}
NETRAL = {
    "hero","map","tim","build","match","item","mode","skill",
    "game","player","update","event"
}
NEGASI = {"tidak", "bukan", "nggak", "ga", "gak", "tak", "belum"}

MULTIWORD = {
    "tim beban": "negatif",
    "server jelek": "negatif",
    "lemot banget": "negatif",
    "bagus banget": "positif"
}

def detect_multiword(text):
    for phrase, label in MULTIWORD.items():
        if phrase in text:
            return label
    return None

def get_lexicon_label(word):
    if word in POSITIF:
        return "positif"
    elif word in NEGATIF:
        return "negatif"
    elif word in NETRAL:
        return "netral"
    return "netral"

# -------------------------
# Hybrid Sentiment Labeling
# -------------------------
def hybrid_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"label": "netral", "score": 0.0}

    txt = preprocess_text(text)
    words = txt.split()

    mw = detect_multiword(txt)
    if mw:
        return {"label": mw, "score": 0.85}

    score = {"positif": 0, "negatif": 0, "netral": 0}

    for i, w in enumerate(words):
        label = get_lexicon_label(w)

        if i > 0 and words[i - 1] in NEGASI:
            if label == "positif":
                label = "negatif"
            elif label == "negatif":
                label = "positif"

        if label == "positif":
            score["positif"] += 2
        elif label == "negatif":
            score["negatif"] += 2
        else:
            score["netral"] += 0.5

    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.03:
        score["positif"] += abs(polarity) * 3
    elif polarity < -0.03:
        score["negatif"] += abs(polarity) * 3
    else:
        score["netral"] += 0.5

    final_label = max(score, key=score.get)
    total = sum(score.values())
    confidence = score[final_label] / total if total else 0

    return {"label": final_label, "score": round(confidence, 3)}

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.title("📊 Analisis Sentimen Pengguna Mobile Legends")

    file = st.file_uploader(
        "Unggah dataset (.xlsx / .csv)",
        type=["xlsx", "csv"]
    )

    if file is None:
        st.info("Silakan unggah dataset terlebih dahulu.")
        return

    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # ✅ FIX UTAMA DI SINI
    text_col = next(
        (c for c in [
            "full_text", "stemmed_text", "clean_text",
            "text", "komentar", "tweet"
        ] if c in df.columns),
        None
    )

    if text_col is None:
        st.error("Kolom teks tidak ditemukan (contoh: full_text).")
        return

    st.success(f"Dataset dimuat — {len(df)} data")
    st.dataframe(df[[text_col]].head())

    # Labeling
    df["stemmed_text"] = df[text_col].astype(str).apply(preprocess_text)
    result = df["stemmed_text"].apply(hybrid_sentiment)
    df["sentiment_label"] = result.apply(lambda x: x["label"])
    df["confidence_score"] = result.apply(lambda x: x["score"])

    # Distribusi
    st.subheader("📈 Distribusi Sentimen")
    order = ["positif", "netral", "negatif"]
    counts = df["sentiment_label"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(3, 3), dpi=160)
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#f1c40f", "#e74c3c"]
    )
    st.pyplot(fig)

    # Model Naive Bayes
    X = df["stemmed_text"]
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ("nb", MultinomialNB(class_prior=[1/3, 1/3, 1/3]))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📊 Evaluasi Model")
    st.write("Akurasi:", accuracy_score(y_test, y_pred))
    st.write("F1-score:", f1_score(y_test, y_pred, average="weighted"))

    # Wordcloud
    st.subheader("☁️ Wordcloud")
    palette = {
        "positif": ["#2ecc71"],
        "netral": ["#f1c40f"],
        "negatif": ["#e74c3c"]
    }

    for sent in order:
        text_data = " ".join(df[df["sentiment_label"] == sent]["stemmed_text"])
        if text_data:
            wc = WordCloud(
                background_color="white",
                color_func=lambda *args, **kwargs: random.choice(palette[sent])
            ).generate(text_data)

            st.write(sent.capitalize())
            st.image(wc.to_array())

    st.subheader("🔍 Contoh Data")
    st.dataframe(df[["stemmed_text", "sentiment_label", "confidence_score"]].head(15))


if __name__ == "__main__":
    main()
