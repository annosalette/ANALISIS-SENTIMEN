# ===========================================
# streamlit_sentiment_app.py (FINAL - KONSISTEN SKRIPSI)
# ===========================================
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
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
# Streamlit App
# -------------------------
def main():
    st.title("📊 Analisis Sentimen Pengguna Mobile Legends")

    file = st.file_uploader("Unggah dataset (.xlsx / .csv)", type=["xlsx", "csv"])
    if file is None:
        st.info("Silakan unggah dataset terlebih dahulu.")
        return

    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # ===== KOLOM TEKS =====
    if "full_text" not in df.columns:
        st.error("Kolom 'full_text' tidak ditemukan.")
        return

    # ===== KOLOM LABEL =====
    if "sentiment_label" not in df.columns:
        st.error("Kolom 'sentiment_label' tidak ditemukan. Dataset harus sudah dilabeli.")
        return

    df["stemmed_text"] = df["full_text"].astype(str).apply(preprocess_text)

    st.success(f"Dataset dimuat: {len(df)} data")
    st.dataframe(df[["full_text", "sentiment_label"]].head())

    # ===================================================
    # DISTRIBUSI SENTIMEN (PASTI SAMA)
    # ===================================================
    st.subheader("📈 Distribusi Sentimen")

    order = ["positif", "netral", "negatif"]
    counts = df["sentiment_label"].value_counts().reindex(order, fill_value=0)

    st.dataframe(counts.rename_axis("Sentimen").reset_index(name="Jumlah"))

    fig, ax = plt.subplots(figsize=(3,3), dpi=180)
    ax.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#5dade2", "#e74c3c"]
    )
    st.pyplot(fig)

    # ===================================================
    # MODEL NAÏVE BAYES
    # ===================================================
    X = df["stemmed_text"]
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1,2),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ("nb", MultinomialNB())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📊 Evaluasi Model")
    st.write("Akurasi:", round(accuracy_score(y_test, y_pred), 4))
    st.write("F1-Score:", round(f1_score(y_test, y_pred, average="weighted"), 4))

    cm = confusion_matrix(y_test, y_pred, labels=order)
    fig_cm, ax_cm = plt.subplots(figsize=(3,3), dpi=180)
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=order, yticklabels=order,
                cmap="Blues", cbar=False)
    st.pyplot(fig_cm)

    # ===================================================
    # WORDCLOUD
    # ===================================================
    st.subheader("☁️ Wordcloud per Sentimen")

    colors = {
        "positif": "#2ecc71",
        "netral": "#f1c40f",
        "negatif": "#e74c3c"
    }

    for sent in order:
        text = " ".join(df[df["sentiment_label"] == sent]["stemmed_text"])
        if text:
            wc = WordCloud(
                background_color="white",
                color_func=lambda *args, **kwargs: colors[sent]
            ).generate(text)
            st.write(sent.capitalize())
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

if __name__ == "__main__":
    main()
