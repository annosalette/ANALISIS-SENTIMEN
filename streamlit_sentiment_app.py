# =======================
# visualisasi.py
# =======================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def main():
    # =======================
    # Konfigurasi Halaman
    # =======================
    st.set_page_config(
        page_title="Visualisasi Sentimen Mobile Legends",
        page_icon="📊",
        layout="wide"
    )

    st.title("📊 Visualisasi Analisis Sentimen Mobile Legends")
    st.markdown("Menu ini menampilkan **distribusi sentimen, confusion matrix, dan wordcloud per sentimen**.")

    # =======================
    # Upload Dataset
    # =======================
    uploaded_file = st.file_uploader(
        "📂 Upload dataset hasil pelabelan (.xlsx)",
        type=["xlsx"]
    )

    if uploaded_file is None:
        st.info("Silakan upload dataset terlebih dahulu.")
        return

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    # =======================
    # Auto Deteksi Kolom
    # =======================
    text_candidates = ["stemmed_text", "clean_text", "full_text", "text", "comment"]
    label_candidates = ["sentiment_label", "sentiment", "label", "sentimen", "kelas"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        st.error("❌ Kolom teks atau label sentimen tidak ditemukan.")
        st.write("Kolom tersedia:", df.columns.tolist())
        return

    df = df.rename(columns={
        text_col: "stemmed_text",
        label_col: "sentiment_label"
    })

    df = df.dropna(subset=["stemmed_text", "sentiment_label"])

    # =======================
    # Informasi Dataset
    # =======================
    st.subheader("📌 Informasi Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Jumlah data:", len(df))
        st.write("Distribusi label:")
        st.dataframe(df["sentiment_label"].value_counts())

    with col2:
        fig, ax = plt.subplots()
        df["sentiment_label"].value_counts().plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=140,
            ax=ax
        )
        ax.set_ylabel("")
        ax.set_title("Distribusi Sentimen")
        st.pyplot(fig)

    # =======================
    # Confusion Matrix (NB + TF-IDF)
    # =======================
    st.subheader("📊 Confusion Matrix")

    X = df["stemmed_text"].astype(str)
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ("classifier", MultinomialNB())
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax_cm
    )
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    ax_cm.set_title("Confusion Matrix - Naïve Bayes (TF-IDF)")
    st.pyplot(fig_cm)

    # =======================
    # WordCloud Keseluruhan
    # =======================
    st.subheader("☁️ WordCloud Keseluruhan")

    text_all = " ".join(df["stemmed_text"].astype(str))
    wc_all = WordCloud(
        width=900,
        height=400,
        background_color="white",
        max_words=150
    ).generate(text_all)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wc_all, interpolation="bilinear")
    ax_wc.axis("off")
    st.pyplot(fig_wc)

    # =======================
    # WordCloud per Sentimen
    # =======================
    st.subheader("☁️ WordCloud per Sentimen")

    colors = {
        "positif": "Greens",
        "negatif": "Reds",
        "netral": "Greys"
    }

    cols = st.columns(3)

    for i, label in enumerate(colors.keys()):
        subset = df[df["sentiment_label"] == label]

        if subset.empty:
            continue

        text_label = " ".join(subset["stemmed_text"].astype(str))

        wc = WordCloud(
            width=700,
            height=350,
            background_color="white",
            max_words=120,
            colormap=colors[label]
        ).generate(text_label)

        with cols[i]:
            st.markdown(f"**{label.capitalize()}**")
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    st.markdown("---")
    st.markdown("📌 **Menu Visualisasi — Penelitian Skripsi Analisis Sentimen Mobile Legends**")


if __name__ == "__main__":
    main()
