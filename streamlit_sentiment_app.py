# =========================================================
# 📊 VISUALISASI & EVALUASI MODEL SENTIMEN
# Metode: TF-IDF + Multinomial Naïve Bayes
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

def main():
    # =========================================================
    # 🎯 JUDUL HALAMAN
    # =========================================================
    st.title("📊 Visualisasi & Evaluasi Model Sentimen")
    st.markdown("Metode **TF-IDF + Naïve Bayes** (Dataset Penelitian Skripsi)")

    # =========================================================
    # 📂 UPLOAD DATASET
    # =========================================================
    uploaded_file = st.file_uploader(
        "📂 Upload dataset hasil pelabelan sentimen (.xlsx)",
        type=["xlsx"]
    )

    if uploaded_file is None:
        st.info("Silakan upload dataset berlabel terlebih dahulu.")
        return

    # Membaca dataset
    df = pd.read_excel(uploaded_file)

    # Normalisasi nama kolom
    df.columns = df.columns.str.strip().str.lower()

    # =========================================================
    # 🔍 AUTO-DETEKSI KOLOM TEKS & LABEL
    # =========================================================
    text_candidates = ["stemmed_text", "clean_text", "full_text", "text"]
    label_candidates = ["sentiment_label", "sentiment", "label"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    # Validasi kolom
    if text_col is None or label_col is None:
        st.error("❌ Kolom teks atau label sentimen tidak ditemukan.")
        st.write("Kolom yang tersedia:", df.columns.tolist())
        return

    # Samakan nama kolom agar konsisten
    df = df.rename(columns={
        text_col: "stemmed_text",
        label_col: "sentiment_label"
    })

    # Hapus data kosong
    df = df.dropna(subset=["stemmed_text", "sentiment_label"])

    # =========================================================
    # 📊 INFORMASI DATASET
    # =========================================================
    st.subheader("📌 Informasi Dataset")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Total Data:** {len(df)}")
        st.write("**Distribusi Label Sentimen:**")
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

    # =========================================================
    # 🔀 SPLIT DATA TRAINING & TESTING
    # =========================================================
    X = df["stemmed_text"].astype(str)
    y = df["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =========================================================
    # ⚙️ PIPELINE TF-IDF + NAÏVE BAYES
    # =========================================================
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

    # =========================================================
    # 🧠 TRAINING MODEL
    # =========================================================
    with st.spinner("🔄 Melatih model Naïve Bayes..."):
        pipeline.fit(X_train, y_train)

    # =========================================================
    # 📈 EVALUASI MODEL
    # =========================================================
    y_pred = pipeline.predict(X_test)

    st.subheader("📈 Hasil Evaluasi Model")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Akurasi", round(accuracy_score(y_test, y_pred), 4))
        st.metric(
            "F1-Score (Weighted)",
            round(f1_score(y_test, y_pred, average="weighted"), 4)
        )

    with col4:
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

    # =========================================================
    # 📊 CONFUSION MATRIX
    # =========================================================
    st.subheader("📊 Confusion Matrix")

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
    ax_cm.set_title("Confusion Matrix - TF-IDF + Naïve Bayes")
    st.pyplot(fig_cm)


# =========================================================
# 🚀 EKSEKUSI
# =========================================================
if __name__ == "__main__":
    main()
