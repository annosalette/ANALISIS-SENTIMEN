# =========================================================
# 📊 VISUALISASI ANALISIS SENTIMEN MOBILE LEGENDS
#     TF-IDF + NAÏVE BAYES (FINAL + ENHANCED)
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from wordcloud import WordCloud
from nltk.corpus import stopwords

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

# =========================================================
# MAIN FUNCTION
# =========================================================
def main():

    # -----------------------------------------------------
    # KONFIGURASI HALAMAN
    # -----------------------------------------------------
    st.set_page_config(
        page_title="Analisis Sentimen Mobile Legends",
        page_icon="🎮",
        layout="wide"
    )

    st.title("🎮 Visualisasi Analisis Sentimen Mobile Legends")
    st.markdown("Evaluasi model Naïve Bayes")

    # -----------------------------------------------------
    # STOPWORDS
    # -----------------------------------------------------
    nltk.download("stopwords")
    stop_words = stopwords.words("indonesian")

    # -----------------------------------------------------
    # EMOJI SENTIMEN
    # -----------------------------------------------------
    sentiment_emoji = {
        "positif": "😊",
        "negatif": "😡",
        "netral": "😐"
    }

    # -----------------------------------------------------
    # UPLOAD DATASET
    # -----------------------------------------------------
    uploaded_file = st.file_uploader(
        "📂 Upload dataset hasil pelabelan (.xlsx)",
        type=["xlsx"]
    )

    if uploaded_file is None:
        st.info("Silakan upload dataset terlebih dahulu.")
        return

    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    # -----------------------------------------------------
    # AUTO DETEKSI KOLOM
    # -----------------------------------------------------
    text_candidates = ["stemmed_text", "clean_text", "text", "comment"]
    label_candidates = ["sentiment_label", "sentiment", "label", "sentimen"]

    text_col = next((c for c in text_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if text_col is None or label_col is None:
        st.error("❌ Kolom teks atau label tidak ditemukan.")
        st.write("Kolom tersedia:", df.columns.tolist())
        return

    df = df.rename(columns={
        text_col: "stemmed_text",
        label_col: "sentiment_label"
    })

    df_raw = df.copy()
    df_model = df.dropna(subset=["stemmed_text", "sentiment_label"])

    # -----------------------------------------------------
    # INFORMASI & DISTRIBUSI SENTIMEN
    # -----------------------------------------------------
    st.subheader("📌 Informasi & Distribusi Sentimen")

    sentiment_counts = df_raw["sentiment_label"].value_counts()
    sentiment_percent = sentiment_counts / sentiment_counts.sum() * 100

    cols = st.columns(len(sentiment_counts))
    for col, (label, count) in zip(cols, sentiment_counts.items()):
        emoji = sentiment_emoji.get(label, "📊")
        col.metric(
            label=f"{emoji} {label.capitalize()}",
            value=count,
            delta=f"{sentiment_percent[label]:.1f}%"
        )

    st.markdown("---")

    # -----------------------------------------------------
    # BAR CHART DISTRIBUSI
    # -----------------------------------------------------
    st.subheader("📊 Distribusi Sentimen (Bar Chart)")

    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    sns.barplot(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        palette="Set2",
        ax=ax_bar
    )
    ax_bar.set_xlabel("Sentimen")
    ax_bar.set_ylabel("Jumlah Data")
    ax_bar.set_title("Distribusi Sentimen")
    st.pyplot(fig_bar)

    # -----------------------------------------------------
    # WORDCLOUD PER SENTIMEN
    # -----------------------------------------------------
    st.subheader("☁️ WordCloud Berdasarkan Sentimen")

    sentiments = sorted(df_raw["sentiment_label"].dropna().unique())
    cols_wc = st.columns(len(sentiments))

    for col, sentiment in zip(cols_wc, sentiments):
        with col:
            emoji = sentiment_emoji.get(sentiment, "")
            st.markdown(f"**{emoji} {sentiment.capitalize()}**")

            text_data = " ".join(
                df_raw[df_raw["sentiment_label"] == sentiment]["stemmed_text"]
                .dropna()
                .astype(str)
                .values
            )

            if text_data.strip() == "":
                st.info("Tidak ada teks.")
            else:
                wc = WordCloud(
                    width=500,
                    height=300,
                    background_color="white",
                    max_words=120,
                    collocations=False
                ).generate(text_data)

                fig_wc, ax_wc = plt.subplots()
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

    # -----------------------------------------------------
    # SPLIT DATA
    # -----------------------------------------------------
    X = df_model["stemmed_text"].astype(str)
    y = df_model["sentiment_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    labels = sorted(y.unique())
    st.write("Urutan Label:", labels)

    # -----------------------------------------------------
    # PIPELINE TF-IDF + NAÏVE BAYES
    # -----------------------------------------------------
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True,
            stop_words=stop_words,
            max_features=5000
        )),
        ("classifier", MultinomialNB(
            class_prior=[0.3, 0.4, 0.3]
        ))
    ])

    with st.spinner("🔄 Melatih model Naïve Bayes..."):
        pipeline.fit(X_train, y_train)

    # -----------------------------------------------------
    # EVALUASI MODEL
    # -----------------------------------------------------
    y_pred = pipeline.predict(X_test)

    st.subheader("📈 Evaluasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Akurasi", round(accuracy_score(y_test, y_pred), 4))
        st.metric("F1-Score (Weighted)", round(
            f1_score(y_test, y_pred, average="weighted"), 4)
        )

    with col2:
        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

    # -----------------------------------------------------
    # CONFUSION MATRIX (LEBIH KECIL)
    # -----------------------------------------------------
    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 10},
        ax=ax_cm
    )
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    ax_cm.set_title("Confusion Matrix - Naïve Bayes")
    st.pyplot(fig_cm)

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("📌 ** Naïve Bayes| Skripsi**")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    main()
