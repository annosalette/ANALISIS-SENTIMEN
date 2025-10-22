# ===============================================================
# streamlit_sentiment_app.py — Final (Upload Manual)
# ===============================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

# ---------------------------------------------------------------
# Judul Aplikasi
# ---------------------------------------------------------------
def main():
    st.set_page_config(page_title="Analisis Sentimen Mobile Legends", layout="wide")
    st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER MENGGUNAKAN ALGORITMA NAÏVE BAYES")
    st.markdown("""

    """)

    # -----------------------------------------------------------
    # Upload Dataset
    # -----------------------------------------------------------
    st.subheader("📂 Upload Dataset ")
    uploaded = st.file_uploader("Upload file .xlsx (harus memiliki kolom 'stemmed_text' & 'sentiment_label')", type=["xlsx"])

    if not uploaded:
        st.info("Silakan upload file Excel berlabel hasil dari proses pelabelan di Colab.")
        return

    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    df.columns = df.columns.str.strip().str.lower()

    if "stemmed_text" not in df.columns or "sentiment_label" not in df.columns:
        st.error("Dataset harus memiliki kolom 'stemmed_text' dan 'sentiment_label'.")
        return

    df = df.dropna(subset=["stemmed_text", "sentiment_label"]).reset_index(drop=True)
    st.success(f"Dataset dimuat: {len(df)} data.")
    st.dataframe(df.head(10))

    # -----------------------------------------------------------
    # TF-IDF + Naïve Bayes
    # -----------------------------------------------------------
    X = df["stemmed_text"].astype(str)
    y = df["sentiment_label"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1,2),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ("nb", MultinomialNB())
    ])

    with st.spinner("🔄 Melatih model Naïve Bayes..."):
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.subheader("📊 Evaluasi Model")
    st.write(f"- **Akurasi:** {acc:.4f}")
    st.write(f"- **F1 Score (weighted):** {f1:.4f}")
    st.text("Laporan Klasifikasi:")
    st.text(classification_report(y_test, y_pred, digits=4))

    # -----------------------------------------------------------
    # Confusion Matrix (kecil & rapi)
    # -----------------------------------------------------------
    st.subheader("📉 Confusion Matrix")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar=False, annot_kws={"size": 9}, ax=ax_cm)
    ax_cm.set_xlabel("Prediksi", fontsize=9)
    ax_cm.set_ylabel("Aktual", fontsize=9)
    ax_cm.set_title("Confusion Matrix", fontsize=10, pad=6)
    plt.tight_layout()
    st.pyplot(fig_cm)

    # -----------------------------------------------------------
    # Distribusi Label (Bar + Pie + Tabel)
    # -----------------------------------------------------------
    st.subheader("📈 Distribusi Sentimen (Seluruh Dataset)")
    display_order = ["positif", "netral", "negatif"]
    counts = df["sentiment_label"].value_counts().reindex(display_order, fill_value=0)

    counts_df = counts.reset_index()
    counts_df.columns = ["sentimen", "jumlah"]
    st.table(counts_df)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Bar Chart**")
        st.bar_chart(counts)
    with col2:
        st.markdown("**Pie Chart**")
        fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]  # hijau, kuning, merah
        ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%",
                   startangle=90, colors=colors, textprops={"fontsize": 9})
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

    # -----------------------------------------------------------
    # WordCloud per Sentimen (ukuran kecil)
    # -----------------------------------------------------------
    st.subheader("☁️ Wordcloud per Sentimen")
    cmap = {"positif": "Greens", "netral": "Purples", "negatif": "Reds"}
    for label in display_order:
        text_data = " ".join(df[df["sentiment_label"] == label]["stemmed_text"])
        if len(text_data.strip()) == 0:
            continue
        wc = WordCloud(width=400, height=250, background_color="white",
                       colormap=cmap[label]).generate(text_data)
        st.write(f"**{label.capitalize()}** — total: {counts[label]} data")
        fig_wc, ax_wc = plt.subplots(figsize=(4, 2.5))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)



if __name__ == "__main__":
    main()
