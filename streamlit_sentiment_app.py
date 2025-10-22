# ============================================================
# 📘 streamlit_sentiment_app.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from textblob import TextBlob

nltk.download("punkt")
nltk.download("wordnet")

# ===============================
# 1️⃣ PREPROCESSING
# ===============================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # hanya huruf
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ===============================
# 2️⃣ AUTO LABELING DENGAN TEXTBLOB (Backup)
# ===============================
def auto_label_with_textblob(df, text_col="clean_text"):
    status = []
    total_positif = total_negatif = total_netral = 0

    for tweet in df[text_col]:
        analysis = TextBlob(str(tweet))
        if analysis.sentiment.polarity > 0.0:
            status.append("positif")
            total_positif += 1
        elif analysis.sentiment.polarity == 0.0:
            status.append("netral")
            total_netral += 1
        else:
            status.append("negatif")
            total_negatif += 1

    df["klasifikasi"] = status
    return df, total_positif, total_netral, total_negatif

# ===============================
# 3️⃣ STREAMLIT APP
# ===============================
def main():
    st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER MENGGUNAKAN ALGORITMA NAÏVE BAYES")

    vectorizer = TfidfVectorizer()
    model = MultinomialNB()

    # ===============================
    # 4️⃣ UPLOAD DATASET BARU
    # ===============================
    st.subheader("📂 Upload Dataset untuk Analisis")
    uploaded_file = st.file_uploader("Upload file .xlsx", type=["xlsx"])

    if uploaded_file:
        df_new = pd.read_excel(uploaded_file)

        # Cari kolom teks utama
        text_col = None
        for candidate in ["stemmed_text", "clean_text", "full_text"]:
            if candidate in df_new.columns:
                text_col = candidate
                break

        if text_col is None:
            st.error("Dataset harus memiliki salah satu kolom teks: 'stemmed_text', 'clean_text', atau 'full_text'.")
            st.stop()

        # Buat kolom clean_text
        if text_col != "clean_text":
            df_new["clean_text"] = df_new[text_col].astype(str).apply(preprocess_text)
        else:
            df_new["clean_text"] = df_new["clean_text"].astype(str)

        # Auto labeling jika belum ada kolom klasifikasi
        if "klasifikasi" not in df_new.columns:
            st.info("Kolom 'klasifikasi' tidak ditemukan → Label otomatis dibuat dengan TextBlob.")
            df_new, pos, net, neg = auto_label_with_textblob(df_new, text_col="clean_text")
            st.success(f"Label otomatis selesai: Positif={pos}, Netral={net}, Negatif={neg}")

        # ===============================
        # 5️⃣ Split Data & Training
        # ===============================
        X = df_new["clean_text"].astype(str)
        y = df_new["klasifikasi"].astype(str)

        X_vec = vectorizer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.subheader("📊 Evaluasi Model Naïve Bayes")
        st.write("**Akurasi:**", round(acc, 4))
        st.text(report)

        # ===============================
        # 6️⃣ Confusion Matrix (kecil)
        # ===============================
        st.subheader("📉 Confusion Matrix")
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)

        fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax_cm,
            cbar=False,
            annot_kws={"size": 9}
        )
        ax_cm.set_xlabel("Prediksi", fontsize=9)
        ax_cm.set_ylabel("Aktual", fontsize=9)
        ax_cm.set_title("Confusion Matrix - Naïve Bayes", fontsize=10, pad=8)
        plt.tight_layout()
        st.pyplot(fig_cm)

        # ===============================
        # 7️⃣ Distribusi Sentimen
        # ===============================
        st.subheader("📊 Distribusi Sentimen")
        df_new["predicted_sentiment"] = model.predict(X_vec)
        sentiment_counts = df_new["predicted_sentiment"].value_counts()

        st.markdown("**Jumlah Distribusi Sentimen:**")
        st.dataframe(sentiment_counts.rename_axis("Sentimen").reset_index().rename(columns={"predicted_sentiment": "Jumlah"}))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Distribusi (Bar Chart)**")
            st.bar_chart(sentiment_counts)

        with col2:
            st.markdown("**Distribusi (Pie Chart)**")
            colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # Hijau, Kuning, Merah
            fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
            ax_pie.pie(
                sentiment_counts,
                labels=sentiment_counts.index,
                autopct='%1.1f%%',
                startangle=140,
                colors=colors,
                textprops={'fontsize': 9}
            )
            ax_pie.axis("equal")
            st.pyplot(fig_pie)

        # ===============================
        # 8️⃣ Wordcloud per Sentimen (kecil)
        # ===============================
        st.subheader("☁️ Wordcloud per Sentimen")
        sentiments = df_new["predicted_sentiment"].unique()
        color_map = {"positif": "Greens", "netral": "Purples", "negatif": "Reds"}

        for sent in sentiments:
            text_data = " ".join(df_new[df_new["predicted_sentiment"] == sent]["clean_text"])
            if text_data.strip():
                wc = WordCloud(
                    width=400,
                    height=250,
                    background_color="white",
                    colormap=color_map.get(sent.lower(), "Blues")
                ).generate(text_data)
                st.write(f"**Sentimen: {sent}**")
                fig_wc, ax_wc = plt.subplots(figsize=(4, 2.5))
                ax_wc.imshow(wc, interpolation="bilinear")
                ax_wc.axis("off")
                st.pyplot(fig_wc)

# Panggil main() saat file dijalankan langsung
if __name__ == "__main__":
    main()
