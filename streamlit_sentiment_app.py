# ===============================================================
# 🧠 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER
#    MENGGUNAKAN ALGORITMA NAÏVE BAYES (TF-IDF)
# ===============================================================

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
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from collections import Counter

# ===============================================================
# 1️⃣ Judul Utama
# ===============================================================
st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER MENGGUNAKAN ALGORITMA NAÏVE BAYES (TF-IDF)")

st.markdown("""
Aplikasi ini menganalisis sentimen komentar pengguna **Mobile Legends** dengan 
algoritma **Naïve Bayes** dan teknik ekstraksi fitur **TF-IDF**.
""")

# ===============================================================
# 2️⃣ Kamus Kata (Jika Auto-Label Diperlukan)
# ===============================================================
positif_words = [
    "bagus","seru","keren","hebat","mantap","menang","gg","top","lancar",
    "op","meta","pro","senang","suka","jago","menarik","terbaik","asik",
    "cepat","puas","kompak","support","win","mantul","legend","unggul","mantep",
    "oke","solid","bagus banget","gokil","epic","fun","stabil","menyenangkan"
]
negatif_words = [
    "buruk","jelek","lag","noob","kesal","marah","toxic","parah","kalah",
    "susah","ngebug","nerf","lemot","ngehang","ngeframe","bete","down",
    "ngefreeze","lelet","gagal","ngecrash","ampas","bodoh","ngelek","error",
    "kecewa","anjir","anjay","kurang","rusak","tidak puas","ngegas","burik",
    "sampah","tim beban","parah banget","lemot banget","server jelek"
]
netral_words = [
    "hero","map","rank","tim","build","match","battle","item","mode","skill",
    "tank","mage","marksman","assassin","support","fighter","mlbb","draft",
    "push","mid","lane","turret","minion","buff","farm","jungle","ulti",
    "player","game","combo","ranked","classic","solo","timing","update",
    "event","grafik","gameplay","akun","emblem","server","patch","fps","ping"
]

def get_sentiment_manual(word):
    if word in positif_words:
        return "positif"
    elif word in negatif_words:
        return "negatif"
    elif word in netral_words:
        return "netral"
    else:
        return "netral"

# ===============================================================
# 3️⃣ Upload Dataset
# ===============================================================
st.subheader("📂 Upload Dataset (.xlsx) dengan Kolom 'stemmed_text' dan (opsional) 'sentiment_label'")
uploaded_file = st.file_uploader("Unggah dataset Anda:", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    if 'stemmed_text' not in df.columns:
        st.error("❌ Kolom 'stemmed_text' tidak ditemukan di dataset.")
        st.stop()

    # Jika belum ada label, buat otomatis dengan kamus kata
    if 'sentiment_label' not in df.columns:
        st.info("Kolom 'sentiment_label' tidak ditemukan. Melakukan auto-labeling dengan kamus kata...")
        temp_labeled = []
        for text in df['stemmed_text'].astype(str):
            words = [w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text)]
            sentiments = [get_sentiment_manual(w) for w in words]
            if 'positif' in sentiments:
                temp_labeled.append('positif')
            elif 'negatif' in sentiments:
                temp_labeled.append('negatif')
            else:
                temp_labeled.append('netral')
        df['sentiment_label'] = temp_labeled
        st.success("✅ Proses auto-labeling selesai!")

    st.write(f"Total data: **{len(df)}**")
    st.dataframe(df.head(10))

    # ===============================================================
    # 4️⃣ Split Data & Pipeline Naïve Bayes (TF-IDF)
    # ===============================================================
    X = df['stemmed_text'].astype(str)
    y = df['sentiment_label'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.85,
            sublinear_tf=True
        )),
        ('classifier', MultinomialNB())
    ])

    nb_pipeline.fit(X_train, y_train)
    y_pred = nb_pipeline.predict(X_test)

    # ===============================================================
    # 5️⃣ Evaluasi Model
    # ===============================================================
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.subheader("📊 Evaluasi Model Naïve Bayes (TF-IDF)")
    st.write(f"**Akurasi Model:** {round(acc, 4)}")
    st.write(f"**F1-Score:** {round(f1, 4)}")
    st.text(classification_report(y_test, y_pred))

    # ===============================================================
    # 6️⃣ Confusion Matrix
    # ===============================================================
    st.subheader("📉 Confusion Matrix")
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig_cm, ax_cm = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel("Prediksi")
    ax_cm.set_ylabel("Aktual")
    ax_cm.set_title("Confusion Matrix - Naïve Bayes (TF-IDF)")
    st.pyplot(fig_cm)

    # ===============================================================
    # 7️⃣ Distribusi Sentimen (Bar + Pie)
    # ===============================================================
    st.subheader("📈 Distribusi Sentimen pada Dataset")
    counts = df["sentiment_label"].value_counts()
    all_labels = ["positif", "netral", "negatif"]
    counts = counts.reindex(all_labels, fill_value=0)

    st.bar_chart(counts)

    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(
        counts,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#f1c40f", "#e74c3c"]
    )
    ax_pie.axis("equal")
    ax_pie.set_title("Distribusi Sentimen Komentar Mobile Legends")
    st.pyplot(fig_pie)

    # ===============================================================
    # 8️⃣ Wordcloud per Sentimen
    # ===============================================================
    st.subheader("☁️ Wordcloud per Sentimen")
    colors = {"positif": "Greens", "negatif": "Reds", "netral": "Blues"}
    for sent in all_labels:
        text_data = " ".join(df[df["sentiment_label"] == sent]["stemmed_text"])
        if text_data.strip():
            wc = WordCloud(width=800, height=400, colormap=colors[sent],
                           background_color="white").generate(text_data)
            st.write(f"**Sentimen: {sent.capitalize()}**")
            fig, ax = plt.subplots()
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

    # ===============================================================
    # 9️⃣ Download Hasil
    # ===============================================================
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Hasil Analisis Sentimen (CSV)",
        data=csv,
        file_name="hasil_sentimen_naivebayes_tfidf.csv",
        mime="text/csv",
    )
