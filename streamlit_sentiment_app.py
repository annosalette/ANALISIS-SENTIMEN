# =========================================================
# 📊 STREAMLIT APLIKASI ANALISIS SENTIMEN MOBILE LEGENDS
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

# =========================================================
# ⚙️ KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="Analisis Sentimen Mobile Legends",
    page_icon="🎮",
    layout="wide"
)

st.title("🎮 Analisis Sentimen Mobile Legends")
st.markdown("Aplikasi klasifikasi sentimen menggunakan **TF-IDF + Naïve Bayes**")

# =========================================================
# 📂 UPLOAD DATASET
# =========================================================
uploaded_file = st.file_uploader(
    "📂 Upload dataset hasil pelabelan (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Silakan upload dataset terlebih dahulu.")
    st.stop()

df = pd.read_excel(uploaded_file)
df.columns = df.columns.str.strip().str.lower()

# =========================================================
# 🔍 AUTO-DETEKSI KOLOM TEKS & LABEL
# =========================================================
text_candidates = ["stemmed_text", "clean_text", "full_text", "text", "comment"]
label_candidates = ["sentiment_label", "sentiment", "label", "sentimen", "kelas"]

text_col = next((c for c in text_candidates if c in df.columns), None)
label_col = next((c for c in label_candidates if c in df.columns), None)

if text_col is None or label_col is None:
    st.error("❌ Kolom teks atau label tidak ditemukan.")
    st.write("Kolom yang tersedia:", df.columns.tolist())
    st.stop()

# Normalisasi nama kolom agar konsisten
df = df.rename(columns={
    text_col: "stemmed_text",
    label_col: "sentiment_label"
})

df = df.dropna(subset=["stemmed_text", "sentiment_label"])

# =========================================================
# 📊 INFO DATASET
# =========================================================
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

# =========================================================
# 🔀 SPLIT DATA
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
# 🧠 TRAIN MODEL
# =========================================================
with st.spinner("🔄 Melatih model..."):
    pipeline.fit(X_train, y_train)

# =========================================================
# 📈 EVALUASI MODEL
# =========================================================
y_pred = pipeline.predict(X_test)

st.subheader("📈 Hasil Evaluasi Model")

col3, col4 = st.columns(2)
with col3:
    st.metric("Akurasi", round(accuracy_score(y_test, y_pred), 4))
    st.metric("F1-Score (Weighted)", round(f1_score(y_test, y_pred, average="weighted"), 4))

with col4:
    st.text("Laporan Klasifikasi")
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
ax_cm.set_title("Confusion Matrix - Naïve Bayes (TF-IDF)")
st.pyplot(fig_cm)

# =========================================================
# 🔮 PREDIKSI TEKS BARU
# =========================================================
st.subheader("🔮 Prediksi Sentimen Teks Baru")

user_input = st.text_area(
    "Masukkan teks tweet / komentar:",
    height=120
)

if st.button("Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong.")
    else:
        prediction = pipeline.predict([user_input])[0]
        st.success(f"🎯 Hasil Prediksi Sentimen: **{prediction.upper()}**")

# =========================================================
# ✅ FOOTER
# =========================================================
st.markdown("---")
st.markdown("📌 **TF-IDF + Naïve Bayes | Penelitian Skripsi**")
