# ===========================================
# STREAMLIT SENTIMENT APP (FINAL - SINKRON NOTEBOOK)
# ===========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random

# -------------------------------------------
# Streamlit Config
# -------------------------------------------
st.set_page_config(
    page_title="Analisis Sentimen Mobile Legends",
    layout="centered"
)

st.title("📊 Analisis Sentimen Pengguna Mobile Legends")
st.write(
    "Aplikasi ini menampilkan hasil klasifikasi sentimen "
    "berdasarkan data yang telah dilabeli pada tahap penelitian."
)

# -------------------------------------------
# Upload Dataset
# -------------------------------------------
file = st.file_uploader(
    "Unggah dataset hasil pelabelan (Excel / CSV)",
    type=["xlsx", "csv"]
)

if file is None:
    st.info("Silakan unggah dataset hasil pelabelan dari notebook penelitian.")
    st.stop()

# -------------------------------------------
# Load Data
# -------------------------------------------
try:
    df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    st.stop()

df.columns = df.columns.str.strip().str.lower()

# Validasi kolom wajib
required_cols = {"stemmed_text", "sentiment_label"}
if not required_cols.issubset(df.columns):
    st.error(
        "Dataset harus memiliki kolom: 'stemmed_text' dan 'sentiment_label'.\n"
        "Gunakan file hasil pelabelan dari notebook penelitian."
    )
    st.stop()

df = df.dropna(subset=["stemmed_text", "sentiment_label"])

st.success(f"Dataset berhasil dimuat ✅ — Total {len(df)} data")
st.dataframe(df[["stemmed_text", "sentiment_label"]].head(5))

# -------------------------------------------
# Distribusi Sentimen (ASLI PENELITIAN)
# -------------------------------------------
st.subheader("📈 Distribusi Sentimen")

order = ["positif", "netral", "negatif"]
sentiment_counts = df["sentiment_label"].value_counts().reindex(order, fill_value=0)

col1, col2 = st.columns([1, 1.2])

with col1:
    st.dataframe(
        sentiment_counts
        .rename_axis("Sentimen")
        .reset_index(name="Jumlah Data")
    )

with col2:
    fig, ax = plt.subplots(figsize=(3, 3), dpi=180)
    ax.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ecc71", "#f1c40f", "#e74c3c"],
        textprops={"fontsize": 8}
    )
    ax.set_title("Distribusi Sentimen")
    st.pyplot(fig)

# -------------------------------------------
# Model Naïve Bayes (TF-IDF)
# -------------------------------------------
st.subheader("📊 Evaluasi Model Naïve Bayes")

X = df["stemmed_text"]
y = df["sentiment_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True
    )),
    ("nb", MultinomialNB())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")

st.write(f"**Akurasi Model :** {acc:.4f}")
st.write(f"**F1-Score :** {f1:.4f}")

# -------------------------------------------
# Confusion Matrix
# -------------------------------------------
cm = confusion_matrix(y_test, y_pred, labels=order)

fig_cm, ax_cm = plt.subplots(figsize=(2.8, 2.2), dpi=180)
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="YlGnBu",
    xticklabels=order,
    yticklabels=order,
    cbar=False,
    annot_kws={"size": 8}
)

ax_cm.set_xlabel("Prediksi")
ax_cm.set_ylabel("Aktual")
ax_cm.set_title("Confusion Matrix Naïve Bayes")
plt.tight_layout()
st.pyplot(fig_cm)

# -------------------------------------------
# WordCloud per Sentimen
# -------------------------------------------
st.subheader("☁️ WordCloud per Sentimen")

palette = {
    "positif": ["#2ecc71"],
    "netral": ["#f1c40f"],
    "negatif": ["#e74c3c"]
}

def color_func(sent):
    return lambda *args, **kwargs: random.choice(palette[sent])

for sent in order:
    text_data = " ".join(df[df["sentiment_label"] == sent]["stemmed_text"])
    if text_data.strip():
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=150,
            collocations=False,
            color_func=color_func(sent)
        ).generate(text_data)

        st.write(f"**Sentimen: {sent.capitalize()}**")
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

# -------------------------------------------
# Preview Data
# -------------------------------------------
st.subheader("🔍 Contoh Data Hasil Penelitian")
st.dataframe(
    df[["stemmed_text", "sentiment_label"]].sample(20, random_state=42)
)
