# ===========================================
# streamlit_sentiment_app.py (HD Compact Visual + Multiwarna)
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
from collections import Counter
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
# Lexicon + Negation
# -------------------------
POSITIF = set(["bagus","seru","keren","hebat","mantap","menang","gg","top","lancar","suka","jago","menarik","terbaik","asik","puas","oke","legend","unggul"])
NEGATIF = set(["buruk","jelek","lag","noob","toxic","kalah","lemot","kecewa","marah","bete","down","ngehang","ngeframe","error","ampas","sampah","parah"])
NETRAL = set(["hero","map","tim","build","match","battle","item","mode","skill","tank","mage","game","player","update","event","grafik"])
NEGASI = set(["tidak", "bukan", "nggak", "ga", "gak", "tak", "belum"])
MULTIWORD = {"tim beban": "negatif", "server jelek": "negatif", "lemot banget": "negatif", "bagus banget": "positif"}

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
# Hybrid Sentiment Function
# -------------------------
def hybrid_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"label": "netral", "score": 0.0}

    txt = preprocess_text(text)
    words = txt.split()
    mw = detect_multiword(txt)
    labels = []

    for i, w in enumerate(words):
        label = get_lexicon_label(w)
        if i > 0 and words[i - 1] in NEGASI:
            if label == "positif": label = "negatif"
            elif label == "negatif": label = "positif"
        labels.append(label)

    if mw:
        labels.append(mw)

    counts = Counter(labels)
    lex_label = max(counts, key=counts.get)
    lex_conf = counts[lex_label] / sum(counts.values())

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        blob_label = "positif"
    elif polarity < -0.05:
        blob_label = "negatif"
    else:
        blob_label = "netral"
    blob_conf = abs(polarity)

    if lex_label == blob_label:
        final_label = lex_label
        conf = (lex_conf + blob_conf) / 2
    else:
        final_label = blob_label if blob_conf > 0.25 else lex_label
        conf = max(lex_conf, blob_conf)
    return {"label": final_label, "score": round(conf, 3)}

# -------------------------
# Streamlit Main App
# -------------------------
def main():
    st.title("📊 Analisis Sentimen Pengguna Mobile Legends")
    st.write("Upload dataset mentah (Excel/CSV) untuk dilabeli otomatis menggunakan metode **Hybrid Lexicon + Naïve Bayes**.")

    file = st.file_uploader("Unggah file dataset (.xlsx / .csv)", type=["xlsx", "csv"])
    if file is None:
        st.info("Silakan unggah dataset terlebih dahulu.")
        return

    try:
        df = pd.read_excel(file) if file.name.endswith(".xlsx") else pd.read_csv(file)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    df.columns = df.columns.str.strip().str.lower()
    text_col = next((c for c in ["stemmed_text", "clean_text", "text", "komentar", "tweet"] if c in df.columns), None)
    if text_col is None:
        st.error("Kolom teks tidak ditemukan. Pastikan ada kolom seperti 'stemmed_text' atau 'text'.")
        return

    st.success(f"Dataset berhasil dimuat ✅ — Total {len(df)} data")
    st.dataframe(df[[text_col]].head(5))

    # Labeling otomatis
    st.info("Melabeli data... mohon tunggu sebentar.")
    df["stemmed_text"] = df[text_col].astype(str).apply(preprocess_text)
    result = df["stemmed_text"].apply(hybrid_sentiment)
    df["sentiment_label"] = result.apply(lambda x: x["label"])
    df["confidence_score"] = result.apply(lambda x: x["score"])

    # Distribusi Sentimen
    st.subheader("📈 Distribusi Sentimen")
    order = ["positif", "netral", "negatif"]
    counts = df["sentiment_label"].value_counts().reindex(order, fill_value=0)

    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.write("Jumlah Data per Sentimen:")
        st.dataframe(counts.rename_axis("Label").reset_index().rename(columns={0: "Jumlah"}))
    with col2:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=180)
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
               colors=["#2ecc71", "#5dade2", "#e74c3c"], textprops={"fontsize": 8})
        st.pyplot(fig, use_container_width=True)

    # Model Naïve Bayes
    X = df["stemmed_text"]
    y = df["sentiment_label"]
    if len(y.unique()) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
        model = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.85)),
            ("nb", MultinomialNB())
        ])
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.subheader("📊 Evaluasi Model Naïve Bayes")
        st.write(f"**Akurasi Model :** {acc:.4f}")
        st.write(f"**F1-Score :** {f1:.4f}")

        # Confusion Matrix (compact + responsif)
        cm = confusion_matrix(y_test, y_pred, labels=order)
        fig_cm, ax_cm = plt.subplots(figsize=(2.6, 2.0), dpi=180)
        sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                    xticklabels=order, yticklabels=order, cbar=False, annot_kws={"size": 8})
        ax_cm.set_xlabel("Prediksi", fontsize=8)
        ax_cm.set_ylabel("Aktual", fontsize=8)
        ax_cm.set_title("Confusion Matrix", fontsize=9, pad=6)
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)

  # WordCloud Multiwarna
st.subheader("☁️ WordCloud Sentimen (Multiwarna & Responsif)")

# Palet warna yang sama dengan pie chart
palette = ["#2ecc71", "#f1c40f", "#e74c3c"]

def pie_palette_color_func(word=None, font_size=None, position=None,
                           orientation=None, font_path=None, random_state=None):
    """Pewarnaan kata berdasarkan palet warna pie chart"""
    return random.choice(palette)

for lbl in order:
    text_data = " ".join(df[df["sentiment_label"] == lbl]["stemmed_text"])
    if not text_data.strip():
        continue
    wc = WordCloud(
        width=480, height=260,
        background_color="white",
        color_func=pie_palette_color_func,  # Warna disesuaikan
        max_words=150,
        collocations=False,
        prefer_horizontal=0.9
    ).generate(text_data)

    st.markdown(f"**{lbl.capitalize()}** ({counts[lbl]} data)")
    fig, ax = plt.subplots(figsize=(2.8, 1.8), dpi=180)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)



    # Preview Data
    st.subheader("🔍 Contoh Hasil Pelabelan")
    st.dataframe(df[["stemmed_text", "sentiment_label", "confidence_score"]].head(20))

if __name__ == "__main__":
    main()
