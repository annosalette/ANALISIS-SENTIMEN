import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from textblob import TextBlob
import nltk

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# ======================
# 🔧 PREPROCESS
# ======================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"@[\w_]+", " ", s)
    s = re.sub(r"#", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

POSITIF = {"bagus", "keren", "hebat", "menang", "mantap", "pro", "asik", "cepat", "puas", "support", "gg", "win"}
NEGATIF = {"buruk", "jelek", "lag", "noob", "toxic", "kalah", "lemot", "ngeframe", "bete", "error", "ampas", "kecewa"}
NETRAL = {"hero", "rank", "tim", "build", "mode", "mlbb", "item", "update", "akun", "server"}
NEGASI = {"tidak", "bukan", "gak", "ga", "nggak", "tak", "ndak"}

# ======================
# 💬 LEXICON METHOD
# ======================
def lexicon_sentiment(text):
    words = text.split()
    label = []
    for i, w in enumerate(words):
        lbl = "netral"
        if w in POSITIF:
            lbl = "positif"
        elif w in NEGATIF:
            lbl = "negatif"
        elif w in NETRAL:
            lbl = "netral"
        if i > 0 and words[i - 1] in NEGASI:
            if lbl == "positif":
                lbl = "negatif"
            elif lbl == "negatif":
                lbl = "positif"
        label.append(lbl)
    vc = pd.Series(label).value_counts()
    return vc.idxmax() if len(vc) > 0 else "netral"

# ======================
# ⚙️ HYBRID SENTIMENT
# ======================
def hybrid_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return {"label": "netral", "score": 0.0}
    t = preprocess_text(text)
    lex = lexicon_sentiment(t)
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol > 0.05:
        tb = "positif"
    elif pol < -0.05:
        tb = "negatif"
    else:
        tb = "netral"
    final = tb if tb != "netral" else lex
    conf = abs(pol)
    return {"label": final, "score": conf}

# ======================
# 🚀 STREAMLIT MAIN
# ======================
def main():
    st.set_page_config(page_title="Analisis Sentimen Hybrid", layout="wide")
    st.title("📊 Analisis Sentimen Pengguna Mobile Legends di Twitter")

    uploaded = st.file_uploader("📂 Upload Dataset Mentah (.xlsx / .csv)", type=["xlsx", "csv"])
    if not uploaded:
        st.info("Silakan unggah file data komentar mentah terlebih dahulu.")
        return

    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    df.columns = df.columns.str.lower()
    text_col = next((c for c in ["stemmed_text", "full_text", "text", "komentar", "tweet", "ulasan"] if c in df.columns), None)
    if not text_col:
        st.error("Kolom teks tidak ditemukan. Pastikan ada kolom seperti 'full_text' atau 'text'.")
        return

    st.success(f"Dataset berhasil dimuat — total {len(df)} data.")
    st.dataframe(df[[text_col]].head(5))

    # Preprocessing & Labeling
    df["stemmed_text"] = df[text_col].astype(str).apply(preprocess_text)
    with st.spinner("Melakukan pelabelan sentimen..."):
        res = df["stemmed_text"].apply(hybrid_sentiment)
        df["sentiment_label"] = res.apply(lambda x: x["label"])
        df["confidence_score"] = res.apply(lambda x: x["score"])

    # ======================
    # 📊 DISTRIBUSI SENTIMEN
    # ======================
    st.subheader("📈 Distribusi Sentimen")
    counts = df["sentiment_label"].value_counts().reindex(["positif", "netral", "negatif"], fill_value=0)
    st.table(counts.rename_axis("Sentimen").reset_index(name="Jumlah"))

    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(counts)
    with col2:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
               colors=["#2ecc71", "#f1c40f", "#e74c3c"], textprops={"fontsize": 8})
        ax.axis("equal")
        st.pyplot(fig)

    # ======================
    # 🤖 TRAINING NAIVE BAYES
    # ======================
    tmp = df[["stemmed_text", "sentiment_label"]].dropna()
    X, y = tmp["stemmed_text"], tmp["sentiment_label"]
    if len(y.unique()) < 2:
        st.warning("Hanya satu kelas terdeteksi, model tidak dapat dilatih.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("nb", MultinomialNB())])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.markdown(f"### 🔹 Akurasi Model : **{acc:.4f}**")
    st.markdown(f"### 🔹 F1-Score : **{f1:.4f}**")

    st.text("Laporan Klasifikasi:")
    st.text(classification_report(y_test, y_pred, digits=2))

    # ======================
    # 📉 CONFUSION MATRIX
    # ======================
    st.subheader("📉 Confusion Matrix (Mini)")
    cm = confusion_matrix(y_test, y_pred, labels=["positif", "netral", "negatif"])
    fig_cm, ax = plt.subplots(figsize=(3, 2.5))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
                xticklabels=["positif", "netral", "negatif"],
                yticklabels=["positif", "netral", "negatif"],
                cbar=False, annot_kws={"size": 8}, ax=ax)
    ax.set_xlabel("Prediksi", fontsize=8)
    ax.set_ylabel("Aktual", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_cm)

    # ======================
    # ☁️ WORDCLOUD (kecil)
    # ======================
    st.subheader("☁️ WordCloud per Sentimen (Mini)")
    cmap = {"positif": "Greens", "netral": "Purples", "negatif": "Reds"}
    for sent in ["positif", "netral", "negatif"]:
        teks = " ".join(df[df["sentiment_label"] == sent]["stemmed_text"].astype(str))
        if not teks.strip():
            continue
        wc = WordCloud(width=250, height=150, background_color="white",
                       colormap=cmap[sent], max_words=60, collocations=False).generate(teks)
        st.markdown(f"**Sentimen: {sent.capitalize()} ({counts.get(sent,0)} data)**")
        fig_wc, ax_wc = plt.subplots(figsize=(2.5, 1.6))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
if __name__ == "__main__":
    main()
