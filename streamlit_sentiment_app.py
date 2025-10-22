# streamlit_sentiment_app.py
"""
Streamlit app:
- Terima dataset mentah (.xlsx / .csv), deteksi kolom teks otomatis
- Pra-pemrosesan sederhana
- Pelabelan hybrid (lexicon + multiword + negation + TextBlob) sesuai metode Colab
- Training Naïve Bayes (TF-IDF) untuk evaluasi (train/test split)
- Visualisasi: confusion matrix (small), distribution (bar+pie), wordcloud (small 400x300)
- Download hasil pelabelan (.xlsx / .csv)
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
from io import BytesIO
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
from textblob import TextBlob
import nltk

# Ensure nltk punkt is available for TextBlob tokenization if needed
nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# -------------------------
# Utility / preprocessing
# -------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    s = str(text)
    s = s.lower()
    # remove urls
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    # remove mentions, hashtags symbols but keep words
    s = re.sub(r"@[\w_]+", " ", s)
    s = re.sub(r"#", " ", s)
    # keep letters and basic punctuation for negation detection (but remove extras)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------
# Lexicon + multiword + negation
# -------------------------
POSITIF_WORDS = set([
    "bagus","seru","keren","hebat","mantap","menang","gg","top","lancar",
    "op","meta","pro","senang","suka","jago","menarik","terbaik","asik",
    "cepat","puas","kompak","support","win","mantul","legend","unggul","mantep",
    "oke","solid","bagusbanget","bagus_banget","bagus banget","gokil","epic","fun","stabil","menyenangkan"
])
NEGATIF_WORDS = set([
    "buruk","jelek","lag","noob","kesal","marah","toxic","parah","kalah",
    "susah","ngebug","nerf","lemot","ngehang","ngeframe","bete","down",
    "ngefreeze","lelet","gagal","ngecrash","ampas","bodoh","ngelek","error",
    "kecewa","anjir","anjay","kurang","rusak","tidak_puas","tidakpuas","ngegas","burik",
    "sampah","timbeban","tim beban","parah banget","lemot banget","server jelek"
])
NETRAL_WORDS = set([
    "hero","map","rank","tim","build","match","battle","item","mode","skill",
    "tank","mage","marksman","assassin","support","fighter","mlbb","draft",
    "push","mid","lane","turret","minion","buff","farm","jungle","ulti",
    "player","game","combo","ranked","classic","solo","timing","update",
    "event","grafik","gameplay","akun","emblem","server","patch","fps","ping"
])

NEGATIONS = set(["tidak", "bukan", "nggak", "ga", "gak", "tak", "ndak", "belum", "anti"])

# Multiword phrases that should be detected as a whole
MULTIWORD_PHRASES = {
    "tim beban": "negatif",
    "server jelek": "negatif",
    "lemot banget": "negatif",
    "bagus banget": "positif",
    "parah banget": "negatif",
    "tidak puas": "negatif",
    "mantul banget": "positif",
    "ampas banget": "negatif"
}

def detect_multiword(text):
    for phrase, label in MULTIWORD_PHRASES.items():
        if phrase in text:
            return label
    return None

def lexicon_lookup(word):
    w = word.replace(" ", "")
    if word in POSITIF_WORDS or w in POSITIF_WORDS:
        return "positif"
    if word in NEGATIF_WORDS or w in NEGATIF_WORDS:
        return "negatif"
    if word in NETRAL_WORDS or w in NETRAL_WORDS:
        return "netral"
    return None

# -------------------------
# Hybrid sentiment function (follows Colab logic)
# -------------------------
def hybrid_sentiment(text):
    """
    Return dict: {'label': label, 'score': confidence}
    Uses lexicon (with negation & multiword) + TextBlob polarity + simple voting/confidence combiner.
    """
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return {"label": "netral", "score": 0.0}

    txt = preprocess_text(text)
    # multiword detection (higher priority)
    mw = detect_multiword(txt)
    words = txt.split()
    lex_labels = []

    for i, w in enumerate(words):
        # handle simple negation: if previous word is negation, flip
        label = lexicon_lookup(w)
        if label is None:
            label = "netral"
        # flip if previous token is negation
        if i > 0 and words[i-1] in NEGATIONS:
            if label == "positif":
                label = "negatif"
            elif label == "negatif":
                label = "positif"
        lex_labels.append(label)

    if mw:
        lex_labels.append(mw)

    counts = pd.Series(lex_labels).value_counts()
    total = counts.sum()
    if total == 0:
        lex_label = "netral"
        lex_conf = 0.0
    else:
        lex_label = counts.idxmax()
        lex_conf = counts.max() / total

    # TextBlob polarity
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        blob_label = "positif"
    elif polarity < -0.05:
        blob_label = "negatif"
    else:
        blob_label = "netral"
    blob_conf = abs(polarity)

    # Combine rules (same logic as your Colab hybrid)
    if lex_label == blob_label:
        final_label = lex_label
        confidence = round((lex_conf + blob_conf) / 2, 4)
    else:
        if blob_conf > 0.25:
            final_label = blob_label
            confidence = round(blob_conf, 4)
        else:
            final_label = lex_label
            confidence = round(lex_conf, 4)

    return {"label": final_label, "score": confidence}

# -------------------------
# Main Streamlit app
# -------------------------
def main():
    st.set_page_config(page_title="Analisis Sentimen (Hybrid Naive Bayes)", layout="wide")
    st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER")
    st.markdown("Upload dataset mentah (Excel/CSV). Aplikasi akan otomatis melakukan preprocessing, pelabelan hybrid (lexicon + TextBlob), melatih Naïve Bayes (TF-IDF) untuk evaluasi, lalu menampilkan visualisasi dan hasil yang dapat diunduh.")

    uploaded = st.file_uploader("Unggah file (.xlsx atau .csv)", type=["xlsx", "csv"])
    if uploaded is None:
        st.info("Silakan unggah file Excel (.xlsx) atau CSV hasil crawling (mengandung kolom teks seperti 'full_text' / 'text').")
        return

    # read file
    try:
        if str(uploaded.type).endswith("csv") or str(uploaded.name).lower().endswith(".csv"):
            df_raw = pd.read_csv(uploaded)
        else:
            df_raw = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    # normalize columns
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # detect text column
    text_col = None
    for cand in ["stemmed_text", "clean_text", "full_text", "text", "komentar", "tweet", "ulasan"]:
        if cand in df_raw.columns:
            text_col = cand
            break
    if text_col is None:
        st.error("Tidak menemukan kolom teks. Pastikan dataset punya kolom seperti 'full_text', 'text', 'stemmed_text', atau 'komentar'.")
        return

    st.success(f"Berhasil memuat dataset — total baris: {len(df_raw)}")
    st.write("Contoh baris (preview):")
    st.dataframe(df_raw[[text_col]].head(6))

    # Preprocess: create stemmed_text if not exist
    if "stemmed_text" not in df_raw.columns:
        st.info("Membuat kolom 'stemmed_text' (preprocessing sederhana)...")
        df_raw["stemmed_text"] = df_raw[text_col].astype(str).apply(preprocess_text)
    else:
        df_raw["stemmed_text"] = df_raw["stemmed_text"].astype(str).apply(preprocess_text)

    # Apply hybrid labeling
    st.info("Melabeli data secara hybrid (lexicon + TextBlob)... (akan memakan beberapa detik untuk dataset besar)")
    with st.spinner("Melabeli..."):
        labels = []
        scores = []
        for txt in df_raw["stemmed_text"].astype(str):
            out = hybrid_sentiment(txt)
            labels.append(out["label"])
            scores.append(out["score"])
        df_raw["sentiment_label"] = labels
        df_raw["confidence_score"] = scores

    st.success("Pelabelan selesai.")
    # show distribution
    display_order = ["positif", "netral", "negatif"]
    counts = df_raw["sentiment_label"].value_counts().reindex(display_order, fill_value=0)

    st.subheader("📈 Distribusi Sentimen (hasil pelabelan)")
    counts_df = counts.reset_index()
    counts_df.columns = ["sentiment", "count"]
    st.table(counts_df)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Bar chart**")
        st.bar_chart(counts)
    with col2:
        st.markdown("**Pie chart**")
        fig_pie, ax_pie = plt.subplots(figsize=(3, 3))
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors, textprops={"fontsize": 9})
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

    # Prepare data for NB training - use temp_df like Colab where text is stemmed_text and label is sentiment_label
    temp_df = df_raw[["stemmed_text", "sentiment_label"]].dropna().copy()
    temp_df.columns = ["text", "label"]

    # Train-test split & pipeline (if at least 2 classes present)
    unique_labels = temp_df["label"].unique()
    if len(unique_labels) < 2:
        st.warning("Hanya ditemukan satu kelas pada hasil pelabelan. Tidak dapat melatih model Naïve Bayes untuk evaluasi.")
    else:
        st.info("Melatih model Naïve Bayes (TF-IDF) untuk evaluasi (train/test split 80:20)...")
        X = temp_df["text"].astype(str)
        y = temp_df["label"].astype(str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=3, max_df=0.85, sublinear_tf=True)),
            ("nb", MultinomialNB())
        ])

        with st.spinner("Melatih pipeline..."):
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        st.subheader("📊 Evaluasi Model (Naïve Bayes - TF-IDF)")
        st.write(f"- Akurasi: **{acc:.4f}**")
        st.write(f"- F1 (weighted): **{f1:.4f}**")
        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred, digits=4))

        # Confusion matrix (small)
        st.subheader("📉 Confusion Matrix")
        labels_sorted = display_order
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
        fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_sorted, yticklabels=labels_sorted, cbar=False, annot_kws={"size": 9}, ax=ax_cm)
        ax_cm.set_xlabel("Prediksi", fontsize=9)
        ax_cm.set_ylabel("Aktual", fontsize=9)
        ax_cm.set_title("Confusion Matrix", fontsize=10, pad=6)
        plt.tight_layout()
        st.pyplot(fig_cm)

    # Wordcloud per sentiment (SMALL: 400x300)
    st.subheader("☁️ WordCloud per Sentimen (ukuran kecil)")
    wc_width, wc_height = 400, 300
    cmap = {"positif": "Greens", "netral": "Purples", "negatif": "Reds"}
    for label in display_order:
        text_data = " ".join(df_raw[df_raw["sentiment_label"] == label]["stemmed_text"].astype(str).tolist())
        if text_data.strip() == "":
            st.write(f"**{label.capitalize()}** — tidak ada teks untuk wordcloud.")
            continue
        wc = WordCloud(width=wc_width, height=wc_height, background_color="white", colormap=cmap[label]).generate(text_data)
        st.write(f"**{label.capitalize()}** — total: {counts.get(label,0)}")
        fig_wc, ax_wc = plt.subplots(figsize=(4, 3))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # Download labeled results (xlsx and csv)
    st.subheader("💾 Unduh Hasil Pelabelan")
    try:
        towrite = BytesIO()
        # prefer xlsx
        with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
            df_raw.to_excel(writer, index=False, sheet_name="labelled")
        towrite.seek(0)
        st.download_button(label="Download (Excel .xlsx)", data=towrite, file_name="hasil_pelabelan.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        # fallback csv
        st.download_button(label="Download (CSV)", data=df_raw.to_csv(index=False).encode("utf-8"), file_name="hasil_pelabelan.csv", mime="text/csv")

    # also provide preview of top 20 labeled rows
    st.subheader("🔎 Contoh Hasil Pelabelan (20 baris pertama)")
    st.dataframe(df_raw[["stemmed_text", "sentiment_label", "confidence_score"]].head(20))


if __name__ == "__main__":
    main()
