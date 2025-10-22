# streamlit_sentiment_app.py
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from collections import Counter

# -------------------------
# Preprocessing Text
# -------------------------
def preprocess_text(text):
    if pd.isna(text):
        return ""
    s = str(text).lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"@[\w_]+|#", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------
# Lexicon + Negation
# -------------------------
POSITIF = {
    "bagus", "seru", "keren", "hebat", "mantap", "menang", "gg", "top",
    "lancar", "suka", "jago", "menarik", "terbaik", "asik", "puas", "oke", "legend", "unggul"
}
NEGATIF = {
    "buruk", "jelek", "lag", "noob", "toxic", "kalah", "lemot", "kecewa",
    "marah", "bete", "down", "ngehang", "ngeframe", "error", "ampas", "sampah", "parah"
}
NETRAL = {
    "hero", "map", "tim", "build", "match", "battle", "item", "mode", "skill",
    "tank", "mage", "game", "player", "update", "event", "grafik"
}
NEGASI = {"tidak", "bukan", "nggak", "ga", "gak", "tak", "ndak", "belum"}
MULTIWORD = {
    "tim beban": "negatif",
    "server jelek": "negatif",
    "lemot banget": "negatif",
    "bagus banget": "positif"
}

def detect_multiword(text):
    for phrase, label in MULTIWORD.items():
        if phrase in text:
            return label
    return None

def get_lexicon_label(word):
    if word in POSITIF:
        return "positif"
    if word in NEGATIF:
        return "negatif"
    if word in NETRAL:
        return "netral"
    return "netral"

# -------------------------
# Hybrid Sentiment Function
# -------------------------
def hybrid_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return {"label": "netral", "score": 0.0}

    txt = preprocess_text(text)
    mw = detect_multiword(txt)
    words = txt.split()
    labels = []

    for i, w in enumerate(words):
        label = get_lexicon_label(w)
        if i > 0 and words[i - 1] in NEGASI:
            if label == "positif":
                label = "negatif"
            elif label == "negatif":
                label = "positif"
        labels.append(label)

    if mw:
        labels.append(mw)

    counts = Counter(labels)
    if len(counts) == 0:
        lex_label = "netral"
        lex_conf = 0.0
    else:
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
# Streamlit main function
# -------------------------
def main():
    st.title("📊 Analisis Sentimen Pengguna Mobile Legends ")
    st.write("Upload dataset mentah (.xlsx / .csv).")

    uploaded = st.file_uploader("Unggah file (.xlsx atau .csv)", type=["xlsx", "csv"])
    if uploaded is None:
        st.info("Silakan unggah file dataset.")
        return

    # load file
    try:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    df.columns = df.columns.str.strip().str.lower()
    text_col = None
    for c in ["stemmed_text", "clean_text", "full_text", "text", "komentar", "tweet", "ulasan"]:
        if c in df.columns:
            text_col = c
            break

    if text_col is None:
        st.error("Kolom teks tidak ditemukan. Pastikan ada kolom 'full_text' / 'text' / 'stemmed_text'.")
        return

    st.success(f"Dataset dimuat: {len(df)} baris.")
    st.dataframe(df[[text_col]].head(6))

    # preprocess/create stemmed_text
    if "stemmed_text" not in df.columns:
        df["stemmed_text"] = df[text_col].astype(str).apply(preprocess_text)
    else:
        df["stemmed_text"] = df["stemmed_text"].astype(str).apply(preprocess_text)

    # hybrid labeling
    st.info("Melabeli data (hybrid lexicon + TextBlob)...")
    with st.spinner("Melabeli..."):
        outs = df["stemmed_text"].astype(str).apply(hybrid_sentiment)
    df["sentiment_label"] = outs.apply(lambda x: x["label"])
    df["confidence_score"] = outs.apply(lambda x: x["score"])

    # distribution
    st.subheader("📈 Distribusi Sentimen")
    order = ["positif", "netral", "negatif"]
    counts = df["sentiment_label"].value_counts().reindex(order, fill_value=0)
    st.write(f"Total: {int(counts.sum())}  |  Positif: {counts['positif']}  |  Netral: {counts['netral']}  |  Negatif: {counts['negatif']}")

    col1, col2 = st.columns([1, 1])
    with col1:
        fig_bar, ax_bar = plt.subplots(figsize=(3, 2.5))
        ax_bar.bar(order, counts.values, color=["#2ecc71", "#f1c40f", "#e74c3c"])
        for i, v in enumerate(counts.values):
            ax_bar.text(i, v + max(1, int(max(counts.values)*0.02)), str(int(v)), ha="center", fontsize=8)
        ax_bar.set_title("Distribusi Sentimen", fontsize=9)
        st.pyplot(fig_bar)
    with col2:
        fig_pie, ax_pie = plt.subplots(figsize=(2.2, 2.2))
        ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#2ecc71", "#f1c40f", "#e74c3c"], textprops={"fontsize": 8})
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

    # prepare for training & evaluation
    temp = df[["stemmed_text", "sentiment_label"]].dropna()
    if temp["sentiment_label"].nunique() < 2:
        st.warning("Hanya satu kelas ditemukan, tidak dapat melakukan pelatihan/evaluasi model.")
        return

    st.info("Melatih model Naïve Bayes (TF-IDF) untuk evaluasi (train/test split 80:20)...")
    X = temp["stemmed_text"].astype(str)
    y = temp["sentiment_label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=3, max_df=0.85, sublinear_tf=True)),
        ("nb", MultinomialNB())
    ])

    with st.spinner("Melatih pipeline..."):
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    st.subheader("📊 Evaluasi Model")
    st.write(f"**Akurasi:** {acc:.4f}")
    st.write(f"**F1-score (weighted):** {f1:.4f}")
    st.text("Classification report:")
    st.text(classification_report(y_test, y_pred, digits=4))

    # confusion matrix (compact)
    st.subheader("📉 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(2.2, 2.2))
    cm = confusion_matrix(y_test, y_pred, labels=order)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=order, yticklabels=order, cbar=False, annot_kws={"size": 7}, ax=ax_cm)
    ax_cm.set_xlabel("Prediksi", fontsize=7)
    ax_cm.set_ylabel("Aktual", fontsize=7)
    st.pyplot(fig_cm)

    # small wordclouds
    st.subheader("☁️ WordCloud per Sentimen (compact)")
    cmap = {"positif": "Greens", "netral": "Purples", "negatif": "Reds"}
    for label in order:
        text_data = " ".join(df[df["sentiment_label"] == label]["stemmed_text"].astype(str).tolist())
        if not text_data.strip():
            st.write(f"**{label.capitalize()}** — tidak ada teks.")
            continue
        wc = WordCloud(width=250, height=150, background_color="white", colormap=cmap[label]).generate(text_data)
        st.markdown(f"**{label.capitalize()}** — total: {int(counts[label])}")
        fig_wc, ax_wc = plt.subplots(figsize=(2.6, 1.6))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # preview
    st.subheader("🔎 Contoh Hasil Pelabelan (10 baris)")
    st.dataframe(df[["stemmed_text", "sentiment_label", "confidence_score"]].head(10))

# allow import + call from app.py
if __name__ == "__main__":
    main()
