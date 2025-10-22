# streamlit_sentiment_app.py — versi kecil dan rapi
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import nltk

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# ---------- Preprocessing ----------
def preprocess_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"@[\w_]+|#", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- Lexicon & Negation ----------
POSITIF = {"bagus","seru","keren","hebat","mantap","menang","gg","top","puas","asik"}
NEGATIF = {"buruk","jelek","lag","noob","toxic","kalah","lemot","sampah","ampas","marah"}
NETRAL = {"hero","game","rank","ml","tim","build","mode","player"}
NEGASI = {"tidak","bukan","ga","gak","nggak","tak"}

def hybrid_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return {"label":"netral","score":0.0}
    t = preprocess_text(text)
    words = t.split()
    lex_labels = []
    for i, w in enumerate(words):
        if w in POSITIF: label="positif"
        elif w in NEGATIF: label="negatif"
        elif w in NETRAL: label="netral"
        else: label="netral"
        if i>0 and words[i-1] in NEGASI:
            label = "negatif" if label=="positif" else "positif"
        lex_labels.append(label)
    counts = pd.Series(lex_labels).value_counts()
    lex_label = counts.idxmax() if not counts.empty else "netral"
    lex_conf = counts.max()/sum(counts) if not counts.empty else 0.0

    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    blob_label = "positif" if pol>0.05 else "negatif" if pol<-0.05 else "netral"
    blob_conf = abs(pol)
    if lex_label==blob_label:
        return {"label":lex_label,"score":round((lex_conf+blob_conf)/2,3)}
    return {"label":blob_label if blob_conf>0.25 else lex_label,"score":round(max(blob_conf,lex_conf),3)}

# ---------- MAIN APP ----------
def main():
    st.set_page_config(page_title="Analisis Sentimen ML", layout="wide")
    st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER")
    uploaded = st.file_uploader("Unggah dataset (.xlsx / .csv)", type=["xlsx","csv"])
    if not uploaded: return st.info("Silakan upload dataset")

    df = pd.read_excel(uploaded) if uploaded.name.endswith(".xlsx") else pd.read_csv(uploaded)
    df.columns = df.columns.str.lower().str.strip()
    text_col = next((c for c in ["full_text","stemmed_text","text","komentar","tweet"] if c in df.columns), None)
    if not text_col: return st.error("Kolom teks tidak ditemukan!")

    df["stemmed_text"] = df[text_col].astype(str).apply(preprocess_text)
    st.write("📄 Contoh data:", df[[text_col]].head(5))

    with st.spinner("Melabeli data..."):
        hasil = df["stemmed_text"].apply(hybrid_sentiment)
    df["sentiment_label"] = hasil.apply(lambda x: x["label"])
    df["confidence_score"] = hasil.apply(lambda x: x["score"])

    counts = df["sentiment_label"].value_counts().reindex(["positif","netral","negatif"], fill_value=0)
    st.subheader("📈 Distribusi Sentimen")
    st.write(counts)
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(counts)
    with col2:
        fig, ax = plt.subplots(figsize=(2.8,2.8))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["#2ecc71","#f1c40f","#e74c3c"], textprops={"fontsize":8})
        ax.axis("equal")
        st.pyplot(fig)

    # ---------- Naive Bayes Eval ----------
    st.subheader("📊 Evaluasi Model Naïve Bayes")
    X_train,X_test,y_train,y_test = train_test_split(df["stemmed_text"],df["sentiment_label"],test_size=0.2,random_state=42,stratify=df["sentiment_label"])
    model = Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=0.85,sublinear_tf=True)),("nb",MultinomialNB())])
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred,average="weighted")
    st.markdown(f"**Akurasi Model :** `{acc:.4f}`  \n**F1-Score :** `{f1:.4f}`")
    st.text(classification_report(y_test,y_pred))

    cm = confusion_matrix(y_test,y_pred,labels=["positif","netral","negatif"])
    fig_cm, ax_cm = plt.subplots(figsize=(3,2.5))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=["positif","netral","negatif"],yticklabels=["positif","netral","negatif"],ax=ax_cm)
    ax_cm.set_xlabel("Prediksi",fontsize=8)
    ax_cm.set_ylabel("Aktual",fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_cm)

    # ---------- WordCloud ----------
    st.subheader("☁️ WordCloud per Sentimen (Mini Size)")
    cmap = {"positif":"Greens","netral":"Purples","negatif":"Reds"}
    for lbl in ["positif","netral","negatif"]:
        teks = " ".join(df[df["sentiment_label"]==lbl]["stemmed_text"])
        if not teks.strip():
            continue
        wc = WordCloud(width=250,height=200,background_color="white",colormap=cmap[lbl]).generate(teks)
        st.markdown(f"**{lbl.capitalize()} (jumlah: {counts.get(lbl,0)})**")
        fig_wc, ax_wc = plt.subplots(figsize=(2.5,2))
        ax_wc.imshow(wc,interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

if __name__=="__main__":
    main()
