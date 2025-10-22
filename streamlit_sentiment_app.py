# streamlit_sentiment_app.py — versi ULTRA COMPACT
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
from sklearn.metrics import accuracy_score, confusion_matrix
from textblob import TextBlob
import nltk

nltk.download("punkt", quiet=True)

# -------------------------
# Text cleaning
# -------------------------
def preprocess_text(text):
    if pd.isna(text): return ""
    s = str(text).lower()
    s = re.sub(r"http\S+|www\S+|https\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# -------------------------
# Lexicon dictionaries
# -------------------------
POSITIF = {"bagus","keren","hebat","mantap","menang","gg","top","senang","suka","terbaik","asik","cepat","puas","legend"}
NEGATIF = {"buruk","jelek","lag","noob","marah","toxic","kalah","susah","ngebug","lemot","ngehang","kecewa","ampas","bodoh","rusak"}
NETRAL  = {"hero","map","rank","tim","mode","player","game","update","event","server"}

NEGATIONS = {"tidak","bukan","gak","ga","tak","belum"}
MULTI = {"tim beban":"negatif","server jelek":"negatif","bagus banget":"positif"}

def lexicon(word):
    if word in POSITIF: return "positif"
    if word in NEGATIF: return "negatif"
    if word in NETRAL:  return "netral"
    return "netral"

def detect_multi(txt):
    for k,v in MULTI.items():
        if k in txt: return v
    return None

def hybrid_sentiment(text):
    if not text: return {"label":"netral","score":0.0}
    txt = preprocess_text(text)
    mw = detect_multi(txt)
    words = txt.split()
    labs = []
    for i,w in enumerate(words):
        lbl = lexicon(w)
        if i>0 and words[i-1] in NEGATIONS:
            if lbl=="positif": lbl="negatif"
            elif lbl=="negatif": lbl="positif"
        labs.append(lbl)
    if mw: labs.append(mw)
    from collections import Counter
    c = Counter(labs)
    lex_lbl = max(c,key=c.get)
    lex_conf = c[lex_lbl]/sum(c.values())
    pol = TextBlob(text).sentiment.polarity
    blob_lbl = "positif" if pol>0.05 else "negatif" if pol<-0.05 else "netral"
    blob_conf = abs(pol)
    if lex_lbl==blob_lbl: return {"label":lex_lbl,"score":round((lex_conf+blob_conf)/2,3)}
    return {"label":blob_lbl if blob_conf>0.25 else lex_lbl,"score":round(max(lex_conf,blob_conf),3)}

# -------------------------
# Streamlit App
# -------------------------
def main():
    st.set_page_config(page_title="Analisis Sentimen ML", layout="wide")
    st.title("📊 Analisis Sentimen Pengguna Mobile Legends (Hybrid + Naive Bayes)")
    uploaded = st.file_uploader("Unggah dataset (.xlsx/.csv)", type=["xlsx","csv"])
    if not uploaded: 
        st.info("Silakan unggah file data ulasan.")
        return

    # load
    df = pd.read_excel(uploaded) if uploaded.name.endswith("xlsx") else pd.read_csv(uploaded)
    df.columns = df.columns.str.lower().str.strip()
    text_col = next((c for c in ["text","full_text","stemmed_text","komentar","tweet","ulasan"] if c in df.columns), None)
    if not text_col:
        st.error("Tidak ditemukan kolom teks.")
        return

    df["stemmed_text"] = df[text_col].astype(str).apply(preprocess_text)

    st.info("Melabeli data...")
    out = [hybrid_sentiment(t) for t in df["stemmed_text"]]
    df["sentiment_label"] = [x["label"] for x in out]
    df["confidence_score"] = [x["score"] for x in out]

    order = ["positif","netral","negatif"]
    counts = df["sentiment_label"].value_counts().reindex(order,fill_value=0)

    st.subheader("📈 Distribusi Sentimen")
    col1, col2 = st.columns(2)
    with col1:
        st.bar_chart(counts)
    with col2:
        fig, ax = plt.subplots(figsize=(2,2))
        ax.pie(counts, labels=counts.index, autopct="%1.0f%%", startangle=90,
               colors=["#2ecc71","#f1c40f","#e74c3c"], textprops={"fontsize":7})
        ax.axis("equal")
        st.pyplot(fig)

    # Naive Bayes evaluation
    temp = df[["stemmed_text","sentiment_label"]].dropna()
    if len(temp["sentiment_label"].unique())>=2:
        X_train,X_test,y_train,y_test=train_test_split(temp["stemmed_text"],temp["sentiment_label"],test_size=0.2,random_state=42,stratify=temp["sentiment_label"])
        pipe = Pipeline([("tfidf",TfidfVectorizer(min_df=3,max_df=0.85,ngram_range=(1,2))),("nb",MultinomialNB())])
        pipe.fit(X_train,y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        st.subheader(f"📊 Evaluasi Model (Akurasi: {acc:.3f})")
        cm = confusion_matrix(y_test,y_pred,labels=order)
        fig2, ax2 = plt.subplots(figsize=(2,2))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=order,yticklabels=order,cbar=False,ax=ax2,annot_kws={"size":7})
        ax2.set_xlabel("Prediksi",fontsize=7)
        ax2.set_ylabel("Aktual",fontsize=7)
        st.pyplot(fig2)
    else:
        st.warning("Data hanya 1 kelas, tidak dapat dievaluasi.")

    # Small WordCloud
    st.subheader("☁️ WordCloud per Sentimen")
    cmap = {"positif":"Greens","netral":"Purples","negatif":"Reds"}
    for s in order:
        txt = " ".join(df[df["sentiment_label"]==s]["stemmed_text"])
        if not txt.strip(): continue
        wc = WordCloud(width=250, height=150, background_color="white", colormap=cmap[s]).generate(txt)
        fig, ax = plt.subplots(figsize=(2.5,1.8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.write(f"**{s.capitalize()}** — total: {counts[s]}")
        st.pyplot(fig)

    st.subheader("📄 Contoh Hasil (10 Baris)")
    st.dataframe(df[["stemmed_text","sentiment_label","confidence_score"]].head(10))

if __name__ == "__main__":
    main()
