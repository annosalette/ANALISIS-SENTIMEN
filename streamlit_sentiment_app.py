# streamlit_sentiment_app.py (versi compact visual)
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
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from textblob import TextBlob
import nltk

nltk.download("punkt", quiet=True)
nltk.download("wordnet", quiet=True)

# -------------------------
# Utility / preprocessing
# -------------------------
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

# -------------------------
# Lexicon + multiword + negation
# -------------------------
POSITIF_WORDS = set([
    "bagus","seru","keren","hebat","mantap","menang","gg","top","lancar",
    "op","meta","pro","senang","suka","jago","menarik","terbaik","asik",
    "cepat","puas","kompak","support","win","mantul","legend","unggul","mantep",
    "oke","solid","bagus banget","gokil","epic","fun","stabil","menyenangkan"
])
NEGATIF_WORDS = set([
    "buruk","jelek","lag","noob","kesal","marah","toxic","parah","kalah",
    "susah","ngebug","nerf","lemot","ngehang","ngeframe","bete","down",
    "ngefreeze","lelet","gagal","ngecrash","ampas","bodoh","error","kecewa",
    "kurang","rusak","tidak puas","burik","sampah","tim beban","parah banget",
    "lemot banget","server jelek"
])
NETRAL_WORDS = set([
    "hero","map","rank","tim","build","match","battle","item","mode","skill",
    "tank","mage","marksman","assassin","support","fighter","mlbb","draft",
    "push","mid","lane","turret","minion","buff","farm","jungle","ulti",
    "player","game","combo","ranked","classic","solo","update","event",
    "grafik","gameplay","akun","emblem","server","patch","fps","ping"
])

NEGATIONS = set(["tidak","bukan","nggak","ga","gak","tak","ndak","belum"])
MULTIWORD_PHRASES = {
    "tim beban":"negatif","server jelek":"negatif","lemot banget":"negatif",
    "bagus banget":"positif","parah banget":"negatif","tidak puas":"negatif"
}

def detect_multiword(text):
    for phrase,label in MULTIWORD_PHRASES.items():
        if phrase in text:
            return label
    return None

def lexicon_lookup(word):
    if word in POSITIF_WORDS: return "positif"
    if word in NEGATIF_WORDS: return "negatif"
    if word in NETRAL_WORDS:  return "netral"
    return None

# -------------------------
# Hybrid sentiment (lexicon + blob)
# -------------------------
def hybrid_sentiment(text):
    if pd.isna(text) or not isinstance(text,str) or text.strip()=="":
        return {"label":"netral","score":0.0}
    txt = preprocess_text(text)
    mw = detect_multiword(txt)
    words = txt.split()
    lex_labels = []
    for i,w in enumerate(words):
        lbl = lexicon_lookup(w) or "netral"
        if i>0 and words[i-1] in NEGATIONS:
            if lbl=="positif": lbl="negatif"
            elif lbl=="negatif": lbl="positif"
        lex_labels.append(lbl)
    if mw: lex_labels.append(mw)
    if len(lex_labels)==0: return {"label":"netral","score":0.0}
    counts = pd.Series(lex_labels).value_counts()
    lex_label = counts.idxmax()
    lex_conf = counts.max()/len(lex_labels)
    blob = TextBlob(text)
    pol = blob.sentiment.polarity
    if pol>0.05: blob_label="positif"
    elif pol<-0.05: blob_label="negatif"
    else: blob_label="netral"
    blob_conf = abs(pol)
    if lex_label==blob_label:
        final,conf=lex_label,(lex_conf+blob_conf)/2
    elif blob_conf>0.25:
        final,conf=blob_label,blob_conf
    else:
        final,conf=lex_label,lex_conf
    return {"label":final,"score":round(conf,4)}

# -------------------------
# Streamlit main app
# -------------------------
def main():
    st.set_page_config(page_title="Analisis Sentimen",layout="wide")
    st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER MENGGUNAKAN ALGORITMA NAÏVE BAYES")
    st.caption("Upload dataset mentah (.xlsx/.csv) ")

    uploaded = st.file_uploader("Unggah file (.xlsx atau .csv)",type=["xlsx","csv"])
    if not uploaded: 
        st.info("Silakan unggah file data ulasan atau tweet.")
        return

    try:
        if uploaded.name.endswith(".csv"):
            df=pd.read_csv(uploaded)
        else:
            df=pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    df.columns=df.columns.str.lower().str.strip()
    text_col=None
    for c in ["stemmed_text","clean_text","full_text","text","komentar","tweet","ulasan"]:
        if c in df.columns: text_col=c;break
    if not text_col:
        st.error("Kolom teks tidak ditemukan! Pastikan ada 'text' / 'stemmed_text'.")
        return

    st.success(f"Dataset terbaca: {len(df)} baris.")
    st.dataframe(df[[text_col]].head(5))

    df["stemmed_text"]=df[text_col].astype(str).apply(preprocess_text)

    st.info("Melakukan pelabelan hybrid (lexicon + TextBlob)...")
    with st.spinner("Processing..."):
        hasil=[hybrid_sentiment(t) for t in df["stemmed_text"]]
    df["sentiment_label"]=[h["label"] for h in hasil]
    df["confidence_score"]=[h["score"] for h in hasil]

    display_order=["positif","netral","negatif"]
    counts=df["sentiment_label"].value_counts().reindex(display_order,fill_value=0)

    st.subheader("📈 Distribusi Sentimen")
    col1,col2=st.columns(2)
    with col1:
        st.bar_chart(counts)
    with col2:
        fig,ax=plt.subplots(figsize=(2.5,2.5))
        ax.pie(counts,labels=counts.index,autopct="%1.1f%%",colors=["#2ecc71","#f1c40f","#e74c3c"],startangle=90,textprops={"fontsize":8})
        ax.axis("equal")
        st.pyplot(fig)

    # Training NB
    temp=df[["stemmed_text","sentiment_label"]].dropna()
    if len(temp["sentiment_label"].unique())>=2:
        X_train,X_test,y_train,y_test=train_test_split(temp["stemmed_text"],temp["sentiment_label"],test_size=0.2,random_state=42,stratify=temp["sentiment_label"])
        pipe=Pipeline([("tfidf",TfidfVectorizer(ngram_range=(1,2),min_df=3,max_df=0.85,sublinear_tf=True)),("nb",MultinomialNB())])
        pipe.fit(X_train,y_train)
        y_pred=pipe.predict(X_test)
        acc=accuracy_score(y_test,y_pred)
        st.subheader("📊 Evaluasi Model Naïve Bayes (TF-IDF)")
        st.write(f"Akurasi: **{acc:.4f}**")
        cm=confusion_matrix(y_test,y_pred,labels=display_order)
        fig_cm,ax_cm=plt.subplots(figsize=(2.5,2.5))
        sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=display_order,yticklabels=display_order,cbar=False,ax=ax_cm)
        ax_cm.set_xlabel("Prediksi",fontsize=8)
        ax_cm.set_ylabel("Aktual",fontsize=8)
        st.pyplot(fig_cm)
    else:
        st.warning("Data hanya berisi satu kelas, model tidak dapat dilatih.")

    # Wordcloud kecil
    st.subheader("☁️ WordCloud per Sentimen")
    cmap={"positif":"Greens","netral":"Purples","negatif":"Reds"}
    for s in display_order:
        txt=" ".join(df[df["sentiment_label"]==s]["stemmed_text"])
        if not txt.strip(): continue
        wc=WordCloud(width=300,height=200,background_color="white",colormap=cmap[s]).generate(txt)
        fig,ax=plt.subplots(figsize=(3,2))
        ax.imshow(wc,interpolation="bilinear")
        ax.axis("off")
        st.write(f"**{s.capitalize()}** — total: {counts[s]}")
        st.pyplot(fig)

    # Tabel hasil
    st.subheader("📄 Contoh 20 Baris")
    st.dataframe(df[["stemmed_text","sentiment_label","confidence_score"]].head(20))

if __name__=="__main__":
    main()
