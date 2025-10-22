# ===========================================
# streamlit_sentiment_app.py (FINAL)
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
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
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
# Streamlit Main
# -------------------------
def main():
    st.title("📊 ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER MENGGUNAKAN ALGORITMA NAÏVE BAYES ")
    st.write("Upload dataset mentah (Excel/CSV).")

    file = st.file_uploader("Unggah file dataset (.xlsx / .csv)", type=["xlsx", "csv"])
    if file is None:
        st.info("Silakan unggah dataset terlebih dahulu.")
        return

    # Load dataset
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

    # Labeling
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
        fig, ax = plt.subplots(figsize=(2.8, 2.8))
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90,
               colors=["#2ecc71", "#f1c40f", "#e74c3c"], textprops={"fontsize": 8})
        st.pyplot(fig)

    # Train/Test Naive Bayes
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
        st.write(f"**Akurasi:** {acc:.4f}")
        st.write(f"**F1-Score (weighted):** {f1:.4f}")

        cm = confusion_matrix(y_test, y_pred, labels=order)
        fig_cm, ax_cm = plt.subplots(figsize=(3, 2.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=order, yticklabels=order, cbar=False, annot_kws={"size": 8})
        ax_cm.set_xlabel("Prediksi", fontsize=8)
        ax_cm.set_ylabel("Aktual", fontsize=8)
        ax_cm.set_title("Confusion Matrix", fontsize=9)
        st.pyplot(fig_cm)

    # WordCloud Kecil
    st.subheader("☁️ WordCloud Sentimen")
    for lbl, color in zip(order, ["Greens", "Purples", "Reds"]):
        text_data = " ".join(df[df["sentiment_label"] == lbl]["stemmed_text"])
        if not text_data.strip():
            continue
        wc = WordCloud(width=250, height=200, background_color="white", colormap=color).generate(text_data)
        st.markdown(f"**{lbl.capitalize()}** ({counts[lbl]} data)")
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig)

    # Preview
    st.subheader("🔍 Contoh Hasil Pelabelan")
    st.dataframe(df[["stemmed_text", "sentiment_label", "confidence_score"]].head(20))

if __name__ == "__main__":
    main()
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

# ========== Preprocessing ==========

def preprocess_text(text):
if pd.isna(text):
return ""
s = str(text).lower()
s = re.sub(r"http\S+|www\S+", " ", s)
s = re.sub(r"@[\w_]+|#", " ", s)
s = re.sub(r"[^a-z0-9\s']", " ", s)
s = re.sub(r"\s+", " ", s).strip()
return s

# ========== Lexicon ==========

POSITIF = {"bagus","seru","keren","mantap","hebat","menang","gg","lancar","senang","suka","terbaik","puas","mantul"}
NEGATIF = {"buruk","jelek","lag","noob","kesal","marah","toxic","kalah","lemot","ngehang","ngeframe","ngelek","ampas","kecewa","error","sampah"}
NETRAL = {"hero","tim","build","rank","item","mode","skill","player","game","event","update"}
NEGATIONS = {"tidak","bukan","nggak","ga","gak","tak","ndak","belum"}

def hybrid_sentiment(text):
if not isinstance(text, str) or not text.strip():
return {"label": "netral", "score": 0.0}
txt = preprocess_text(text)
words = txt.split()
lex_labels = []
for i, w in enumerate(words):
if w in POSITIF:
label = "positif"
elif w in NEGATIF:
label = "negatif"
elif w in NETRAL:
label = "netral"
else:
label = "netral"
if i > 0 and words[i-1] in NEGATIONS:
if label == "positif": label = "negatif"
elif label == "negatif": label = "positif"
lex_labels.append(label)
counts = pd.Series(lex_labels).value_counts()
lex_label = counts.idxmax()
lex_conf = counts.max()/len(lex_labels)

```
blob = TextBlob(text)
pol = blob.sentiment.polarity
blob_label = "positif" if pol>0.05 else "negatif" if pol<-0.05 else "netral"
blob_conf = abs(pol)

if lex_label==blob_label:
    return {"label":lex_label,"score":round((lex_conf+blob_conf)/2,3)}
return {"label":blob_label if blob_conf>0.25 else lex_label,"score":round(max(lex_conf,blob_conf),3)}
```

# ========== Main App ==========

def main():
st.set_page_config(page_title="Analisis Sentimen", layout="wide")
st.title("📊 Analisis Sentimen Pengguna Mobile Legends (Hybrid + Naïve Bayes)")

```
uploaded = st.file_uploader("Unggah dataset mentah (.xlsx/.csv)", type=["xlsx","csv"])
if not uploaded:
    st.info("Silakan unggah dataset mentah (contoh: hasil crawling Twitter).")
    return

try:
    df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
except Exception as e:
    st.error(f"Gagal membaca file: {e}")
    return

df.columns = df.columns.str.lower().str.strip()
text_col = next((c for c in ["stemmed_text","full_text","text","komentar","tweet"] if c in df.columns), None)
if not text_col:
    st.error("Kolom teks tidak ditemukan! Pastikan ada kolom seperti 'text' atau 'full_text'.")
    return

df["stemmed_text"] = df[text_col].astype(str).apply(preprocess_text)
st.success(f"Dataset berhasil dimuat — total {len(df)} data")
st.dataframe(df.head(5))

# ---------- Labeling ----------
st.info("Melabeli data secara hybrid (lexicon + TextBlob)...")
with st.spinner("Proses pelabelan..."):
    result = df["stemmed_text"].apply(hybrid_sentiment)
df["sentiment_label"] = result.apply(lambda x: x["label"])
df["confidence_score"] = result.apply(lambda x: x["score"])

# ---------- Distribusi ----------
st.subheader("📊 Distribusi Sentimen (Hasil Pelabelan)")
display_order = ["positif", "netral", "negatif"]
counts = df["sentiment_label"].value_counts().reindex(display_order, fill_value=0)
total = int(counts.sum())
st.write(f"**Total Data:** {total} | Positif: {counts['positif']} | Netral: {counts['netral']} | Negatif: {counts['negatif']}")

col1, col2 = st.columns([1,1])
with col1:
    fig_bar, ax = plt.subplots(figsize=(3,2.5))
    ax.bar(counts.index, counts.values, color=['#2ecc71','#f1c40f','#e74c3c'])
    for i,v in enumerate(counts.values): ax.text(i,v+3,str(v),ha='center',fontsize=8)
    ax.set_title("Distribusi Sentimen",fontsize=9)
    st.pyplot(fig_bar)
with col2:
    fig_pie, ax = plt.subplots(figsize=(2.5,2.5))
    ax.pie(counts,labels=counts.index,autopct="%1.1f%%",colors=['#2ecc71','#f1c40f','#e74c3c'],textprops={'fontsize':8})
    ax.axis("equal"); st.pyplot(fig_pie)

# ---------- Training Naïve Bayes ----------
if df["sentiment_label"].nunique()<2:
    st.warning("Hanya satu label, tidak bisa evaluasi model.")
    return

st.info("Melatih model Naïve Bayes (TF-IDF)...")
X_train, X_test, y_train, y_test = train_test_split(df["stemmed_text"], df["sentiment_label"], test_size=0.2, stratify=df["sentiment_label"], random_state=42)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True,ngram_range=(1,2),min_df=3,max_df=0.85,sublinear_tf=True)),
    ("nb", MultinomialNB())
])
pipe.fit(X_train,y_train)
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred,average="weighted")

st.subheader("📈 Hasil Evaluasi Model (Naïve Bayes - TF-IDF)")
st.markdown(f"**🔹 Akurasi Model :** {acc:.4f}  \n**🔹 F1-Score      :** {f1:.4f}")
st.text("Laporan Klasifikasi:")
st.text(classification_report(y_test,y_pred,digits=4))

st.subheader("📉 Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(3,2.5))
cm = confusion_matrix(y_test,y_pred,labels=display_order)
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=display_order,yticklabels=display_order,cbar=False,ax=ax_cm)
ax_cm.set_xlabel("Prediksi",fontsize=8)
ax_cm.set_ylabel("Aktual",fontsize=8)
plt.tight_layout(); st.pyplot(fig_cm)

# ---------- WordCloud ----------
st.subheader("☁️ WordCloud per Sentimen (Kecil)")
for label in display_order:
    text_data = " ".join(df[df["sentiment_label"]==label]["stemmed_text"])
    if not text_data.strip(): continue
    wc = WordCloud(width=250,height=180,background_color="white",colormap={"positif":"Greens","netral":"Purples","negatif":"Reds"}[label]).generate(text_data)
    st.markdown(f"**{label.capitalize()}** (total: {counts[label]})")
    fig_wc, ax_wc = plt.subplots(figsize=(2.8,1.8))
    ax_wc.imshow(wc,interpolation="bilinear"); ax_wc.axis("off"); st.pyplot(fig_wc)

# ---------- Preview ----------
st.subheader("🔎 Contoh 10 Data Berlabel")
st.dataframe(df[["stemmed_text","sentiment_label","confidence_score"]].head(10))
```

if **name**=="**main**":
main()
