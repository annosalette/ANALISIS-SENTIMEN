# ===============================================================
# streamlit_sentiment_app.py
# TF-IDF + MultinomialNB Streamlit app (mirror dari Colab notebook)
# ===============================================================

import streamlit as st
import pandas as pd
import re
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

# ---------------------------
# 1) Judul
# ---------------------------
def main():
    st.set_page_config(page_title="Analisis Sentimen (Naïve Bayes TF-IDF)", layout="wide")
    st.title("📊ANALISIS SENTIMEN PENGGUNA MOBILE LEGENDS DI TWITTER MENGGUNAKAN ALGORITMA NAÏVE BAYES")
    st.write(
      
    )

    # ---------------------------
    # 2) Kamus kata (untuk auto-labeling jika perlu)
    #    (gunakan lexicon yang sama seperti di notebook)
    # ---------------------------
    positif_words = {
        "bagus","seru","keren","hebat","mantap","menang","gg","top","lancar",
        "op","meta","pro","senang","suka","jago","menarik","terbaik","asik",
        "cepat","puas","kompak","support","win","mantul","legend","unggul","mantep",
        "oke","solid","bagus banget","gokil","epic","fun","stabil","menyenangkan"
    }
    negatif_words = {
        "buruk","jelek","lag","noob","kesal","marah","toxic","parah","kalah",
        "susah","ngebug","nerf","lemot","ngehang","ngeframe","bete","down",
        "ngefreeze","lelet","gagal","ngecrash","ampas","bodoh","ngelek","error",
        "kecewa","anjir","anjay","kurang","rusak","tidak puas","ngegas","burik",
        "sampah","tim beban","parah banget","lemot banget","server jelek"
    }
    netral_words = {
        "hero","map","rank","tim","build","match","battle","item","mode","skill",
        "tank","mage","marksman","assassin","support","fighter","mlbb","draft",
        "push","mid","lane","turret","minion","buff","farm","jungle","ulti",
        "player","game","combo","ranked","classic","solo","timing","update",
        "event","grafik","gameplay","akun","emblem","server","patch","fps","ping"
    }

    def lexicon_label(word):
        """Return 'positif'|'negatif'|'netral' based on lists (default netral)."""
        w = word.lower()
        if w in positif_words:
            return "positif"
        if w in negatif_words:
            return "negatif"
        return "netral"

    # ---------------------------
    # 3) Upload file
    # ---------------------------
    st.subheader("📂 Upload Dataset (.xlsx) — pastikan ada kolom 'stemmed_text' (atau 'clean_text')")
    uploaded = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

    if not uploaded:
        st.info("Silakan unggah file .xlsx yang berisi kolom 'stemmed_text' (atau 'clean_text').")
        return

    # load
    try:
        df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # find text column
    text_col = None
    for c in ["stemmed_text", "clean_text", "full_text", "text", "komentar"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        st.error("Kolom teks tidak ditemukan. Pastikan dataset memiliki 'stemmed_text' atau 'clean_text' atau 'full_text'.")
        return

    st.success(f"Dataset dimuat — total baris: {len(df)}")
    st.dataframe(df.head(8))

    # ---------------------------
    # 4) jika tidak ada label, auto-labeling dengan kamus
    # ---------------------------
    label_col = None
    for c in ["sentiment_label", "klasifikasi", "label"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is None:
        st.info("Tidak menemukan kolom label. Melakukan auto-labeling berbasis kamus kata...")
        labels = []
        for txt in df[text_col].fillna("").astype(str):
            words = re.findall(r"\b[a-zA-Z]+\b", txt.lower())
            tags = [lexicon_label(w) for w in words]
            if "positif" in tags:
                labels.append("positif")
            elif "negatif" in tags:
                labels.append("negatif")
            else:
                labels.append("netral")
        df["sentiment_label"] = labels
        label_col = "sentiment_label"
        st.success("Auto-labeling selesai.")
    else:
        # normalize label column name
        df = df.rename(columns={label_col: "sentiment_label"})
        label_col = "sentiment_label"

    # ensure text column used for training — create cleaned column for vectorizer
    df["text_for_model"] = df[text_col].astype(str)

    # drop NaNs in label/text for training
    df = df.dropna(subset=["text_for_model", "sentiment_label"]).reset_index(drop=True)
    if len(df) == 0:
        st.error("Tidak ada data setelah pembersihan. Periksa dataset.")
        return

    st.markdown("**Ringkasan label (sample 8 baris):**")
    st.dataframe(df[[text_col, "sentiment_label"]].head(8))

    # ---------------------------
    # 5) Train-test split & pipeline (TF-IDF + MultinomialNB)
    #    TF-IDF configuration same as notebook: ngram (1,2), min_df=3, max_df=0.85, sublinear_tf=True
    # ---------------------------
    X = df["text_for_model"].astype(str)
    y = df["sentiment_label"].astype(str)

    # ensure stratify is possible: if only one class present, stratify can't be used
    unique_classes = y.unique()
    if len(unique_classes) < 2:
        st.warning("Dataset hanya berisi 1 kelas. Tidak dapat melakukan split stratified. Akan menampilkan statistik saja.")
        do_train = False
    else:
        do_train = True

    if do_train:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1,2),
                min_df=3,
                max_df=0.85,
                sublinear_tf=True
            )),
            ("nb", MultinomialNB())
        ])

        with st.spinner("Melatih model Naïve Bayes (TF-IDF)..."):
            pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        # metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.subheader("📊 Evaluasi Model (Train/Test Split)")
        st.write(f"- **Akurasi:** {acc:.4f}")
        st.write(f"- **F1 (weighted):** {f1:.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred, digits=4))

        # ---------------------------
        # 6) Confusion matrix — small figure
        # ---------------------------
        st.subheader("📉 Confusion Matrix")
        labels_sorted = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

        fig_cm, ax_cm = plt.subplots(figsize=(3.5, 3))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_sorted, yticklabels=labels_sorted,
            cbar=False, annot_kws={"size": 9}, ax=ax_cm
        )
        ax_cm.set_xlabel("Prediksi", fontsize=9)
        ax_cm.set_ylabel("Aktual", fontsize=9)
        ax_cm.set_title("Confusion Matrix", fontsize=10, pad=6)
        plt.tight_layout()
        st.pyplot(fig_cm)

    else:
        st.info("Lewati pelatihan model karena dataset hanya memiliki satu kelas. Menampilkan ringkasan distribusi saja.")

    # ---------------------------
    # 7) Prediksi seluruh dataset (pakai pipeline jika sudah dilatih)
    # ---------------------------
    if do_train:
        df["predicted_label"] = pipeline.predict(X)
    else:
        # fallback: predicted = label (since no model)
        df["predicted_label"] = df["sentiment_label"]

    # normalize order of labels for display
    display_order = ["positif", "netral", "negatif"]
    counts = df["predicted_label"].value_counts().reindex(display_order, fill_value=0)

    # ---------------------------
    # 8) Distribusi — Bar + Pie + jumlah tabel
    # ---------------------------
    st.subheader("📈 Distribusi Sentimen (prediksi pada seluruh dataset)")
    # show dataframe of counts (compact)
    counts_df = counts.reset_index()
    counts_df.columns = ["sentiment", "count"]
    st.table(counts_df)  # small table with counts

    # bar + pie side-by-side
    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("**Bar chart**")
        st.bar_chart(counts)
    with col2:
        st.markdown("**Pie chart**")
        fig_pie, ax_pie = plt.subplots(figsize=(3,3))
        colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
        ax_pie.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors, textprops={"fontsize":9})
        ax_pie.axis("equal")
        st.pyplot(fig_pie)

    # ---------------------------
    # 9) Wordcloud per kelas (small size)
    # ---------------------------
    st.subheader("☁️ Wordcloud per Sentimen (prediksi)")
    wc_sizes = {"figsize": (4,2.5), "width":400, "height":250}
    cmap = {"positif":"Greens", "netral":"Purples", "negatif":"Reds"}
    for label in display_order:
        text_data = " ".join(df[df["predicted_label"] == label]["text_for_model"].astype(str))
        if len(text_data.strip()) == 0:
            continue
        wc = WordCloud(width=wc_sizes["width"], height=wc_sizes["height"], background_color="white", colormap=cmap[label]).generate(text_data)
        st.write(f"**{label.capitalize()}** — total: {counts.loc[label]} data")
        fig_wc, ax_wc = plt.subplots(figsize=wc_sizes["figsize"])
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)

    # ---------------------------
    # 10) Download hasil
    # ---------------------------
    st.subheader("💾 Unduh Hasil")
    result_cols = [text_col, "sentiment_label", "predicted_label"]
    # if pipeline trained, include model details (not saving model file here)
    st.download_button(
        label="Download CSV hasil (text,label,predicted)",
        data=df[result_cols].to_csv(index=False).encode("utf-8"),
        file_name="hasil_sentimen_streamlit.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
