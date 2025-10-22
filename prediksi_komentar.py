# ======================= 
# prediksi_komentar.py 
# =======================

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def main():
    # =======================
    # Konfigurasi halaman
    # =======================
    st.set_page_config(
        page_title="Sentiment Analysis Mobile Legends",
        page_icon="🎮"
    )

    # =======================
    # Judul aplikasi
    # =======================
    st.title("🎮 Sentiment Analysis Komentar Mobile Legends")
    @st.cache_resource(show_spinner=True)
    def load_model():
        try:
            pretrained = "mdhugol/indonesia-bert-sentiment-classification"
            model = AutoModelForSequenceClassification.from_pretrained(pretrained)
            tokenizer = AutoTokenizer.from_pretrained(pretrained)
            return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        except Exception as e:
            st.error(f"Gagal memuat model IndoBERT: {e}")
            return None

    sentiment_pipeline = load_model()

    if sentiment_pipeline is None:
        st.stop()

    label_index = {
        'LABEL_0': 'Positif',
        'LABEL_1': 'Netral',
        'LABEL_2': 'Negatif'
    }

    # =======================
    # Input komentar
    # =======================
    user_text = st.text_area("Masukkan komentar di sini:")

    if st.button("Prediksi Sentimen"):
        if user_text.strip() != "":
            try:
                # Prediksi sentimen
                result = sentiment_pipeline(user_text)
                label = label_index.get(result[0]['label'], result[0]['label'])
                score = result[0]['score']

                # =======================
                # Tampilkan hasil
                # =======================
                st.subheader("📊 Hasil Prediksi")
                st.write(f"**Komentar:** {user_text}")
                st.write(f"**Sentimen:** {label}")
                st.write(f"**Confidence:** {score * 100:.2f}%")

                # Pesan sesuai sentimen
                if label == "Positif":
                    st.success("Komentar ini bernada **positif** 🎉")
                elif label == "Netral":
                    st.info("Komentar ini bersifat **netral** 😐")
                else:
                    st.error("Komentar ini bernada **negatif** 😠")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses prediksi: {e}")
        else:
            st.warning("⚠️ Silakan masukkan komentar terlebih dahulu.")

if __name__ == "__main__":
    main()
