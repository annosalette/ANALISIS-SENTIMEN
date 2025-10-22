# ===============================
# app.py — Streamlit Main App
# ===============================
import sys, os
import streamlit as st

# Pastikan direktori kerja aktif dikenali (agar impor tidak error di Streamlit Cloud)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import dua modul fitur
import prediksi_komentar
import streamlit_sentiment_app

# ===============================
# Konfigurasi dasar aplikasi
# ===============================
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# Sidebar Navigation
# ===============================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3330/3330314.png", width=80)
    st.title("📂 Navigasi Utama")

    menu = st.radio(
        "Pilih Halaman:",
        ("🏠 Beranda", "🔍 Prediksi Komentar", "📈 Analisis Sentimen"),
        index=0
    )

# ===============================
# Halaman: Beranda
# ===============================
if menu == "🏠 Beranda":
    st.markdown(
        """
        # 💬 Aplikasi Analisis Sentimen
        Aplikasi ini dibuat oleh **Septiano Dwiyanto Salette (22.22.2647)**  
        sebagai syarat untuk kelulusan **Sidang Skripsi**.

        ---
        ### 🧠 Fitur Aplikasi:
        - **🔍 Prediksi Komentar:**  
          Untuk memprediksi sentimen (positif, netral, negatif) dari komentar pengguna secara otomatis.  
        - **📈 Analisis Sentimen:**  
          Untuk melakukan analisis lebih mendalam, visualisasi hasil, dan evaluasi model Machine Learning.
        """
    )

# ===============================
# Halaman: Prediksi Komentar
# ===============================
elif menu == "🔍 Prediksi Komentar":
    st.markdown("<h2 style='color:#2196F3;'>🔍 Prediksi Komentar</h2>", unsafe_allow_html=True)
    st.info("Masukkan komentar pengguna Mobile Legends dan dapatkan hasil prediksi sentimennya.")
    try:
        prediksi_komentar.main()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat fitur Prediksi Komentar:\n\n{e}")

# ===============================
# Halaman: Analisis Sentimen
# ===============================
elif menu == "📈 Analisis Sentimen":
    st.markdown("<h2 style='color:#4CAF50;'>📈 Analisis Sentimen</h2>", unsafe_allow_html=True)
    st.info("Unggah dataset dan lakukan analisis serta visualisasi sentimen pengguna.")
    try:
        streamlit_sentiment_app.main()
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan saat memuat fitur Analisis Sentimen:\n\n{e}")
