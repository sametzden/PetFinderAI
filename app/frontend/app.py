import streamlit as st
import requests
from PIL import Image
import io
import base64
import os

# Docker'da 'backend' ismini, lokalde localhost kullanır
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_URL = f"{API_URL}/predict"
UPLOAD_URL = f"{API_URL}/upload_to_gallery"

st.set_page_config(page_title="PetFinder AI", layout="wide", page_icon="🐾")

st.sidebar.title("🐾 Menü")
mode = st.sidebar.radio("Seçim Yapın:", ["🔍 Kayıp Hayvan Ara", "➕ Yeni Hayvan Ekle"])

if mode == "🔍 Kayıp Hayvan Ara":
    st.title("🔍 Kayıp Dostunu Bul")
    st.markdown("Yüklediğiniz fotoğrafa en çok benzeyenleri veritabanında arar.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("Aranan Hayvanın Resmi", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            st.image(uploaded_file, caption="Sorgu", width=300)
            if st.button("Benzerleri Getir", type="primary"):
                with st.spinner("Taranıyor..."):
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}
                    try:
                        res = requests.post(PREDICT_URL, files=files).json()
                        st.session_state['results'] = res
                        st.success(f"Irk: {res['prediction']} (%{res['confidence']*100:.1f})")
                    except Exception as e:
                        st.error(f"Hata: {e}")

    with col2:
        if 'results' in st.session_state:
            res = st.session_state['results']
            st.subheader("Eşleşen Sonuçlar")
            if not res['similar_pets']:
                st.info("Henüz benzer bir kayıt bulunamadı.")
            
            cols = st.columns(3)
            for i, item in enumerate(res['similar_pets']):
                col = cols[i % 3]
                img_data = base64.b64decode(item['image_base64'])
                with col:
                    st.image(Image.open(io.BytesIO(img_data)), use_column_width=True)
                    st.caption(f"{item['filename']} (Skor: {item['score']:.2f})")

elif mode == "➕ Yeni Hayvan Ekle":
    st.title("➕ Veritabanına Kayıt Ekle")
    st.markdown("Barınağa veya sisteme yeni gelen hayvanları buraya yükleyin.")
    
    new_files = st.file_uploader("Fotoğrafları Seç (Çoklu Seçim)", accept_multiple_files=True, type=["jpg", "png"])
    
    if st.button("Kaydet ve İndeksle"):
        if new_files:
            progress_bar = st.progress(0)
            for i, file in enumerate(new_files):
                file.seek(0)
                files = {"file": file}
                try:
                    requests.post(UPLOAD_URL, files=files)
                except Exception as e:
                    st.error(f"{file.name} yüklenemedi: {e}")
                progress_bar.progress((i + 1) / len(new_files))
            st.success("Tüm dosyalar MinIO'ya yüklendi ve indekslendi! 🚀")
        else:
            st.warning("Lütfen dosya seçin.")