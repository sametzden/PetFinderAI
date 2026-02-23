import streamlit as st
import requests
from PIL import Image
import io
import os
import base64

# --- AYARLAR ---
# Docker içinde "backend" ismiyle konuşuruz.
API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="Pet Finder AI (AWS Edition)", page_icon="☁️", layout="wide")

st.title("☁️ Pet Retrieval System (AWS S3 Powered)")
st.markdown("Yapay zeka ile kayıp dostlarımızı bulalım. Veriler artık Amazon Bulutunda güvende!")

# --- SEKMELER ---
tab1, tab2 = st.tabs(["🔍 Arama Yap (Search)", "➕ İlan / Veri Ekle (Upload)"])

# --- TAB 1: ARAMA ---
with tab1:
    st.header("Kayıp Bir Hayvanı Arat")
    
    # Arama Modu Seçimi
    search_mode = st.radio("Arama Tipi:", ["test", "real"], horizontal=True, 
                          format_func=lambda x: "🧪 Test Verisi" if x == "test" else "🌍 Gerçek İlanlar")
    
    uploaded_file = st.file_uploader("Resmi buraya sürükle", type=["jpg", "png", "jpeg"], key="search")
    
    if uploaded_file is not None:
        # Resmi Göster
        image = Image.open(uploaded_file)
        st.image(image, caption="Aranan Resim", width=300)
        
        if st.button("🔍 Benzerlerini Bul", key="btn_search"):
            with st.spinner("AWS S3 ve Qdrant taranıyor..."):
                try:
                    # Backend'e Dosya Gönder
                    files = {"file": uploaded_file.getvalue()}
                    data = {"mode": search_mode}
                    
                    response = requests.post(f"{API_URL}/predict", files=files, data=data)
                    
                    if response.status_code == 200:
                        resp_json = response.json()
                        
                        # Hata Kontrolü (YOLO vb.)
                        if resp_json.get("error"):
                            st.error(f"Hata: {resp_json.get('debug_info')}")
                        else:
                            # Tahmin Bilgisi
                            pred_class = resp_json.get("prediction", "Bilinmiyor")
                            conf = resp_json.get("confidence", 0.0)
                            st.info(f"🤖 Model Tahmini: **{pred_class}** (Güven: %{conf*100:.1f})")
                            
                            results = resp_json.get("similar_pets", [])
                            
                            if not results:
                                st.warning("Eşleşen sonuç bulunamadı.")
                            else:
                                st.success(f"En benzer {len(results)} sonuç bulundu!")
                                
                                # Sonuçları Yan Yana Göster (3'erli satırlar halinde)
                                cols = st.columns(3)
                                for idx, res in enumerate(results):
                                    col = cols[idx % 3]
                                    with col:
                                        # Base64 Resim Çözme
                                        img_b64 = res.get("image_base64")
                                        if img_b64:
                                            img_data = base64.b64decode(img_b64)
                                            st.image(img_data, use_container_width=True)
                                        else:
                                            st.image("https://via.placeholder.com/300?text=Resim+Yok", use_container_width=True)
                                        
                                        st.caption(f"Benzerlik: %{res['score']*100:.1f}")
                                        st.markdown(f"**Dosya:** `{res['filename']}`")
                                        
                                        if search_mode == "real":
                                            st.markdown(f"👤 **Sahibi:** {res.get('owner_name')}")
                                            st.markdown(f"📞 **İletişim:** {res.get('contact_info')}")
                                            st.markdown(f"📍 **Şehir:** {res.get('city')}")

                    else:
                        st.error(f"Sunucu Hatası: {response.text}")
                        
                except Exception as e:
                    st.error(f"Bağlantı Hatası: {e}")

# --- TAB 2: YÜKLEME ---
with tab2:
    st.header("Sisteme Veri Yükle")
    
    upload_type = st.radio("Yükleme Tipi:", ["test", "ad"], horizontal=True,
                          format_func=lambda x: "🧪 Test Verisi (Eğitim için)" if x == "test" else "📢 İlan Oluştur (Canlı)")
    
    with st.form("upload_form"):
        uploaded_index = st.file_uploader("Resim Seç", type=["jpg", "png", "jpeg"])
        
        # İlan ise detay iste
        owner_name = st.text_input("Ad Soyad") if upload_type == "ad" else None
        contact_info = st.text_input("Telefon / E-posta") if upload_type == "ad" else None
        city = st.text_input("Şehir") if upload_type == "ad" else None
        desc = st.text_area("Açıklama") if upload_type == "ad" else None
        
        submitted = st.form_submit_button("☁️ AWS S3'e ve Veritabanına Kaydet")
        
        if submitted:
            if uploaded_index:
                with st.spinner("Yükleniyor..."):
                    try:
                        files = {"file": (uploaded_index.name, uploaded_index.getvalue(), uploaded_index.type)}
                        data = {"upload_type": upload_type}
                        
                        if upload_type == "ad":
                            data.update({
                                "owner_name": owner_name,
                                "contact_info": contact_info,
                                "city": city,
                                "description": desc,
                                "status": "active"
                            })
                        
                        response = requests.post(f"{API_URL}/upload_to_gallery", files=files, data=data)
                        
                        if response.status_code == 200:
                            st.balloons()
                            st.success("✅ Başarıyla Kaydedildi! Resim AWS S3'te, Vektör Qdrant'ta.")
                        else:
                            st.error(f"Hata: {response.text}")
                    except Exception as e:
                        st.error(f"Bağlantı Hatası: {e}")
            else:
                st.warning("Lütfen bir resim seçin.")