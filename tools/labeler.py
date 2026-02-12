import streamlit as st
from minio import Minio
import os
import io
from dotenv import load_dotenv
from PIL import Image
import uuid

# 1. Ayarları Yükle
load_dotenv() # .env dosyasını okur

# MinIO Bağlantısı
minio_client = Minio(
    os.getenv("MINIO_ENDPOINT", "localhost:9000").replace("minio", "localhost"), # Docker dışından bağlandığımız için localhost
    access_key=os.getenv("MINIO_ROOT_USER"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    secure=False
)

SOURCE_BUCKET = os.getenv("MINIO_BUCKET_TEST", "pet-test") # Hangi kovadan veri çekeceğiz? (Test veya İlanlar)
TRAIN_DIR = "data/train" # Hedef klasör

# 2. Sınıfları Otomatik Bul (Klasör isimlerinden)
try:
    CLASSES = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    CLASSES.sort()
except FileNotFoundError:
    st.error(f"❌ '{TRAIN_DIR}' klasörü bulunamadı! Lütfen doğru yerde çalıştırdığınızdan emin olun.")
    st.stop()

# 3. Streamlit Arayüzü
st.set_page_config(page_title="Veri Rafinerisi 🏭", layout="centered")
st.title("🏭 Veri Etiketleme İstasyonu")

# Oturum Durumu (Sıradaki resmi tutmak için)
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

def get_next_image():
    """MinIO'dan henüz işlenmemiş bir resim getir"""
    try:
        objects = minio_client.list_objects(SOURCE_BUCKET)
        for obj in objects:
            # Basit bir kontrol: Dosya adı zaten eğitim setinde var mı? (Çoklu klasör taraması basitçe)
            is_exist = False
            for cls in CLASSES:
                if os.path.exists(os.path.join(TRAIN_DIR, cls, obj.object_name)):
                    is_exist = True
                    break
            
            if not is_exist:
                return obj.object_name
        return None
    except Exception as e:
        st.error(f"MinIO Hatası: {e}")
        return None

# Yeni resim yükle (Eğer yoksa)
if st.session_state.current_file is None:
    st.session_state.current_file = get_next_image()

# Arayüz Akışı
if st.session_state.current_file:
    file_name = st.session_state.current_file
    st.info(f"İncelenen Dosya: `{file_name}`")
    
    # Resmi Göster
    try:
        response = minio_client.get_object(SOURCE_BUCKET, file_name)
        img_data = response.read()
        image = Image.open(io.BytesIO(img_data))
        st.image(image, use_container_width=True) # use_column_width yerine use_container_width (yeni sürüm)
        response.close()
        response.release_conn()
        
        # Etiketleme Formu
        with st.form("label_form"):
            selected_class = st.selectbox("Bu hangi ırk?", CLASSES)
            
            c1, c2, c3 = st.columns(3)
            with c1:
                save_btn = st.form_submit_button("✅ Kaydet & Eğitime Ekle", type="primary")
            with c2:
                skip_btn = st.form_submit_button("⏭️ Pas Geç")
            with c3:
                trash_btn = st.form_submit_button("🗑️ Çöp (Gereksiz)")
            
            if save_btn:
                # 1. Resmi Kaydet
                save_path = os.path.join(TRAIN_DIR, selected_class, file_name)
                with open(save_path, "wb") as f:
                    f.write(img_data)
                
                st.success(f"✅ {selected_class} olarak kaydedildi!")
                st.session_state.current_file = None # Sıradakine geç
                st.rerun()
                
            if skip_btn:
                st.warning("Pas geçildi.")
                st.session_state.current_file = None
                st.rerun()

            if trash_btn:
                # Burada MinIO'dan silmiyoruz (uygulama bozulmasın diye), sadece geçiyoruz.
                # İstersen 'ignore_list' yapılabilir ama şimdilik pas geçmek yeterli.
                st.error("Çöpe atıldı (Pas geçildi).")
                st.session_state.current_file = None
                st.rerun()
                
    except Exception as e:
        st.error(f"Resim yüklenemedi: {e}")
        # Hatalıysa geç
        st.session_state.current_file = None
        st.rerun()

else:
    st.balloons()
    st.success("🎉 Harika! MinIO'daki tüm yeni resimler etiketlendi.")
    if st.button("🔄 Tekrar Kontrol Et"):
        st.rerun()