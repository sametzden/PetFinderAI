import streamlit as st
import requests

st.set_page_config(page_title="PetFinder AI", page_icon="🐾", layout="wide")
st.markdown("<style>.css-15zrgzn {display: none} .css-1629p8f h1 a {display: none} a.anchor-link {display: none}</style>", unsafe_allow_html=True)

BASE_URL = "http://backend:8000"

st.sidebar.title("🐾 Menü")
menu = st.sidebar.radio("Seçiniz:", ["🎮 Modeli Test Et", "📢 İlan Ver (Kayıp/Bulundu)", "🔍 İlanlarda Ara"])

# --- 1. OYUN ALANI (TEST) ---
if menu == "🎮 Modeli Test Et":
    st.title("🧪 Yapay Zeka Test Alanı")
    st.info("Yüklediğiniz fotoğraf veritabanına KAYDEDİLMEZ, sadece mevcut test verileriyle karşılaştırılır.")
    
    uploaded_file = st.file_uploader("Bir fotoğraf yükleyin", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Sizin Resminiz", use_container_width=True)
            
            # ANALİZ BUTONU
            if st.button("🔍 Analiz Et", type="primary"):
                with st.spinner("Vektör uzayı taranıyor..."):
                    files = {"file": uploaded_file.getvalue()}
                    data = {"mode": "test"} 
                    try:
                        res = requests.post(f"{BASE_URL}/predict", files=files, data=data).json()
                        st.session_state['test_res'] = res
                    except: st.error("Sunucu Hatası")

            # --- YENİ: KATKI BUTONU ---
            st.divider()
            st.markdown("##### 🤝 Modele Destek Ol")
            if st.button("💾 Bu Veriyi Veritabanına Bağışla"):
                with st.spinner("Veri setine ekleniyor..."):
                    # Resmi 'test' tipiyle upload servisine gönderiyoruz
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    data = {
                        "upload_type": "test", # Test veri setine ekle
                        "status": "User Contrib",
                        "description": "Kullanıcı tarafından test sırasında bağışlandı."
                    }
                    try:
                        res = requests.post(f"{BASE_URL}/upload_to_gallery", files=files, data=data)
                        if res.status_code == 200:
                            st.success("Teşekkürler! Veriniz anonim olarak test havuzuna eklendi.")
                        else:
                            st.error("Kayıt başarısız.")
                    except Exception as e:
                        st.error(f"Hata: {e}")

        with col2:
            if 'test_res' in st.session_state:
                res = st.session_state['test_res']
                if res.get("error"):
                    st.error(f"Hata: {res.get('debug_info')}")
                else:
                    st.success(f"Tahmin: **{res['prediction']}** (%{res['confidence']*100:.1f})")
                    
                    st.divider()
                    st.markdown("### 🧬 Veritabanı Eşleşmeleri")
                    
                    if not res['similar_pets']:
                        st.warning("Veritabanında benzer kayıt bulunamadı.")
                    else:
                        for pet in res['similar_pets']:
                            with st.container():
                                c_img, c_info = st.columns([1, 2])
                                with c_img:
                                    st.image(f"data:image/jpeg;base64,{pet['image_base64']}", use_container_width=True)
                                with c_info:
                                    st.write(f"**Benzerlik Skoru:** %{pet['score']*100:.1f}")
                                    st.caption(f"Kaynak: {pet['status']}")
                                st.divider()

# --- 2. İLAN VERME (GERÇEK) ---
elif menu == "📢 İlan Ver (Kayıp/Bulundu)":
    st.title("📝 İlan Oluştur")
    st.warning("Buraya girdiğiniz bilgiler halka açık olarak aranabilir olacaktır.")
    
    with st.form("ilan_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Fotoğraf", type=["jpg", "png"])
            if uploaded_file: st.image(uploaded_file, width=200)
        with col2:
            owner = st.text_input("Ad Soyad")
            contact = st.text_input("İletişim (Tel/Email)")
            city = st.selectbox("Şehir", ["İstanbul", "Ankara", "İzmir", "Diğer"])
            status = st.selectbox("Durum", ["Kayıp", "Bulundu", "Sahiplendirme"])
            desc = st.text_area("Açıklama")
            
        if st.form_submit_button("Yayınla"):
            if uploaded_file and contact:
                files = {"file": uploaded_file.getvalue()}
                data = {
                    "upload_type": "ad", # İlan tipi
                    "owner_name": owner, "contact_info": contact,
                    "city": city, "status": status, "description": desc
                }
                try:
                    requests.post(f"{BASE_URL}/upload_to_gallery", files=files, data=data)
                    st.success("✅ İlanınız başarıyla yayınlandı!")
                except: st.error("Hata oluştu")
            else: st.error("Fotoğraf ve İletişim zorunludur!")

# --- 3. İLANLARDA ARAMA (GERÇEK) ---
elif menu == "🔍 İlanlarda Ara":
    st.title("🕵️‍♂️ Kayıp Eşleştirme Sistemi")
    st.markdown("Elinizdeki fotoğrafı yükleyin, **veritabanındaki gerçek ilanlarla** eşleştirelim.")
    
    uploaded_file = st.file_uploader("Aranan hayvanın fotoğrafı", type=["jpg", "png"])
    
    if uploaded_file:
        st.image(uploaded_file, width=300)
        if st.button("Veritabanında Tara"):
            with st.spinner("İlanlar taranıyor..."):
                files = {"file": uploaded_file.getvalue()}
                data = {"mode": "real"} # Gerçek arama modu
                try:
                    res = requests.post(f"{BASE_URL}/predict", files=files, data=data).json()
                    
                    if res.get("error"):
                        st.error("Hayvan tespit edilemedi.")
                    else:
                        st.success(f"Bu bir **{res['prediction']}**. İşte benzer ilanlar:")
                        
                        if not res['similar_pets']:
                            st.warning("⚠️ Eşleşen ilan bulunamadı.")
                        else:
                            for pet in res['similar_pets']:
                                with st.container():
                                    c1, c2 = st.columns([1, 3])
                                    with c1:
                                        st.image(f"data:image/jpeg;base64,{pet['image_base64']}", use_container_width=True)
                                    with c2:
                                        st.subheader(f"{pet['status']} - {pet['city']}")
                                        st.write(f"**Benzerlik:** %{pet['score']*100:.1f}")
                                        st.write(f"**İletişim:** {pet.get('contact_info')}")
                                        st.write(f"**Sahibi:** {pet.get('owner_name')}")
                                        st.info(f"Not: {pet.get('description')}")
                                    st.divider()
                except Exception as e: st.error(f"Hata: {e}")