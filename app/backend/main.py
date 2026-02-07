import io
import os
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model, applications
from sklearn.neighbors import NearestNeighbors
from minio import Minio

# --- AYARLAR ---
MODEL_PATH = "models/final_breed_model.weights.h5"
MINIO_ENDPOINT = "minio:9000"  # Docker içinden MinIO'ya ulaşım adresi
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "pet-gallery" # Resimlerin tutulacağı kova ismi
NUM_CLASSES = 37

app = FastAPI()

# --- GLOBAL DEĞİŞKENLER ---
model = None
minio_client = None
gallery_embeddings = [] 
gallery_filenames = []  
nbrs_engine = None      

class_names = [
    'Abyssinian', 'American_Bulldog', 'American_Pit_Bull_Terrier', 'Basset_Hound',
    'Beagle', 'Bengal', 'Birman', 'Bombay', 'Boxer', 'British_Shorthair',
    'Chihuahua', 'Egyptian_Mau', 'English_Cocker_Spaniel', 'English_Setter',
    'German_Shorthaired', 'Great_Pyrenees', 'Havanese', 'Japanese_Chin',
    'Keeshond', 'Leonberger', 'Maine_Coon', 'Miniature_Pinscher', 'Newfoundland',
    'Persian', 'Pomeranian', 'Pug', 'Ragdoll', 'Russian_Blue', 'Saint_Bernard',
    'Samoyed', 'Scottish_Terrier', 'Shiba_Inu', 'Siamese', 'Sphynx',
    'Staffordshire_Bull_Terrier', 'Wheaten_Terrier', 'Yorkshire_Terrier'
]

# --- KATMAN VE MODEL ---
class L2Normalization(layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
    def get_config(self):
        return super(L2Normalization, self).get_config()

def build_model():
    img_input = layers.Input(shape=(224, 224, 3))
    base_cnn = applications.MobileNetV2(
        input_shape=(224, 224, 3), weights=None, include_top=False
    )
    x = base_cnn(img_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    embedding = layers.Dense(128, name="embedding")(x)
    normalized_embedding = L2Normalization(name='std_embedding')(embedding)
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name="class_output")(x)
    return Model(inputs=img_input, outputs=[normalized_embedding, class_output])

# --- YARDIMCI FONKSİYONLAR ---
def process_image_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = preprocess_input(np.expand_dims(np.array(img), axis=0))
    return img_array

def refresh_knn():
    """Embedding listesi güncellendiğinde KNN motorunu yeniler"""
    global nbrs_engine, gallery_embeddings
    if len(gallery_embeddings) > 0:
        nbrs_engine = NearestNeighbors(n_neighbors=min(5, len(gallery_embeddings)), metric='euclidean')
        nbrs_engine.fit(np.array(gallery_embeddings))
        print(f"🔄 KNN güncellendi: {len(gallery_embeddings)} resim.")

# --- STARTUP: MINIO BAĞLANTISI VE İNDEKSLEME ---
@app.on_event("startup")
async def startup_event():
    global model, minio_client, gallery_embeddings, gallery_filenames
    
    # 1. Modeli Yükle
    model = build_model()
    try:
        model.load_weights(MODEL_PATH)
        print("✅ Model yüklendi!")
    except Exception as e:
        print(f"❌ Model hatası: {e}")

    # 2. MinIO Bağlantısı
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        
        # Kova (Bucket) yoksa oluştur
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
            print(f"📦 Yeni kova oluşturuldu: {BUCKET_NAME}")
        
        # 3. Mevcut Resimleri İndeksle
        print("📂 MinIO taranıyor...")
        objects = minio_client.list_objects(BUCKET_NAME)
        
        for obj in objects:
            try:
                # Resmi RAM'e indir
                response = minio_client.get_object(BUCKET_NAME, obj.object_name)
                img_data = response.read()
                response.close()
                response.release_conn()
                
                # Embedding çıkar
                img_array = process_image_bytes(img_data)
                preds = model.predict(img_array, verbose=0)
                
                gallery_embeddings.append(preds[0][0])
                gallery_filenames.append(obj.object_name)
            except Exception as e:
                print(f"Hata ({obj.object_name}): {e}")
        
        refresh_knn()
        
    except Exception as e:
        print(f"❌ MinIO Bağlantı Hatası: {e}")

# --- API ENDPOINTS ---

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Sorgu resmi işle
    content = await file.read()
    img_array = process_image_bytes(content)
    
    # 2. Tahmin
    embedding, class_probs = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(class_probs[0]))
    current_emb = embedding[0]

    # 3. MinIO'dan Benzerleri Bul
    similar_pets = []
    if nbrs_engine is not None:
        dists, indices = nbrs_engine.kneighbors([current_emb])
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = dists[0][i]
            filename = gallery_filenames[idx]
            
            # Resmi MinIO'dan çekip Base64 yapıp Frontend'e atıyoruz
            try:
                response = minio_client.get_object(BUCKET_NAME, filename)
                img_bytes = response.read()
                response.close()
                response.release_conn()
                b64_img = base64.b64encode(img_bytes).decode('utf-8')
                
                similar_pets.append({
                    "filename": filename,
                    "score": float(1 / (1 + dist)),
                    "image_base64": b64_img
                })
            except:
                pass

    return {
        "prediction": class_names[pred_idx],
        "confidence": float(class_probs[0][pred_idx]),
        "embedding_sample": current_emb[:5].tolist(),
        "similar_pets": similar_pets
    }

@app.post("/upload_to_gallery")
async def upload_to_gallery(file: UploadFile = File(...)):
    """Yeni bir resmi galeriye (MinIO'ya) ekler ve indeksler"""
    try:
        content = await file.read()
        
        # 1. MinIO'ya Kaydet
        # Dosya boyutunu bulmak için stream'i kullanıyoruz
        file_like = io.BytesIO(content)
        minio_client.put_object(
            BUCKET_NAME, 
            file.filename, 
            file_like, 
            length=len(content),
            content_type=file.content_type
        )
        
        # 2. Embedding Hesapla ve Listeye Ekle
        img_array = process_image_bytes(content)
        preds = model.predict(img_array, verbose=0)
        
        gallery_embeddings.append(preds[0][0])
        gallery_filenames.append(file.filename)
        
        # 3. İndeksi Güncelle
        refresh_knn()
        
        return {"message": f"{file.filename} başarıyla eklendi ve indekslendi!"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))