import io
import os
import base64
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model, applications
import boto3 # <--- MinIO gitti, Boto3 geldi
from botocore.exceptions import NoCredentialsError
from qdrant_client import QdrantClient, models
from ultralytics import YOLO

# --- AYARLAR (AWS S3 VE QDRANT) ---
MODEL_PATH = "models/final_breed_model.weights.h5"

# AWS AYARLARI
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-central-1")
# Senin oluşturduğun ana kova ismini buraya yaz (veya env'den al)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "pet-retrieval-storage-samet")

# S3 İÇİNDEKİ KLASÖRLER
FOLDER_TEST = "gallery/"     # Eski 'pet-test' yerine
FOLDER_ADS = "ads-images/"   # Eski 'pet-ads' yerine

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pet_vectors")
VECTOR_SIZE = 128
NUM_CLASSES = 37

app = FastAPI()

# --- GLOBAL DEĞİŞKENLER ---
model = None
yolo_model = None
s3_client = None # <--- MinIO client yerine S3 client
qdrant_client = None

# Sınıf İsimleri (Aynı)
class_names = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
    'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
    'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
    'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
    'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
    'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
    'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
    'wheaten_terrier', 'yorkshire_terrier'
]

# --- YARDIMCI FONKSİYONLAR ---
class L2Normalization(layers.Layer):
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

def build_model():
    img_input = layers.Input(shape=(224, 224, 3))
    base_cnn = applications.MobileNetV2(input_shape=(224, 224, 3), weights=None, include_top=False)
    x = base_cnn(img_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    embedding = layers.Dense(128, name="embedding")(x)
    normalized_embedding = L2Normalization(name='std_embedding')(embedding)
    class_output = layers.Dense(NUM_CLASSES, activation='softmax', name="class_output")(x)
    return Model(inputs=img_input, outputs=[normalized_embedding, class_output])

def process_image_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_resized = img.resize((224, 224))
    img_array = preprocess_input(np.expand_dims(np.array(img_resized), axis=0))
    return img, img_array

def check_yolo(pil_img):
    if not yolo_model: return True, "YOLO Yok", 0.0
    results = yolo_model(pil_img, verbose=False)
    detected_label = "Nesne Yok"; max_conf = 0.0; is_animal = False
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0]); cls_id = int(box.cls[0]); label = yolo_model.names[cls_id]
            if conf > max_conf: max_conf = conf; detected_label = label
            if cls_id in [15, 16] and conf > 0.4: is_animal = True
    return is_animal, detected_label, max_conf

# --- STARTUP ---
@app.on_event("startup")
async def startup_event():
    global model, yolo_model, s3_client, qdrant_client
    
    # Modelleri Yükle
    try:
        model = build_model(); model.load_weights(MODEL_PATH); print("✅ Ana Model yüklendi!")
        yolo_model = YOLO("yolov8n.pt"); print("✅ Bekçi Model (YOLO) yüklendi!")
    except Exception as e: print(f"❌ Model Hatası: {e}")

    # AWS S3 Bağlantısı
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        print("✅ AWS S3 Bağlantısı Başarılı!")
    except Exception as e:
        print(f"❌ AWS S3 Hatası: {e}")

    # Qdrant
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.create_collection(COLLECTION_NAME, vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.EUCLID))
    except: pass

# --- ENDPOINTS ---
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    mode: str = Form("test")
):
    # 1. Dosyayı Oku
    content = await file.read()
    pil_img, img_array = process_image_bytes(content)
    
    # 2. Bekçi Kontrolü
    is_animal, detected_label, conf = check_yolo(pil_img)
    if not is_animal:
        return {"error": True, "debug_info": f"Tespit: {detected_label}", "prediction": "Bilinmiyor", "similar_pets": []}
    
    # 3. Model Tahmini
    embedding, class_probs = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(class_probs[0]))
    embedding_vector = embedding[0].tolist()

    # 4. Benzerleri Ara
    similar_pets = []
    if qdrant_client:
        try:
            search_filter = None
            if mode == "real":
                search_filter = models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="ad"))])
            else:
                search_filter = models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="test"))])

            search_response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding_vector,
                query_filter=search_filter,
                limit=5
            )

            for result in search_response.points:
                payload = result.payload
                # ÖNEMLİ: S3'teki tam yol "s3_key" olarak saklanmalı.
                # Eğer eski veri varsa uyum için filename kullanılır.
                s3_key = payload.get("s3_key")
                
                # Eğer yeni sistemle kaydedilmemişse (eski veriyse) manuel oluşturmayı dene
                if not s3_key:
                    filename = payload.get("filename")
                    bucket_type = payload.get("bucket") # Eski sistemdeki bucket adı
                    folder = FOLDER_ADS if bucket_type == "pet-ads" else FOLDER_TEST
                    s3_key = f"{folder}{filename}"

                score = result.score
                sim_score = 1 / (1 + score) if score >= 0 else 0
                
                try:
                    # AWS S3'ten Resmi Çek
                    obj = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    file_content = obj['Body'].read()
                    b64_img = base64.b64encode(file_content).decode('utf-8')
                    
                    pet_data = {
                        "filename": s3_key,
                        "score": sim_score,
                        "image_base64": b64_img,
                        "status": payload.get("status", "Bilinmiyor"),
                        "city": payload.get("city", "-"),
                    }
                    
                    if mode == "real":
                        pet_data.update({
                            "contact_info": payload.get("contact_info", "Belirtilmemiş"),
                            "owner_name": payload.get("owner_name", "Anonim"),
                            "description": payload.get("description", "")
                        })
                    else:
                        pet_data.update({
                            "contact_info": "🔒 Test Verisi",
                            "owner_name": "🔒 Sistem",
                            "description": "Bu görsel test veri setinden gelmektedir."
                        })

                    similar_pets.append(pet_data)
                except Exception as e:
                    print(f"S3 Okuma Hatası ({s3_key}): {e}")
                    
        except Exception as e: print(f"Arama Hatası: {e}")

    return {
        "prediction": class_names[pred_idx],
        "confidence": float(class_probs[0][pred_idx]),
        "similar_pets": similar_pets,
        "error": False,
        "debug_info": detected_label
    }


@app.post("/upload_to_gallery")
async def upload_to_gallery(
    file: UploadFile = File(...),
    upload_type: str = Form("test"),
    owner_name: str = Form(None),
    contact_info: str = Form(None),
    city: str = Form(None),
    status: str = Form(None),
    description: str = Form(None)
):
    try:
        content = await file.read()
        pil_img, img_array = process_image_bytes(content)
        is_animal, detected_label, conf = check_yolo(pil_img)

        if not is_animal:
            raise HTTPException(status_code=400, detail=f"Sadece hayvan yükleyin! ({detected_label})")

        # Dosya Yolu Belirleme (Klasör Mantığı)
        folder = FOLDER_ADS if upload_type == "ad" else FOLDER_TEST
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        s3_full_key = f"{folder}{unique_filename}" # Örn: gallery/uuid_kedi.jpg
        
        # AWS S3'e Yükleme
        file_like = io.BytesIO(content)
        s3_client.upload_fileobj(
            file_like,
            S3_BUCKET_NAME,
            s3_full_key,
            ExtraArgs={'ContentType': file.content_type or "image/jpeg"}
        )
        
        # Vektör Çıkarma
        preds = model.predict(img_array, verbose=0)
        embedding_vector = preds[0][0].tolist()
        
        # Payload Hazırla
        payload_data = {
            "s3_key": s3_full_key, # Artık dosyanın tam yolunu tutuyoruz
            "filename": unique_filename,
            "bucket_name": S3_BUCKET_NAME,
            "type": upload_type,
            "timestamp": str(uuid.uuid1().time)
        }
        
        if upload_type == "ad":
            payload_data.update({
                "owner_name": owner_name,
                "contact_info": contact_info,
                "city": city,
                "status": status,
                "description": description
            })

        qdrant_client.upsert(
            collection_name=COLLECTION_NAME, wait=True,
            points=[models.PointStruct(id=str(uuid.uuid4()), vector=embedding_vector, payload=payload_data)]
        )
        return {"message": "S3'e Başarıyla Kaydedildi!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))