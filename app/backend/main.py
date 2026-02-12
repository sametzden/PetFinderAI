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
from minio import Minio
from qdrant_client import QdrantClient, models
from ultralytics import YOLO

# --- AYARLAR ---
MODEL_PATH = "models/final_breed_model.weights.h5"
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ROOT_USER", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_ROOT_PASSWORD", "minioadmin")
# İKİ AYRI KOVA (BUCKET)
TEST_BUCKET = os.getenv("MINIO_BUCKET_TEST", "pet-test")
ADS_BUCKET = os.getenv("MINIO_BUCKET_ADS", "pet-ads")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pet_vectors")
VECTOR_SIZE = 128
NUM_CLASSES = 37

app = FastAPI()

# --- GLOBAL DEĞİŞKENLER (Model, Client vb.) ---
model = None
yolo_model = None
minio_client = None
qdrant_client = None

# Sınıf İsimleri (Aynı kalacak)
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

# --- YARDIMCI FONKSİYONLAR (Model Build, Preprocess, YOLO) ---
# ... (Buralar aynı kalıyor, sadece L2Normalization ve check_yolo fonksiyonlarını koru) ...
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
    global model, yolo_model, minio_client, qdrant_client
    
    # Modelleri Yükle
    try:
        model = build_model(); model.load_weights(MODEL_PATH); print("✅ Ana Model yüklendi!")
        yolo_model = YOLO("yolov8n.pt"); print("✅ Bekçi Model (YOLO) yüklendi!")
    except Exception as e: print(f"❌ Model Hatası: {e}")

    # MinIO (İki kova oluşturuyoruz)
    try:
        minio_client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False)
        if not minio_client.bucket_exists(TEST_BUCKET): minio_client.make_bucket(TEST_BUCKET)
        if not minio_client.bucket_exists(ADS_BUCKET): minio_client.make_bucket(ADS_BUCKET)
    except: pass

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
    mode: str = Form("test") # 'test' veya 'real'
):
    # 1. Dosyayı Oku
    content = await file.read()
    pil_img, img_array = process_image_bytes(content)
    
    # 2. Bekçi Kontrolü
    is_animal, detected_label, conf = check_yolo(pil_img)
    if not is_animal:
        return {
            "error": True, 
            "debug_info": f"Tespit: {detected_label}", 
            "prediction": "Bilinmiyor", 
            "similar_pets": []
        }
    
    # 3. Model Tahmini (Sadece Vektör Çıkarıyoruz, KAYDETMİYORUZ)
    embedding, class_probs = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(class_probs[0]))
    embedding_vector = embedding[0].tolist()

    # --- BURADAKİ OTOMATİK KAYIT KODLARINI SİLDİK --- ❌

    # 4. Benzerleri Ara
    similar_pets = []
    if qdrant_client:
        try:
            # FİLTRELEME MANTIĞI (Keskin Ayrım)
            search_filter = None
            
            if mode == "real":
                # Sadece GERÇEK İlanlar
                search_filter = models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="ad"))]
                )
            else: # mode == "test"
                # Sadece TEST Verileri (Senin yüklediklerin)
                search_filter = models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="test"))]
                )

            search_result = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=embedding_vector,
                query_filter=search_filter,
                limit=5
            )

            for result in search_result:
                payload = result.payload
                filename = payload.get("filename")
                bucket = payload.get("bucket", TEST_BUCKET) # Varsayılan
                
                score = result.score
                sim_score = 1 / (1 + score) if score >= 0 else 0
                
                try:
                    resp = minio_client.get_object(bucket, filename)
                    b64_img = base64.b64encode(resp.read()).decode('utf-8')
                    resp.close(); resp.release_conn()
                    
                    pet_data = {
                        "filename": filename,
                        "score": sim_score,
                        "image_base64": b64_img,
                        "status": payload.get("status", "Bilinmiyor"),
                        "city": payload.get("city", "-"),
                        "bucket": bucket
                    }
                    
                    # Gizlilik Mantığı
                    if mode == "real":
                        pet_data["contact_info"] = payload.get("contact_info", "Belirtilmemiş")
                        pet_data["owner_name"] = payload.get("owner_name", "Anonim")
                        pet_data["description"] = payload.get("description", "")
                    else:
                        pet_data["contact_info"] = "🔒 Test Verisi"
                        pet_data["owner_name"] = "🔒 Sistem"
                        pet_data["description"] = "Bu görsel test veri setinden gelmektedir."

                    similar_pets.append(pet_data)
                except: pass
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
    upload_type: str = Form("test"), # 'test' veya 'ad' (ilan)
    # Aşağıdakiler opsiyonel (sadece ilan ise dolu gelir)
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

        # Hangi kovaya gidecek?
        target_bucket = ADS_BUCKET if upload_type == "ad" else TEST_BUCKET
        
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_like = io.BytesIO(content)
        minio_client.put_object(target_bucket, unique_filename, file_like, length=len(content), content_type=file.content_type)
        
        preds = model.predict(img_array, verbose=0)
        embedding_vector = preds[0][0].tolist()
        
        # Payload Hazırla
        payload_data = {
            "filename": unique_filename,
            "bucket": target_bucket,
            "type": upload_type, # 'test' veya 'ad'
            "timestamp": str(uuid.uuid1().time)
        }
        
        # Eğer ilansa ekstra bilgileri ekle
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
        return {"message": "Kayıt Başarılı!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))