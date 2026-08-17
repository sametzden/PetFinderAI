import io
import os
import base64
import uuid
import glob
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model, applications
from qdrant_client import QdrantClient, models
from ultralytics import YOLO

# --- AYARLAR (LOKAL DEPOLAMA VE QDRANT) ---
MODEL_PATH = "models/final_breed_model.weights.h5"

# GÖRSEL DEPOLAMA (Yerel Diskte - S3 gerekmez)
GALLERY_DIR = os.getenv("GALLERY_DIR", "gallery")
FOLDER_TEST = "test"   # gallery/test  -> demo/test verisi
FOLDER_ADS = "ads"     # gallery/ads   -> kullanıcı ilanları

# QDRANT (Embedded / dosya tabanlı - ayrı bir sunucu gerekmez)
QDRANT_PATH = os.getenv("QDRANT_PATH", "qdrant_data")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pet_vectors")
VECTOR_SIZE = 128
NUM_CLASSES = 37

app = FastAPI()

# --- GLOBAL DEĞİŞKENLER ---
model = None
yolo_model = None
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

def gallery_path(folder: str, filename: str) -> str:
    return os.path.join(GALLERY_DIR, folder, filename)

def seed_gallery_if_empty():
    """İlk açılışta gallery/test klasöründeki örnek görselleri Qdrant'a indeksler."""
    count = qdrant_client.count(collection_name=COLLECTION_NAME, exact=True).count
    if count > 0:
        print(f"ℹ️ Qdrant zaten dolu ({count} vektör), seed atlanıyor.")
        return

    test_dir = os.path.join(GALLERY_DIR, FOLDER_TEST)
    image_paths = sorted(glob.glob(os.path.join(test_dir, "*.jpg"))) + \
                  sorted(glob.glob(os.path.join(test_dir, "*.jpeg"))) + \
                  sorted(glob.glob(os.path.join(test_dir, "*.png")))

    if not image_paths:
        print(f"⚠️ Seed edilecek görsel bulunamadı: {test_dir}")
        return

    points = []
    for path in image_paths:
        with open(path, "rb") as f:
            content = f.read()
        _, img_array = process_image_bytes(content)
        embedding, _ = model.predict(img_array, verbose=0)
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding[0].tolist(),
            payload={
                "filename": os.path.basename(path),
                "folder": FOLDER_TEST,
                "type": "test",
            }
        ))

    qdrant_client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
    print(f"✅ {len(points)} demo görsel Qdrant'a indekslendi.")

# --- STARTUP ---
@app.on_event("startup")
async def startup_event():
    global model, yolo_model, qdrant_client

    # Klasörleri hazırla
    os.makedirs(gallery_path(FOLDER_TEST, ""), exist_ok=True)
    os.makedirs(gallery_path(FOLDER_ADS, ""), exist_ok=True)

    # Modelleri Yükle
    try:
        model = build_model(); model.load_weights(MODEL_PATH); print("✅ Ana Model yüklendi!")
        yolo_model = YOLO("yolov8n.pt"); print("✅ Bekçi Model (YOLO) yüklendi!")
    except Exception as e: print(f"❌ Model Hatası: {e}")

    # Qdrant (embedded / dosya tabanlı - sunucu gerekmez)
    try:
        qdrant_client = QdrantClient(path=QDRANT_PATH)
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.create_collection(COLLECTION_NAME, vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.EUCLID))
        if model is not None:
            seed_gallery_if_empty()
        print("✅ Qdrant (embedded) hazır!")
    except Exception as e:
        print(f"❌ Qdrant Hatası: {e}")

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
            target_folder = FOLDER_ADS if mode == "real" else FOLDER_TEST
            search_filter = models.Filter(must=[models.FieldCondition(key="type", match=models.MatchValue(value="ad" if mode == "real" else "test"))])

            search_response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding_vector,
                query_filter=search_filter,
                limit=5
            )

            for result in search_response.points:
                payload = result.payload
                filename = payload.get("filename")
                folder = payload.get("folder", target_folder)
                local_path = gallery_path(folder, filename)

                score = result.score
                sim_score = 1 / (1 + score) if score >= 0 else 0

                try:
                    with open(local_path, "rb") as f:
                        file_content = f.read()
                    b64_img = base64.b64encode(file_content).decode('utf-8')

                    pet_data = {
                        "filename": filename,
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
                    print(f"Görsel Okuma Hatası ({local_path}): {e}")

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
        local_path = gallery_path(folder, unique_filename)

        # Yerel Diske Kaydet
        with open(local_path, "wb") as f:
            f.write(content)

        # Vektör Çıkarma
        preds = model.predict(img_array, verbose=0)
        embedding_vector = preds[0][0].tolist()

        # Payload Hazırla
        payload_data = {
            "filename": unique_filename,
            "folder": folder,
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
        return {"message": "Kaydedildi!"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
