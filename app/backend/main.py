import io
import os
import base64
import uuid
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model, applications
from minio import Minio
from qdrant_client import QdrantClient, models # models modülünü içeri alıyoruz

# --- AYARLAR ---
MODEL_PATH = "models/final_breed_model.weights.h5"
MINIO_ENDPOINT = "minio:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "pet-gallery"
QDRANT_HOST = "qdrant"
QDRANT_PORT = 6333
COLLECTION_NAME = "pet_vectors"
VECTOR_SIZE = 128
NUM_CLASSES = 37

app = FastAPI()

# --- GLOBAL DEĞİŞKENLER ---
model = None
minio_client = None
qdrant_client = None

class_names = [
    'Abyssinian',
    'Bengal',
    'Birman',
    'Bombay',
    'British_Shorthair',
    'Egyptian_Mau',
    'Maine_Coon',
    'Persian',
    'Ragdoll',
    'Russian_Blue',
    'Siamese',
    'Sphynx',
    'american_bulldog',
    'american_pit_bull_terrier',
    'basset_hound',
    'beagle',
    'boxer',
    'chihuahua',
    'english_cocker_spaniel',
    'english_setter',
    'german_shorthaired',
    'great_pyrenees',
    'havanese',
    'japanese_chin',
    'keeshond',
    'leonberger',
    'miniature_pinscher',
    'newfoundland',
    'pomeranian',
    'pug',
    'saint_bernard',
    'samoyed',
    'scottish_terrier',
    'shiba_inu',
    'staffordshire_bull_terrier',
    'wheaten_terrier',
    'yorkshire_terrier'
]

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

def process_image_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = preprocess_input(np.expand_dims(np.array(img), axis=0))
    return img_array

@app.on_event("startup")
async def startup_event():
    global model, minio_client, qdrant_client
    
    # 1. Model
    model = build_model()
    try:
        model.load_weights(MODEL_PATH)
        print("✅ Model yüklendi!")
    except Exception as e:
        print(f"❌ Model hatası: {e}")

    # 2. MinIO
    try:
        minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        if not minio_client.bucket_exists(BUCKET_NAME):
            minio_client.make_bucket(BUCKET_NAME)
    except Exception as e:
        print(f"❌ MinIO Hatası: {e}")

    # 3. Qdrant
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        
        # Koleksiyon kontrolü
        # DÜZELTME: collection_exists metodunu doğru kullanalım
        collections = qdrant_client.get_collections()
        exists = any(c.name == COLLECTION_NAME for c in collections.collections)

        if not exists:
            if not exists:
            # DÜZELTME BURADA: .Euclid yerine .EUCLID yaptık
                qdrant_client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.EUCLID)
                )
                print(f"🗄️ Qdrant: Yeni koleksiyon oluşturuldu ({COLLECTION_NAME})")
            
            # Senkronizasyon
            print("🔄 MinIO verileri Qdrant'a aktarılıyor...")
            objects = minio_client.list_objects(BUCKET_NAME)
            points = []
            
            for obj in objects:
                try:
                    response = minio_client.get_object(BUCKET_NAME, obj.object_name)
                    img_data = response.read()
                    response.close()
                    response.release_conn()
                    
                    img_array = process_image_bytes(img_data)
                    preds = model.predict(img_array, verbose=0)
                    embedding_vector = preds[0][0].tolist()
                    
                    points.append(models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding_vector,
                        payload={"filename": obj.object_name}
                    ))
                except Exception as e:
                    print(f"Atlandı: {e}")
            
            if points:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    wait=True,
                    points=points
                )
                print(f"✅ {len(points)} resim Qdrant'a indekslendi!")
        else:
            print("✅ Qdrant: Koleksiyon zaten var.")
            
    except Exception as e:
        print(f"❌ Qdrant Hatası (Startup): {e}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    img_array = process_image_bytes(content)
    
    embedding, class_probs = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(class_probs[0]))
    current_emb = embedding[0].tolist()

    similar_pets = []
    if qdrant_client:
        try:
            # DÜZELTME: search metodu garanti çalışacak
            search_result = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=current_emb,
                limit=5
            )
            
            for result in search_result:
                filename = result.payload["filename"]
                score = result.score
                
                # Qdrant'ın Euclidean skoru mesafedir (düşük = iyi). 
                # Bunu benzerliğe çeviriyoruz.
                similarity_score = 1 / (1 + score) if score >= 0 else 0

                try:
                    response = minio_client.get_object(BUCKET_NAME, filename)
                    img_bytes = response.read()
                    response.close()
                    response.release_conn()
                    b64_img = base64.b64encode(img_bytes).decode('utf-8')
                    
                    similar_pets.append({
                        "filename": filename,
                        "score": float(similarity_score),
                        "image_base64": b64_img
                    })
                except:
                    pass
        except Exception as e:
            print(f"❌ Arama Hatası: {e}")

    return {
        "prediction": class_names[pred_idx],
        "confidence": float(class_probs[0][pred_idx]),
        "similar_pets": similar_pets
    }

@app.post("/upload_to_gallery")
async def upload_to_gallery(file: UploadFile = File(...)):
    try:
        content = await file.read()
        file_like = io.BytesIO(content)
        minio_client.put_object(
            BUCKET_NAME, file.filename, file_like, length=len(content), content_type=file.content_type
        )
        
        img_array = process_image_bytes(content)
        preds = model.predict(img_array, verbose=0)
        embedding_vector = preds[0][0].tolist()
        
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding_vector,
                    payload={"filename": file.filename}
                )
            ]
        )
        return {"message": "Eklendi!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))