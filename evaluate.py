import os
import numpy as np
import tensorflow as tf
import yaml
import json
from tqdm import tqdm
from train import create_model, L2Normalization, load_and_process_image # train.py'den fonksiyonları alıyoruz

# 1. Ayarları Yükle
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

DATA_PATH = params["data_path"]
MODEL_PATH = params["model_output"]
IMG_SIZE = params["img_size"]
SAMPLE_SIZE = 500 # Test için kaç resim kullanalım? (Hız için sınırlı tutuyoruz)

def get_all_images(data_dir):
    image_paths = []
    labels = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for label_idx, cls in enumerate(classes):
        cls_path = os.path.join(data_dir, cls)
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            image_paths.append(f)
            labels.append(label_idx)
            
    return image_paths, labels, len(classes)

def compute_metrics():
    print("🔍 Değerlendirme Başlıyor...")
    
    # Verileri topla
    paths, labels, num_classes = get_all_images(DATA_PATH)
    
    # Hız için rastgele örneklem al (Shuffle)
    combined = list(zip(paths, labels))
    np.random.shuffle(combined)
    paths, labels = zip(*combined)
    
    paths = paths[:SAMPLE_SIZE]
    labels = np.array(labels[:SAMPLE_SIZE])
    
    print(f"📊 Toplam {len(paths)} resim üzerinde benzerlik testi yapılacak.")

    # Modeli Yükle
    # Sadece 'base_model' lazım (Embedding veren kısım)
    full_model, _ = create_model(num_classes=num_classes)
    full_model.load_weights(MODEL_PATH) 
    
    # Sadece embedding çıktısını veren bir model oluşturalım
    # full_model output: [embedding, classification] -> Biz 0. indexi istiyoruz
    embedding_model = tf.keras.Model(inputs=full_model.input, outputs=full_model.outputs[0])

    # Embeddingleri Çıkar
    embeddings = []
    for p in tqdm(paths, desc="Vektörler çıkarılıyor"):
        img = load_and_process_image(p)
        img = tf.expand_dims(img, axis=0)
        emb = embedding_model.predict(img, verbose=0)
        embeddings.append(emb[0])
    
    embeddings = np.array(embeddings)
    
    # Benzerlik Matrisi (Cosine Similarity yerine Dot Product - çünkü L2 Normalize yaptık)
    # Matris çarpımı ile herkesin herkese uzaklığını buluyoruz
    sim_matrix = np.dot(embeddings, embeddings.T)
    np.fill_diagonal(sim_matrix, -1) # Kendisini bulmasın

    # Recall Hesapla
    recall_1_count = 0
    recall_5_count = 0
    
    for i in range(len(embeddings)):
        # En benzer K indeksleri bul (Büyükten küçüğe sırala)
        sorted_indices = np.argsort(sim_matrix[i])[::-1]
        
        # En benzer 1. resmin etiketi aynı mı?
        if labels[sorted_indices[0]] == labels[i]:
            recall_1_count += 1
            
        # İlk 5 resimden en az biri aynı mı?
        top_5_indices = sorted_indices[:5]
        if any(labels[idx] == labels[i] for idx in top_5_indices):
            recall_5_count += 1

    recall_1 = recall_1_count / len(embeddings)
    recall_5 = recall_5_count / len(embeddings)
    
    print(f"\n🏆 SONUÇLAR:")
    print(f"Recall@1: {recall_1:.4f} (En benzer resim doğru mu?)")
    print(f"Recall@5: {recall_5:.4f} (En benzer 5 resimden biri doğru mu?)")
    
    # Sonuçları Kaydet
    metrics = {
        "recall_1": recall_1,
        "recall_5": recall_5
    }
    
    with open("eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    compute_metrics()