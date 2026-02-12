import tensorflow as tf
from tensorflow.keras import layers, Model, applications, optimizers
import os
import random
import yaml
import json
import numpy as np

# 1. Parametreleri Yükle
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)["train"]

IMG_SIZE = params["img_size"]
BATCH_SIZE = params["batch_size"]
DATA_PATH = params["data_path"]
EPOCHS = params["epochs"]

# --- CUSTOM KATMANLAR (Senin Kodundan) ---
class L2Normalization(layers.Layer):
    """L2 normalization layer - H5 formatında kaydedilebilir"""
    def __init__(self, **kwargs):
        super(L2Normalization, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def get_config(self):
        return super(L2Normalization, self).get_config()

class TripletLossLayer(layers.Layer):
    def __init__(self, alpha=0.2, loss_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.loss_weight = loss_weight

    def call(self, inputs):
        anchor, positive, negative = inputs
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
        basic_loss = pos_dist - neg_dist + self.alpha
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        self.add_loss(loss * self.loss_weight)
        return anchor

# --- MODEL MİMARİSİ ---
def create_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=10):
    # Base Model
    base_cnn = applications.MobileNetV2(
        input_shape=input_shape, weights='imagenet', include_top=False
    )
    base_cnn.trainable = False # Transfer Learning

    # Ortak Katmanlar
    img_input = layers.Input(shape=input_shape)
    x = base_cnn(img_input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)

    # Embedding Çıkışı (Benzerlik için)
    embedding = layers.Dense(params["embedding_dim"], name="embedding")(x)
    normalized_embedding = L2Normalization(name='std_embedding')(embedding)

    # Sınıflandırma Çıkışı (Classification Head)
    class_output = layers.Dense(num_classes, activation='softmax', name="class_output")(x)

    # Çıkarım (Inference) Modeli
    base_model = Model(inputs=img_input, outputs=[normalized_embedding, class_output])

    # --- EĞİTİM MODELİ (Triplet Inputs) ---
    input_anchor = layers.Input(shape=input_shape, name="anchor")
    input_positive = layers.Input(shape=input_shape, name="positive")
    input_negative = layers.Input(shape=input_shape, name="negative")

    emb_a, class_a = base_model(input_anchor)
    emb_p, _ = base_model(input_positive)
    emb_n, _ = base_model(input_negative)

    # Triplet Loss Katmanı
    TripletLossLayer(alpha=params["alpha"], loss_weight=params["loss_weight"])([emb_a, emb_p, emb_n])

    # Eğitim Modeli (Giriş: 3 Resim, Çıkış: Sınıf Tahmini)
    training_model = Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=class_a,
        name="training_model"
    )

    return base_model, training_model

# --- VERİ YÜKLEME VE GENERATOR ---
def load_and_process_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return applications.mobilenet_v2.preprocess_input(img)

def get_class_data(directory):
    class_indices = {}
    classes = sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])
    
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(files) > 1: # En az 2 resim şart
            class_indices[cls] = files
            
    return class_indices, classes

def triplet_generator(files_dict, classes, batch_size=32):
    class_names = list(files_dict.keys())
    while True:
        anchors, positives, negatives, labels = [], [], [], []
        
        for _ in range(batch_size):
            anchor_cls = random.choice(class_names)
            if len(files_dict[anchor_cls]) < 2: continue
            
            # Anchor & Positive
            a_path, p_path = random.sample(files_dict[anchor_cls], 2)
            
            # Negative
            neg_cls = random.choice([c for c in class_names if c != anchor_cls])
            n_path = random.choice(files_dict[neg_cls])
            
            anchors.append(load_and_process_image(a_path))
            positives.append(load_and_process_image(p_path))
            negatives.append(load_and_process_image(n_path))
            labels.append(classes.index(anchor_cls))
            
        yield (
            (tf.stack(anchors), tf.stack(positives), tf.stack(negatives)),
            tf.convert_to_tensor(labels)
        )

# --- ANA EĞİTİM AKIŞI ---
if __name__ == "__main__":
    print(f"📂 Veri yolu: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Veri bulunamadı: {DATA_PATH}. Lütfen 'dvc pull' yapın.")

    files_dict, classes = get_class_data(DATA_PATH)
    NUM_CLASSES = len(classes)
    print(f"🐶 Sınıf Sayısı: {NUM_CLASSES}")

    # Dataset Oluştur
    train_ds = tf.data.Dataset.from_generator(
        lambda: triplet_generator(files_dict, classes, BATCH_SIZE),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).prefetch(tf.data.AUTOTUNE)

    # Modeli Hazırla
    base_model, training_model = create_model(num_classes=NUM_CLASSES)
    
    training_model.compile(
        optimizer=optimizers.Adam(params["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"🚀 Eğitim Başlıyor... Epochs: {EPOCHS}")
    history = training_model.fit(
        train_ds,
        steps_per_epoch=params["steps_per_epoch"],
        epochs=EPOCHS
    )

    # Kaydet
    if not os.path.exists("models"):
        os.makedirs("models")
        
    base_model.save_weights(params["model_output"])
    print(f"✅ Model ağırlıkları kaydedildi: {params['model_output']}")

    # Metrikleri Kaydet (DVC için)
    metrics = {
        "accuracy": float(history.history['accuracy'][-1]),
        "loss": float(history.history['loss'][-1])
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)