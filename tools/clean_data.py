import os
from PIL import Image
import warnings

# Veri yolu (params.yaml'dan da okutabilirdik ama pratik olsun)
DATA_DIR = "data/train"

print(f"🧹 Temizlik başlıyor: {DATA_DIR}")
deleted_count = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            try:
                # Resmi açmayı dene, bozuksa hata verir
                with Image.open(file_path) as img:
                    img.verify() 
            except (IOError, SyntaxError) as e:
                print(f"❌ Bozuk dosya siliniyor: {file_path}")
                os.remove(file_path)
                deleted_count += 1

print(f"✨ Bitti! Toplam silinen bozuk dosya: {deleted_count}")