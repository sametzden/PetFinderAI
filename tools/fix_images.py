import os
from PIL import Image
import warnings

# Resimlerin olduğu klasör
DATA_DIR = "data/train"
print(f"🔧 Tamirat başlıyor: {DATA_DIR}")

fixed_count = 0
deleted_count = 0
error_count = 0

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(root, file)
            
            try:
                # Resmi aç ve zorla yükle (Load data)
                with Image.open(file_path) as img:
                    img.load() # Veriyi belleğe çek (Bozuksa burada patlar)
                    
                    # RGB moda çevir (PNG'deki şeffaflık veya CMYK sorunlarını çözer)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Kendi üzerine tekrar kaydet (Bu işlem dosyayı temizler)
                    img.save(file_path, "JPEG", quality=95)
                    fixed_count += 1
                    
            except Exception as e:
                # Eğer açarken veya kaydederken hata verirse, dosya cidden bozuktur. Sil.
                print(f"❌ Kurtarılamadı, siliniyor: {file_path} | Hata: {e}")
                try:
                    os.remove(file_path)
                    deleted_count += 1
                except:
                    pass

print(f"\n✨ Rapor:")
print(f"✅ Onarılan/Yenilenen: {fixed_count}")
print(f"🗑️ Silinen (Çöp): {deleted_count}")