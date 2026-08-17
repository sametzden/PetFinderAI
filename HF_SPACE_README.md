---
title: PetFinder AI
emoji: 🐾
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🐾 PetFinder AI — Kayıp Evcil Hayvan Bulma Sistemi

Bir evcil hayvan fotoğrafı yükleyin; derin metrik öğrenme modeli görseli
vektöre çevirir, ırkını tahmin eder ve galerideki en görsel-benzer
kayıtları getirir.

- **Model:** MobileNetV2 tabanlı embedding + triplet loss (Recall@1 %83.8)
- **Bekçi model:** YOLOv8n (yüklenen görselin gerçekten bir hayvan olduğunu doğrular)
- **Vektör arama:** Qdrant (embedded)

Kaynak kod ve eğitim pipeline'ı: https://github.com/sametzden/PetFinderAI
