#!/usr/bin/env bash
# Bu repoyu bir Hugging Face Space'e (Docker SDK) deploy eder.
#
# Kullanım:
#   1) huggingface.co üzerinde "Create new Space" -> SDK: Docker -> boş bir Space oluştur.
#   2) huggingface-cli login  (veya git için HF access token'ını hazırla)
#   3) ./scripts/deploy_hf_space.sh https://huggingface.co/spaces/<kullanici-adi>/<space-adi>
#
# Not: Bu script Space repo'suna force-push yapar (GitHub reponu etkilemez,
# sadece belirttiğin HF Space'in git remote'unu günceller).
set -euo pipefail

SPACE_URL="${1:?Kullanım: $0 <hf-space-git-url>  (örn: https://huggingface.co/spaces/kullanici/petfinder-ai)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="$(mktemp -d)"

echo "📦 Deploy dosyaları hazırlanıyor: $STAGE_DIR"
cp -r "$REPO_ROOT/app" "$STAGE_DIR/app"
cp -r "$REPO_ROOT/models" "$STAGE_DIR/models"
cp "$REPO_ROOT/Dockerfile" "$REPO_ROOT/supervisord.conf" "$STAGE_DIR/"
cp "$REPO_ROOT/HF_SPACE_README.md" "$STAGE_DIR/README.md"

find "$STAGE_DIR" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$STAGE_DIR/models" -type f ! -name "final_breed_model.weights.h5" -delete
rm -rf "$STAGE_DIR/app/backend/gallery/ads"
mkdir -p "$STAGE_DIR/app/backend/gallery/ads"

cd "$STAGE_DIR"
git init -q
git add .
git commit -q -m "Deploy PetFinder AI"
git branch -M main
git remote add space "$SPACE_URL"
git push --force space main

echo "✅ Push tamamlandı: $SPACE_URL"
echo "🧹 Geçici klasörü silmek istersen: rm -rf $STAGE_DIR"
