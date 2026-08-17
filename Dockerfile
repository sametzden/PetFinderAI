# Tek konteynerde Backend (FastAPI) + Frontend (Streamlit) — Hugging Face Spaces için.
# Yerel geliştirmede docker-compose.yml (ayrı backend/frontend konteynerleri) kullanılabilir;
# bu Dockerfile ise tek portlu barındırma ortamları (HF Spaces gibi) içindir.
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app/backend/requirements.txt backend-requirements.txt
COPY app/frontend/requirements.txt frontend-requirements.txt
RUN pip install --upgrade pip && \
    pip install --default-timeout=2000 --retries=10 --no-cache-dir \
        -r backend-requirements.txt \
        -r frontend-requirements.txt

COPY app/backend/ /app/backend/
COPY app/frontend/ /app/frontend/
COPY models/*.h5 /app/backend/models/

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

ENV API_URL=http://localhost:8000
# HF Spaces varsayılan olarak 7860 portunu bekler.
EXPOSE 7860

CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
