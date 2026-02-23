FROM python:3.10-slim

WORKDIR /app

# Requirements dosyasını kopyala
COPY requirements.txt .


RUN pip install --upgrade pip && \
    pip install --default-timeout=2000 --retries=10 --no-cache-dir -r requirements.txt


RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Kodları kopyala
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]