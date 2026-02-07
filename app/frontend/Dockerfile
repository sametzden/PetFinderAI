# Temel imaj
FROM python:3.10-slim

WORKDIR /app

# Kütüphaneleri kopyala ve kur
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kodları kopyala
COPY . .

# Streamlit portunu aç
EXPOSE 8501

# Uygulamayı başlat
CMD ["streamlit", "run", "app/frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]