FROM python:3.12-slim

# Build tools required for dlib (used by face-recognition)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (split so dlib compile layer is cached separately)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir dlib \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code and pre-downloaded YOLO weights
COPY . .

# Persistent face database lives in a volume mounted at /app/face_db
VOLUME ["/app/face_db"]

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
