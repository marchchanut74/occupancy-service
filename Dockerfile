FROM python:3.11-slim

WORKDIR /app

# System deps:
#   libglib2.0-0 — ยังจำเป็นสำหรับ opencv-python-headless (GLib runtime)
#   libgl1 ไม่จำเป็น เพราะ headless ไม่ใช้ OpenGL
RUN apt-get update && \
    apt-get install -y --no-install-recommends libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Directories
RUN mkdir -p /data/snapshots /app/data /app/logs /app/masks

EXPOSE 8000

CMD ["python", "-m", "occupancy_service.main", "--config", "config.yaml"]
