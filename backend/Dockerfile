FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_ENV=production \
    STAMPEDE_MODEL_PATH=yolov8m.pt \
    DEPLOYMENT_PROFILE=standard \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgomp1 libgl1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "gunicorn -w 1 --threads 8 -k gthread -b 0.0.0.0:${PORT:-7860} wsgi:app --timeout 180"]
