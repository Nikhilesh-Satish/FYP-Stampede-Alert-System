import os
import gdown
from ultralytics import YOLO

FILE_ID = "1im4qMsoiBrEEs_EyUGofjz68_qdYcSJl"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "stampede_model.pt")

# ✅ Global cache so model loads only once
_cached_model = None


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found locally...")
        print("Downloading from Google Drive...")

        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

        # ✅ Check file downloaded correctly
        if os.path.getsize(MODEL_PATH) < 5_000_000:
            raise RuntimeError("❌ Download failed: file too small (not a real .pt model)")

        print("✅ Model downloaded successfully!")

    else:
        print("✅ Model already exists locally.")


def load_model():
    global _cached_model

    if _cached_model is None:
        download_model()

        print("🚀 Loading YOLO model...")
        _cached_model = YOLO(MODEL_PATH)

        print("✅ Model loaded successfully!")

    return _cached_model
