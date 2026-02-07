import os
import gdown
from ultralytics import YOLO

FILE_ID = "1im4qMsoiBrEEs_EyUGofjz68_qdYcSJl"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "stampede_model.pt")


def download_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        print("📥 Model not found locally...")
        print("Downloading from Google Drive...")

        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

        print("✅ Model downloaded successfully!")

    else:
        print("✅ Model already exists locally.")


def load_model():
    download_model()

    print("🚀 Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("✅ Model loaded successfully!")
    return model
