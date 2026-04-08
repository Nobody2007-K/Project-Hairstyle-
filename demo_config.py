 import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Model paths
MODEL_PATH = BASE_DIR / "models" / "face_shape_b0_260_final.keras"
TFLITE_MODEL_PATH = BASE_DIR / "models" / "face_shape_b0_260_final.tflite"

# Image settings
IMG_SIZE = 260
TOP_MARGIN = 0.35
BOTTOM_MARGIN = 0.60
SIDE_MARGIN = 0.28

# Upload settings
UPLOAD_FOLDER = BASE_DIR.parent / "frontend" / "static" / "uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Face shape classes
CLASS_NAMES = ["Heart", "Oblong", "Oval", "Round", "Square"]

# Create upload folder
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
