import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "indices")
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "videos"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "texts"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "audio"), exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Index directory: {INDEX_DIR}")
print(f"Data directory: {DATA_DIR}")

FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")
CHUNK_SIZE = 512
MODEL_NAME = "openai/clip-vit-large-patch14"

AUDIO_MODEL = "openai/whisper-small"
AUDIO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VIDEO_FRAME_INTERVAL = 5 
VIDEO_SIMILARITY_THRESHOLD = 0.95
VIDEO_AUDIO_PADDING = 1.0