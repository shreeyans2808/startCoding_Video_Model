import torch
import faiss
import pickle
import numpy as np
import cv2
import librosa
import uuid
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from transformers import (
    CLIPProcessor, 
    CLIPModel,
    WhisperForConditionalGeneration,
    WhisperProcessor
)
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
clip_model.eval()

whisper_processor = WhisperProcessor.from_pretrained(AUDIO_MODEL)
whisper_model = WhisperForConditionalGeneration.from_pretrained(AUDIO_MODEL).to(AUDIO_DEVICE)
whisper_model.eval()
EMBEDDING_DIM = 768

def load_or_create_index():
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        assert index.d == EMBEDDING_DIM
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        print(" Loaded existing index and metadata")
    except (FileNotFoundError, RuntimeError, AssertionError) as e:
        print(f" {str(e)} - Creating new index")
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = {}
        print(f" Created new index with dimension {EMBEDDING_DIM}")
    return index, metadata

def save_index(index, metadata):
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f" Saved index and metadata")

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    audio, sr = librosa.load(video_path, sr=16000)
    duration = librosa.get_duration(y=audio, sr=sr)
    
    frames = []
    previous_frame = None
    current_time = 0
    
    while current_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time*1000)
        ret, frame = cap.read()
        if not ret:
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if previous_frame is not None:
            similarity = ssim(previous_frame, gray_frame)
            if similarity > VIDEO_SIMILARITY_THRESHOLD:
                current_time += VIDEO_FRAME_INTERVAL
                continue
                
        start_time = max(0, current_time - VIDEO_AUDIO_PADDING)
        end_time = min(duration, current_time + VIDEO_AUDIO_PADDING)
        audio_chunk = audio[int(start_time*sr):int(end_time*sr)]
        
        frames.append({
            "frame_time": current_time,
            "frame_image": Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
            "audio_start": start_time,
            "audio_end": end_time,
            "audio_chunk": audio_chunk
        })
        
        previous_frame = gray_frame
        current_time += VIDEO_FRAME_INTERVAL
        
    cap.release()
    return frames

def transcribe_audio(audio_chunk, sr=16000):
    try:
        inputs = whisper_processor(
            audio_chunk, 
            sampling_rate=sr, 
            return_tensors="pt"
        ).input_features.to(AUDIO_DEVICE)
        
        with torch.no_grad():
            predicted_ids = whisper_model.generate(inputs)
        
        return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
    except Exception as e:
        print(f"Audio transcription failed: {str(e)}")
        return None

def embed_text(text):
    inputs = clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**inputs)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
    return text_emb.cpu().squeeze(0).numpy()

def embed_image(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        img_emb = clip_model.get_image_features(**inputs)
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
    return img_emb.cpu().squeeze(0).numpy()

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks