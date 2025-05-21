import os
from utils import *
from config import *
from PIL import Image
import librosa
import uuid

def add_content(content, content_type, source=None):
    index, metadata = load_or_create_index()
    try:
        if content_type == "text":
            chunks = chunk_text(content)
            source = source or "custom_text"
        elif content_type == "image":
            if isinstance(content, str):
                if not os.path.exists(content):
                    raise FileNotFoundError(f"Image not found: {content}")
                chunks = [Image.open(content).convert("RGB")]
                source = os.path.basename(content)
            elif isinstance(content, Image.Image):
                chunks = [content.convert("RGB")]
                source = source or "custom_image"
            else:
                raise ValueError("Image content must be a file path or PIL Image")
        elif content_type == "audio":
            if not os.path.exists(content):
                raise FileNotFoundError(f"Audio not found: {content}")
            audio, sr = librosa.load(content, sr=16000)
            text = transcribe_audio(audio, sr)
            chunks = chunk_text(text) if text else []
            source = os.path.basename(content)
        elif content_type == "video":
            if not os.path.exists(content):
                raise FileNotFoundError(f"Video not found: {content}")
            video_frames = process_video(content)
            source = os.path.basename(content)
            chunks = video_frames
        else:
            raise ValueError(f"Unsupported content type: {content_type}")

        for chunk in chunks:
            if content_type == "video":
                frame_emb = embed_image(chunk["frame_image"])
                audio_text = transcribe_audio(chunk["audio_chunk"])
                text_emb = embed_text(audio_text) if audio_text else None

                doc_id = str(uuid.uuid4())

                metadata[len(metadata)] = {
                    "type": "video_frame",
                    "content": chunk["frame_image"],
                    "source": source,
                    "timestamp": chunk["frame_time"],
                    "doc_id": doc_id,
                    "linked_audio": f"{doc_id}_audio"
                }
                index.add(frame_emb.reshape(1, -1))

                if text_emb is not None:
                    metadata[len(metadata)] = {
                        "type": "video_audio",
                        "content": audio_text,
                        "source": source,
                        "timestamp": chunk["frame_time"],
                        "doc_id": f"{doc_id}_audio",
                        "linked_frame": doc_id
                    }
                    index.add(text_emb.reshape(1, -1))
            else:
                emb = embed_image(chunk) if content_type == "image" else embed_text(chunk)
                metadata[len(metadata)] = {
                    "type": content_type,
                    "content": chunk,
                    "source": source,
                    "timestamp": None
                }
                index.add(emb.reshape(1, -1))

        save_index(index, metadata)
        print(f" Added {len(chunks)} {content_type} chunks from {source}")
        return True

    except Exception as e:
        print(f" Error adding {content_type}: {str(e)}")
        return False

def add_file(file_path, content_type=None):
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        return False
        
    if content_type is None:
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.txt', '.md', '.html', '.htm']:
            content_type = "text"
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
            content_type = "image"
        elif ext in ['.mp3', '.wav', '.ogg', '.m4a']:
            content_type = "audio"
        elif ext in ['.mp4', '.avi', '.mov', '.wmv']:
            content_type = "video"
        else:
            print(f" Unsupported file type: {ext}")
            return False
    
    if content_type == "text":
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return add_content(content, "text", file_path)
        except Exception as e:
            print(f" Error reading text file: {str(e)}")
            return False
    
    return add_content(file_path, content_type)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add content to the search index")
    parser.add_argument("path", help="Path to the file or content to add")
    parser.add_argument("--type", choices=["text", "image", "audio", "video"], 
                       help="Content type (auto-detected if not specified)")
    
    args = parser.parse_args()
    add_file(args.path, args.type)
