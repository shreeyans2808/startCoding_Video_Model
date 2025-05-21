import os
import cv2
import numpy as np
from groq import Groq
from PIL import Image
import logging
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
import textwrap
from dotenv import load_dotenv
import datetime
import requests
from duckduckgo_search import DDGS

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    from search_index import get_text_content, search_index
except ImportError:
    logger.warning("search_index module not found. Implementing placeholder functions.")
    
    def get_text_content(query, top_k=3):
        logger.warning("Using placeholder get_text_content function")
        return [f"Content about {query}"]
    
    def search_index(query, content_type, top_k=1):
        logger.warning("Using placeholder search_index function")
        return []

class MovieGenerator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        self.groq_client = Groq(api_key=api_key)
        
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            logger.info("FFmpeg is available")
            self.ffmpeg_available = True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Video creation will be limited.")
            self.ffmpeg_available = False
    
    def search_image(self, query):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=1))
                if not results:
                    logger.warning(f"No images found for query: {query}")
                    return None
                
                image_url = results[0]['image']
                
                response = requests.get(image_url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    if image.size != (512, 512):
                        image = image.resize((512, 512), Image.Resampling.LANCZOS)
                    
                    return image
                else:
                    logger.error(f"Failed to download image from {image_url}")
                    return None
        except Exception as e:
            logger.error(f"Failed to search/download image: {str(e)}")
            return None
    
    def get_search_context(self, query, top_k=8):
        try:
            results = search_index(query, "text", top_k=top_k)
            if not results:
                logger.warning("No search results found")
                return [query]
            
            context = []
            for result in results:
                if result["type"] == "text":
                    context.append(result["content"])
            
            if not context:
                logger.warning("No text content found in search results")
                return [query]
            
            return context
        except Exception as e:
            logger.error(f"Failed to get search context: {str(e)}")
            return [query]
    
    def generate_script(self, query, context):
        try:
            combined_context = "\n".join(context)
            
            prompt = f"""Based on the following context about {query}, generate a 1-minute educational script.
            The script should be engaging and informative, and should cover the most important aspects from the context.
            Make sure the script is exactly 60 seconds long when read at a normal pace.
            
            Context:
            {combined_context}
            
            Generate a script that flows naturally and covers the key points."""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.7,
                max_tokens=1000
            )
            
            script = response.choices[0].message.content
            logger.info("Successfully generated script")
            return script
        except Exception as e:
            logger.error(f"Failed to generate script: {str(e)}")
            return None
    
    def summarize_chunk(self, chunk):
        try:
            prompt = f"""Summarize the following text in 2-3 short sentences, focusing on the key points:
            {chunk}"""
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.7,
                max_tokens=100
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Failed to summarize chunk: {str(e)}")
            return chunk
    
    def split_script_into_chunks(self, script, num_chunks=12):
        if not script:
            return []
            
        words = script.split()
        words_per_chunk = max(1, len(words) // num_chunks)
        chunks = []
        
        for i in range(0, len(words), words_per_chunk):
            chunk = ' '.join(words[i:i + words_per_chunk])
            summary = self.summarize_chunk(chunk)
            chunks.append({
                'original': chunk,
                'summary': summary
            })
        
        return chunks
    
    def get_or_generate_image(self, text_chunk):
        try:
            results = search_index(text_chunk, "image", top_k=10)

            for result in results:
                if result["type"] == "image":
                    try:
                        image = Image.open(result["content"])
                        if image.mode != 'RGB':
                            image = image.convert('RGB')
                        
                        if image.size != (512, 512):
                            image = image.resize((512, 512), Image.Resampling.LANCZOS)
                        
                        logger.info("Found image from search index")
                        return image
                    except Exception as e:
                        logger.warning(f"Failed to load image from index: {str(e)}")
                        continue
            logger.info("No image found in index, searching DuckDuckGo")
            image = self.search_image(text_chunk)
            if image:
                return image
            
            logger.warning("No image found, using blank image as fallback")
            return Image.new('RGB', (512, 512), color='white')
        except Exception as e:
            logger.error(f"Failed to get image: {str(e)}")
            return Image.new('RGB', (512, 512), color='white')
    
    def add_text_to_image(self, image, text, subtitle):
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
        height, width = img_cv.shape[:2]
        overlay = img_cv.copy()

        wrapper = textwrap.TextWrapper(width=50)
        text_lines = wrapper.wrap(text)
        text_height = 30 * len(text_lines) + 20
        subtitle_lines = wrapper.wrap(subtitle)
        subtitle_height = 30 * len(subtitle_lines) + 20
        cv2.rectangle(overlay, (0, height - text_height - subtitle_height - 20), 
                     (width, height), (0, 0, 0), -1)
        
        alpha = 0.7
        img_cv = cv2.addWeighted(overlay, alpha, img_cv, 1 - alpha, 0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        font_color = (255, 255, 255)
        
        y_position = height - subtitle_height - 10 - 30 * (len(text_lines) - 1)
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            x_position = (width - text_size[0]) // 2
            cv2.putText(img_cv, line, (x_position, y_position), 
                       font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            y_position += 30
        
        y_position = height - 10 - 30 * (len(subtitle_lines) - 1)
        for line in subtitle_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            x_position = (width - text_size[0]) // 2
            cv2.putText(img_cv, line, (x_position, y_position), 
                       font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            y_position += 30
        
        return img_cv
    
    def create_video_with_opencv(self, image_paths, output_path, duration_per_image=5):
        if not image_paths:
            raise ValueError("No images provided for video creation")

        img = cv2.imread(image_paths[0])
        height, width, layers = img.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 24
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for img_path in image_paths:
            img = cv2.imread(img_path)
            for _ in range(duration_per_image * fps):
                out.write(img)
        out.release()
        logger.info(f"Created video at {output_path}")
        return output_path
    
    def create_video_with_ffmpeg(self, image_dir, output_path, duration_per_image=5):
        if not self.ffmpeg_available:
            raise RuntimeError("FFmpeg is not available. Cannot create video.")
        
        image_list_file = os.path.join(image_dir, "images.txt")
        with open(image_list_file, 'w') as f:
            for img_file in sorted(os.listdir(image_dir)):
                if img_file.endswith('.jpg') or img_file.endswith('.png'):
                    f.write(f"file '{img_file}'\n")
                    f.write(f"duration {duration_per_image}\n")
        
        try:
            cmd = [
                "ffmpeg",
                "-y", 
                "-f", "concat",
                "-safe", "0",
                "-i", image_list_file,
                "-vsync", "vfr",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, check=True, cwd=image_dir)
            logger.info(f"Created video with FFmpeg at {output_path}")
            return output_path
        except subprocess.SubprocessError as e:
            logger.error(f"FFmpeg error: {str(e)}")
            raise
    
    def create_video(self, query, output_path=None):
        try:
            if output_path is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")

            with tempfile.TemporaryDirectory() as temp_dir:
                context = self.get_search_context(query, top_k=8)

                script = self.generate_script(query, context)
                if not script:
                    logger.error("Failed to generate script. Cannot proceed with video creation.")
                    return None

                chunks = self.split_script_into_chunks(script, num_chunks=12)
                script_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(output_path))[0]}_script.txt")
                with open(script_path, 'w') as f:
                    f.write("Full Script:\n")
                    f.write(script)
                    f.write("\n\nChunks with Summaries:\n")
                    for i, chunk in enumerate(chunks):
                        f.write(f"\nChunk {i+1}:\n")
                        f.write(f"Original: {chunk['original']}\n")
                        f.write(f"Summary: {chunk['summary']}\n")

                frame_paths = []
                for i, chunk in enumerate(chunks):
                    logger.info(f"Processing chunk {i+1}/{len(chunks)}")

                    pil_image = self.get_or_generate_image(chunk['summary'])

                    img_with_text = self.add_text_to_image(
                        pil_image, 
                        chunk['original'],
                        chunk['summary']
                    )

                    frame_path = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
                    cv2.imwrite(frame_path, img_with_text)
                    frame_paths.append(frame_path)
                if self.ffmpeg_available:
                    return self.create_video_with_ffmpeg(temp_dir, output_path)
                else:
                    return self.create_video_with_opencv(frame_paths, output_path)
                
        except Exception as e:
            logger.error(f"Failed to create video: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        missing_deps = []
        try:
            import cv2
        except ImportError:
            missing_deps.append("opencv-python")
            
        try:
            import groq
        except ImportError:
            missing_deps.append("groq")
            
        try:
            import duckduckgo_search
        except ImportError:
            missing_deps.append("duckduckgo-search")
        
        if missing_deps:
            print("Missing dependencies detected. Please install them with pip:")
            for dep in missing_deps:
                print(f"pip install {dep}")
            exit(1)

        generator = MovieGenerator()
        query = "how does a piston work"
        output_path = generator.create_video(query)
        if output_path and os.path.exists(output_path):
            print(f"Video created successfully at: {output_path}")
            print(f"Script saved as: {os.path.splitext(output_path)[0]}_script.txt")
        else:
            print("Failed to create video. Check the logs for details.")
    except Exception as e:
        logger.error(f"Failed to run movie generator: {str(e)}", exc_info=True)