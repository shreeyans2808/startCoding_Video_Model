import os
from groq import Groq
from search_index import search_index
import json
from dotenv import load_dotenv
import numpy as np
from duckduckgo_search import DDGS
from moviepy import ImageClip, concatenate_videoclips, CompositeVideoClip, AudioFileClip
import requests
from PIL import Image, ImageDraw, ImageFont
import tempfile
from gtts import gTTS
import subprocess

# Load environment variables
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_key_context(query):
    """Extract the most important word or phrase from the user's query"""
    prompt = f"""Extract the single most important word or short phrase from this query that would help in finding relevant images.
    This should be the main subject or topic that would be most visually distinctive.
    
    Query: {query}
    
    Most important word/phrase:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that extracts key context from queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        max_tokens=50
    )
    
    return completion.choices[0].message.content.strip()

def format_query_with_groq(user_input, purpose="search", context=""):
    """Format the user's query using Groq"""
    if purpose == "image_search":
        prompt = f"""Given the following text and context, create multiple image search queries.
        Create 3 different search queries:
        1. A specific, detailed query
        2. A broader query
        3. A simple, keyword-based query
        
        Context: {context}
        Text: {user_input}
        
        Return the queries in this format:
        specific: [specific query]
        broad: [broad query]
        simple: [simple query]"""
    else:
        prompt = f"""Format this query into a clear search query:
        User query: {user_input}
        Formatted search query:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that formats queries."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.3,
        max_tokens=200
    )
    
    if purpose == "image_search":
        response = completion.choices[0].message.content.strip()
        queries = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                queries[key.strip()] = value.strip()
        return queries
    else:
        return completion.choices[0].message.content.strip()

def clean_text(text):
    """Remove time-related words from text"""
    if not text:
        return text
    return ' '.join([word for word in str(text).split() 
                   if word.lower() not in ['5', 'seconds', 'second']])

def download_image(url):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(response.content)
            temp_file.close()
            return temp_file.name
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
    return None

def generate_audio_for_sentence(sentence, output_path):
    """Generate clean audio without time mentions"""
    try:
        clean_sentence = clean_text(sentence)
        if not clean_sentence:
            return False
            
        tts = gTTS(text=clean_sentence, lang='en', slow=False)
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

def create_text_image(text, image_size, font_size=24):
    """Create text overlay without time mentions"""
    try:
        clean_text_content = clean_text(text)
        if not clean_text_content:
            return None

        text_img = Image.new('RGBA', image_size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_img)
        
        try:
            font_paths = [
                "/System/Library/Fonts/Arial.ttf",
                "/Windows/Fonts/arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, font_size)
                        break
                    except Exception:
                        continue
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
        
        words = clean_text_content.split()
        lines = []
        current_line = []
        
        for word in words:
            test_line = ' '.join(current_line + [word])
            try:
                bbox = draw.textbbox((0, 0), test_line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(test_line) * (font_size * 0.6)
            
            if text_width <= image_size[0] - 40:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        line_height = font_size + 5
        total_text_height = len(lines) * line_height
        start_y = max(10, image_size[1] - total_text_height - 20)
        
        bg_y = start_y - 10
        bg_height = total_text_height + 20
        draw.rectangle([(0, bg_y), (image_size[0], bg_y + bg_height)], 
                      fill=(0, 0, 0, 180))
        
        for i, line in enumerate(lines):
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                text_width = bbox[2] - bbox[0]
            except:
                text_width = len(line) * (font_size * 0.6)
            
            x = max(20, (image_size[0] - text_width) // 2)
            y = start_y + i * line_height
            
            outline_width = 1
            for dx in [-outline_width, 0, outline_width]:
                for dy in [-outline_width, 0, outline_width]:
                    if dx != 0 or dy != 0:
                        draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
            
            draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
        
        temp_text_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        text_img.save(temp_text_file.name, 'PNG')
        temp_text_file.close()
        return temp_text_file.name
        
    except Exception as e:
        print(f"Error creating text image: {str(e)}")
        return None

def get_audio_duration(audio_path):
    """Get duration of audio file"""
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        return duration
    except Exception:
        return 5.0

def find_relevant_image(sentence, context=""):
    """Find the most relevant image for a given sentence"""
    print(f"\nSearching for image for sentence: {sentence}")
    
    print("Trying DuckDuckGo search first...")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(keywords=f"{context} {sentence}", max_results=5))
            if results and results[0].get('image'):
                print(f"Found image in DuckDuckGo: {results[0]['image']}")
                return {
                    "image_path": results[0]['image'],
                    "score": None,
                    "source": "duckduckgo",
                    "type": "image"
                }
    except Exception as e:
        print(f"DuckDuckGo search failed: {str(e)}")
    
    print("Trying index search...")
    try:
        results = search_index(sentence, "text", top_k=5)
        for result in results:
            if result.get("type") in ["image", "video_frame"]:
                print(f"Found image in index: {result.get('content')}")
                return {
                    "image_path": result.get("content"),
                    "score": result.get("score"),
                    "source": "index",
                    "type": result.get("type")
                }
    except Exception as e:
        print(f"Index search failed: {str(e)}")
    
    print("No image found in any source")
    return None

def generate_video_script(results, formatted_query):
    """Generate script without time announcements"""
    serializable_results = []
    for result in results:
        serializable_result = {}
        for key, value in result.items():
            if isinstance(value, np.float32):
                serializable_result[key] = float(value)
            else:
                serializable_result[key] = value
        serializable_results.append(serializable_result)
    
    prompt = f"""Create a 1-minute video script (12 lines, 5 seconds each) based on these results.
    Focus on key information, flow naturally, and NEVER mention time durations.
    
    Query: {formatted_query}
    
    Results:
    {json.dumps(serializable_results, indent=2)}
    
    Clean script (12 lines, no time mentions):"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "Create engaging video scripts without time announcements."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        max_tokens=500
    )
    
    script = completion.choices[0].message.content.strip()
    return '\n'.join([line for line in script.split('\n') 
                     if 'second' not in line.lower() and '5' not in line])

def create_video(script_with_images, output_path="output_video.mp4"):
    """Create video without time announcements"""
    try:
        if not script_with_images:
            print("No content for video creation")
            return

        temp_dir = tempfile.mkdtemp()
        audio_files = []
        video_clips = []
        text_image_files = []

        # Generate clean audio files
        for i, item in enumerate(script_with_images):
            if item.get('sentence'):
                audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")
                if generate_audio_for_sentence(item['sentence'], audio_path):
                    audio_files.append(audio_path)
                else:
                    audio_files.append(None)
            else:
                audio_files.append(None)

        # Create video clips
        for i, item in enumerate(script_with_images):
            if not item.get('image'):
                continue
                
            image_path = download_image(item['image'])
            if not image_path:
                continue
                
            try:
                clip_duration = 5.0  # Fixed 5-second duration
                if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                    clip_duration = get_audio_duration(audio_files[i])

                img_clip = ImageClip(image_path, duration=clip_duration)
                
                with Image.open(image_path) as img:
                    img_size = img.size
                
                if item.get('sentence'):
                    text_image_path = create_text_image(item['sentence'], img_size)
                    if text_image_path:
                        text_image_files.append(text_image_path)
                        text_clip = ImageClip(text_image_path, duration=clip_duration)
                        video_clip = CompositeVideoClip([img_clip, text_clip])
                    else:
                        video_clip = img_clip
                else:
                    video_clip = img_clip
                
                if i < len(audio_files) and audio_files[i] and os.path.exists(audio_files[i]):
                    try:
                        audio_clip = AudioFileClip(audio_files[i])
                        video_clip.audio = audio_clip
                    except Exception:
                        pass
                
                video_clips.append(video_clip)
                os.unlink(image_path)
                
            except Exception:
                if os.path.exists(image_path):
                    os.unlink(image_path)
                continue

        if video_clips:
            final_clip = concatenate_videoclips(video_clips, method="compose")
            final_clip.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                threads=4,
                preset='medium',
                bitrate='2000k'
            )
            final_clip.close()
            
        # Cleanup
        for file in audio_files + text_image_files:
            if file and os.path.exists(file):
                os.unlink(file)
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass
            
    except Exception as e:
        print(f"Error creating video: {str(e)}")

def main():
    user_input = input("Enter your search query: ")
    
    key_context = extract_key_context(user_input)
    formatted_query = format_query_with_groq(user_input)
    if isinstance(formatted_query, dict):
        formatted_query = formatted_query.get('specific', user_input)
    
    try:
        results = search_index(formatted_query, "text", top_k=10)
        if results:
            script = generate_video_script(results, formatted_query)
            sentences = [s.strip() for s in script.split('\n') if s.strip()]
            
            script_with_images = []
            for sentence in sentences:
                image_result = find_relevant_image(sentence, key_context)
                script_with_images.append({
                    "sentence": sentence,
                    "image": image_result["image_path"] if image_result else None
                })
            
            create_video(script_with_images)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()