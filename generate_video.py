import os
from groq import Groq
from search_index import search_index
import json
from dotenv import load_dotenv
import numpy as np
from duckduckgo_search import DDGS
import cv2
import requests
from PIL import Image
from io import BytesIO
import tempfile
from gtts import gTTS
import subprocess
import time

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def download_image(url):
    """Download image from URL and save to temporary file"""
    try:
        print(f"Attempting to download image from: {url}")
        response = requests.get(url, timeout=10)  # Added timeout
        if response.status_code == 200:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(response.content)
            temp_file.close()
            print(f"Successfully downloaded image to: {temp_file.name}")
            return temp_file.name
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error downloading image: {str(e)}")
    return None

def add_subtitle_to_frame(frame, text):
    """Add subtitle text to a frame"""
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Adjusted font size
    font_thickness = 2
    font_color = (255, 255, 255)  # White color
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Ensure text fits within frame width
    while text_width > width - 40:  # Leave 20px margin on each side
        font_scale *= 0.9
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Calculate text position (centered at bottom)
    text_x = (width - text_width) // 2
    text_y = height - 30  # Moved up slightly
    
    # Create a semi-transparent overlay for the text background
    overlay = frame.copy()
    cv2.rectangle(overlay, 
                 (text_x - 10, text_y - text_height - 10),
                 (text_x + text_width + 10, text_y + 10),
                 (0, 0, 0),
                 -1)
    
    # Blend the overlay with the original frame
    alpha = 0.7  # Transparency factor (0.0 to 1.0)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
    
    return frame

def generate_audio_for_sentence(sentence, output_path):
    """Generate audio for a sentence using gTTS"""
    try:
        # Generate audio with faster speed
        tts = gTTS(text=sentence, lang='en', slow=False)
        
        # Save the audio file
        tts.save(output_path)
        
        # Use FFmpeg to speed up the audio
        temp_output = output_path + ".temp.mp3"
        subprocess.run([
            'ffmpeg', '-y',
            '-i', output_path,
            '-filter:a', 'atempo=1',  # Speed up by 30%
            '-vn',  # No video
            temp_output
        ], check=True)
        
        # Replace original file with sped up version
        os.replace(temp_output, output_path)
        
        return True
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False

def create_video(script_with_images, output_path="output_video.mp4", fps=24, duration_per_image=5):
    """Create video from images and add subtitles with audio"""
    try:
        if not script_with_images or not any(item['image'] for item in script_with_images):
            print("No valid images found for video creation")
            return

        # Create temporary directory for audio files
        temp_dir = tempfile.mkdtemp()
        audio_files = []

        # Get the first valid image to determine dimensions
        first_image = None
        for item in script_with_images:
            if item['image']:
                print(f"\nTrying to download image: {item['image']}")
                first_image = cv2.imread(download_image(item['image']))
                if first_image is not None:
                    print("Successfully loaded first image")
                    break
                else:
                    print("Failed to load first image")
        
        if first_image is None:
            print("Could not read any images")
            return

        # Get video dimensions from first image
        height, width = first_image.shape[:2]
        print(f"Video dimensions: {width}x{height}")
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = os.path.join(temp_dir, "temp_video.mp4")
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        
        # Calculate frames for 5 seconds per image
        frames_per_image = fps * duration_per_image  # 24 fps * 5 seconds = 120 frames per image
        
        print(f"Creating video with {len(script_with_images)} images")
        print(f"Frames per image: {frames_per_image}")
        print(f"Total duration: {len(script_with_images) * duration_per_image} seconds")
        
        # First, generate all audio files
        print("\nGenerating audio for each sentence...")
        for i, item in enumerate(script_with_images):
            audio_path = os.path.join(temp_dir, f"audio_{i}.mp3")
            if generate_audio_for_sentence(item['sentence'], audio_path):
                audio_files.append(audio_path)
                print(f"Generated audio for sentence {i+1}/{len(script_with_images)}")
        
        # Then process each image
        print("\nCreating video frames...")
        for i, item in enumerate(script_with_images):
            print(f"\nProcessing image {i+1}/{len(script_with_images)}")
            if item['image']:
                # Download and read image
                image_path = download_image(item['image'])
                if image_path:
                    print(f"Downloaded image to: {image_path}")
                    frame = cv2.imread(image_path)
                    if frame is not None:
                        print("Successfully read image")
                        # Resize frame if necessary
                        if frame.shape[:2] != (height, width):
                            print(f"Resizing image from {frame.shape[:2]} to {(height, width)}")
                            frame = cv2.resize(frame, (width, height))
                        
                        # Add subtitle
                        frame = add_subtitle_to_frame(frame, item['sentence'])
                        
                        # Write frame for exactly 5 seconds
                        print(f"Writing {frames_per_image} frames")
                        for _ in range(frames_per_image):
                            out.write(frame.copy())
                        
                        # Clean up
                        os.unlink(image_path)
                        print("Cleaned up image file")
                    else:
                        print(f"Warning: Could not read image for sentence {i+1}")
                else:
                    print(f"Warning: Could not download image for sentence {i+1}")
            else:
                print(f"Warning: No image provided for sentence {i+1}")
        
        # Release video writer
        out.release()
        
        # Combine all audio files into one
        print("\nCombining audio files...")
        combined_audio_path = os.path.join(temp_dir, "combined_audio.mp3")
        
        # Create a file list for FFmpeg
        with open(os.path.join(temp_dir, "concat.txt"), 'w') as f:
            for audio_file in audio_files:
                f.write(f"file '{audio_file}'\n")
        
        # Use FFmpeg to concatenate audio files
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', os.path.join(temp_dir, "concat.txt"),
            '-c', 'copy',
            combined_audio_path
        ], check=True)
        
        # Combine video with audio using FFmpeg
        print("\nCombining video with audio...")
        subprocess.run([
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', combined_audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-strict', 'experimental',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-shortest',  # Ensure output duration matches the shortest input
            output_path
        ], check=True)
        
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        for file in audio_files:
            os.unlink(file)
        os.unlink(combined_audio_path)
        os.unlink(os.path.join(temp_dir, "concat.txt"))
        os.unlink(temp_video_path)
        os.rmdir(temp_dir)
        
        print(f"\nVideo has been created successfully: {output_path}")
        print(f"Video duration: {len(script_with_images) * duration_per_image} seconds")
        
    except Exception as e:
        print(f"Error creating video: {str(e)}")
        # Clean up on error
        try:
            for file in audio_files:
                if os.path.exists(file):
                    os.unlink(file)
            if os.path.exists(combined_audio_path):
                os.unlink(combined_audio_path)
            if os.path.exists(temp_video_path):
                os.unlink(temp_video_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except:
            pass

def format_query_with_groq(user_input, purpose="image_search", context=""):
    """Format the user's query using Groq to make it more search-friendly"""
    if purpose == "image_search":
        prompt = f"""Given the following text and context, create multiple image search queries that will find the most relevant images.
        Create 3 different search queries:
        1. A specific, detailed query focusing on the main subject
        2. A broader query that captures the general concept
        3. A simple, keyword-based query
        
        Focus on visual elements that would make good images.
        Keep queries concise but descriptive.
        
        Context: {context}
        Text: {user_input}
        
        Return the queries in this format:
        specific: [specific query]
        broad: [broad query]
        simple: [simple query]"""
    else:
        prompt = f"""Given the following user query, format it into a clear and concise search query that will help find relevant information. 
        Keep the core meaning but make it more search-friendly.
        
        User query: {user_input}
        
        Formatted search query:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that formats queries for optimal search results."
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
        # Parse the response to get different query types
        response = completion.choices[0].message.content.strip()
        queries = {}
        for line in response.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                queries[key.strip()] = value.strip()
        return queries
    else:
        return completion.choices[0].message.content.strip()

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

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

def search_duckduckgo_images(query, context=""):
    """Search for images using DuckDuckGo with improved parameters"""
    try:
        with DDGS() as ddgs:
            # Get multiple query variations
            query_variations = format_query_with_groq(query, purpose="image_search", context=context)
            print(f"Generated query variations: {query_variations}")
            
            # Define search parameters to try
            search_params = [
                # High quality search
                {
                    'max_results': 5,
                    'safesearch': 'on',
                    'size': 'Large',
                    'type_image': 'photo',
                    'layout': 'Wide',
                    'license_image': 'share'
                },
                # Medium quality search
                {
                    'max_results': 5,
                    'safesearch': 'on',
                    'size': 'Large'
                },
                # Basic search
                {
                    'max_results': 5
                }
            ]
            
            # Try each query variation with different parameters
            for query_type, search_query in query_variations.items():
                print(f"\nTrying {query_type} query: {search_query}")
                
                for params in search_params:
                    try:
                        print(f"Searching with parameters: {params}")
                        results = list(ddgs.images(search_query, **params))
                        
                        if results:
                            print(f"Found {len(results)} images")
                            # Try each result until we find a valid image
                            for result in results:
                                if result.get('image'):
                                    print(f"Found valid image: {result['image']}")
                                    return result['image']
                        else:
                            print("No images found with current parameters")
                    except Exception as e:
                        print(f"Error with current search attempt: {str(e)}")
                        continue
            
            # If all attempts fail, try one last time with the original query
            print("\nTrying final fallback with original query...")
            try:
                results = list(ddgs.images(f"{context} {query}", max_results=5))
                if results and results[0].get('image'):
                    print("Found image in final fallback attempt")
                    return results[0]['image']
            except Exception as e:
                print(f"Final fallback attempt failed: {str(e)}")
            
            print("All search attempts failed to find images")
            return None
            
    except Exception as e:
        print(f"Error in DuckDuckGo search: {str(e)}")
        return None

def find_relevant_image(sentence, context=""):
    """Find the most relevant image for a given sentence"""
    print(f"\nSearching for image for sentence: {sentence}")
    
    # First try DuckDuckGo search
    print("Trying DuckDuckGo search first...")
    duckduckgo_image = search_duckduckgo_images(sentence, context)
    if duckduckgo_image:
        print(f"Found image in DuckDuckGo: {duckduckgo_image}")
        return {
            "sentence": sentence,
            "image_path": duckduckgo_image,
            "score": None,
            "source": "duckduckgo",
            "type": "image"
        }
    
    # If DuckDuckGo fails, try the index
    print("DuckDuckGo search failed, trying index...")
    # Use the specific query for index search
    query_variations = format_query_with_groq(sentence, purpose="image_search", context=context)
    if isinstance(query_variations, dict) and 'specific' in query_variations:
        image_query = query_variations['specific']
    else:
        image_query = sentence  # Fallback to original sentence if query formatting fails
    
    print(f"Formatted search query for index: {image_query}")
    
    try:
        results = search_index(image_query, "text", top_k=10)
        print(f"Found {len(results)} results in index")
        
        for result in results:
            if result.get("type") in ["image", "video_frame"]:
                print(f"Found image in index: {result.get('content')}")
                return {
                    "sentence": sentence,
                    "image_path": result.get("content"),
                    "score": result.get("score"),
                    "source": "index",
                    "type": result.get("type")
                }
    except Exception as e:
        print(f"Error searching index: {str(e)}")
    
    print("No image found in any source")
    return None

def generate_video_script(results, formatted_query):
    """Generate a 1-minute video script from search results"""
    # Convert results to JSON-serializable format
    serializable_results = convert_to_serializable(results)
    
    prompt = f"""Create a concise 1-minute video script based on these search results. 
    Each line should take 5 seconds to speak (about 12-15 words per line).
    Total script should be 12 lines (1 minute total).
    Focus on the most important information from the results.
    Make it flow naturally and be engaging.
    
    Search Query: {formatted_query}
    
    Search Results:
    {json.dumps(serializable_results, indent=2)}
    
    Create a script with exactly 12 lines, each line taking 5 seconds to speak:"""
    
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a professional script writer who creates concise, engaging video scripts."
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
    
    return completion.choices[0].message.content.strip()

def display_results(results):
    """Display the search results in a formatted way"""
    if not results:
        print("No results found.")
        return
        
    print("\n=== Search Results ===")
    for result in results:
        print(f"\nRank {result['rank']} | Score: {result['score']:.4f}")
        print(f"Type: {result['type']}")
        print(f"Content: {result['content'][:200]}...")
        if result.get('timestamp'):
            print(f"Timestamp: {result['timestamp']}")
        if result.get('source'):
            print(f"Source: {result['source']}")
        print("-" * 50)

def main():
    # Get user input
    user_input = input("Enter your search query: ")
    
    # Extract key context from user's query
    key_context = extract_key_context(user_input)
    print(f"\nExtracted key context: {key_context}")
    
    # Format query using Groq for initial search
    formatted_query = format_query_with_groq(user_input)
    if isinstance(formatted_query, dict):
        # If we got a dictionary, use the specific query
        formatted_query = formatted_query.get('specific', user_input)
    print(f"\nFormatted query: {formatted_query}")
    
    # Search the index
    try:
        results = search_index(formatted_query, "text", top_k=10)
        # Display results
        display_results(results)
        
        # Generate and display video script
        if results:
            print("\n=== Generated Video Script (1 minute) ===")
            print("Each line takes 5 seconds to speak:")
            print("-" * 50)
            script = generate_video_script(results, formatted_query)
            print(script)
            print("-" * 50)
            
            # Split script into sentences and find relevant images
            print("\n=== Finding Relevant Images for Each Sentence ===")
            sentences = [s.strip() for s in script.split('\n') if s.strip()]
            script_with_images = []
            
            # Use the key context for all image searches
            for sentence in sentences:
                print(f"\nProcessing sentence: {sentence}")
                image_result = find_relevant_image(sentence, key_context)
                if image_result:
                    source = image_result['source']
                    score_info = f" (Score: {image_result['score']:.4f})" if image_result['score'] is not None else ""
                    print(f"Found relevant image from {source}: {image_result['image_path']}{score_info}")
                    script_with_images.append({
                        "sentence": sentence,
                        "image": image_result["image_path"],
                        "score": image_result["score"],
                        "source": source
                    })
                else:
                    print("No relevant image found")
                    script_with_images.append({
                        "sentence": sentence,
                        "image": None,
                        "score": None,
                        "source": None
                    })
            
            # Display final script with images
            print("\n=== Final Script with Images ===")
            for item in script_with_images:
                print(f"\nSentence: {item['sentence']}")
                if item['image']:
                    score_info = f" (Relevance Score: {item['score']:.4f})" if item['score'] is not None else ""
                    print(f"Image: {item['image']} (Source: {item['source']}){score_info}")
                else:
                    print("No matching image found")
                print("-" * 50)
            
            # Generate the video
            print("\n=== Generating Video ===")
            create_video(script_with_images)
    except Exception as e:
        print(f"Error during search or processing: {str(e)}")
        print("No results found.")

if __name__ == "__main__":
    main()
