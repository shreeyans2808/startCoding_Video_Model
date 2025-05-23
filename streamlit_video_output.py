import os
from add_to_index import add_file
from config import *
import logging
from pathlib import Path
import streamlit as st
import time
from datetime import datetime
import faiss
import pickle
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from movie_generator import MovieGenerator
from utils import embed_text, embed_image, EMBEDDING_DIM
from duckduckgo_search import DDGS
import re
import nltk
from nltk.tokenize import sent_tokenize
import groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = groq.Client(api_key=os.getenv("GROQ_API_KEY"))

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_index():
    """Load the FAISS index and metadata."""
    try:
        # Check if index exists
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            logger.info("No existing index found. Please index some content first.")
            return None, None
            
        # Load existing index
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
    except Exception as e:
        logger.error(f"Error loading index: {str(e)}")
        return None, None

def search_index(query, index, metadata, k=10):
    """Search the index for similar content."""
    try:
        # Get query embedding
        query_emb = embed_text(query)
        
        # Search the index
        distances, indices = index.search(query_emb.reshape(1, -1), k)
        
        # Get results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx in metadata:
                results.append({
                    'metadata': metadata[idx],
                    'score': float(distances[0][i])
                })
        
        return results
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        return []

def generate_image_search_query(text_chunk):
    """Generate an optimized search query for images based on the text chunk."""
    try:
        # Extract key nouns and phrases from the text chunk
        words = text_chunk.split()
        # Remove common words and keep meaningful content
        filtered_words = [w for w in words if len(w) > 3 and w.lower() not in ['this', 'that', 'with', 'from', 'have', 'will', 'would', 'could', 'should']]
        
        # Take the first 3-4 meaningful words for the search
        search_terms = filtered_words[:4]
        search_query = ' '.join(search_terms)
        
        # Add context words to improve search results
        search_query = f"high quality {search_query} professional"
        
        logger.info(f"Generated image search query: {search_query}")
        return search_query
    except Exception as e:
        logger.error(f"Error generating image search query: {str(e)}")
        return text_chunk[:50]  # Fallback to first 50 characters if error occurs

def search_image_online(query):
    """Search for an image using DuckDuckGo."""
    try:
        logger.info(f"Searching DuckDuckGo for: {query}")
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=5))  # Get more results to choose the best
            
            if results:
                # Try to find the best image (preferably landscape orientation)
                best_image = None
                for result in results:
                    try:
                        image_url = result['image']
                        response = requests.get(image_url, timeout=10)
                        if response.status_code == 200:
                            image = Image.open(BytesIO(response.content))
                            # Convert to RGB if necessary
                            if image.mode != 'RGB':
                                image = image.convert('RGB')
                            
                            # Check if image is landscape orientation
                            if image.width > image.height:
                                best_image = image
                                break
                            elif best_image is None:
                                best_image = image
                    except Exception as e:
                        logger.error(f"Error processing image: {str(e)}")
                        continue
                
                if best_image:
                    # Resize image to match video dimensions while maintaining aspect ratio
                    target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
                    image_ratio = best_image.width / best_image.height
                    
                    if image_ratio > target_ratio:
                        # Image is wider than target ratio
                        new_width = int(best_image.height * target_ratio)
                        new_height = best_image.height
                    else:
                        # Image is taller than target ratio
                        new_width = best_image.width
                        new_height = int(best_image.width / target_ratio)
                    
                    best_image = best_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    return best_image
            else:
                logger.warning("No images found in DuckDuckGo search")
            
    except Exception as e:
        logger.error(f"Error searching image online: {str(e)}")
    return None

def format_subtitles(text, max_chars=SUBTITLE_MAX_CHARS_PER_LINE):
    """Format text into subtitle lines with proper length."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= max_chars:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines

def split_script_into_chunks(script, chunk_duration=5):
    """Split script into chunks of approximately chunk_duration seconds."""
    # Split script into sentences
    sentences = sent_tokenize(script)
    
    chunks = []
    current_chunk = []
    current_duration = 0
    
    # Approximate words per second (average speaking rate)
    words_per_second = 2.5
    
    for sentence in sentences:
        words = sentence.split()
        sentence_duration = len(words) / words_per_second
        
        if current_duration + sentence_duration <= chunk_duration:
            current_chunk.append(sentence)
            current_duration += sentence_duration
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_duration = sentence_duration
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def generate_video_from_index(progress_bar, status_text, query, context=""):
    """Generate a video using the indexed content."""
    try:
        # Initialize movie generator
        movie_gen = MovieGenerator()
        
        # Load index
        index, metadata = load_index()
        if index is None or metadata is None:
            st.error("Failed to load index")
            return False
        
        # Generate script using Groq
        status_text.text("Generating script with Groq...")
        script = generate_script_with_groq(query)
        if not script:
            st.error("Failed to generate script")
            return False
        
        # Split script into chunks
        script_chunks = split_script_into_chunks(script)
        
        # Find images for each chunk
        status_text.text("Finding images for each part of the script...")
        images = []
        subtitles = []
        
        for i, chunk in enumerate(script_chunks):
            progress = (i + 1) / len(script_chunks)
            progress_bar.progress(progress)
            status_text.text(f"Processing chunk {i + 1}/{len(script_chunks)}")
            
            # Generate image search query using Groq
            image_query = generate_search_query_with_groq(chunk)
            
            # Search index for related content
            results = search_index(image_query, index, metadata)
            
            # Find the most relevant image
            image = None
            if results:
                for result in results:
                    if result['metadata']['type'] == 'image':
                        image = result['metadata']['content']
                        break
            
            # If no image found in index, search online
            if image is None:
                status_text.text(f"No relevant image found in index for chunk {i + 1}, searching online...")
                image = search_image_online(image_query)
            
            if image is None:
                st.error(f"Failed to find a suitable image for chunk {i + 1}")
                return False
            
            images.append(image)
            
            # Format subtitles for this chunk
            subtitle_lines = format_subtitles(chunk)
            subtitles.append(subtitle_lines)
        
        # Create video with subtitles
        status_text.text("Creating video with subtitles...")
        output_path = os.path.join(OUTPUT_DIR, f"generated_video_{int(time.time())}.mp4")
        
        # Create video with subtitles
        success = movie_gen.create_video(
            images=images,
            subtitles=subtitles,
            output_path=output_path,
            duration_per_image=5  # 5 seconds per image
        )
        
        if success:
            st.success(f"Video generated successfully! Saved to: {output_path}")
            return True
        else:
            st.error("Failed to create video")
            return False
            
    except Exception as e:
        logger.error(f"Error generating video: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        return False

def index_directory(directory_path, progress_bar, status_text):
    """Index all files in the given directory and its subdirectories."""
    try:
        # Initialize FAISS index
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        metadata = {}
        current_index = 0

        # Get all files in directory and subdirectories
        all_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                all_files.append(os.path.join(root, file))

        total_files = len(all_files)
        if total_files == 0:
            st.error("No files found in the specified directory")
            return False

        # Process each file
        for i, file_path in enumerate(all_files):
            try:
                progress = (i + 1) / total_files
                progress_bar.progress(progress)
                status_text.text(f"Processing file {i + 1}/{total_files}: {os.path.basename(file_path)}")

                # Add file to index
                success = add_file(file_path, index, metadata, current_index)
                if success:
                    current_index += 1

            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue

        if current_index == 0:
            st.error("No files were successfully indexed")
            return False

        # Save index and metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_path = os.path.join(INDEX_DIR, f"index_{timestamp}.faiss")
        metadata_path = os.path.join(INDEX_DIR, f"metadata_{timestamp}.pkl")

        # Create index directory if it doesn't exist
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Save the index and metadata
        faiss.write_index(index, index_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        st.success(f"Successfully indexed {current_index} files. Index saved as {os.path.basename(index_path)}")
        return True

    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        st.error(f"Error during indexing: {str(e)}")
        return False

def generate_script_with_groq(query):
    """Generate a 1-minute video script using Groq."""
    try:
        prompt = f"""Create a concise, engaging script for a 1-minute video about: {query}
        The script should be informative and well-structured. Only provide the raw text of the script, 
        no timestamps or effects. Keep it natural and conversational."""
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=500
        )
        
        script = response.choices[0].message.content.strip()
        return script
    except Exception as e:
        logger.error(f"Error generating script with Groq: {str(e)}")
        return None

def generate_search_query_with_groq(text_chunk):
    """Generate an optimized search query for images using Groq."""
    try:
        prompt = f"""Given this text chunk: "{text_chunk}"
        Generate a concise, effective search query to find a relevant image. 
        The query should be specific and focused on the main visual elements."""
        
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.7,
            max_tokens=100
        )
        
        query = response.choices[0].message.content.strip()
        return query
    except Exception as e:
        logger.error(f"Error generating search query with Groq: {str(e)}")
        return text_chunk[:50]  # Fallback to first 50 characters if error occurs

def main():
    st.title("Directory Indexer")
    
    # Get directory path from user
    directory_path = st.text_input("Enter the directory path to index:")
    
    if st.button("Start Indexing"):
        if not directory_path:
            st.error("Please enter a directory path")
            return
        
        if not os.path.exists(directory_path):
            st.error("Directory does not exist")
            return
        
        # Create progress bar and status text
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Start indexing
        success = index_directory(directory_path, progress_bar, status_text)
        
        if success:
            st.success("Indexing completed successfully!")
        else:
            st.error("Indexing failed")

if __name__ == "__main__":
    main()
