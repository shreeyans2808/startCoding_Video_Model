import streamlit as st
import os
import tempfile
from video_gen5 import (
    extract_key_context,
    format_query_with_groq,
    search_index,
    generate_video_script,
    find_relevant_image,
    create_video,
    display_results
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(page_title="AI Video Generator", layout="wide")
st.title("AI Video Generator")

# Session state to track progress
if 'progress' not in st.session_state:
    st.session_state.progress = {
        'status': 'ready',  # ready, working, complete, error
        'current_step': '',
        'message': '',
        'script': '',
        'script_with_images': [],
        'video_path': None
    }

def update_progress(status, step, message):
    st.session_state.progress['status'] = status
    st.session_state.progress['current_step'] = step
    st.session_state.progress['message'] = message
    st.experimental_rerun()

def generate_video_from_prompt(user_input):
    try:
        # Step 1: Extract key context
        update_progress('working', 'extract_context', "Extracting key context from your query...")
        key_context = extract_key_context(user_input)
        
        # Step 2: Format query
        update_progress('working', 'format_query', "Formatting search query...")
        formatted_query = format_query_with_groq(user_input)
        if isinstance(formatted_query, dict):
            formatted_query = formatted_query.get('specific', user_input)
        
        # Step 3: Search index
        update_progress('working', 'search_index', "Searching for relevant content...")
        results = search_index(formatted_query, "text", top_k=10)
        
        if not results:
            update_progress('error', '', "No results found for your query.")
            return
        
        # Step 4: Generate script
        update_progress('working', 'generate_script', "Generating video script...")
        script = generate_video_script(results, formatted_query)
        st.session_state.progress['script'] = script
        
        # Step 5: Find images for each sentence
        update_progress('working', 'find_images', "Finding relevant images for each sentence...")
        sentences = [s.strip() for s in script.split('\n') if s.strip()]
        script_with_images = []
        
        for sentence in sentences:
            image_result = find_relevant_image(sentence, key_context)
            if image_result:
                script_with_images.append({
                    "sentence": sentence,
                    "image": image_result["image_path"],
                    "score": image_result["score"],
                    "source": image_result["source"]
                })
            else:
                script_with_images.append({
                    "sentence": sentence,
                    "image": None,
                    "score": None,
                    "source": None
                })
        
        st.session_state.progress['script_with_images'] = script_with_images
        
        # Step 6: Create video
        update_progress('working', 'create_video', "Generating video...")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_path = temp_video.name
        
        create_video(script_with_images, output_path=video_path)
        st.session_state.progress['video_path'] = video_path
        
        update_progress('complete', '', "Video generation complete!")
        
    except Exception as e:
        update_progress('error', '', f"An error occurred: {str(e)}")

# Main UI
with st.form("video_generator_form"):
    user_input = st.text_area("Enter your topic or prompt for the video:", 
                            height=100,
                            placeholder="e.g. 'The history of artificial intelligence'")
    
    submitted = st.form_submit_button("Generate Video")
    
    if submitted and user_input.strip():
        generate_video_from_prompt(user_input.strip())

# Progress and results display
if st.session_state.progress['status'] != 'ready':
    st.subheader("Progress")
    
    if st.session_state.progress['status'] == 'working':
        with st.spinner(st.session_state.progress['message']):
            time.sleep(0.5)  # Just to show the spinner
    
    elif st.session_state.progress['status'] == 'error':
        st.error(st.session_state.progress['message'])
    
    elif st.session_state.progress['status'] == 'complete':
        st.success(st.session_state.progress['message'])
        
        # Show the generated script
        st.subheader("Generated Script")
        st.text_area("Script", 
                    value=st.session_state.progress['script'], 
                    height=300,
                    disabled=True)
        
        # Show the video
        st.subheader("Generated Video")
        if st.session_state.progress['video_path']:
            video_file = open(st.session_state.progress['video_path'], 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)
            
            # Download button
            st.download_button(
                label="Download Video",
                data=video_bytes,
                file_name="generated_video.mp4",
                mime="video/mp4"
            )
            
            # Clean up
            try:
                os.unlink(st.session_state.progress['video_path'])
            except:
                pass