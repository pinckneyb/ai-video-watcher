"""
AI Video Watcher - Main Streamlit Application
A generic video analysis app using GPT-4o to narrate video content.
"""

import streamlit as st
import os
import json
from typing import List, Dict, Any, Optional
import time
from datetime import datetime

# Import our modules
from video_processor import VideoProcessor, FrameBatchProcessor
from gpt4o_client import GPT4oClient
from profiles import ProfileManager
from utils import (
    parse_timestamp, format_timestamp, validate_time_range,
    save_transcript, save_timeline_json, estimate_processing_time,
    extract_video_info, sanitize_filename, extract_audio_from_video,
    transcribe_audio_with_whisper, transcribe_audio_with_diarization
)

# API key management
import json
import os

# Page configuration
st.set_page_config(
    page_title="AI Video Watcher",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def validate_api_key(api_key: str) -> bool:
    """Validate OpenAI API key by making a test call."""
    try:
        test_client = GPT4oClient(api_key=api_key)
        # Make a simple test call to validate the key
        response = test_client.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        return True
    except Exception as e:
        print(f"API key validation failed: {e}")
        return False

def is_api_key_still_valid(client: GPT4oClient) -> bool:
    """Check if the current API key is still valid."""
    try:
        # Make a simple test call
        response = client.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        return True
    except Exception as e:
        print(f"API key validation check failed: {e}")
        return False

def save_api_key(api_key: str):
    """Save API key to config file."""
    # Try to save in project root first, fallback to user home
    config_file = "config.json"
    
    try:
        config = {"openai_api_key": api_key}
        with open(config_file, "w") as f:
            json.dump(config, f)
    except Exception as e:
        print(f"Could not save to project root, using home directory: {e}")
        # Fallback to user home directory
        config_dir = os.path.expanduser("~/.ai_video_watcher")
        os.makedirs(config_dir, exist_ok=True)
        config_file = os.path.join(config_dir, "config.json")
        
        config = {"openai_api_key": api_key}
        with open(config_file, "w") as f:
            json.dump(config, f)

def load_api_key() -> str:
    """Load API key from config file."""
    # Try to load from project root first
    config_file = "config.json"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("openai_api_key", "")
        except Exception as e:
            print(f"Error loading config from project root: {e}")
    
    # Fallback to user home directory
    config_dir = os.path.expanduser("~/.ai_video_watcher")
    config_file = os.path.join(config_dir, "config.json")
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("openai_api_key", "")
        except Exception as e:
            print(f"Error loading config from home directory: {e}")
            return ""
    return ""

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .frame-preview {
        max-width: 200px;
        border: 2px solid #ddd;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎥 AI Video Watcher</h1>', unsafe_allow_html=True)
    st.markdown("### Let GPT-4o watch and narrate your videos with intelligent frame-by-frame analysis")
    
    # Initialize session state
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = None
    if 'gpt4o_client' not in st.session_state:
        st.session_state.gpt4o_client = None
    if 'current_profile' not in st.session_state:
        st.session_state.current_profile = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'transcript' not in st.session_state:
        st.session_state.transcript = ""
    if 'events' not in st.session_state:
        st.session_state.events = []
    if 'saved_api_key' not in st.session_state:
        st.session_state.saved_api_key = load_api_key()
    if 'enhanced_narrative' not in st.session_state:
        st.session_state.enhanced_narrative = ""
    if 'audio_transcript' not in st.session_state:
        st.session_state.audio_transcript = ""
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Profile selection
        profile_manager = ProfileManager()
        available_profiles = profile_manager.get_available_profiles()
        selected_profile = st.selectbox(
            "Select Analysis Profile",
            available_profiles,
            format_func=lambda x: profile_manager.get_profile(x)["name"]
        )
        
        if selected_profile:
            profile = profile_manager.get_profile(selected_profile)
            st.session_state.current_profile = profile
            st.info(f"**{profile['name']}**: {profile['description']}")
        
        # Video settings
        st.subheader("📹 Video Settings")
        fps = st.slider("Frames per second", 0.5, 5.0, 1.0, 0.1)
        batch_size = st.slider("Batch size", 3, 15, 5, 1)
        
        # Audio transcription settings
        st.subheader("🎵 Audio Transcription")
        enable_whisper = st.checkbox(
            "Enable OpenAI Whisper transcription",
            help="Extract and transcribe audio track (if present) using Whisper API"
        )
        
        if enable_whisper:
            enable_diarization = st.checkbox(
                "Enable speaker diarization",
                help="Identify different speakers in the audio (requires pyannote.audio)"
            )
        
        # Concurrency settings
        st.subheader("⚡ Concurrency Settings")
        max_concurrent_batches = st.slider("Max concurrent batches", 1, 10, 3, 1, 
                                         help="Higher values = faster processing (requires OpenAI Tier 4+)")
        
        # Rescan settings
        st.subheader("🔍 Rescan Settings")
        rescan_fps = st.slider("Rescan FPS", 5.0, 20.0, 10.0, 1.0)
        
        # API key input
        st.subheader("🔑 OpenAI API Key")
        
        # Show saved API key status
        if st.session_state.saved_api_key:
            st.info("🔑 API key found from previous session")
            if st.button("🗑️ Clear saved API key"):
                st.session_state.saved_api_key = ""
                st.session_state.gpt4o_client = None
                save_api_key("")  # Clear saved key
                st.rerun()
        
        # API key input field
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            value=st.session_state.saved_api_key,
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        # Validate and save API key
        if api_key and api_key != st.session_state.saved_api_key:
            with st.spinner("Validating API key..."):
                if validate_api_key(api_key):
                    st.session_state.saved_api_key = api_key
                    save_api_key(api_key)
                    st.session_state.gpt4o_client = GPT4oClient(api_key=api_key)
                    st.success("✅ API key validated and saved!")
                else:
                    st.error("❌ Invalid API key. Please check and try again.")
                    st.session_state.gpt4o_client = None
        
        # Show current API key status
        elif st.session_state.saved_api_key and st.session_state.gpt4o_client:
            # Check if the API key is still valid
            if is_api_key_still_valid(st.session_state.gpt4o_client):
                st.success("✅ API key configured and validated")
            else:
                st.warning("⚠️ API key may have expired. Please re-enter your key.")
                st.session_state.gpt4o_client = None
        elif st.session_state.saved_api_key:
            # Try to recreate client with saved key
            try:
                st.session_state.gpt4o_client = GPT4oClient(api_key=st.session_state.saved_api_key)
                st.success("✅ API key loaded from previous session")
            except Exception as e:
                st.error(f"❌ Error loading saved API key: {e}")
                st.session_state.saved_api_key = ""
                save_api_key("")
                st.session_state.gpt4o_client = None
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Video Input")
        
        # Video input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload Video File", "Enter Video URL"],
            horizontal=True
        )
        
        video_source = None
        
        if input_method == "Upload Video File":
            uploaded_file = st.file_uploader(
                "Upload a video file",
                type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
                help="Supported formats: MP4, AVI, MOV, MKV, WMV"
            )
            
            if uploaded_file is not None:
                video_source = uploaded_file.read()
                st.success(f"✅ Video uploaded: {uploaded_file.name}")
                
                # Show video info
                with st.expander("📊 Video Information"):
                    # Save temporarily to get info
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(video_source)
                    
                    video_info = extract_video_info(temp_path)
                    if "error" not in video_info:
                        st.json(video_info)
                        st.info(f"Estimated processing time: {estimate_processing_time(video_info['duration'], fps, batch_size)}")
                    
                    # Clean up temp file
                    os.remove(temp_path)
        
        else:  # URL input
            video_url = st.text_input(
                "Enter video URL",
                placeholder="https://example.com/video.mp4",
                help="Direct link to video file"
            )
            
            if video_url:
                video_source = video_url
                st.success(f"✅ Video URL set: {video_url}")
        
        # Load video button
        if video_source and st.button("🚀 Load Video", type="primary"):
            with st.spinner("Loading video..."):
                try:
                    st.session_state.video_processor = VideoProcessor()
                    success = st.session_state.video_processor.load_video(video_source, fps)
                    
                    if success:
                        # Verify video properties were loaded
                        if st.session_state.video_processor.duration and st.session_state.video_processor.duration > 0:
                            st.success("✅ Video loaded successfully!")
                            st.info(f"Duration: {format_timestamp(st.session_state.video_processor.duration)}")
                            st.info(f"Total frames: {int(st.session_state.video_processor.duration * fps)}")
                            st.session_state.analysis_complete = False
                            st.session_state.transcript = ""
                            st.session_state.events = []
                            
                                                    # Reset GPT-4o client context
                        if st.session_state.gpt4o_client:
                            st.session_state.gpt4o_client.reset_context()
                        
                        # Handle audio transcription if enabled
                        if enable_whisper and st.session_state.gpt4o_client:
                            with st.spinner("Transcribing audio with Whisper..."):
                                try:
                                    # Save video temporarily for audio extraction
                                    temp_video_path = f"temp_video_{uploaded_file.name}"
                                    with open(temp_video_path, "wb") as f:
                                        f.write(video_source)
                                    
                                    # Extract and transcribe audio
                                    audio_path = extract_audio_from_video(temp_video_path)
                                    if audio_path and os.path.exists(audio_path):
                                        if enable_diarization:
                                            audio_transcript = transcribe_audio_with_diarization(
                                                audio_path, 
                                                st.session_state.gpt4o_client.api_key
                                            )
                                        else:
                                            audio_transcript = transcribe_audio_with_whisper(
                                                audio_path, 
                                                st.session_state.gpt4o_client.api_key
                                            )
                                        if audio_transcript:
                                            st.session_state.audio_transcript = audio_transcript
                                            st.success("✅ Audio transcription completed!")
                                        else:
                                            st.warning("⚠️ Audio transcription failed")
                                    
                                    # Clean up temp files
                                    if os.path.exists(temp_video_path):
                                        os.remove(temp_video_path)
                                    if audio_path and os.path.exists(audio_path):
                                        os.remove(audio_path)
                                        
                                except Exception as e:
                                    st.warning(f"⚠️ Audio transcription error: {e}")
                        
                        # Auto-start analysis if profile is selected
                        if st.session_state.current_profile:
                            st.rerun()
                        else:
                            st.error("❌ Video loaded but duration is invalid")
                            st.session_state.video_processor = None
                    else:
                        st.error("❌ Failed to load video")
                        
                except Exception as e:
                    st.error(f"❌ Error loading video: {e}")
                    st.session_state.video_processor = None
    
    with col2:
        st.header("📊 Status")
        
        if st.session_state.video_processor:
            st.success("✅ Video loaded")
            if st.session_state.video_processor.duration:
                st.metric("Duration", f"{format_timestamp(st.session_state.video_processor.duration)}")
                st.metric("Total frames", f"{int(st.session_state.video_processor.duration * fps)}")
                st.metric("Batches", f"{(int(st.session_state.video_processor.duration * fps) + batch_size - 1) // batch_size}")
        else:
            st.info("⏳ No video loaded")
        
        if st.session_state.gpt4o_client:
            st.success("✅ API configured")
        else:
            st.warning("⚠️ API not configured")
        
        # Concurrency status
        if max_concurrent_batches > 1:
            st.info(f"⚡ Concurrency: {max_concurrent_batches} batches")
            st.metric("Speed Boost", f"~{max_concurrent_batches}x faster")
        else:
            st.info("🐌 Sequential processing")
    
    # Analysis section
    if st.session_state.video_processor and st.session_state.gpt4o_client:
        st.header("🧠 Video Analysis")
        
        # Start analysis button
        if st.button("🎬 Start Analysis", type="primary", disabled=st.session_state.analysis_complete):
            if st.session_state.current_profile:
                start_analysis(fps, batch_size, max_concurrent_batches)
            else:
                st.error("Please select an analysis profile")
        
        # Show analysis progress
        if st.session_state.transcript:
            st.subheader("📝 Analysis Results")
            
            # Display transcript
            with st.expander("📖 Full Transcript", expanded=True):
                st.markdown(st.session_state.transcript)
            
            # Display events timeline
            if st.session_state.events:
                with st.expander("📅 Events Timeline"):
                    for event in st.session_state.events:
                        st.json(event)
            
            # Display audio transcription if available
            if st.session_state.audio_transcript:
                st.subheader("🎵 Audio Transcription (Whisper)")
                with st.expander("📝 Spoken Content", expanded=True):
                    st.markdown(st.session_state.audio_transcript)
                    
                    # Download audio transcript
                    if st.button("📄 Download Audio Transcript (MD)"):
                        filename = save_transcript(st.session_state.audio_transcript, prefix="audio_")
                        st.success(f"✅ Audio transcript saved as {filename}")
                        st.download_button(
                            label="📥 Download Audio Transcript",
                            data=st.session_state.audio_transcript,
                            file_name=filename,
                            mime="text/markdown"
                        )
            
            # Enhanced narrative section
            if st.session_state.enhanced_narrative:
                st.subheader("✨ Enhanced Narrative (GPT-5)")
                with st.expander("📖 Coherent Story", expanded=True):
                    st.markdown(st.session_state.enhanced_narrative)
                
                # Download enhanced narrative
                if st.button("📄 Download Enhanced Narrative (MD)"):
                    filename = save_transcript(st.session_state.enhanced_narrative, prefix="enhanced_")
                    st.success(f"✅ Enhanced narrative saved as {filename}")
                    st.download_button(
                        label="📥 Download Enhanced Narrative",
                        data=st.session_state.enhanced_narrative,
                        file_name=filename,
                        mime="text/markdown"
                    )
                
                # Regenerate button
                if st.button("🔄 Regenerate Enhanced Narrative"):
                    with st.spinner("Regenerating enhanced narrative..."):
                        enhanced_narrative = create_coherent_narrative(
                            st.session_state.transcript, 
                            st.session_state.events,
                            st.session_state.gpt4o_client.api_key,
                            st.session_state.get('audio_transcript', '')
                        )
                        if enhanced_narrative:
                            st.session_state.enhanced_narrative = enhanced_narrative
                            st.success("✨ Enhanced narrative regenerated!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to regenerate enhanced narrative")
            else:
                # Manual trigger button if no enhanced narrative exists
                if st.button("✨ Create Enhanced Narrative with GPT-5"):
                    with st.spinner("Creating enhanced narrative..."):
                        enhanced_narrative = create_coherent_narrative(
                            st.session_state.transcript, 
                            st.session_state.events,
                            st.session_state.gpt4o_client.api_key,
                            st.session_state.get('audio_transcript', '')
                        )
                        if enhanced_narrative:
                            st.session_state.enhanced_narrative = enhanced_narrative
                            st.success("✨ Enhanced narrative created!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to create enhanced narrative")
            
            # Download options
            st.subheader("💾 Download Results")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                if st.button("📄 Download Transcript (MD)"):
                    filename = save_transcript(st.session_state.transcript)
                    st.success(f"✅ Transcript saved as {filename}")
                    st.download_button(
                        label="📥 Download Transcript",
                        data=st.session_state.transcript,
                        file_name=filename,
                        mime="text/markdown"
                    )
            
            with col_dl2:
                if st.button("📊 Download Timeline (JSON)"):
                    filename = save_timeline_json(st.session_state.events)
                    st.success(f"✅ Timeline saved as {filename}")
                    st.download_button(
                        label="📥 Download Timeline",
                        data=json.dumps(st.session_state.events, indent=2),
                        file_name=filename,
                        mime="application/json"
                    )
    
    # Rescan section
    if st.session_state.video_processor and st.session_state.gpt4o_client and st.session_state.analysis_complete:
        st.header("🔍 Rescan Segment")
        
        col_rescan1, col_rescan2 = st.columns(2)
        
        with col_rescan1:
            start_time = st.text_input(
                "Start time (HH:MM:SS)",
                placeholder="00:01:30",
                help="Start time for rescan segment"
            )
        
        with col_rescan2:
            end_time = st.text_input(
                "End time (HH:MM:SS)",
                placeholder="00:02:00",
                help="End time for rescan segment"
            )
        
        if start_time and end_time:
            if validate_time_range(start_time, end_time, st.session_state.video_processor.duration):
                st.success("✅ Valid time range")
                
                if st.button("🔍 Rescan Segment", type="primary"):
                    rescan_segment(start_time, end_time, rescan_fps)
            else:
                st.error("❌ Invalid time range")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**AI Video Watcher** - Powered by GPT-4o | "
        "Built with Streamlit, OpenCV, and OpenAI"
    )

def start_analysis(fps: float, batch_size: int, max_concurrent_batches: int = 1):
    """Start the video analysis process with optional concurrency."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        video_processor = st.session_state.video_processor
        gpt4o_client = st.session_state.gpt4o_client
        profile = st.session_state.current_profile
        
        # Extract all frames
        status_text.text("Extracting frames...")
        
        # Debug info
        st.info(f"Video duration: {video_processor.duration} seconds")
        st.info(f"Target FPS: {fps}")
        
        try:
            frames = video_processor.extract_frames(custom_fps=fps)
            
            if not frames:
                st.error("No frames extracted from video")
                return
                
        except Exception as e:
            st.error(f"❌ Frame extraction failed: {e}")
            return
        
        # Create batches
        batch_processor = FrameBatchProcessor(batch_size=batch_size)
        batches = batch_processor.create_batches(frames)
        
        total_batches = len(batches)
        st.info(f"Processing {len(frames)} frames in {total_batches} batches...")
        
        if max_concurrent_batches > 1:
            st.info(f"⚡ Using concurrent processing: {max_concurrent_batches} batches simultaneously")
            start_time = time.time()
            # Use concurrent processing
            process_batches_concurrently(batches, gpt4o_client, profile, progress_bar, status_text, total_batches)
            end_time = time.time()
            processing_time = end_time - start_time
            st.success(f"⚡ Concurrent processing completed in {processing_time:.2f} seconds")
        else:
            st.info("🐌 Using sequential processing")
            start_time = time.time()
            # Use sequential processing
            process_batches_sequentially(batches, gpt4o_client, profile, progress_bar, status_text, total_batches)
            end_time = time.time()
            processing_time = end_time - start_time
            st.success(f"🐌 Sequential processing completed in {processing_time:.2f} seconds")
        
        # Analysis complete
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        # Store results
        st.session_state.transcript = gpt4o_client.get_full_transcript()
        st.session_state.events = gpt4o_client.get_event_timeline()
        st.session_state.analysis_complete = True
        
        st.success("🎉 Video analysis completed successfully!")
        
        # Post-process with GPT-5 for coherent narrative
        if st.button("🔄 Enhance Narrative with GPT-5", type="secondary"):
            with st.spinner("Creating coherent narrative with GPT-5..."):
                enhanced_narrative = create_coherent_narrative(
                    st.session_state.transcript, 
                    st.session_state.events,
                    gpt4o_client.api_key,
                    st.session_state.get('audio_transcript', '')
                )
                if enhanced_narrative:
                    st.session_state.enhanced_narrative = enhanced_narrative
                    st.success("✨ Enhanced narrative created!")
                else:
                    st.error("❌ Failed to create enhanced narrative")
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        status_text.text("❌ Analysis failed")

def process_batches_sequentially(batches, gpt4o_client, profile, progress_bar, status_text, total_batches):
    """Process batches sequentially (original method)."""
    for i, batch in enumerate(batches):
        status_text.text(f"Processing batch {i+1}/{total_batches}...")
        
        # Analyze batch with GPT-4o
        narrative, events = gpt4o_client.analyze_frames(
            batch, 
            profile, 
            gpt4o_client.context_state
        )
        
        # Update context
        gpt4o_client.update_context(narrative, events)
        
        # Update progress
        progress = (i + 1) / total_batches
        progress_bar.progress(progress)
        
        # Small delay to show progress
        time.sleep(0.1)

def process_batches_concurrently(batches, gpt4o_client, profile, progress_bar, status_text, total_batches):
    """Process batches concurrently using ThreadPoolExecutor."""
    import concurrent.futures
    import threading
    
    # Use a lock to protect context updates
    context_lock = threading.Lock()
    
    # Create a thread pool for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all batch processing tasks
        future_to_batch = {}
        for i, batch in enumerate(batches):
            future = executor.submit(
                process_single_batch_concurrent,
                batch, gpt4o_client, profile, i, total_batches, context_lock
            )
            future_to_batch[future] = i
        
        # Process completed tasks as they finish
        completed_batches = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_index = future_to_batch[future]
            try:
                narrative, events = future.result()
                
                # Update context with thread safety
                with context_lock:
                    gpt4o_client.update_context(narrative, events)
                
                completed_batches += 1
                progress = completed_batches / total_batches
                progress_bar.progress(progress)
                status_text.text(f"Completed {completed_batches}/{total_batches} batches...")
                
            except Exception as e:
                st.error(f"❌ Batch {batch_index + 1} failed: {e}")
                completed_batches += 1
                progress = completed_batches / total_batches
                progress_bar.progress(progress)

def process_single_batch_concurrent(batch, gpt4o_client, profile, batch_index, total_batches, context_lock):
    """Process a single batch for concurrent execution."""
    try:
        # Create a temporary client for this batch to avoid conflicts
        temp_client = GPT4oClient(api_key=gpt4o_client.api_key)
        
        # Get current context state safely
        with context_lock:
            current_context = gpt4o_client.context_state
        
        # Analyze batch with GPT-4o
        narrative, events = temp_client.analyze_frames(
            batch, 
            profile, 
            current_context  # Use current context state
        )
        
        return narrative, events
        
    except Exception as e:
        st.error(f"❌ Error in concurrent batch {batch_index + 1}: {e}")
        return f"Error in batch {batch_index + 1}: {e}", []

def create_coherent_narrative(raw_transcript: str, events: List[Dict], api_key: str, audio_transcript: str = "") -> str:
    """Create a coherent, continuous narrative using GPT-5."""
    try:
        # Create a GPT-5 client for narrative enhancement
        gpt5_client = GPT4oClient(api_key=api_key)
        
        # Prepare the enhancement prompt
        enhancement_prompt = f"""You are a master storyteller and video analyst. Your task is to transform raw, batch-by-batch video analysis and audio transcription into a coherent, continuous narrative that reads like a polished story.

RAW TRANSCRIPT:
{raw_transcript}

EVENTS TIMELINE:
{json.dumps(events, indent=2)}

AUDIO TRANSCRIPT:
{audio_transcript if audio_transcript else "No audio transcription available."}

INSTRUCTIONS:
1. **Chronological Flow**: Maintain strict chronological order from start to finish
2. **Character Continuity**: Track characters/objects across scenes, noting when they appear/disappear
3. **Setting Continuity**: Maintain awareness of locations and spatial relationships
4. **Cause and Effect**: Connect events logically, explaining why things happen
5. **Narrative Coherence**: Fill gaps, resolve contradictions, and create smooth transitions
6. **Audio-Visual Coordination**: Seamlessly weave together visual actions and spoken dialogue
7. **Temporal Synchronization**: Align visual events with corresponding audio when possible
8. **Dialogue Integration**: Naturally incorporate spoken words into the narrative flow
9. **Absolute Specificity**: Describe EVERYTHING with concrete, specific details
10. **No Generic Language**: Eliminate phrases like "a figure," "someone," "a person"
11. **Physical Description**: Specify gender, age, body type, clothing, expressions, posture
12. **Direct Observation**: If someone is obese, beautiful, ugly, old, young - say so directly
13. **No Political Correctness**: Describe what you see without filtering or softening
14. **Condensation**: Condense elements within chunks or across chunks to tighten the narrative
15. **Imputation**: Impute what happens between actions to create smooth flow

OUTPUT FORMAT:
- Write in third-person narrative style
- Use present tense for immediacy
- NO timestamps or time markers - let the story flow naturally
- Break into logical paragraphs based on story beats, not time chunks
- Create smooth scene transitions
- Focus on visual description and action
- Avoid color commentary, purple prose, or excessive adjectives
- End with a conclusion that ties everything together

CRITICAL: Every person, object, or action must be described with absolute specificity. No generalizations, no generic terms, no vague descriptions. If you see a "middle-aged obese man in a stained blue t-shirt with a scowl on his face," write exactly that. If you see a "young beautiful woman with long blonde hair wearing a red dress," write exactly that. Be direct, specific, and unfiltered in your descriptions.

NO POETIC LANGUAGE: Avoid phrases like "as if he's measuring the space" or "like he's hitting a beat." Stick to rich, full detailing of what we can actually see. If there is text (overlaid or in scene), transcribe it exactly. Focus on concrete visual details - the scene itself playing out in prose, richly described, will be enough.

Create a tight, continuous story that flows naturally without time constraints, seamlessly coordinating visual actions with spoken dialogue. When audio is available, weave the spoken words naturally into the narrative, creating a rich, multi-sensory experience that captures both what is seen and what is heard. Focus on what happens visually and how events connect logically, with every detail rendered in concrete, specific terms."""

        # Make API call to GPT-5
        response = gpt5_client.client.chat.completions.create(
            model="gpt-5",  # Use GPT-5 for enhanced narrative
            messages=[
                {
                    "role": "user",
                    "content": enhancement_prompt
                }
            ],
            max_completion_tokens=4000
        )
        
        enhanced_narrative = response.choices[0].message.content
        
        # Ensure proper encoding
        if isinstance(enhanced_narrative, str):
            enhanced_narrative = enhanced_narrative.encode('utf-8', errors='replace').decode('utf-8')
        
        return enhanced_narrative
        
    except Exception as e:
        st.error(f"❌ Error creating enhanced narrative: {e}")
        return ""

def rescan_segment(start_time: str, end_time: str, rescan_fps: float):
    """Rescan a specific video segment at higher detail."""
    
    try:
        video_processor = st.session_state.video_processor
        gpt4o_client = st.session_state.gpt4o_client
        profile = st.session_state.current_profile
        
        # Parse timestamps
        start_sec = parse_timestamp(start_time)
        end_sec = parse_timestamp(end_time)
        
        with st.spinner(f"Rescanning segment {start_time} to {end_time}..."):
            # Extract frames at higher FPS for rescan
            rescan_frames = video_processor.extract_frames(
                start_time=start_sec,
                end_time=end_sec,
                custom_fps=rescan_fps
            )
            
            if not rescan_frames:
                st.error("No frames extracted for rescan segment")
                return
            
            # Analyze with rescan prompt
            detailed_narrative, detailed_events = gpt4o_client.rescan_segment(
                start_time, end_time, rescan_frames, profile
            )
            
            # Display rescan results
            st.subheader(f"🔍 Rescan Results: {start_time} - {end_time}")
            
            with st.expander("📖 Detailed Narrative", expanded=True):
                st.markdown(detailed_narrative)
            
            if detailed_events:
                with st.expander("📅 Detailed Events"):
                    for event in detailed_events:
                        st.json(event)
            
            st.success(f"✅ Segment rescan completed with {len(rescan_frames)} frames at {rescan_fps} FPS")
            
    except Exception as e:
        st.error(f"❌ Rescan failed: {e}")

if __name__ == "__main__":
    main()
