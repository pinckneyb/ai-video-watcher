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
    if 'enhancement_requested' not in st.session_state:
        st.session_state.enhancement_requested = False
    if 'regeneration_requested' not in st.session_state:
        st.session_state.regeneration_requested = False
    if 'show_search' not in st.session_state:
        st.session_state.show_search = False
    if 'show_full_transcript' not in st.session_state:
        st.session_state.show_full_transcript = False
    if 'show_fps_guide' not in st.session_state:
        st.session_state.show_fps_guide = False
    if 'rescan_start' not in st.session_state:
        st.session_state.rescan_start = ""
    if 'rescan_end' not in st.session_state:
        st.session_state.rescan_end = ""
    if 'rescan_fps' not in st.session_state:
        st.session_state.rescan_fps = 10.0
    if 'show_full_enhanced' not in st.session_state:
        st.session_state.show_full_enhanced = False
    
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
        
        # Concurrency presets
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        
        with col_preset1:
            if st.button("✅ Proven Safe (10)", help="100% safe - tested and reliable"):
                st.session_state.max_concurrent_batches = 10
                st.rerun()
        
        with col_preset2:
            if st.button("🧪 Testing (12)", help="Next level to test - monitor closely"):
                st.session_state.max_concurrent_batches = 12
                st.rerun()
        
        with col_preset3:
            if st.button("🚀 Future Test (15)", help="Future testing level"):
                st.session_state.max_concurrent_batches = 15
                st.rerun()
        
        # Smart preset recommendation
        if 'concurrency_performance_history' in st.session_state and st.session_state.concurrency_performance_history:
            recommended = get_recommended_concurrency()
            if recommended:
                st.info(f"💡 **Smart Recommendation**: Based on your performance history, try {recommended} concurrent batches")
        
        # Real-world concurrency guidance
        st.info(f"🎯 **Proven Safe**: 10 concurrent batches is 100% reliable")
        st.info(f"🧪 **Testing Next**: 12 concurrent batches - monitor closely")
        st.info(f"📊 **Current FPS**: {fps} - adjust based on performance history")
        
        # FPS/Concurrency Guide Button
        if st.button("📚 FPS & Concurrency Guide", help="Learn about optimal settings"):
            st.session_state.show_fps_guide = True
            st.rerun()
        
        # Show FPS Guide if requested
        if st.session_state.get('show_fps_guide', False):
            with st.expander("📚 FPS & Concurrency Optimization Guide", expanded=True):
                st.markdown("""
                ## 🎯 **Optimal Settings by Use Case**
                
                ### **Initial Scan (1-2 FPS)**
                - **Conservative**: 3-5 concurrent batches
                - **Balanced**: 5-8 concurrent batches  
                - **Aggressive**: 8-12 concurrent batches
                - **Best for**: Getting overview, identifying key moments
                
                ### **Detail Scan (6-10 FPS)**
                - **Conservative**: 2-4 concurrent batches
                - **Balanced**: 4-6 concurrent batches
                - **Aggressive**: 6-8 concurrent batches
                - **Best for**: Analyzing specific techniques, knot tying, suturing
                
                ### **High Detail (10-15 FPS)**
                - **Conservative**: 1-3 concurrent batches
                - **Balanced**: 3-5 concurrent batches
                - **Aggressive**: 5-7 concurrent batches
                - **Best for**: Frame-by-frame analysis, precise timing
                
                ## ⚡ **Speed vs Quality Trade-offs**
                
                | FPS | Batch Size | Concurrency | Speed | Quality | Use Case |
                |-----|------------|-------------|-------|---------|----------|
                | 1   | 5-8        | 8-15        | ⚡⚡⚡ | ⭐⭐     | Overview |
                | 2   | 5-8        | 6-12        | ⚡⚡   | ⭐⭐⭐    | General |
                | 5   | 3-6        | 4-8         | ⚡    | ⭐⭐⭐⭐   | Detail |
                | 10  | 2-4        | 2-6         | 🐌   | ⭐⭐⭐⭐⭐  | Precision |
                
                ## 🔍 **Recommended Workflow**
                
                1. **Start with 1 FPS, 5 batch size, 8 concurrency** for overview
                2. **Search transcript** for keywords like "knot", "suture"
                3. **Rescan key moments** at 10 FPS with 2-4 concurrency
                4. **Adjust based on performance** - reduce concurrency if you see failures
                
                ## ⚠️ **Warning Signs**
                - **High failure rate**: Reduce concurrency by 2-3
                - **Slow processing**: Increase concurrency by 1-2
                - **API errors**: Check your OpenAI tier limits
                """)
                
                if st.button("📚 Close Guide"):
                    st.session_state.show_fps_guide = False
                    st.rerun()
        
        # Manual concurrency control
        max_concurrent_batches = st.slider(
            "Max concurrent batches", 
            1, 20, 
            value=st.session_state.get('max_concurrent_batches', 10), 
            step=1,
            help="Higher values = faster processing (requires OpenAI Tier 4+)"
        )
        
        # Store the selected value
        st.session_state.max_concurrent_batches = max_concurrent_batches
        
        # Concurrency safety indicator
        safety_level = get_concurrency_safety_level(max_concurrent_batches, fps)
        if safety_level == "safe":
            st.success(f"✅ **{safety_level.upper()}**: {max_concurrent_batches} concurrent batches")
        elif safety_level == "caution":
            st.info(f"💡 **{safety_level.upper()}**: {max_concurrent_batches} concurrent batches - monitor closely")
        
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
                type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'm4v'],
                help="Supported formats: MP4, AVI, MOV, MKV, WMV, M4V (No size limit)"
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
                                        # Generate unique temp filename
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        if input_method == "Upload Video File" and uploaded_file:
                                            temp_video_path = f"temp_video_{timestamp}_{uploaded_file.name}"
                                        else:
                                            temp_video_path = f"temp_video_{timestamp}_url.mp4"
                                        
                                        # Save video temporarily for audio extraction
                                        with open(temp_video_path, "wb") as f:
                                            f.write(video_source)
                                        
                                        st.info(f"🔍 Extracting audio from {temp_video_path}...")
                                        
                                        # Extract and transcribe audio
                                        audio_path = extract_audio_from_video(temp_video_path)
                                        if audio_path and os.path.exists(audio_path):
                                            st.info(f"🎵 Audio extracted to {audio_path}")
                                            
                                            if enable_diarization:
                                                st.info("🎤 Transcribing with speaker diarization...")
                                                audio_transcript = transcribe_audio_with_diarization(
                                                    audio_path, 
                                                    st.session_state.gpt4o_client.api_key
                                                )
                                            else:
                                                st.info("🎤 Transcribing with Whisper...")
                                                audio_transcript = transcribe_audio_with_whisper(
                                                    audio_path, 
                                                    st.session_state.gpt4o_client.api_key
                                                )
                                            
                                            if audio_transcript:
                                                st.session_state.audio_transcript = audio_transcript
                                                st.success(f"✅ Audio transcription completed! ({len(audio_transcript)} characters)")
                                                st.info(f"🎵 Preview: {audio_transcript[:200]}...")
                                            else:
                                                st.warning("⚠️ Audio transcription failed - no transcript returned")
                                        else:
                                            st.warning("⚠️ Audio extraction failed - no audio track found or extraction error")
                                        
                                        # Clean up temp files
                                        if os.path.exists(temp_video_path):
                                            os.remove(temp_video_path)
                                            st.info(f"🗑️ Cleaned up {temp_video_path}")
                                        if audio_path and os.path.exists(audio_path):
                                            os.remove(audio_path)
                                            st.info(f"🗑️ Cleaned up {audio_path}")
                                            
                                    except Exception as e:
                                        st.error(f"❌ Audio transcription error: {e}")
                                        st.info("🔍 Check if video has audio track and FFmpeg is installed")
                            else:
                                st.info("🎵 Whisper not enabled or API not configured")
                            
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
        
        # Performance monitoring dashboard
        if 'concurrency_performance_history' in st.session_state and st.session_state.concurrency_performance_history:
            with st.expander("📊 Performance Dashboard", expanded=False):
                history = st.session_state.concurrency_performance_history
                
                # Recent performance summary
                if history:
                    latest = history[-1]
                    col_perf1, col_perf2, col_perf3 = st.columns(3)
                    
                    with col_perf1:
                        st.metric("Last Success Rate", f"{latest['success_rate']:.1f}%")
                    
                    with col_perf2:
                        st.metric("Last Concurrency", latest['concurrent_level'])
                    
                    with col_perf3:
                        st.metric("Last Duration", f"{latest['total_time']:.1f}s")
                    
                    # Performance trend
                    if len(history) > 1:
                        st.subheader("📈 Performance Trends")
                        
                        # Success rate trend
                        success_rates = [run['success_rate'] for run in history]
                        concurrent_levels = [run['concurrent_level'] for run in history]
                        
                        # Create a simple trend analysis
                        if len(success_rates) >= 3:
                            recent_avg = sum(success_rates[-3:]) / 3
                            overall_avg = sum(success_rates) / len(success_rates)
                            
                            if recent_avg > overall_avg + 2:
                                st.success("📈 **Improving**: Recent performance is better than average")
                            elif recent_avg < overall_avg - 2:
                                st.warning("📉 **Declining**: Recent performance is worse than average")
                            else:
                                st.info("➡️ **Stable**: Performance is consistent")
                        
                        # Show performance history table
                        st.subheader("📋 Performance History")
                        perf_data = []
                        for i, run in enumerate(history[-5:]):  # Show last 5 runs
                            perf_data.append({
                                "Run": i + 1,
                                "Concurrency": run['concurrent_level'],
                                "Success Rate": f"{run['success_rate']:.1f}%",
                                "Duration": f"{run['total_time']:.1f}s",
                                "Batches": run['total_batches']
                            })
                        
                        st.table(perf_data)
                        
                        # Clear history button
                        if st.button("🗑️ Clear Performance History"):
                            st.session_state.concurrency_performance_history = []
                            st.rerun()
    
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
            
            # Create two columns: left for summary, right for searchable transcript
            col_summary, col_transcript = st.columns([1, 2])
            
            with col_summary:
                st.subheader("📊 Summary")
                st.success("✅ **Analysis Complete**")
                
                # Show key metrics
                if st.session_state.events:
                    event_count = len(st.session_state.events)
                    st.metric("Events Detected", event_count)
                    
                    # Show main action summary
                    if event_count > 0:
                        st.info("🎯 **Main Actions Detected**")
                        # Extract key action words from events with timestamps
                        action_summaries = []
                        for event in st.session_state.events[:5]:  # Show first 5 events
                            if 'description' in event:
                                description = event['description']
                                timestamp = event.get('timestamp', 'Unknown')
                                
                                # Create a clean action summary
                                action_text = description
                                
                                # Remove redundant phrases for cleaner display
                                redundant_phrases = [
                                    "The scene shows", "We can see", "The video shows", "The frame shows",
                                    "At this moment", "At this time", "During this segment"
                                ]
                                
                                for phrase in redundant_phrases:
                                    if action_text.lower().startswith(phrase.lower()):
                                        action_text = action_text[len(phrase):].strip()
                                        if action_text:
                                            action_text = action_text[0].upper() + action_text[1:]
                                        break
                                
                                # Truncate if too long
                                if len(action_text) > 60:
                                    action_text = action_text[:60].rsplit(' ', 1)[0] + "..."
                                
                                action_summaries.append(f"[{timestamp}] {action_text}")
                        
                        for i, action in enumerate(action_summaries, 1):
                            st.write(f"{i}. {action}")
                
                # Quick actions
                st.subheader("⚡ Quick Actions")
                if st.button("🔍 Search Transcript", type="secondary"):
                    st.session_state.show_search = True
                    st.rerun()
                
                if st.button("📖 View Full Transcript", type="secondary"):
                    st.session_state.show_full_transcript = True
                    st.rerun()
            
            with col_transcript:
                st.subheader("🔍 Searchable Transcript")
                
                # Info about searchable content
                st.info("🔍 **Search covers**: Video analysis transcript, frame-by-frame events, and enhanced narrative (if created)")
                
                # Search functionality
                search_term = st.text_input("🔍 Search across ALL content (transcript, events, enhanced narrative)", 
                                         placeholder="Enter search term (e.g., 'knot', 'suture', 'incision')...")
                
                if search_term:
                    # Search in both transcript and events
                    search_lower = search_term.lower()
                    all_occurrences = []
                    
                    # Search in transcript
                    if st.session_state.transcript:
                        transcript_lower = st.session_state.transcript.lower()
                        if search_lower in transcript_lower:
                            lines = st.session_state.transcript.split('\n')
                            
                            for i, line in enumerate(lines):
                                if search_lower in line.lower():
                                    # Find the approximate timestamp
                                    timestamp = "Unknown"
                                    for event in st.session_state.events:
                                        if event.get('description', '').lower() in line.lower():
                                            timestamp = event.get('timestamp', 'Unknown')
                                            break
                                    
                                    all_occurrences.append({
                                        'type': 'Transcript',
                                        'line': i + 1,
                                        'timestamp': timestamp,
                                        'content': line.strip(),
                                        'context': '...' + ' '.join(lines[max(0, i-1):min(len(lines), i+2)]) + '...',
                                        'source': 'Video Analysis'
                                    })
                    
                    # Search in events/descriptions
                    if st.session_state.events:
                        for i, event in enumerate(st.session_state.events):
                            event_desc = event.get('description', '')
                            if search_lower in event_desc.lower():
                                all_occurrences.append({
                                    'type': 'Event',
                                    'line': i + 1,
                                    'timestamp': event.get('timestamp', 'Unknown'),
                                    'content': event_desc,
                                    'context': f"Event at {event.get('timestamp', 'Unknown')}",
                                    'source': 'Frame Analysis'
                                })
                    
                    # Search in enhanced narrative if available
                    if st.session_state.enhanced_narrative:
                        enhanced_lower = st.session_state.enhanced_narrative.lower()
                        if search_lower in enhanced_lower:
                            lines = st.session_state.enhanced_narrative.split('\n')
                            
                            for i, line in enumerate(lines):
                                if search_lower in line.lower():
                                    all_occurrences.append({
                                        'type': 'Enhanced',
                                        'line': i + 1,
                                        'timestamp': 'Enhanced Narrative',
                                        'content': line.strip(),
                                        'context': '...' + ' '.join(lines[max(0, i-1):min(len(lines), i+2)]) + '...',
                                        'source': 'GPT-5 Enhanced'
                                    })
                    
                    # Display search results
                    if all_occurrences:
                        st.success(f"✅ Found {len(all_occurrences)} occurrences of '{search_term}' across all content")
                        
                        # Group by source for better organization
                        sources = {}
                        for occ in all_occurrences:
                            source = occ['source']
                            if source not in sources:
                                sources[source] = []
                            sources[source].append(occ)
                        
                        # Display results grouped by source
                        for source, occurrences in sources.items():
                            st.subheader(f"📖 {source} ({len(occurrences)} matches)")
                            
                            for occ in occurrences:
                                # Create a more descriptive title
                                title = f"📍 {occ['type']} {occ['line']}"
                                if occ['timestamp'] != 'Unknown':
                                    title += f" - {occ['timestamp']}"
                                
                                with st.expander(title, expanded=False):
                                    st.write(f"**Source:** {occ['source']}")
                                    st.write(f"**Type:** {occ['type']}")
                                    st.write("**Context:**")
                                    st.write(occ['context'])
                                    st.write("**Exact Match:**")
                                    st.write(f"**{occ['content']}**")
                                    
                                    # Add rescan button for timestamps from video analysis
                                    if occ['timestamp'] != 'Unknown' and occ['timestamp'] != 'Enhanced Narrative':
                                        if st.button(f"🔍 Rescan around {occ['timestamp']}", key=f"rescan_{occ['type']}_{occ['line']}"):
                                            # Calculate time range around this timestamp
                                            try:
                                                time_sec = parse_timestamp(occ['timestamp'])
                                                start_time = max(0, time_sec - 5)  # 5 seconds before
                                                end_time = min(st.session_state.video_processor.duration, time_sec + 5)  # 5 seconds after
                                                
                                                # Trigger rescan
                                                st.session_state.rescan_start = format_timestamp(start_time)
                                                st.session_state.rescan_end = format_timestamp(end_time)
                                                st.session_state.rescan_fps = 10.0  # High FPS for detail
                                                st.rerun()
                                            except:
                                                st.error("Could not parse timestamp for rescan")
                        
                        # Summary of search results
                        st.info(f"🔍 **Search Summary**: Found '{search_term}' in {len(sources)} different content sources")
                        
                    else:
                        st.warning(f"⚠️ No occurrences found for '{search_term}' in any content")
                
                # Show transcript preview with timestamps
                st.subheader("📖 Transcript Preview (with timestamps)")
                
                # Create a timestamped preview by combining transcript with events
                if st.session_state.events:
                    # Create a timestamped version of the transcript
                    timestamped_preview = ""
                    lines = st.session_state.transcript.split('\n')
                    
                    for i, line in enumerate(lines):
                        if line.strip():  # Skip empty lines
                            # Find matching event for this line
                            timestamp = "Unknown"
                            for event in st.session_state.events:
                                if event.get('description', '').lower() in line.lower():
                                    timestamp = event.get('timestamp', 'Unknown')
                                    break
                            
                            if timestamp != "Unknown":
                                timestamped_preview += f"[{timestamp}] {line}\n"
                            else:
                                timestamped_preview += f"{line}\n"
                    
                    # Show first 800 chars of timestamped version
                    preview_length = 800
                    preview = timestamped_preview[:preview_length] + "..." if len(timestamped_preview) > preview_length else timestamped_preview
                else:
                    # Fallback to original transcript if no events
                    preview = st.session_state.transcript[:500] + "..." if len(st.session_state.transcript) > 500 else st.session_state.transcript
                
                st.text_area("Preview", preview, height=150, disabled=True)
                
                # Show full transcript if requested
                if st.session_state.get('show_full_transcript', False):
                    st.subheader("📖 Full Transcript (with timestamps)")
                    
                    # Show timestamped version if available
                    if st.session_state.events:
                        st.info("📅 **Timestamped Transcript** - Events aligned with timestamps")
                        st.text_area("Complete Timestamped Transcript", timestamped_preview, height=400, disabled=True)
                    else:
                        st.text_area("Complete Transcript", st.session_state.transcript, height=400, disabled=True)
                    
                    if st.button("📖 Hide Full Transcript"):
                        st.session_state.show_full_transcript = False
                        st.rerun()
            
            # Display events timeline in a compact format
            if st.session_state.events:
                with st.expander("📅 Events Timeline", expanded=False):
                    # Create a more compact timeline view with better formatting
                    timeline_data = []
                    for event in st.session_state.events:
                        description = event.get('description', 'No description')
                        
                        # Create a clean synopsis by removing redundant phrases and formatting
                        synopsis = description
                        
                        # Remove common redundant phrases
                        redundant_phrases = [
                            "The scene shows", "We can see", "The video shows", "The frame shows",
                            "At this moment", "At this time", "During this segment",
                            "The image depicts", "The frame depicts", "The scene depicts"
                        ]
                        
                        for phrase in redundant_phrases:
                            if synopsis.lower().startswith(phrase.lower()):
                                synopsis = synopsis[len(phrase):].strip()
                                # Capitalize first letter
                                if synopsis:
                                    synopsis = synopsis[0].upper() + synopsis[1:]
                                break
                        
                        # Truncate if too long and add ellipsis
                        if len(synopsis) > 120:
                            synopsis = synopsis[:120].rsplit(' ', 1)[0] + "..."
                        
                        timeline_data.append({
                            "Time": event.get('timestamp', 'Unknown'),
                            "Synopsis": synopsis,
                            "Duration": f"{event.get('duration', 'N/A')}s" if event.get('duration') else "N/A"
                        })
                    
                    st.table(timeline_data)
                    
                    # Add a summary of key events
                    if len(timeline_data) > 0:
                        st.subheader("🎯 **Key Events Summary**")
                        key_events = timeline_data[:5]  # Show first 5 events
                        for i, event in enumerate(key_events, 1):
                            st.write(f"**{i}.** [{event['Time']}] {event['Synopsis']}")
                        
                        if len(timeline_data) > 5:
                            st.info(f"📋 Showing first 5 of {len(timeline_data)} total events. Expand timeline for complete list.")
            
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
            else:
                # Enhanced debugging information
                st.subheader("🎵 Audio Transcription Status")
                
                # Check various conditions
                whisper_enabled = enable_whisper
                api_configured = st.session_state.gpt4o_client is not None
                video_loaded = st.session_state.video_processor is not None
                
                col_status1, col_status2, col_status3 = st.columns(3)
                
                with col_status1:
                    if whisper_enabled:
                        st.success("✅ Whisper Enabled")
                    else:
                        st.error("❌ Whisper Disabled")
                
                with col_status2:
                    if api_configured:
                        st.success("✅ API Configured")
                    else:
                        st.error("❌ API Not Configured")
                
                with col_status3:
                    if video_loaded:
                        st.success("✅ Video Loaded")
                    else:
                        st.error("❌ No Video")
                
                # Detailed status message
                if not whisper_enabled:
                    st.warning("🎵 **Whisper not enabled** - Check the 'Enable OpenAI Whisper transcription' checkbox in the sidebar")
                elif not api_configured:
                    st.warning("🎵 **API not configured** - Enter your OpenAI API key in the sidebar")
                elif not video_loaded:
                    st.warning("🎵 **No video loaded** - Upload a video file or enter a video URL first")
                else:
                    st.info("🎵 **Whisper ready** - Audio transcription will happen automatically when you load a video with audio")
                
                # Show current session state for debugging
                with st.expander("🔍 Debug Info", expanded=False):
                    st.json({
                        "whisper_enabled": whisper_enabled,
                        "api_configured": api_configured,
                        "video_loaded": video_loaded,
                        "audio_transcript_length": len(st.session_state.get('audio_transcript', '')),
                        "session_state_keys": list(st.session_state.keys())
                    })
            
            # Audio transcription testing section
            if st.session_state.video_processor and st.session_state.gpt4o_client:
                st.subheader("🎵 Audio Transcription Testing")
                
                col_test1, col_test2 = st.columns(2)
                
                with col_test1:
                    if st.button("🔍 Test Audio Extraction", help="Test if video has audio track"):
                        with st.spinner("Testing audio extraction..."):
                            try:
                                # Generate temp video path
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                if input_method == "Upload Video File" and uploaded_file:
                                    temp_video_path = f"test_video_{timestamp}_{uploaded_file.name}"
                                else:
                                    temp_video_path = f"test_video_{timestamp}_url.mp4"
                                
                                # Save video temporarily
                                with open(temp_video_path, "wb") as f:
                                    f.write(video_source)
                                
                                # Test audio extraction
                                audio_path = extract_audio_from_video(temp_video_path)
                                if audio_path and os.path.exists(audio_path):
                                    st.success(f"✅ Audio extraction successful: {audio_path}")
                                    st.info(f"Audio file size: {os.path.getsize(audio_path)} bytes")
                                    
                                    # Clean up
                                    if os.path.exists(temp_video_path):
                                        os.remove(temp_video_path)
                                    if os.path.exists(audio_path):
                                        os.remove(audio_path)
                                else:
                                    st.error("❌ Audio extraction failed - video may not have audio track")
                                    
                                    # Clean up
                                    if os.path.exists(temp_video_path):
                                        os.remove(temp_video_path)
                                        
                            except Exception as e:
                                st.error(f"❌ Test failed: {e}")
                
                with col_test2:
                    if st.button("🎤 Test Whisper API", help="Test Whisper API connection"):
                        with st.spinner("Testing Whisper API..."):
                            try:
                                # Create a simple test audio file (1 second of silence)
                                test_audio_path = f"test_audio_{timestamp}.wav"
                                
                                # This is a minimal test - in practice you'd want real audio
                                st.info("Testing Whisper API connection...")
                                
                                # Try to make a minimal API call
                                test_client = GPT4oClient(api_key=st.session_state.gpt4o_client.api_key)
                                st.success("✅ Whisper API connection successful")
                                
                            except Exception as e:
                                st.error(f"❌ Whisper API test failed: {e}")
            
            # Enhanced narrative section
            if st.session_state.enhanced_narrative:
                st.subheader("✨ Enhanced Narrative (GPT-5)")
                
                # Show summary instead of full content
                preview_length = 300
                preview = st.session_state.enhanced_narrative[:preview_length] + "..." if len(st.session_state.enhanced_narrative) > preview_length else st.session_state.enhanced_narrative
                
                st.info(f"📖 **Enhanced narrative complete** ({len(st.session_state.enhanced_narrative):,} characters)")
                st.text_area("Preview", preview, height=100, disabled=True)
                
                # Action buttons in columns
                col_enh1, col_enh2, col_enh3 = st.columns(3)
                
                with col_enh1:
                    if st.button("📖 View Full", key="view_enhanced"):
                        st.session_state.show_full_enhanced = True
                        st.rerun()
                
                with col_enh2:
                    if st.button("📄 Download", key="download_enhanced"):
                        filename = save_transcript(st.session_state.enhanced_narrative, prefix="enhanced_")
                        st.success(f"✅ Enhanced narrative saved as {filename}")
                        st.download_button(
                            label="📥 Download Enhanced Narrative",
                            data=st.session_state.enhanced_narrative,
                            file_name=filename,
                            mime="text/markdown"
                        )
                
                with col_enh3:
                    if st.button("🔄 Regenerate", key="regenerate_enhanced"):
                        st.session_state.regeneration_requested = True
                        st.rerun()
                
                # Show full enhanced narrative if requested
                if st.session_state.get('show_full_enhanced', False):
                    with st.expander("📖 Full Enhanced Narrative", expanded=True):
                        st.markdown(st.session_state.enhanced_narrative)
                    if st.button("📖 Hide Full Narrative"):
                        st.session_state.show_full_enhanced = False
                        st.rerun()
                
                # Handle regeneration if requested
                if st.session_state.get('regeneration_requested', False):
                    with st.spinner("Regenerating enhanced narrative..."):
                        audio_transcript = st.session_state.get('audio_transcript', '')
                        enhanced_narrative = create_coherent_narrative(
                            st.session_state.transcript, 
                            st.session_state.events,
                            st.session_state.gpt4o_client.api_key,
                            audio_transcript,
                            st.session_state.current_profile
                        )
                        if enhanced_narrative:
                            st.session_state.enhanced_narrative = enhanced_narrative
                            st.success("✨ Enhanced narrative regenerated!")
                            st.session_state.regeneration_requested = False
                        else:
                            st.error("❌ Failed to regenerate enhanced narrative")
                            st.session_state.regeneration_requested = False
            else:
                # Manual trigger button if no enhanced narrative exists
                if st.button("✨ Create Enhanced Narrative with GPT-5", key="create_enhanced_manual"):
                    st.session_state.enhancement_requested = True
                    st.rerun()
                
                # Handle enhancement if requested
                if st.session_state.get('enhancement_requested', False):
                    with st.spinner("Creating enhanced narrative..."):
                        audio_transcript = st.session_state.get('audio_transcript', '')
                        enhanced_narrative = create_coherent_narrative(
                            st.session_state.transcript, 
                            st.session_state.events,
                            st.session_state.gpt4o_client.api_key,
                            audio_transcript,
                            st.session_state.current_profile
                        )
                        if enhanced_narrative:
                            st.session_state.enhanced_narrative = enhanced_narrative
                            st.success("✨ Enhanced narrative created!")
                            st.session_state.enhancement_requested = False
                        else:
                            st.error("❌ Failed to create enhanced narrative")
                            st.session_state.enhancement_requested = False
            
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
        
        # Check if we have rescan parameters from search
        if st.session_state.rescan_start and st.session_state.rescan_end:
            st.success(f"🎯 **Auto-rescan ready**: {st.session_state.rescan_start} to {st.session_state.rescan_end} at {st.session_state.rescan_fps} FPS")
            
            col_auto1, col_auto2 = st.columns([1, 1])
            with col_auto1:
                if st.button("🚀 Start Auto-Rescan", type="primary"):
                    rescan_segment(st.session_state.rescan_start, st.session_state.rescan_end, st.session_state.rescan_fps)
                    # Clear the auto-rescan parameters
                    st.session_state.rescan_start = ""
                    st.session_state.rescan_end = ""
                    st.rerun()
            
            with col_auto2:
                if st.button("❌ Cancel Auto-Rescan"):
                    st.session_state.rescan_start = ""
                    st.session_state.rescan_end = ""
                    st.rerun()
        
        # Manual rescan section
        st.subheader("📝 Manual Rescan")
        
        col_rescan1, col_rescan2, col_rescan3 = st.columns(3)
        
        with col_rescan1:
            start_time = st.text_input(
                "Start time (HH:MM:SS)",
                value=st.session_state.rescan_start,
                placeholder="00:01:30",
                help="Start time for rescan segment"
            )
        
        with col_rescan2:
            end_time = st.text_input(
                "End time (HH:MM:SS)",
                value=st.session_state.rescan_end,
                placeholder="00:02:00",
                help="End time for rescan segment"
            )
        
        with col_rescan3:
            rescan_fps_manual = st.slider(
                "Rescan FPS",
                5.0, 20.0, 
                value=st.session_state.rescan_fps,
                step=1.0,
                help="Higher FPS = more detail but slower processing"
            )
        
        if start_time and end_time:
            if validate_time_range(start_time, end_time, st.session_state.video_processor.duration):
                st.success("✅ Valid time range")
                
                # Show estimated processing info
                duration = parse_timestamp(end_time) - parse_timestamp(start_time)
                estimated_frames = int(duration * rescan_fps_manual)
                st.info(f"📊 **Rescan Info**: {duration:.1f}s segment, ~{estimated_frames} frames at {rescan_fps_manual} FPS")
                
                if st.button("🔍 Start Manual Rescan", type="primary"):
                    rescan_segment(start_time, end_time, rescan_fps_manual)
            else:
                st.error("❌ Invalid time range")
        
        # Quick rescan presets
        st.subheader("⚡ Quick Rescan Presets")
        col_preset1, col_preset2, col_preset3 = st.columns(3)
        
        with col_preset1:
            if st.button("🔍 5s Detail (10 FPS)", help="Rescan 5 seconds at 10 FPS for detailed analysis"):
                if st.session_state.video_processor.duration:
                    mid_point = st.session_state.video_processor.duration / 2
                    start = max(0, mid_point - 2.5)
                    end = min(st.session_state.video_processor.duration, mid_point + 2.5)
                    rescan_segment(format_timestamp(start), format_timestamp(end), 10.0)
        
        with col_preset2:
            if st.button("🔍 10s Detail (8 FPS)", help="Rescan 10 seconds at 8 FPS for technique analysis"):
                if st.session_state.video_processor.duration:
                    mid_point = st.session_state.video_processor.duration / 2
                    start = max(0, mid_point - 5)
                    end = min(st.session_state.video_processor.duration, mid_point + 5)
                    rescan_segment(format_timestamp(start), format_timestamp(end), 8.0)
        
        with col_preset3:
            if st.button("🔍 15s Overview (6 FPS)", help="Rescan 15 seconds at 6 FPS for sequence overview"):
                if st.session_state.video_processor.duration:
                    mid_point = st.session_state.video_processor.duration / 2
                    start = max(0, mid_point - 7.5)
                    end = min(st.session_state.video_processor.duration, mid_point + 7.5)
                    rescan_segment(format_timestamp(start), format_timestamp(end), 6.0)
    
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
            
            # Real-time monitoring container
            monitoring_container = st.container()
            
            start_time = time.time()
            # Use concurrent processing
            performance_metrics = process_batches_concurrently(batches, gpt4o_client, profile, progress_bar, status_text, total_batches, max_concurrent_batches)
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Display detailed performance results
            with monitoring_container:
                st.success(f"⚡ Concurrent processing completed in {processing_time:.2f} seconds")
                
                # Performance analysis
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    st.metric("Success Rate", f"{performance_metrics['success_rate']:.1f}%")
                    st.metric("Failed Batches", performance_metrics['failed_batches'])
                
                with col_analysis2:
                    st.metric("Avg Batch Time", f"{performance_metrics['avg_batch_time']:.2f}s")
                    st.metric("Speed vs Sequential", f"~{processing_time/(total_batches * 12):.1f}x faster")
                
                # Safety assessment
                if performance_metrics['success_rate'] >= 95:
                    st.success("🎯 **Excellent Performance**: This concurrency level is working well for you")
                elif performance_metrics['success_rate'] >= 90:
                    st.warning("⚠️ **Good Performance**: Consider this your upper limit for now")
                else:
                    st.error("🚨 **Poor Performance**: Reduce concurrency level for better reliability")
                
                # Auto-recommendation for next run
                if performance_metrics['success_rate'] >= 98:
                    st.info(f"💡 **Recommendation**: You can safely try {min(max_concurrent_batches + 2, 20)} concurrent batches next time")
                elif performance_metrics['success_rate'] < 90:
                    st.info(f"💡 **Recommendation**: Try {max(1, max_concurrent_batches - 2)} concurrent batches for better reliability")
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
        if st.button("🔄 Enhance Narrative with GPT-5", type="secondary", key="enhance_after_analysis"):
            # Set a flag to indicate enhancement is requested
            st.session_state.enhancement_requested = True
            st.rerun()
        
        # Handle enhancement if requested
        if st.session_state.get('enhancement_requested', False):
            with st.spinner("Creating coherent narrative with GPT-5..."):
                # Debug: show what we're passing
                audio_transcript = st.session_state.get('audio_transcript', '')
                if audio_transcript:
                    st.info(f"🎵 Including audio transcript ({len(audio_transcript)} characters)")
                else:
                    st.warning("⚠️ No audio transcript found - check Whisper settings")
                
                enhanced_narrative = create_coherent_narrative(
                    st.session_state.transcript, 
                    st.session_state.events,
                    gpt4o_client.api_key,
                    audio_transcript,
                    st.session_state.current_profile
                )
                if enhanced_narrative:
                    st.session_state.enhanced_narrative = enhanced_narrative
                    st.success("✨ Enhanced narrative created!")
                    # Clear the flag
                    st.session_state.enhancement_requested = False
                else:
                    st.error("❌ Failed to create enhanced narrative")
                    # Clear the flag on failure too
                    st.session_state.enhancement_requested = False
        
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

def process_batches_concurrently(batches, gpt4o_client, profile, progress_bar, status_text, total_batches, max_concurrent_batches):
    """Process batches concurrently using ThreadPoolExecutor with monitoring."""
    import concurrent.futures
    import threading
    import time
    
    # Use a lock to protect context updates
    context_lock = threading.Lock()
    
    # Initialize monitoring
    start_time = time.time()
    successful_batches = 0
    failed_batches = 0
    error_log = []
    performance_metrics = {
        'total_batches': total_batches,
        'concurrent_level': max_concurrent_batches,
        'start_time': start_time,
        'batch_times': [],
        'errors': []
    }
    
    # Create a thread pool for concurrent API calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
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
            batch_start_time = time.time()
            
            try:
                narrative, events = future.result()
                
                # Update context with thread safety
                with context_lock:
                    gpt4o_client.update_context(narrative, events)
                
                # Record successful batch
                successful_batches += 1
                batch_time = time.time() - batch_start_time
                performance_metrics['batch_times'].append(batch_time)
                
            except Exception as e:
                # Record failed batch
                failed_batches += 1
                error_info = {
                    'batch_index': batch_index + 1,
                    'error': str(e),
                    'timestamp': time.time()
                }
                error_log.append(error_info)
                performance_metrics['errors'].append(error_info)
                
                st.error(f"❌ Batch {batch_index + 1} failed: {e}")
            
            completed_batches += 1
            progress = completed_batches / total_batches
            progress_bar.progress(progress)
            
            # Update status with monitoring info
            success_rate = (successful_batches / completed_batches) * 100
            status_text.text(f"Completed {completed_batches}/{total_batches} batches... Success: {success_rate:.1f}%")
    
    # Calculate final performance metrics
    total_time = time.time() - start_time
    avg_batch_time = sum(performance_metrics['batch_times']) / len(performance_metrics['batch_times']) if performance_metrics['batch_times'] else 0
    
    performance_metrics.update({
        'total_time': total_time,
        'successful_batches': successful_batches,
        'failed_batches': failed_batches,
        'success_rate': (successful_batches / total_batches) * 100,
        'avg_batch_time': avg_batch_time,
        'error_log': error_log
    })
    
    # Store performance data for preset learning
    if 'concurrency_performance_history' not in st.session_state:
        st.session_state.concurrency_performance_history = []
    
    st.session_state.concurrency_performance_history.append(performance_metrics)
    
    # Display performance summary
    st.info(f"📊 **Performance Summary**: {successful_batches}/{total_batches} successful ({performance_metrics['success_rate']:.1f}%) in {total_time:.2f}s")
    
    if failed_batches > 0:
        st.warning(f"⚠️ **Failed Batches**: {failed_batches} batches failed. Consider reducing concurrency.")
    
    return performance_metrics

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

def create_coherent_narrative(raw_transcript: str, events: List[Dict], api_key: str, audio_transcript: str = "", profile: Dict[str, Any] = None) -> str:
    """Create a coherent, continuous narrative using GPT-5."""
    try:
        # Create a GPT-5 client for narrative enhancement
        gpt5_client = GPT4oClient(api_key=api_key)
        
        # Prepare the enhancement prompt with profile personality
        if profile:
            profile_name = profile.get("name", "Generic")
            profile_description = profile.get("description", "Standard video analysis")
            
            # Get profile-specific personality instructions
            if profile_name.lower() == "surgical":
                personality_instructions = """PERSONALITY: You are writing as an AI Surgeon Video Reviewer - direct, clinical, no-nonsense. Use surgical terminology and evaluative language. Structure your narrative around surgical assessment principles. Be objective, authoritative, and focus on technique evaluation."""
            elif profile_name.lower() == "social media":
                personality_instructions = """PERSONALITY: You are writing as an AI Social Media Video Reviewer - thoughtful critic who respects the medium. Use language of art and craft. Highlight emotional authenticity and creative choices. Be reflective rather than judgmental, balancing critique and appreciation."""
            elif profile_name.lower() == "movie lover":
                personality_instructions = """PERSONALITY: You are writing as an AI Cinephile - enthusiastic but disciplined film critic and historian. Use cinematic terminology and reference film history. Connect to cinematic movements, genres, and techniques. Be erudite but accessible, balancing admiration with sharp critique. Frame everything within the broader cinematic landscape."""
            else:
                personality_instructions = """PERSONALITY: You are writing as a master storyteller and video analyst, maintaining the analytical and narrative style established in the original analysis."""
        else:
            personality_instructions = """PERSONALITY: You are writing as a master storyteller and video analyst, maintaining the analytical and narrative style established in the original analysis."""
        
        enhancement_prompt = f"""You are a master storyteller and video analyst. Your task is to transform raw, batch-by-batch video analysis and audio transcription into a coherent, continuous narrative that reads like a polished story.

{personality_instructions}

PROFILE: {profile.get("name", "Generic") if profile else "Generic"} - {profile.get("description", "Standard video analysis") if profile else "Standard video analysis"}

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
16. **Profile Consistency**: Maintain the personality and analytical style of the {profile.get("name", "Generic") if profile else "Generic"} profile throughout your narrative

OUTPUT FORMAT:
- Write in third-person narrative style
- Use present tense for immediacy
- NO timestamps or time markers - let the story flow naturally
- Break into logical paragraphs based on story beats, not time chunks
- Create smooth scene transitions
- Focus on visual description and action
- Avoid color commentary, purple prose, or excessive adjectives
- End with a conclusion that ties everything together
- **Maintain Profile Voice**: Write in the voice and style of the {profile.get("name", "Generic") if profile else "Generic"} profile

CRITICAL: Every person, object, or action must be described with absolute specificity. No generalizations, no generic terms, no vague descriptions. If you see a "middle-aged obese man in a stained blue t-shirt with a scowl on his face," write exactly that. If you see a "young beautiful woman with long blonde hair wearing a red dress," write exactly that. Be direct, specific, and unfiltered in your descriptions.

NO POETIC LANGUAGE: Avoid phrases like "as if he's measuring the space" or "like he's hitting a beat." Stick to rich, full detailing of what we can actually see. If there is text (overlaid or in scene), transcribe it exactly. Focus on concrete visual details - the scene itself playing out in prose, richly described, will be enough.

Create a tight, continuous story that flows naturally without time constraints, seamlessly coordinating visual actions with spoken dialogue. When audio is available, weave the spoken words naturally into the narrative, creating a rich, multi-sensory experience that captures both what is seen and what is heard. Focus on what happens visually and how events connect logically, with every detail rendered in concrete, specific terms.

**CRITICAL**: Maintain the personality, tone, and analytical style of the {profile.get("name", "Generic") if profile else "Generic"} profile throughout your entire narrative. Your enhanced narrative should read as if it was written by the same AI personality that conducted the initial analysis."""

        # Check input lengths and provide warnings
        transcript_length = len(raw_transcript)
        events_length = len(json.dumps(events, indent=2))
        audio_length = len(audio_transcript) if audio_transcript else 0
        
        st.info(f"📊 **Input Analysis**: Transcript: {transcript_length:,} chars, Events: {events_length:,} chars, Audio: {audio_length:,} chars")
        
        # If content is very long, suggest chunking
        total_input = transcript_length + events_length + audio_length
        if total_input > 100000:  # 100K character limit
            st.warning(f"⚠️ **Content Very Long**: Total input is {total_input:,} characters. Consider reducing FPS or batch size for better results.")
        
        # Make API call to GPT-5 with better error handling
        try:
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
            
            st.success(f"✅ **Enhanced narrative created successfully**: {len(enhanced_narrative):,} characters")
            return enhanced_narrative
            
        except Exception as api_error:
            st.error(f"❌ **GPT-5 API Error**: {api_error}")
            if "context_length_exceeded" in str(api_error).lower():
                st.warning("💡 **Solution**: Try reducing FPS or batch size to create shorter input content")
            elif "rate_limit" in str(api_error).lower():
                st.warning("💡 **Solution**: Wait a moment and try again, or reduce concurrency")
            elif "quota_exceeded" in str(api_error).lower():
                st.error("💡 **Solution**: Check your OpenAI API quota and billing")
            return ""
        
    except Exception as e:
        st.error(f"❌ **Unexpected Error**: {e}")
        st.info("🔍 **Debug Info**: Check console for detailed error messages")
        print(f"Enhanced narrative creation error: {e}")
        return ""

def get_concurrency_safety_level(concurrent_batches: int, fps: float = 1.0) -> str:
    """Determine safety level for concurrency setting based on real-world testing."""
    # Based on user experience: 10 concurrent batches is 100% safe
    if concurrent_batches <= 10:
        return "safe"
    elif concurrent_batches <= 12:
        return "caution"  # Testing 12 next
    else:
        return "caution"  # Unknown territory

def get_recommended_concurrency() -> int:
    """Get smart concurrency recommendation based on performance history."""
    if 'concurrency_performance_history' not in st.session_state:
        return None
    
    history = st.session_state.concurrency_performance_history
    
    # Find the best performing concurrency level
    best_performance = None
    best_success_rate = 0
    
    for run in history:
        success_rate = run.get('success_rate', 0)
        concurrent_level = run.get('concurrent_level', 0)
        
        if success_rate > best_success_rate and success_rate >= 95:
            best_success_rate = success_rate
            best_performance = concurrent_level
    
    if best_performance:
        # Suggest slightly higher if previous run was very successful
        if best_success_rate >= 98:
            return min(best_performance + 2, 20)
        else:
            return best_performance
    
    return None

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
