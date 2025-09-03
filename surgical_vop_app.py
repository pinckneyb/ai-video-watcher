"""
Surgical VOP (Verification of Proficiency) Assessment Application
Specialized app for evaluating suturing technique videos using structured rubrics.
"""

import streamlit as st
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime
from pathlib import Path

# Import our modules from the main app
from video_processor import VideoProcessor, FrameBatchProcessor
from gpt4o_client import GPT4oClient
from utils import (
    parse_timestamp, format_timestamp, validate_time_range,
    save_transcript, extract_video_info, sanitize_filename
)

# Page configuration
st.set_page_config(
    page_title="Surgical VOP Assessment",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set upload limit to 2GB for large MOV files
import streamlit as st
try:
    st._config.set_option('server.maxUploadSize', 2048)
except:
    # Fallback if config setting fails
    pass

class SuturePatternDetector:
    """Detects suture patterns from folder names and filenames."""
    
    def __init__(self):
        self.patterns = {
            "simple_interrupted": ["simple", "interrupted", "simple_interrupted"],
            "vertical_mattress": ["vertical", "mattress", "vertical_mattress"],
            "subcuticular": ["subcuticular", "running", "continuous"]
        }
    
    def detect_pattern(self, folder_path: str, filename: str) -> Optional[str]:
        """
        Detect suture pattern from folder and filename.
        
        Args:
            folder_path: Path to the folder containing the video
            filename: Name of the video file
            
        Returns:
            Detected pattern ID or None if not detected
        """
        # Combine folder and filename for analysis
        combined_text = f"{folder_path} {filename}".lower()
        
        # Check each pattern
        for pattern_id, keywords in self.patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                return pattern_id
        
        return None
    
    def get_available_patterns(self) -> List[str]:
        """Get list of available pattern IDs."""
        return list(self.patterns.keys())

class RubricEngine:
    """Handles rubric loading and scoring logic."""
    
    def __init__(self, rubric_path: str = "unified_rubric.JSON"):
        self.rubric_data = self._load_rubric(rubric_path)
        self.scale = self.rubric_data["global_policies"]["scale"]
        self.baseline_score = self.rubric_data["global_policies"]["baseline_score_if_unclear"]
    
    def _load_rubric(self, rubric_path: str) -> Dict[str, Any]:
        """Load rubric from JSON file."""
        try:
            with open(rubric_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading rubric: {e}")
            return {}
    
    def get_pattern_rubric(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """Get rubric points for a specific pattern."""
        for pattern in self.rubric_data.get("patterns", []):
            if pattern["id"] == pattern_id:
                return pattern
        return None
    
    def get_assessment_criteria(self, pattern_id: str) -> List[Dict[str, Any]]:
        """Get list of assessment points for a pattern."""
        pattern = self.get_pattern_rubric(pattern_id)
        return pattern["points"] if pattern else []
    
    def calculate_overall_score(self, point_scores: Dict[int, int]) -> Dict[str, Any]:
        """Calculate overall assessment result using VOP-aligned criteria."""
        if not point_scores:
            return {"pass": False, "reason": "No scores provided"}
        
        # VOP pass rule: Pass if every rubric point >= 2; otherwise Remediation
        all_scores_above_2 = all(score >= 2 for score in point_scores.values())
        
        return {
            "pass": all_scores_above_2,
            "total_points": len(point_scores),
            "average_score": sum(point_scores.values()) / len(point_scores),
            "reason": "PASS - Meets VOP competency standards" if all_scores_above_2 else "REMEDIATION - Requires additional training"
        }

class SurgicalAssessmentProfile:
    """Specialized profile for surgical technique assessment."""
    
    def __init__(self, rubric_engine: RubricEngine):
        self.rubric_engine = rubric_engine
        self.narrative_guides = self._load_narrative_guides()
    
    def _load_narrative_guides(self) -> Dict[str, str]:
        """Load the ideal narrative examples for each pattern."""
        guides = {}
        patterns = {
            "simple_interrupted": "simple_interrupted_narrative.txt",
            "vertical_mattress": "vertical_mattress_narrative.txt", 
            "subcuticular": "subcuticular_narrative.txt"
        }
        
        for pattern_id, filename in patterns.items():
            try:
                # Try multiple encodings to handle any encoding issues
                for encoding in ['utf-8', 'utf-8-sig', 'latin1', 'cp1252']:
                    try:
                        with open(filename, 'r', encoding=encoding) as f:
                            guides[pattern_id] = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, use fallback
                    guides[pattern_id] = "Narrative guide not available due to encoding issues."
                    print(f"Could not load narrative guide for {pattern_id}: encoding issues")
            except Exception as e:
                print(f"Could not load narrative guide for {pattern_id}: {e}")
                guides[pattern_id] = "Narrative guide not available."
        
        return guides
    
    def create_assessment_prompt(self, pattern_id: str) -> str:
        """Create GPT-4o prompt for surgical assessment using proven narrative structure."""
        pattern_data = self.rubric_engine.get_pattern_rubric(pattern_id)
        assessment_points = self.rubric_engine.get_assessment_criteria(pattern_id)
        ideal_narrative = self.narrative_guides.get(pattern_id, "No narrative guide available.")
        
        # Build detailed assessment criteria (all points equal weight)
        criteria_text = "\n".join([
            f"**{p['pid']}. {p['title']}**\n"
            f"- What to assess: {p['what_you_assess']}\n"
            f"- Ideal result: {p['ideal_result']}\n"
            for p in assessment_points
        ])
        
        prompt = f"""You are conducting a VOP assessment for {pattern_data['display_name']} suturing. Analyze frames for evidence of each rubric point. Be concise.

RUBRIC POINTS TO ASSESS:
{criteria_text}

IDEAL REFERENCE:
{ideal_narrative[:500]}...

REQUIREMENTS:
1. Assess each rubric point with specific evidence
2. Count hand/instrument departures from work area for efficiency
3. Identify pattern type with confidence (1-5)
4. Be brief - final report will be one page

OUTPUT: Brief observations for each rubric point with timestamps."""

        return prompt
    
    def create_context_condensation_prompt(self) -> str:
        """Create context condensation prompt for surgical assessment continuity."""
        return """Condense surgical assessment progress in 100 words max. Include key technique observations, errors noted, and current suturing phase. Maintain clinical tone."""

def create_surgical_vop_narrative(raw_transcript: str, events: List[Dict], api_key: str, pattern_id: str, rubric_engine: RubricEngine) -> str:
    """Create enhanced surgical VOP narrative using GPT-5 - copied from working app.py structure."""
    try:
        # Get pattern and rubric information
        pattern_data = rubric_engine.get_pattern_rubric(pattern_id)
        assessment_points = rubric_engine.get_assessment_criteria(pattern_id)
        
        # Build rubric criteria for reference
        rubric_criteria = "\n".join([
            f"{p['pid']}. {p['title']}: {p['what_you_assess']} (Ideal: {p['ideal_result']})"
            for p in assessment_points
        ])
        
        # Create a GPT-5 client for narrative enhancement (EXACT COPY from app.py)
        gpt5_client = GPT4oClient(api_key=api_key)
        
        # Use the EXACT SAME structure as the working app.py but for surgical assessment
        personality_instructions = """PERSONALITY: You are writing as an AI Surgeon Video Reviewer - direct, clinical, no-nonsense. Use surgical terminology and evaluative language. Structure your narrative around surgical assessment principles. Be objective, authoritative, and focus on technique evaluation."""
        
        enhancement_prompt = f"""YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE. You are training surgeons who will operate on real patients. Assume EVERY technique has flaws until proven otherwise.

RAW ANALYSIS:
{raw_transcript[:3000]}...

RUBRIC CRITERIA:
{rubric_criteria}

STRICT SCORING GUIDELINES:
- Score 1 = Major deficiencies - technique significantly below standard
- Score 2 = Some deficiencies - technique below standard with notable issues  
- Score 3 = Meets standard - technique is adequate and competent
- Score 4 = Exceeds standard - technique is consistently good with minor areas for improvement
- Score 5 = Exemplary - technique demonstrates mastery and serves as a model

CRITICAL SCORING PHILOSOPHY:
- Score 2 should be your DEFAULT for safe, functional technique
- Score 4 means you would use this video to teach other attendings
- Score 5 means this is among the best technique you've seen in your entire career
- Assume EVERY technique has flaws until proven otherwise
- You are training surgeons who will operate on real patients

For each rubric point, write ONE OR TWO SENTENCES that:
- Describe what you observed in the technique
- Explain why the performance earned its score
- NO timestamps, NO references to inability to judge
- The AI must judge from the evidence available

Then write a summative paragraph that:
- Provides entirely actionable critiques from holistic review
- NO reprise of individual rubric assessments
- NO timestamps, NO uncertainty about visibility
- Must be useful observations and nothing else

MANDATORY SCORING:
RUBRIC_SCORES_START
1: X
2: X  
3: X
4: X
5: X
6: X
7: X
RUBRIC_SCORES_END"""

        # Check input lengths and provide warnings (EXACT COPY from app.py)
        transcript_length = len(raw_transcript)
        events_length = len(json.dumps(events, indent=2))
        
        st.info(f"📊 **Input Analysis**: Transcript: {transcript_length:,} chars, Events: {events_length:,} chars")
        
        # If content is very long, suggest chunking
        total_input = transcript_length + events_length
        if total_input > 100000:  # 100K character limit
            st.warning(f"⚠️ **Content Very Long**: Total input is {total_input:,} characters. Consider reducing FPS or batch size for better results.")
        
        # Make API call to GPT-5 with better error handling (EXACT COPY from app.py)
        try:
            response = gpt5_client.client.chat.completions.create(
                model="gpt-5",  # Use GPT-5 for enhanced narrative
                messages=[
                    {
                        "role": "user",
                        "content": enhancement_prompt
                    }
                ],
                max_completion_tokens=8000,
                reasoning_effort="low"
            )
            
            enhanced_narrative = response.choices[0].message.content
            
            # Ensure proper encoding (EXACT COPY from app.py)
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

def extract_rubric_scores_from_narrative(enhanced_narrative: str) -> Dict[int, int]:
    """Extract numerical scores from GPT-5 enhanced narrative."""
    scores = {}
    
    if not enhanced_narrative:
        return scores
    
    try:
        # Look for the scoring section
        start_marker = "RUBRIC_SCORES_START"
        end_marker = "RUBRIC_SCORES_END"
        
        start_idx = enhanced_narrative.find(start_marker)
        end_idx = enhanced_narrative.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            scores_section = enhanced_narrative[start_idx + len(start_marker):end_idx].strip()
            
            # Parse each line
            for line in scores_section.split('\n'):
                line = line.strip()
                if ':' in line:
                    try:
                        point_id, score_str = line.split(':', 1)
                        point_id = int(point_id.strip())
                        score = int(score_str.strip())
                        
                        # Validate score is in range
                        if 1 <= score <= 5:
                            scores[point_id] = score
                        else:
                            st.warning(f"Score {score} for point {point_id} is out of range (1-5)")
                    except (ValueError, IndexError) as e:
                        st.warning(f"Could not parse score line: {line}")
                        continue
        else:
            st.warning("GPT-5 response did not include proper scoring format - scores will default to 3")
            
    except Exception as e:
        st.error(f"Error extracting scores from narrative: {e}")
    
    return scores

def save_api_key(api_key: str):
    """Save API key to config file."""
    config_file = "surgical_config.json"
    
    try:
        config = {"openai_api_key": api_key}
        with open(config_file, "w") as f:
            json.dump(config, f)
    except Exception as e:
        print(f"Could not save API key: {e}")

def load_api_key() -> str:
    """Load API key from config file."""
    config_file = "surgical_config.json"
    
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                return config.get("openai_api_key", "")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    return ""

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'vop_analysis_complete' not in st.session_state:
        st.session_state.vop_analysis_complete = False
    if 'selected_pattern' not in st.session_state:
        st.session_state.selected_pattern = None
    if 'rubric_scores' not in st.session_state:
        st.session_state.rubric_scores = {}
    if 'assessment_results' not in st.session_state:
        st.session_state.assessment_results = None
    if 'saved_api_key' not in st.session_state:
        st.session_state.saved_api_key = load_api_key()

def detect_pattern_from_upload(uploaded_file) -> Optional[str]:
    """Detect pattern from uploaded file name(s)."""
    if not uploaded_file:
        return None
    
    detector = SuturePatternDetector()
    
    # Handle both single file and multiple files
    if isinstance(uploaded_file, list):
        # For multiple files, use the first file for pattern detection
        if len(uploaded_file) > 0:
            filename = uploaded_file[0].name
        else:
            return None
    else:
        # Single file
        filename = uploaded_file.name
    
    # For uploaded files, we don't have folder context
    return detector.detect_pattern("", filename)

def main():
    """Main application function."""
    
    st.title("🏥 Surgical VOP Assessment")
    st.markdown("*Verification of Proficiency - Suturing Technique Evaluation*")
    
    initialize_session_state()
    
    # Initialize components
    pattern_detector = SuturePatternDetector()
    rubric_engine = RubricEngine()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📋 Assessment Configuration")
        
        # Pattern detection and selection
        st.subheader("🧵 Suture Pattern")
        
        uploaded_file = st.file_uploader(
            "Upload Surgical Video(s)",
            type=['mp4', 'avi', 'mov', 'mkv', 'm4v'],
            accept_multiple_files=True,
            help="Upload one or multiple videos of suturing technique for assessment (up to 2GB each). MOV files supported."
        )
        
        # Add upload status and troubleshooting
        if uploaded_file is not None:
            if isinstance(uploaded_file, list):
                # Multiple files - just show count
                st.info(f"📁 **Files**: {len(uploaded_file)} videos selected")
                for i, file in enumerate(uploaded_file, 1):
                    file_size_mb = len(file.read()) / (1024 * 1024)
                    file.seek(0)  # Reset file pointer
                    if file_size_mb > 2000:  # 2GB limit
                        st.error(f"❌ **File too large**: {file.name} exceeds 2GB limit")
                        uploaded_file = None
                        break
            else:
                # Single file - just show count
                file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
                uploaded_file.seek(0)  # Reset file pointer
                
                if file_size_mb > 2000:  # 2GB limit
                    st.error("❌ **File too large**: Maximum size is 2GB. Please compress your video.")
                    uploaded_file = None
        
        if uploaded_file:
            # Pattern detection for single or multiple files
            if isinstance(uploaded_file, list):
                # Multiple files - detect pattern for each
                st.subheader("🔍 Pattern Detection Results")
                detected_patterns = {}
                for i, file in enumerate(uploaded_file, 1):
                    pattern = detect_pattern_from_upload(file)
                    if pattern:
                        detected_patterns[file.name] = pattern
                        st.success(f"✅ {i}. {file.name}: {pattern.replace('_', ' ').title()}")
                    else:
                        st.warning(f"⚠️ {i}. {file.name}: Pattern not detected")
                
                # Show pattern summary
                if detected_patterns:
                    unique_patterns = list(set(detected_patterns.values()))
                    if len(unique_patterns) == 1:
                        st.info(f"📊 **All files use**: {unique_patterns[0].replace('_', ' ').title()}")
                        default_pattern = unique_patterns[0]
                    else:
                        st.info(f"📊 **Mixed patterns detected**: {', '.join([p.replace('_', ' ').title() for p in unique_patterns])}")
                        st.warning("⚠️ **Note**: Each file will be assessed with its detected pattern")
                        default_pattern = unique_patterns[0]  # Use first pattern as default
                else:
                    st.error("❌ No patterns detected in any files")
                    default_pattern = pattern_detector.get_available_patterns()[0]
                
                # Store individual patterns for processing
                st.session_state.detected_patterns = detected_patterns
                default_index = pattern_detector.get_available_patterns().index(default_pattern)
            else:
                # Single file - original logic
                detected_pattern = detect_pattern_from_upload(uploaded_file)
                if detected_pattern:
                    st.success(f"✅ Detected pattern: {detected_pattern.replace('_', ' ').title()}")
                    default_index = pattern_detector.get_available_patterns().index(detected_pattern)
                else:
                    st.warning("⚠️ Pattern not detected from filename")
                    default_index = 0
                st.session_state.detected_patterns = {uploaded_file.name: detected_pattern} if detected_pattern else {}
            
            # Pattern selection (with override capability)
            selected_pattern = st.selectbox(
                "Confirm/Select Default Suture Pattern:",
                options=pattern_detector.get_available_patterns(),
                index=default_index,
                format_func=lambda x: x.replace('_', ' ').title(),
                help="For multiple files with mixed patterns, this is the default. Each file will use its detected pattern."
            )
            st.session_state.selected_pattern = selected_pattern
            
            # Display selected rubric
            if selected_pattern:
                pattern_data = rubric_engine.get_pattern_rubric(selected_pattern)
                st.info(f"📊 **Assessment Rubric**: {pattern_data['display_name']}")
                
                with st.expander("View Rubric Details"):
                    for point in pattern_data["points"]:
                        st.markdown(f"**{point['pid']}. {point['title']}**")
                        st.markdown(f"*{point['what_you_assess']}*")
        
        # Analysis settings
        st.subheader("⚙️ Analysis Settings")
        fps = st.slider("Analysis FPS", 1.0, 5.0, 5.0, 0.5)
        batch_size = st.slider("Batch Size", 5, 15, 10, 1, help="Number of frames processed together in each batch")
        
        # Concurrency settings
        st.subheader("⚡ Concurrency Settings")
        max_concurrent_batches = st.slider(
            "Concurrent Batches", 
            1, 150, 
            100,  # Default high performance setting
            step=1,
            help="Higher values = faster processing (requires OpenAI Tier 4+). Use 100-150 for maximum speed."
        )
        

        
        # API Key
        st.subheader("🔑 OpenAI API Key")
        
        # Show saved API key status
        if st.session_state.saved_api_key:
            st.info("🔑 API key found from previous session")
            if st.button("🗑️ Clear saved API key"):
                st.session_state.saved_api_key = ""
                save_api_key("")  # Clear saved key
                st.rerun()
        
        # API key input field
        api_key = st.text_input(
            "Enter your OpenAI API Key",
            value=st.session_state.saved_api_key,
            type="password",
            help="Get your API key from https://platform.openai.com/api-keys"
        )
        
        # Save API key if it's new
        if api_key and api_key != st.session_state.saved_api_key:
            st.session_state.saved_api_key = api_key
            save_api_key(api_key)
            st.success("✅ API key saved for future assessments!")
        
        # STOP button
        st.subheader("🛑 App Control")
        if st.button("🛑 STOP Application", type="secondary", help="Gracefully stop the application"):
            st.stop()
    
    # Main content area
    if uploaded_file and st.session_state.selected_pattern and api_key:
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📹 Video Analysis")
            
            # Handle single or multiple files
            if isinstance(uploaded_file, list):
                # Multiple files
                st.info(f"📁 **Processing {len(uploaded_file)} videos**")
                
                # Show video information for each file
                with st.expander("📊 Video Information", expanded=True):
                    temp_video_paths = []
                    for i, file in enumerate(uploaded_file, 1):
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.read())
                        temp_video_paths.append(temp_path)
                        
                        video_info = extract_video_info(temp_path)
                        st.subheader(f"Video {i}: {file.name}")
                        if "error" not in video_info:
                            st.json(video_info)
                        else:
                            st.error(f"Error reading video: {video_info['error']}")
                
                # Start batch analysis
                col_start, col_stop = st.columns([2, 1])
                with col_start:
                    if st.button("🚀 Start Batch VOP Assessment", type="primary"):
                        # Process each video with its detected pattern
                        for i, (file, temp_path) in enumerate(zip(uploaded_file, temp_video_paths), 1):
                            file_pattern = st.session_state.detected_patterns.get(file.name, st.session_state.selected_pattern)
                            st.info(f"Processing {i}/{len(uploaded_file)}: {file.name} ({file_pattern})")
                            start_vop_analysis(temp_path, api_key, fps, batch_size, max_concurrent_batches, file_pattern)
                with col_stop:
                    if st.button("🛑 STOP Batch", type="secondary", help="Stop batch processing"):
                        st.warning("🛑 Batch processing stopped by user")
                        st.stop()
            else:
                # Single file
                with st.expander("📊 Video Information", expanded=True):
                    # Save video temporarily for analysis
                    temp_video_path = f"temp_{uploaded_file.name}"
                    with open(temp_video_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    video_info = extract_video_info(temp_video_path)
                    if "error" not in video_info:
                        st.json(video_info)
                    else:
                        st.error(f"Error reading video: {video_info['error']}")
                
                # Start analysis
                if st.button("🚀 Start VOP Assessment", type="primary"):
                    start_vop_analysis(temp_video_path, api_key, fps, batch_size, max_concurrent_batches)
        
        with col2:
            st.header("📋 Assessment Progress")
            
            # Emergency STOP button
            if st.button("🛑 STOP Analysis", type="secondary", help="Stop current analysis process"):
                st.warning("🛑 Analysis stopped by user")
                st.stop()
            
            if st.session_state.vop_analysis_complete:
                display_assessment_results(rubric_engine)
            else:
                st.info("Click 'Start VOP Assessment' to begin evaluation")
    
    else:
        st.info("👆 Please upload a video, confirm the suture pattern, and enter your API key to begin assessment")
        


def start_vop_analysis(video_path: str, api_key: str, fps: float, batch_size: int, max_concurrent_batches: int = 100, pattern_id: str = None):
    """Start the VOP analysis process using the proven video analysis architecture."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components using proven architecture
        video_processor = VideoProcessor()
        gpt4o_client = GPT4oClient(api_key=api_key)
        rubric_engine = RubricEngine()
        
        # Load video
        status_text.text("Loading video...")
        success = video_processor.load_video(video_path, fps)
        if not success:
            st.error("Failed to load video")
            return
        
        # Extract frames with custom FPS
        status_text.text("Extracting frames...")
        frames = video_processor.extract_frames(custom_fps=fps)
        if not frames:
            st.error("No frames extracted from video")
            return
        
        st.info(f"Video duration: {video_processor.duration} seconds")
        st.info(f"Target FPS: {fps}")
        st.success(f"✅ Extracted {len(frames)} frames at {fps} FPS")
        
        # Use provided pattern or fall back to session state
        current_pattern = pattern_id if pattern_id else st.session_state.selected_pattern
        
        # Create surgical assessment profile using proven profile structure
        assessment_profile = SurgicalAssessmentProfile(rubric_engine)
        surgical_profile = {
            "name": "Surgical VOP",
            "description": "Surgical Verification of Proficiency Assessment",
            "base_prompt": assessment_profile.create_assessment_prompt(current_pattern),
            "context_condensation_prompt": assessment_profile.create_context_condensation_prompt()
        }
        
        # Create batches
        batch_processor = FrameBatchProcessor(batch_size=batch_size)
        batches = batch_processor.create_batches(frames)
        total_batches = len(batches)
        
        st.info(f"Processing {len(frames)} frames in {total_batches} batches...")
        
        # Use concurrent processing if specified
        if max_concurrent_batches > 1:
            st.info(f"⚡ Using concurrent processing: {max_concurrent_batches} batches simultaneously")
            
            # Import the proven concurrent processing function
            from app import process_batches_concurrently
            
            start_time = time.time()
            performance_metrics = process_batches_concurrently(
                batches, gpt4o_client, surgical_profile, progress_bar, status_text, total_batches, max_concurrent_batches
            )
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Display performance metrics
            success_rate = (performance_metrics.get('successful_batches', 0) / total_batches) * 100
            avg_batch_time = sum(performance_metrics.get('batch_times', [0])) / max(len(performance_metrics.get('batch_times', [1])), 1)
            
            st.success(f"✅ Analysis complete in {processing_time:.1f} seconds!")
            st.info(f"📊 Success rate: {success_rate:.1f}% ({performance_metrics.get('successful_batches', 0)}/{total_batches} batches)")
            st.info(f"⚡ Speed: {avg_batch_time:.1f}s average per batch")
            
            # Store performance data for future optimization
            if 'vop_performance_history' not in st.session_state:
                st.session_state.vop_performance_history = []
            
            st.session_state.vop_performance_history.append({
                'concurrent_level': max_concurrent_batches,
                'success_rate': success_rate,
                'processing_time': processing_time,
                'total_batches': total_batches,
                'timestamp': datetime.now().isoformat()
            })
            
        else:
            # Use sequential processing
            from app import process_batches_sequentially
            process_batches_sequentially(batches, gpt4o_client, surgical_profile, progress_bar, status_text, total_batches)
        
        # Get complete analysis using proven narrative building
        full_transcript = gpt4o_client.get_full_transcript()
        event_timeline = gpt4o_client.get_event_timeline()
        
        # Create enhanced narrative using GPT-5
        status_text.text("Creating final surgical assessment with GPT-5...")
        enhanced_narrative = create_surgical_vop_narrative(
            full_transcript, 
            event_timeline, 
            api_key,
            current_pattern,
            rubric_engine
        )
        
        # Extract scores from enhanced narrative
        extracted_scores = extract_rubric_scores_from_narrative(enhanced_narrative)
        
        # Set extracted scores in session state
        if extracted_scores:
            st.session_state.rubric_scores = extracted_scores
            st.success(f"✅ Extracted {len(extracted_scores)} rubric scores from GPT-5 assessment")
        else:
            st.warning("⚠️ Could not extract scores from GPT-5 - manual scoring required")
        
        # Store results in the proven format
        st.session_state.assessment_results = {
            "full_transcript": full_transcript,  # Keep for debugging if needed
            "enhanced_narrative": enhanced_narrative,  # This is what we'll display
            "event_timeline": event_timeline,
            "extracted_scores": extracted_scores,  # Store the extracted scores
            "video_info": {
                "filename": os.path.basename(video_path),
                "pattern": current_pattern,
                "fps": fps,
                "total_frames": len(frames),
                "duration": video_processor.duration
            },
            "performance_metrics": performance_metrics if max_concurrent_batches > 1 else None,
            "timestamp": datetime.now().isoformat()
        }
        
        st.session_state.vop_analysis_complete = True
        status_text.text("✅ Surgical VOP Assessment complete!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        status_text.text("❌ Analysis failed")

def display_assessment_results(rubric_engine: RubricEngine):
    """Display the assessment results using the proven narrative structure."""
    if not st.session_state.assessment_results:
        return
    
    results = st.session_state.assessment_results
    
    st.subheader("📊 Assessment Results")
    
    # Display summary
    st.info(f"**Pattern**: {results['video_info']['pattern'].replace('_', ' ').title()}")
    st.info(f"**Video**: {results['video_info']['filename']}")
    st.info(f"**Duration**: {results['video_info']['duration']:.1f} seconds")
    
    # Performance metrics
    if results.get('performance_metrics'):
        metrics = results['performance_metrics']
        success_rate = (metrics.get('successful_batches', 0) / metrics.get('total_batches', 1)) * 100
        st.success(f"📊 Processing: {success_rate:.1f}% success rate")
    
    # Display GPT-5 enhanced narrative (primary assessment)
    st.subheader("🏥 Final Surgical Assessment")
    if results.get("enhanced_narrative"):
        # Format the enhanced narrative for better readability
        narrative = results["enhanced_narrative"]
        
        # Clean up the text and add proper spacing
        formatted_narrative = narrative.replace('\n\n', '\n\n---\n\n')
        
        with st.expander("📋 Comprehensive VOP Assessment Report", expanded=True):
            st.markdown(
                f"""<div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; 
                border-left: 4px solid #007bff; line-height: 1.6; font-family: 'Arial', sans-serif;">
                {formatted_narrative}
                </div>""", 
                unsafe_allow_html=True
            )
    else:
        st.warning("⚠️ Enhanced narrative not available - showing raw analysis")
        with st.expander("Raw Frame Analysis", expanded=False):
            st.markdown(results.get("full_transcript", "No analysis available"))
    
    # Technical details (collapsed by default)
    with st.expander("🔍 Technical Details & Events", expanded=False):
        if results.get("event_timeline"):
            st.subheader("📅 Technical Events Timeline")
            for event in results["event_timeline"]:
                st.json(event)
        
        st.subheader("📊 Raw Frame Analysis")
        st.markdown(results.get("full_transcript", "No raw analysis available"))
        st.caption("*This is the detailed frame-by-frame analysis that was synthesized into the final assessment above.*")
    
    # Scoring interface
    st.subheader("📝 Assessment Scores")
    pattern_data = rubric_engine.get_pattern_rubric(st.session_state.selected_pattern)
    
    if pattern_data:
        # Check if we have extracted scores from GPT-5
        extracted_scores = results.get("extracted_scores", {})
        
        if extracted_scores:
            st.info("📊 **AI-Generated Scores** - Based on GPT-5 video analysis (adjust if needed):")
        else:
            st.markdown("*Score each rubric point based on the analysis above:*")
        
        for point in pattern_data["points"]:
            # Use extracted score as default, or 3 if not available
            default_score = extracted_scores.get(point['pid'], 3)
            
            score = st.slider(
                f"{point['pid']}. {point['title']}", 
                1, 5, default_score, 
                key=f"score_{point['pid']}",
                help=f"What to assess: {point['what_you_assess']}"
            )
            st.session_state.rubric_scores[point['pid']] = score
    
    # Calculate overall result
    if st.session_state.rubric_scores:
        overall_result = rubric_engine.calculate_overall_score(st.session_state.rubric_scores)
        
        st.subheader("🎯 Overall VOP Assessment")
        
        if overall_result["pass"]:
            st.markdown(
                f"""<div style="background-color: #d4edda; color: #155724; padding: 15px; 
                border-radius: 8px; border: 1px solid #c3e6cb; text-align: center; font-size: 18px; font-weight: bold;">
                ✅ COMPETENCY ACHIEVED - Average Score: {overall_result['average_score']:.1f}/5.0
                </div>""", 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""<div style="background-color: #f8d7da; color: #721c24; padding: 15px; 
                border-radius: 8px; border: 1px solid #f5c6cb; text-align: center; font-size: 18px; font-weight: bold;">
                ❌ COMPETENCY NOT ACHIEVED - {overall_result['reason']}
                </div>""", 
                unsafe_allow_html=True
            )
        
        # PDF report generation button
        if st.button("📄 Generate PDF Report"):
            try:
                from surgical_report_generator import SurgicalVOPReportGenerator
                
                report_generator = SurgicalVOPReportGenerator()
                report_filename = f"VOP_Assessment_{results['video_info']['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                report_path = report_generator.generate_vop_report(
                    results,
                    st.session_state.rubric_scores,
                    overall_result,
                    report_filename
                )
                
                st.success(f"✅ PDF report generated: {report_path}")
                
                # Offer download
                with open(report_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=report_filename,
                        mime="application/pdf"
                    )
                    
            except Exception as e:
                st.error(f"Error generating PDF report: {e}")

if __name__ == "__main__":
    main()
