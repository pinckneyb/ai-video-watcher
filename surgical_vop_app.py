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
        """Calculate overall assessment result."""
        if not point_scores:
            return {"pass": False, "reason": "No scores provided"}
        
        # Get critical points
        pattern_data = self.get_pattern_rubric(st.session_state.get('selected_pattern'))
        critical_points = [p for p in pattern_data["points"] if p.get("critical", False)]
        critical_pids = [p["pid"] for p in critical_points]
        
        # Check pass/fail criteria
        all_scores_above_3 = all(score >= 3 for score in point_scores.values())
        no_critical_below_3 = all(point_scores.get(pid, 3) >= 3 for pid in critical_pids)
        
        passes = all_scores_above_3 and no_critical_below_3
        
        return {
            "pass": passes,
            "total_points": len(point_scores),
            "average_score": sum(point_scores.values()) / len(point_scores),
            "critical_points_passed": no_critical_below_3,
            "reason": "PASS" if passes else "FAIL - Critical points below threshold or overall performance inadequate"
        }

class SurgicalAssessmentProfile:
    """Specialized profile for surgical technique assessment."""
    
    def __init__(self, rubric_engine: RubricEngine):
        self.rubric_engine = rubric_engine
    
    def create_assessment_prompt(self, pattern_id: str) -> str:
        """Create GPT-4o prompt for surgical assessment."""
        pattern_data = self.rubric_engine.get_pattern_rubric(pattern_id)
        assessment_points = self.rubric_engine.get_assessment_criteria(pattern_id)
        
        # Build detailed assessment criteria
        criteria_text = "\n".join([
            f"**Point {p['pid']}: {p['title']}** ({'CRITICAL' if p.get('critical') else 'Standard'})\n"
            f"- What to assess: {p['what_you_assess']}\n"
            f"- Ideal result: {p['ideal_result']}\n"
            for p in assessment_points
        ])
        
        prompt = f"""You are a senior attending surgeon conducting a Verification of Proficiency (VOP) assessment for {pattern_data['display_name']} suturing technique.

Your task is to evaluate this surgical video according to the specific rubric criteria below. You must:

1. Watch the video frames carefully and identify specific moments where each rubric point can be assessed
2. For each rubric point, provide a score from 1-5 and specific timestamp-referenced observations
3. Focus ONLY on what you can actually observe in the video frames
4. Be precise, clinical, and matter-of-fact in your assessment
5. Provide specific timestamps for all observations

ASSESSMENT CRITERIA FOR {pattern_data['display_name'].upper()}:
{criteria_text}

SCORING SCALE:
1 = Unacceptable/Dangerous
2 = Poor technique with significant deficiencies  
3 = Adequate/Baseline acceptable performance
4 = Good technique with minor issues
5 = Excellent/Exemplary technique

CRITICAL REQUIREMENTS:
- Always reference specific timestamps (e.g., "At 00:01:23...")
- Describe exactly what you observe in clinical terms
- Score each rubric point individually
- No generic assessments - everything must be based on direct visual observation

Output format:
- **Assessment for each rubric point** with timestamp references and clinical observations
- **Score justification** for each point (1-5 scale)
- **Overall technical assessment** with specific recommendations

You are given frames in chronological order with timestamps. Analyze systematically."""

        return prompt

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
    """Detect pattern from uploaded file name."""
    if not uploaded_file:
        return None
    
    detector = SuturePatternDetector()
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
            "Upload Surgical Video",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video of suturing technique for assessment"
        )
        
        if uploaded_file:
            # Attempt automatic detection
            detected_pattern = detect_pattern_from_upload(uploaded_file)
            
            if detected_pattern:
                st.success(f"✅ Detected pattern: {detected_pattern.replace('_', ' ').title()}")
                default_index = pattern_detector.get_available_patterns().index(detected_pattern)
            else:
                st.warning("⚠️ Pattern not detected from filename")
                default_index = 0
            
            # Pattern selection (with override capability)
            selected_pattern = st.selectbox(
                "Confirm/Select Suture Pattern:",
                options=pattern_detector.get_available_patterns(),
                index=default_index,
                format_func=lambda x: x.replace('_', ' ').title()
            )
            st.session_state.selected_pattern = selected_pattern
            
            # Display selected rubric
            if selected_pattern:
                pattern_data = rubric_engine.get_pattern_rubric(selected_pattern)
                st.info(f"📊 **Assessment Rubric**: {pattern_data['display_name']}")
                
                with st.expander("View Rubric Details"):
                    for point in pattern_data["points"]:
                        critical_badge = "🔴 CRITICAL" if point.get("critical") else "⚪ Standard"
                        st.markdown(f"**{point['pid']}. {point['title']}** {critical_badge}")
                        st.markdown(f"*{point['what_you_assess']}*")
        
        # Analysis settings
        st.subheader("⚙️ Analysis Settings")
        fps = st.slider("Analysis FPS", 1.0, 5.0, 2.0, 0.5)
        batch_size = st.slider("Batch Size", 3, 12, 6)
        
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
    
    # Main content area
    if uploaded_file and st.session_state.selected_pattern and api_key:
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📹 Video Analysis")
            
            # Video info
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
                start_vop_analysis(temp_video_path, api_key, fps, batch_size)
        
        with col2:
            st.header("📋 Assessment Progress")
            
            if st.session_state.vop_analysis_complete:
                display_assessment_results(rubric_engine)
            else:
                st.info("Click 'Start VOP Assessment' to begin evaluation")
    
    else:
        st.info("👆 Please upload a video, confirm the suture pattern, and enter your API key to begin assessment")

def start_vop_analysis(video_path: str, api_key: str, fps: float, batch_size: int):
    """Start the VOP analysis process."""
    try:
        # Initialize components
        video_processor = VideoProcessor()
        gpt4o_client = GPT4oClient(api_key=api_key)
        rubric_engine = RubricEngine()
        
        # Load video
        with st.spinner("Loading video..."):
            success = video_processor.load_video(video_path, fps)
            if not success:
                st.error("Failed to load video")
                return
        
        # Extract frames
        with st.spinner("Extracting frames..."):
            frames = video_processor.extract_frames()
            if not frames:
                st.error("No frames extracted from video")
                return
        
        st.success(f"✅ Extracted {len(frames)} frames at {fps} FPS")
        
        # Create assessment profile
        assessment_profile = SurgicalAssessmentProfile(rubric_engine)
        prompt = assessment_profile.create_assessment_prompt(st.session_state.selected_pattern)
        
        # Create batches
        batch_processor = FrameBatchProcessor(batch_size)
        batches = batch_processor.create_batches(frames)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process batches
        full_analysis = []
        context_state = ""
        
        for i, batch in enumerate(batches):
            status_text.text(f"Analyzing batch {i+1}/{len(batches)}...")
            
            # Analyze batch with surgical assessment prompt
            narrative, events = gpt4o_client.analyze_frames(
                batch, 
                {"base_prompt": prompt}, 
                context_state
            )
            
            full_analysis.append({
                "batch_id": i,
                "narrative": narrative,
                "events": events,
                "timestamp_range": f"{batch[0]['timestamp']} - {batch[-1]['timestamp']}"
            })
            
            # Update context (simplified for VOP)
            context_state = f"Previous assessment context: {narrative[-500:]}..."
            
            progress_bar.progress((i + 1) / len(batches))
        
        # Store results
        st.session_state.assessment_results = {
            "analysis": full_analysis,
            "video_info": {
                "filename": os.path.basename(video_path),
                "pattern": st.session_state.selected_pattern,
                "fps": fps,
                "total_frames": len(frames)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        st.session_state.vop_analysis_complete = True
        status_text.text("✅ Assessment complete!")
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")

def display_assessment_results(rubric_engine: RubricEngine):
    """Display the assessment results."""
    if not st.session_state.assessment_results:
        return
    
    results = st.session_state.assessment_results
    
    st.subheader("📊 Assessment Results")
    
    # Display summary
    st.info(f"**Pattern**: {results['video_info']['pattern'].replace('_', ' ').title()}")
    st.info(f"**Video**: {results['video_info']['filename']}")
    
    # Display analysis
    for batch_result in results["analysis"]:
        with st.expander(f"Analysis - {batch_result['timestamp_range']}", expanded=True):
            st.markdown(batch_result["narrative"])
    
    # Scoring interface (placeholder for now)
    st.subheader("📝 Manual Scoring")
    pattern_data = rubric_engine.get_pattern_rubric(st.session_state.selected_pattern)
    
    if pattern_data:
        for point in pattern_data["points"]:
            critical_indicator = "🔴" if point.get("critical") else "⚪"
            score = st.slider(
                f"{critical_indicator} {point['title']}", 
                1, 5, 3, 
                key=f"score_{point['pid']}"
            )
            st.session_state.rubric_scores[point['pid']] = score
    
    # Calculate overall result
    if st.session_state.rubric_scores:
        overall_result = rubric_engine.calculate_overall_score(st.session_state.rubric_scores)
        
        if overall_result["pass"]:
            st.success(f"✅ **PASS** - Average Score: {overall_result['average_score']:.1f}/5")
        else:
            st.error(f"❌ **FAIL** - {overall_result['reason']}")

if __name__ == "__main__":
    main()
