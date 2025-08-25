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

# Set upload limit to 1GB
import streamlit as st
import streamlit.web.cli as stcli
st._config.set_option('server.maxUploadSize', 1024)

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
        """Create GPT-4o prompt for surgical assessment using proven narrative structure."""
        pattern_data = self.rubric_engine.get_pattern_rubric(pattern_id)
        assessment_points = self.rubric_engine.get_assessment_criteria(pattern_id)
        
        # Build detailed assessment criteria (without critical/standard distinction)
        criteria_text = "\n".join([
            f"**{p['pid']}. {p['title']}**\n"
            f"- What to assess: {p['what_you_assess']}\n"
            f"- Ideal result: {p['ideal_result']}\n"
            for p in assessment_points
        ])
        
        prompt = f"""You are a senior attending surgeon conducting a high-stakes Verification of Proficiency (VOP) assessment for {pattern_data['display_name']} suturing technique. This is a formal competency evaluation with real consequences.

ASSESSMENT PHILOSOPHY:
- **Standard of Care**: You must evaluate against published surgical standards, not subjective impressions
- **Evidence-Based**: Every observation must be grounded in visual evidence from the frames
- **Uncompromising**: Minor deviations from technique ARE significant - this is professional competency
- **Forensic Detail**: You are building a legal-grade assessment that could be challenged
- **No Benefit of Doubt**: If you cannot clearly see proper technique, it is inadequate

MANDATORY ASSESSMENT REQUIREMENTS:
1. **ALWAYS find evidence for each rubric point** - across a full suturing video, all criteria can be assessed
2. **Use precise measurements** when visible: needle angles, bite distances, knot spacing
3. **Timestamp every significant observation** - be specific to the second when possible
4. **Identify specific errors** - do not use vague terms like "adequate" or "reasonable"
5. **Reference surgical principles** - explain WHY each observation matters clinically
6. **Track progression** - note improvement/deterioration across the procedure
7. **No hedging language** - avoid "appears to," "seems," "possibly" - state what you observe

ASSESSMENT CRITERIA FOR {pattern_data['display_name'].upper()}:
{criteria_text}

DETAILED OBSERVATION PROTOCOL:
- **Needle handling**: Exact grip position, wrist angle, finger placement, tremor/steadiness
- **Tissue interaction**: Exact entry/exit points, tissue deformation, bleeding, trauma signs
- **Spatial relationships**: Bite spacing measurements, symmetry, alignment to wound edges
- **Temporal analysis**: Speed consistency, hesitation patterns, efficiency indicators
- **Instrument technique**: Forceps pressure, needle driver positioning, assistant coordination
- **Environmental factors**: Lighting adequacy, field exposure, contamination risks

CRITICAL EVIDENCE REQUIREMENTS:
- **Quantify everything measurable**: "3mm bite depth" not "appropriate depth"
- **Count repetitions**: "Fourth attempt at knot placement" not "multiple attempts"
- **Measure timing**: "15-second delay" not "brief pause" 
- **Document deviations**: "45-degree needle angle versus standard 90-degree entry"
- **Assess cumulative effect**: How errors compound across the procedure

PROHIBITED ASSESSMENT LANGUAGE:
❌ "Insufficient evidence to assess"
❌ "Cannot determine from available footage"  
❌ "Appears adequate"
❌ "Generally acceptable"
❌ "Would benefit from"
✅ "Demonstrates substandard bite placement at 00:02:15"
✅ "Needle angle consistently 30 degrees off perpendicular"
✅ "Tissue approximation gap of 2-3mm throughout closure"

Output format:
- **Continuous narrative** (clinical, critical, timestamped, forensically detailed)
- **Structured JSON log** with one entry per detected technique event:
  ```json
  [
    {{
      "timestamp": "00:01:12",
      "event": "Needle entry at 45-degree angle instead of required 90-degree perpendicular approach",
      "confidence": 0.95,
      "technique_assessment": "Substandard - creates excessive tissue trauma",
      "rubric_point": 1,
      "clinical_significance": "Increased risk of wound dehiscence due to uneven tissue stress distribution",
      "measurement": "45 degrees vs 90 degree standard"
    }}
  ]
  ```"""

        return prompt
    
    def create_context_condensation_prompt(self) -> str:
        """Create context condensation prompt for surgical assessment continuity."""
        return """You are maintaining a running summary of a surgical video assessment as a senior attending surgeon. 
Your task is to compress the current full assessment into a concise "state summary" that captures: 
- Key surgical events and technique observations (with approximate timestamps) 
- Current surgical phase, instruments in use, and suture type
- Technical errors and deviations from surgical principles noted so far
- Assessment continuity for the next frames

Guidelines: 
- Use no more than 150 words. 
- Preserve chronological flow. 
- Keep timestamps coarse (to the nearest ~10–15 seconds). 
- Focus on surgical assessment continuity and technique progression.
- Use critical, clinical language appropriate for surgical review.

Output format: [Condensed Surgical Assessment State]"""

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
        
        enhancement_prompt = f"""You are a master storyteller and video analyst. Your task is to transform raw, batch-by-batch video analysis into a coherent, continuous narrative that reads like a polished surgical assessment.

{personality_instructions}

PROFILE: Surgical VOP Assessment - {pattern_data['display_name']} Technique Evaluation

ASSESSMENT RUBRIC CRITERIA:
{rubric_criteria}

RAW TRANSCRIPT:
{raw_transcript}

EVENTS TIMELINE:
{json.dumps(events, indent=2)}

AUDIO TRANSCRIPT:
No audio transcription available.

INSTRUCTIONS:
1. **Chronological Flow**: Maintain strict chronological order from start to finish
2. **Character Continuity**: Track the surgeon and instruments across the procedure
3. **Setting Continuity**: Maintain awareness of the surgical field and spatial relationships
4. **Cause and Effect**: Connect surgical actions logically, explaining why techniques succeed or fail
5. **Narrative Coherence**: Fill gaps, resolve contradictions, and create smooth transitions
6. **Surgical Terminology**: Use precise surgical language throughout
7. **Temporal Synchronization**: Align surgical events with timestamps when possible
8. **Absolute Specificity**: Describe EVERYTHING with concrete, specific details
9. **No Generic Language**: Eliminate phrases like "a figure," "someone," "a person"
10. **Physical Description**: Specify exact hand positions, instrument angles, tissue appearance
11. **Direct Observation**: Document exactly what is visible - no interpretation beyond what can be seen
12. **Technical Assessment**: Evaluate each action against surgical standards
13. **Rubric Integration**: Address each rubric point with specific evidence from the video AND provide numerical scores (1-5)
14. **Progressive Analysis**: Track how technique changes throughout the procedure
15. **Clinical Significance**: Connect observations to patient outcomes and surgical principles
16. **Mandatory Scoring**: You MUST provide a numerical score (1-5) for each rubric criterion based on evidence
17. **Profile Consistency**: Maintain surgical assessment voice throughout

OUTPUT FORMAT:
- Write in third-person narrative style
- Use present tense for immediacy
- NO timestamps or time markers - let the story flow naturally
- Break into logical paragraphs based on surgical phases, not time chunks
- Create smooth transitions between surgical actions
- Focus on visual description and technical execution
- Avoid color commentary, purple prose, or excessive adjectives
- End with a conclusion that ties everything together for competency determination
- **Maintain Surgical Voice**: Write in the voice and style of a surgical assessment

CRITICAL: Every person, object, or action must be described with absolute specificity. No generalizations, no generic terms, no vague descriptions. Document exact needle angles, bite depths, knot configurations, and tissue handling with clinical precision.

NO POETIC LANGUAGE: Avoid phrases like "as if measuring" or "like threading." Stick to rich, full detailing of what we can actually see. Focus on concrete surgical details - the procedure itself playing out in prose, richly described with surgical terminology.

Create a tight, continuous surgical narrative that flows naturally without time constraints, focusing on technique evaluation. Address each rubric criterion with specific evidence from the video analysis. Every detail should be rendered in concrete, specific surgical terms.

**MANDATORY SCORING REQUIREMENT**: You MUST conclude your assessment with explicit numerical scores for each rubric point in this format:

RUBRIC SCORES:
Point 1 - [Title]: Score X/5 - [Brief justification with timestamp evidence]
Point 2 - [Title]: Score X/5 - [Brief justification with timestamp evidence]
[Continue for all rubric points]

**CRITICAL**: Maintain the clinical, analytical surgical assessment voice throughout your entire narrative. Your enhanced narrative should read as if it was written by the same surgical AI that conducted the initial frame analysis."""

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
                max_completion_tokens=4000
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
            type=['mp4', 'avi', 'mov', 'mkv', 'm4v'],
            help="Upload a video of suturing technique for assessment (up to 1GB)"
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
        
        # Concurrency settings
        st.subheader("⚡ Concurrency Settings")
        max_concurrent_batches = st.slider(
            "Concurrent Batches", 
            1, 20, 
            10,  # Default to proven safe level
            step=1,
            help="Higher values = faster processing (requires OpenAI Tier 4+)"
        )
        
        # Concurrency guidance
        if max_concurrent_batches <= 10:
            st.success(f"✅ **SAFE**: {max_concurrent_batches} concurrent batches")
        elif max_concurrent_batches <= 12:
            st.info(f"🧪 **TESTING**: {max_concurrent_batches} concurrent batches - monitor closely")
        else:
            st.info(f"🚀 **EXPERIMENTAL**: {max_concurrent_batches} concurrent batches")
        
        # Performance history
        if 'vop_performance_history' in st.session_state and st.session_state.vop_performance_history:
            with st.expander("📊 Performance History"):
                for run in st.session_state.vop_performance_history[-3:]:  # Show last 3 runs
                    st.text(f"Level {run['concurrent_level']}: {run['success_rate']:.1f}% success, {run['processing_time']:.1f}s")
        
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
                start_vop_analysis(temp_video_path, api_key, fps, batch_size, max_concurrent_batches)
        
        with col2:
            st.header("📋 Assessment Progress")
            
            if st.session_state.vop_analysis_complete:
                display_assessment_results(rubric_engine)
            else:
                st.info("Click 'Start VOP Assessment' to begin evaluation")
    
    else:
        st.info("👆 Please upload a video, confirm the suture pattern, and enter your API key to begin assessment")

def start_vop_analysis(video_path: str, api_key: str, fps: float, batch_size: int, max_concurrent_batches: int = 10):
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
        
        # Create surgical assessment profile using proven profile structure
        assessment_profile = SurgicalAssessmentProfile(rubric_engine)
        surgical_profile = {
            "name": "Surgical VOP",
            "description": "Surgical Verification of Proficiency Assessment",
            "base_prompt": assessment_profile.create_assessment_prompt(st.session_state.selected_pattern),
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
            st.session_state.selected_pattern,
            rubric_engine
        )
        
        # Store results in the proven format
        st.session_state.assessment_results = {
            "full_transcript": full_transcript,  # Keep for debugging if needed
            "enhanced_narrative": enhanced_narrative,  # This is what we'll display
            "event_timeline": event_timeline,
            "video_info": {
                "filename": os.path.basename(video_path),
                "pattern": st.session_state.selected_pattern,
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
    st.subheader("📝 Manual Scoring")
    pattern_data = rubric_engine.get_pattern_rubric(st.session_state.selected_pattern)
    
    if pattern_data:
        st.markdown("*Score each rubric point based on the analysis above:*")
        
        for point in pattern_data["points"]:
            score = st.slider(
                f"{point['pid']}. {point['title']}", 
                1, 5, 3, 
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
