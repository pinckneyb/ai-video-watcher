"""
Surgical VOP (Verification of Proficiency) Assessment Application
Specialized app for evaluating suturing technique videos using structured rubrics.
"""

import streamlit as st
import os
import json
import re
import base64
import gc
from typing import List, Dict, Any, Optional, Tuple
import time
from datetime import datetime
from pathlib import Path

# Import our modules from the main app
from video_processor import VideoProcessor, FrameBatchProcessor
from gpt4o_client import GPT4oClient
from gpt5_vision_client import GPT5VisionClient
from utils import (
    parse_timestamp, format_timestamp, validate_time_range,
    save_transcript, extract_video_info, sanitize_filename
)
from batch_manager import BatchManager

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
    
    def __init__(self, rubric_path: str = "surgical-vop-assessment/unified_rubric.JSON"):
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

def _generate_rubric_scores_block(assessment_points) -> str:
    """Generate dynamic RUBRIC_SCORES format based on actual rubric points."""
    score_lines = ["RUBRIC_SCORES_START"]
    for point in assessment_points:
        score_lines.append(f"{point['pid']}: X")
    score_lines.append("RUBRIC_SCORES_END")
    return "\n".join(score_lines)

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
- Score 3 should be your DEFAULT for competent, standard technique
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
{_generate_rubric_scores_block(assessment_points)}"""

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
                        
                        # Use raw score directly - no multiplier adjustment
                        # Validate score is in range
                        if 1 <= score <= 5:
                            scores[point_id] = score
                        else:
                            st.warning(f"Score {score} for point {point_id} is out of range (1-5), using default of 3")
                            scores[point_id] = 3
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
                valid_files = []
                for i, file in enumerate(uploaded_file, 1):
                    try:
                        file_size_mb = len(file.read()) / (1024 * 1024)
                        file.seek(0)  # Reset file pointer
                        if file_size_mb > 2000:  # 2GB limit
                            st.error(f"❌ **File too large**: {file.name} exceeds 2GB limit")
                        else:
                            valid_files.append(file)
                            st.success(f"✅ {file.name}: {file_size_mb:.1f} MB")
                    except Exception as e:
                        st.error(f"❌ **Error reading file**: {file.name} - {str(e)}")
                        continue
                
                # Update uploaded_file to only include valid files
                uploaded_file = valid_files if valid_files else None
            else:
                # Single file - just show count
                try:
                    file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
                    uploaded_file.seek(0)  # Reset file pointer
                    
                    if file_size_mb > 2000:  # 2GB limit
                        st.error("❌ **File too large**: Maximum size is 2GB. Please compress your video.")
                        uploaded_file = None
                    else:
                        st.success(f"✅ {uploaded_file.name}: {file_size_mb:.1f} MB")
                except Exception as e:
                    st.error(f"❌ **Error reading file**: {uploaded_file.name} - {str(e)}")
                    uploaded_file = None
        
        if uploaded_file:
            # Pattern detection for single or multiple files
            if isinstance(uploaded_file, list):
                # Multiple files - detect pattern for each
                st.subheader("🔍 Pattern Detection Results")
                detected_patterns = {}
                for i, file in enumerate(uploaded_file, 1):
                    try:
                        pattern = detect_pattern_from_upload(file)
                        if pattern:
                            detected_patterns[file.name] = pattern
                            st.success(f"✅ {i}. {file.name}: {pattern.replace('_', ' ').title()}")
                        else:
                            st.warning(f"⚠️ {i}. {file.name}: Pattern not detected")
                    except Exception as e:
                        st.error(f"❌ **Error detecting pattern for** {file.name}: {str(e)}")
                        # Continue with other files
                
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
        fps = st.slider("Analysis FPS", 1.0, 5.0, 1.0, 0.5)
        batch_size = st.slider("Batch Size", 5, 15, 15, 1, help="Number of frames processed together in each batch")
        
        # GPT-5 Model Settings with Pass-specific Configuration
        st.subheader("🤖 GPT-5 Model Settings")
        
        # Create tabs for each analysis pass
        pass1_tab, pass2_tab, pass3_tab = st.tabs(["Pass 1", "Pass 2", "Pass 3"])
        
        # Pass 1 Settings (Frame Analysis)
        with pass1_tab:
            st.markdown("**🔍 Frame Analysis**")
            st.caption("AI analyzes each video frame to identify surgical techniques, tool usage, and suturing patterns")
            
            gpt5_model_pass1 = st.selectbox(
                "Model",
                options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
                index=0,  # Default to gpt-5
                help="Choose the GPT-5 model for frame analysis. gpt-5 offers highest quality, gpt-5-mini is balanced, gpt-5-nano is fastest.",
                key="model_pass1"
            )
            st.session_state.gpt5_model_pass1 = gpt5_model_pass1
            
            reasoning_level_pass1 = st.selectbox(
                "Reasoning Level",
                options=["minimal", "low", "medium", "high"],
                index=2,  # Default to medium
                help="Controls computational effort for frame analysis: minimal (speed-optimized), low (balanced), medium (default), high (maximum quality).",
                key="reasoning_pass1"
            )
            st.session_state.reasoning_level_pass1 = reasoning_level_pass1
            
            verbosity_level_pass1 = st.selectbox(
                "Verbosity Level",
                options=["low", "medium", "high"],
                index=1,  # Default to medium
                help="Controls response detail for frame observations: low (terse), medium (balanced), high (verbose explanations).",
                key="verbosity_pass1"
            )
            st.session_state.verbosity_level_pass1 = verbosity_level_pass1
        
        # Pass 2 Settings (Narrative Synthesis)
        with pass2_tab:
            st.markdown("**📝 Narrative Synthesis**")
            st.caption("Combines frame observations into coherent video narrative with temporal flow analysis")
            
            gpt5_model_pass2 = st.selectbox(
                "Model",
                options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
                index=0,  # Default to gpt-5
                help="Choose the GPT-5 model for narrative synthesis. gpt-5 offers highest quality, gpt-5-mini is balanced, gpt-5-nano is fastest.",
                key="model_pass2"
            )
            st.session_state.gpt5_model_pass2 = gpt5_model_pass2
            
            reasoning_level_pass2 = st.selectbox(
                "Reasoning Level",
                options=["minimal", "low", "medium", "high"],
                index=2,  # Default to medium
                help="Controls computational effort for narrative synthesis: minimal (speed-optimized), low (balanced), medium (default), high (maximum quality).",
                key="reasoning_pass2"
            )
            st.session_state.reasoning_level_pass2 = reasoning_level_pass2
            
            verbosity_level_pass2 = st.selectbox(
                "Verbosity Level",
                options=["low", "medium", "high"],
                index=2,  # Default to high for detailed narratives
                help="Controls response detail for narrative synthesis: low (terse), medium (balanced), high (verbose explanations).",
                key="verbosity_pass2"
            )
            st.session_state.verbosity_level_pass2 = verbosity_level_pass2
        
        # Pass 3 Settings (Rubric Assessment)
        with pass3_tab:
            st.markdown("**📊 Rubric Assessment**")
            st.caption("Evaluates video narrative against standardized medical education rubrics with numerical scoring")
            
            gpt5_model_pass3 = st.selectbox(
                "Model",
                options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
                index=0,  # Default to gpt-5
                help="Choose the GPT-5 model for rubric assessment. gpt-5 offers highest quality, gpt-5-mini is balanced, gpt-5-nano is fastest.",
                key="model_pass3"
            )
            st.session_state.gpt5_model_pass3 = gpt5_model_pass3
            
            reasoning_level_pass3 = st.selectbox(
                "Reasoning Level",
                options=["minimal", "low", "medium", "high"],
                index=3,  # Default to high for precise rubric evaluation
                help="Controls computational effort for rubric assessment: minimal (speed-optimized), low (balanced), medium (default), high (maximum quality).",
                key="reasoning_pass3"
            )
            st.session_state.reasoning_level_pass3 = reasoning_level_pass3
            
            verbosity_level_pass3 = st.selectbox(
                "Verbosity Level",
                options=["low", "medium", "high"],
                index=1,  # Default to medium for structured scoring
                help="Controls response detail for rubric scoring: low (terse), medium (balanced), high (verbose explanations).",
                key="verbosity_pass3"
            )
            st.session_state.verbosity_level_pass3 = verbosity_level_pass3
        
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
    
    # Initialize batch manager
    if 'batch_manager' not in st.session_state:
        st.session_state.batch_manager = BatchManager()
    
    # Batch History section  
    st.subheader("📚 Batch History")
    batches = st.session_state.batch_manager.list_batches()
    
    if batches:
        # Show latest batch recovery if available
        latest_batch_id = st.session_state.batch_manager.get_latest_batch_id()
        if latest_batch_id and st.button("🔄 Resume Latest Batch"):
            st.session_state.selected_batch = latest_batch_id
            st.rerun()
        
        # Batch selector
        batch_options = []
        for batch in batches[:10]:  # Show last 10 batches
            batch_options.append(f"{batch['batch_id']} ({batch['video_count']} videos, {batch['status']})")
        
        selected_batch_display = st.selectbox("Select Previous Batch:", ["None"] + batch_options)
        
        if selected_batch_display != "None":
            # Extract batch_id from display string
            selected_batch_id = selected_batch_display.split(' (')[0]
            batch_manifest = st.session_state.batch_manager.get_batch_manifest(selected_batch_id)
            
            if batch_manifest:
                st.markdown(f"### 📊 Batch Details: {selected_batch_id}")
                
                # Batch summary
                total_items = len(batch_manifest["items"])
                completed_items = sum(1 for item in batch_manifest["items"] if item["status"] == "completed")
                failed_items = sum(1 for item in batch_manifest["items"] if item["status"] == "failed")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Videos", total_items)
                with col2:
                    st.metric("Completed", completed_items)
                with col3:
                    st.metric("Failed", failed_items)
                
                # Download entire batch as ZIP
                if completed_items > 0:
                    zip_path = st.session_state.batch_manager.create_batch_zip(selected_batch_id)
                    if zip_path and os.path.exists(zip_path):
                        with open(zip_path, "rb") as f:
                            st.download_button(
                                label=f"📦 Download Complete Batch ZIP ({completed_items} files)",
                                data=f.read(),
                                file_name=f"{selected_batch_id}.zip",
                                mime="application/zip",
                                type="primary"
                            )
                
                # Individual file downloads
                if batch_manifest["items"]:
                    st.markdown("#### 📁 Individual Files")
                    
                    for item in batch_manifest["items"]:
                        if item["status"] == "completed":
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                score = item.get("score", "N/A")
                                st.write(f"**{item['input_name']}** - {item['detected_pattern']} - Score: {score}")
                            
                            with col2:
                                if item["html_path"] and os.path.exists(item["html_path"]):
                                    with open(item["html_path"], "rb") as f:
                                        st.download_button(
                                            label="🌐 HTML",
                                            data=f.read(),
                                            file_name=os.path.basename(item["html_path"]),
                                            mime="text/html",
                                            key=f"html_{selected_batch_id}_{item['input_name']}"
                                        )
                            
                            with col3:
                                if item["narrative_path"] and os.path.exists(item["narrative_path"]):
                                    with open(item["narrative_path"], "rb") as f:
                                        st.download_button(
                                            label="📄 TXT",
                                            data=f.read(),
                                            file_name=os.path.basename(item["narrative_path"]),
                                            mime="text/plain",
                                            key=f"txt_{selected_batch_id}_{item['input_name']}"
                                        )
                        elif item["status"] == "failed":
                            st.error(f"❌ {item['input_name']} - Failed: {item.get('error', 'Unknown error')}")
    else:
        st.info("No previous batches found. Run your first batch assessment to see results here.")
    
    st.divider()
    
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
                    # Ensure temp_videos directory exists
                    os.makedirs("temp_videos", exist_ok=True)
                    
                    temp_video_paths = []
                    for i, file in enumerate(uploaded_file, 1):
                        temp_path = os.path.join("temp_videos", f"temp_{file.name}")
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
                        # Create new batch using BatchManager
                        video_names = [file.name for file in uploaded_file]
                        batch_settings = {
                            "pattern": st.session_state.selected_pattern,
                            "fps": fps,
                            "batch_size": batch_size,
                            "max_concurrent_batches": max_concurrent_batches
                        }
                        
                        # Create batch and get batch ID
                        current_batch_id = st.session_state.batch_manager.create_batch(video_names, batch_settings)
                        st.info(f"🎯 **Batch Created**: {current_batch_id}")
                        
                        # Process each video using the existing working function with isolation
                        for i, (file, temp_path) in enumerate(zip(uploaded_file, temp_video_paths), 1):
                            file_pattern = st.session_state.detected_patterns.get(file.name, st.session_state.selected_pattern)
                            st.info(f"Processing {i}/{len(uploaded_file)}: {file.name} ({file_pattern})")
                            
                            # Update batch status to processing
                            st.session_state.batch_manager.update_item_status(
                                current_batch_id, file.name, "processing"
                            )
                            
                            # Process isolation: Each video in its own try-catch to prevent batch crashes
                            try:
                                # Clear previous video's session state to avoid conflicts
                                if hasattr(st.session_state, 'assessment_results'):
                                    del st.session_state.assessment_results
                                if hasattr(st.session_state, 'vop_analysis_complete'):
                                    del st.session_state.vop_analysis_complete
                                if hasattr(st.session_state, 'final_product_image'):
                                    del st.session_state.final_product_image
                                
                                # Use the existing working function (but suppress individual downloads)
                                st.session_state.suppress_individual_downloads = True
                                start_vop_analysis(temp_path, api_key, fps, batch_size, max_concurrent_batches, file_pattern)
                                
                                # Extract score from assessment results
                                score = None
                                html_path = None
                                narrative_path = None
                                
                                if hasattr(st.session_state, 'assessment_results'):
                                    assessment_results = st.session_state.assessment_results
                                    
                                    # Extract score
                                    if 'rubric_scores' in assessment_results:
                                        scores = assessment_results['rubric_scores']
                                        if scores:
                                            avg_score = sum(scores) / len(scores)
                                            score = f"{avg_score:.1f}/5"
                                    
                                    # Find the generated files
                                    clean_filename = file.name
                                    if clean_filename.startswith("temp_"):
                                        clean_filename = clean_filename[5:]
                                    
                                    # Look for HTML and narrative files
                                    import glob
                                    html_pattern = f"html_reports/VOP_Assessment_{clean_filename}_*.html"
                                    html_files = glob.glob(html_pattern)
                                    if html_files:
                                        html_path = max(html_files, key=os.path.getctime)
                                    
                                    txt_pattern = f"narratives/VOP_Assessment_{clean_filename}_*.txt"
                                    txt_files = glob.glob(txt_pattern)
                                    if txt_files:
                                        narrative_path = max(txt_files, key=os.path.getctime)
                                
                                # Update batch with completed status
                                st.session_state.batch_manager.update_item_status(
                                    current_batch_id, file.name, "completed",
                                    html_path=html_path,
                                    narrative_path=narrative_path,
                                    score=score
                                )
                                
                                st.success(f"✅ Video {i} completed successfully - Score: {score or 'N/A'}")
                                
                            except Exception as e:
                                error_msg = str(e)
                                st.error(f"❌ Video {i} ({file.name}) failed: {error_msg}")
                                
                                # Enhanced error logging for debugging
                                import traceback
                                full_traceback = traceback.format_exc()
                                print(f"ERROR DETAILS for {file.name}:")
                                print(f"Error message: {error_msg}")
                                print(f"Full traceback: {full_traceback}")
                                
                                # Update batch with failed status and truncated error
                                error_summary = error_msg[:200] + "..." if len(error_msg) > 200 else error_msg
                                st.session_state.batch_manager.update_item_status(
                                    current_batch_id, file.name, "failed",
                                    error=error_summary
                                )
                                
                                # Force memory cleanup before next video
                                import gc
                                gc.collect()
                                
                                # Continue processing next video instead of crashing entire batch
                                continue
                            
                            # Add a separator between videos
                            if i < len(uploaded_file):
                                st.divider()
                        
                        # Reset individual download suppression
                        st.session_state.suppress_individual_downloads = False
                        
                        # Get final batch status
                        final_manifest = st.session_state.batch_manager.get_batch_manifest(current_batch_id)
                        if final_manifest:
                            completed_count = sum(1 for item in final_manifest["items"] if item["status"] == "completed")
                            failed_count = sum(1 for item in final_manifest["items"] if item["status"] == "failed")
                            
                            st.success(f"✅ Batch processing complete! ✅ {completed_count} completed, ❌ {failed_count} failed")
                            
                            # Create and offer batch ZIP download
                            if completed_count > 0:
                                zip_path = st.session_state.batch_manager.create_batch_zip(current_batch_id)
                                if zip_path and os.path.exists(zip_path):
                                    st.markdown("### 📦 **Download Complete Batch**")
                                    with open(zip_path, "rb") as f:
                                        st.download_button(
                                            label=f"📦 Download Batch ZIP ({completed_count} assessments)",
                                            data=f.read(),
                                            file_name=f"{current_batch_id}.zip",
                                            mime="application/zip",
                                            type="primary",
                                            help="Contains all HTML reports, TXT files, and summary CSV. Safe to download - won't reset the app!"
                                        )
                            
                            st.info("💡 **All results are safely stored!** Check the 'Batch History' section above to access your files anytime, even after app restarts.")
                        
                with col_stop:
                    if st.button("🛑 STOP Batch", type="secondary", help="Stop batch processing"):
                        st.warning("🛑 Batch processing stopped by user")
                        st.stop()
            else:
                # Single file
                with st.expander("📊 Video Information", expanded=True):
                    # Ensure temp_videos directory exists
                    os.makedirs("temp_videos", exist_ok=True)
                    
                    # Save video temporarily for analysis
                    temp_video_path = os.path.join("temp_videos", f"temp_{uploaded_file.name}")
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
            
            if st.session_state.get('vop_analysis_complete', False):
                display_assessment_results(rubric_engine)
            else:
                st.info("Click 'Start VOP Assessment' to begin evaluation")
        
    else:
        st.info("👆 Please upload a video, confirm the suture pattern, and enter your API key to begin assessment")


def _process_batches_concurrently_gpt5(batches, gpt5_client, profile, progress_bar, status_text, total_batches, max_concurrent_batches):
    """Process batches concurrently using GPT-5 for descriptive analysis"""
    import concurrent.futures
    import threading
    import time
    
    context_lock = threading.Lock()
    successful_batches = 0
    batch_times = []
    
    def process_single_batch_gpt5(batch, profile, batch_idx, total_batches, context_lock):
        """Process a single batch with GPT-5"""
        start_time = time.time()
        
        try:
            with context_lock:
                current_context = gpt5_client.context_state
            
            print(f"DEBUG: Processing batch {batch_idx} with {len(batch)} frames")
            
            # PASS 1: Surgical description of frames
            analysis, descriptions = gpt5_client.pass1_surgical_description(
                batch, current_context,
                model=st.session_state.get('gpt5_model_pass1', 'gpt-5'),
                reasoning_level=st.session_state.get('reasoning_level_pass1', 'low'),
                verbosity_level=st.session_state.get('verbosity_level_pass1', 'medium')
            )
            
            with context_lock:
                gpt5_client.context_state = gpt5_client.context_state
            
            batch_time = time.time() - start_time
            print(f"DEBUG: Batch {batch_idx} completed in {batch_time:.2f}s")
            
            return {
                'success': True,
                'batch_idx': batch_idx,
                'batch_time': batch_time,
                'analysis': analysis
            }
            
        except Exception as e:
            batch_time = time.time() - start_time
            error_msg = f"Batch {batch_idx} failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            return {
                'success': False,
                'batch_idx': batch_idx,
                'batch_time': batch_time,
                'error': error_msg
            }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_batches) as executor:
        future_to_batch = {}
        for i, batch in enumerate(batches):
            future = executor.submit(
                process_single_batch_gpt5,
                batch, profile, i, total_batches, context_lock
            )
            future_to_batch[future] = i
        
        completed = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            result = future.result()
            completed += 1
            
            if result['success']:
                successful_batches += 1
                batch_times.append(result['batch_time'])
            
            # Update progress
            progress = completed / total_batches
            progress_bar.progress(progress)
            status_text.text(f"Processed {completed}/{total_batches} batches...")
            
            # Minimal crash prevention: periodic garbage collection
            if completed % 5 == 0:
                gc.collect()
    
    return {
        'successful_batches': successful_batches,
        'batch_times': batch_times
    }


def _process_batches_sequentially_gpt5(batches, gpt5_client, profile, progress_bar, status_text, total_batches):
    """Process batches sequentially using GPT-5 for descriptive analysis"""
    successful_batches = 0
    batch_times = []
    
    for i, batch in enumerate(batches):
        start_time = time.time()
        
        try:
            print(f"DEBUG: Processing batch {i} with {len(batch)} frames")
            
            # PASS 1: Surgical description of frames
            analysis, descriptions = gpt5_client.pass1_surgical_description(
                batch, gpt5_client.context_state,
                model=st.session_state.get('gpt5_model_pass1', 'gpt-5'),
                reasoning_level=st.session_state.get('reasoning_level_pass1', 'low'),
                verbosity_level=st.session_state.get('verbosity_level_pass1', 'medium')
            )
            
            batch_time = time.time() - start_time
            successful_batches += 1
            batch_times.append(batch_time)
            print(f"DEBUG: Batch {i} completed in {batch_time:.2f}s")
            
        except Exception as e:
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            error_msg = f"Batch {i} failed: {str(e)}"
            print(f"ERROR: {error_msg}")
            print(f"ERROR: Exception type: {type(e).__name__}")
        
        # Update progress
        progress = (i + 1) / total_batches
        progress_bar.progress(progress)
        status_text.text(f"Processed {i + 1}/{total_batches} batches...")
    
    return {
        'successful_batches': successful_batches,
        'batch_times': batch_times
    }


def start_vop_analysis(video_path: str, api_key: str, fps: float, batch_size: int, max_concurrent_batches: int = 100, pattern_id: str = None):
    """Start the VOP analysis process using the proven video analysis architecture."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize components using single GPT-5 approach
        video_processor = VideoProcessor()
        gpt5_client = GPT5VisionClient(api_key=api_key)
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
        
        # Convert frames to base64 for GPT-5 processing with memory optimization
        status_text.text("Converting frames to base64...")
        
        # Memory optimization: Process frames in chunks to avoid memory exhaustion
        import gc
        chunk_size = 10  # Process frames in smaller chunks to prevent memory issues
        
        print(f"DEBUG: Converting {len(frames)} frames to base64 in chunks of {chunk_size}")
        
        for chunk_start in range(0, len(frames), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(frames))
            frame_chunk = frames[chunk_start:chunk_end]
            
            # Convert chunk to base64
            chunk_base64 = video_processor.frames_to_base64(frame_chunk)
            
            # Add base64 data to frames
            for i, frame in enumerate(frame_chunk):
                frame['base64'] = chunk_base64[i]
            
            # Force garbage collection after each chunk
            gc.collect()
            
            print(f"DEBUG: Processed frames {chunk_start+1}-{chunk_end} of {len(frames)}")
        
        print(f"DEBUG: Converted {len(frames)} frames to base64 with memory optimization")
        print(f"DEBUG: First frame base64 length: {len(frames[0]['base64'])}")
        
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
        
        # Process batches using GPT-5 for descriptive analysis
        if max_concurrent_batches > 1:
            st.info(f"⚡ Using concurrent processing: {max_concurrent_batches} batches simultaneously")
            
            start_time = time.time()
            performance_metrics = _process_batches_concurrently_gpt5(
                batches, gpt5_client, surgical_profile, progress_bar, status_text, total_batches, max_concurrent_batches
            )
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Display performance metrics
            success_rate = (performance_metrics.get('successful_batches', 0) / total_batches) * 100
            avg_batch_time = sum(performance_metrics.get('batch_times', [0])) / max(len(performance_metrics.get('batch_times', [1])), 1)
            
            st.success(f"✅ Analysis complete in {processing_time:.1f} seconds!")
            st.info(f"📊 Success rate: {success_rate:.1f}% ({performance_metrics.get('successful_batches', 0)}/{total_batches} batches)")
            st.info(f"⚡ Speed: {avg_batch_time:.1f}s average per batch")
            
        else:
            # Use sequential processing with GPT-5
            _process_batches_sequentially_gpt5(batches, gpt5_client, surgical_profile, progress_bar, status_text, total_batches)
        
        # Check if we have frame descriptions from PASS 1
        if not gpt5_client.frame_descriptions:
            st.error("❌ PASS 1 failed: No frame descriptions generated")
            return
        
        st.success(f"✅ PASS 1 complete: Generated {len(gpt5_client.frame_descriptions)} frame descriptions")
        
        # PASS 2: Create video narrative from frame descriptions
        status_text.text("PASS 2: Creating video narrative with flow analysis...")
        
        # Add meaningful progress indicator for PASS 2
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        try:
            progress_text.text("🔄 Assembling frame descriptions...")
            progress_bar.progress(0.2)
            
            # Show progress as we process
            progress_text.text("🔄 Analyzing temporal flow and motion...")
            progress_bar.progress(0.5)
            
            video_narrative = gpt5_client.pass2_video_narrative(
                gpt5_client.frame_descriptions,
                model=st.session_state.get('gpt5_model_pass2', 'gpt-5'),
                reasoning_level=st.session_state.get('reasoning_level_pass2', 'medium'),
                verbosity_level=st.session_state.get('verbosity_level_pass2', 'high')
            )
            
            progress_text.text("🔄 Generating comprehensive narrative...")
            progress_bar.progress(0.8)
            
            if not video_narrative or video_narrative.startswith("Error"):
                st.error(f"❌ PASS 2 failed: {video_narrative}")
                return
            
            progress_bar.progress(1.0)
            progress_text.text("✅ Video narrative complete!")
            st.success("✅ PASS 2 complete: Video narrative generated")
            
        except Exception as e:
            st.error(f"❌ PASS 2 failed with exception: {str(e)}")
            return
        finally:
            # Clear progress indicators after a brief delay
            time.sleep(1)
            progress_bar.empty()
            progress_text.empty()
        
                    # Export Pass 2 narrative as TXT file immediately (no need to wait for Pass 3)
        try:
            # Create organized folders
            os.makedirs("narratives", exist_ok=True)
            os.makedirs("html_reports", exist_ok=True)
            os.makedirs("temp_videos", exist_ok=True)
            
            # Use original filename, not temp filename
            original_filename = os.path.basename(video_path)
            if original_filename.startswith("temp_"):
                original_filename = original_filename[5:]  # Remove temp_ prefix
            
            video_name = os.path.splitext(original_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = f"VOP_Narrative_{video_name}_{timestamp}.txt"
            txt_path = os.path.join("narratives", txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"SURGICAL VOP ASSESSMENT - PASS 2 NARRATIVE\n")
                f.write(f"Video: {os.path.basename(video_path)}\n")
                f.write(f"Pattern: {current_pattern}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
                f.write("COMPLETE VIDEO NARRATIVE:\n")
                f.write(f"{'='*60}\n")
                f.write(video_narrative)
                f.write(f"\n\n{'='*60}\n")
                f.write("END OF NARRATIVE")
            
            st.success(f"✅ Pass 2 narrative exported to: {txt_filename}")
            
            # Immediate download button for Pass 2 narrative (only if not in batch mode)
            if not st.session_state.get('suppress_individual_downloads', False):
                st.markdown("### 📥 Save Pass 2 Narrative to Your Local Machine")
                st.info("💾 **Save to:** `C:\\CursorAI_folders\\AI_video_watcher\\narratives`")
                with open(txt_path, "rb") as f:
                    st.download_button(
                        label="📄 Download Pass 2 Narrative (TXT)",
                        data=f.read(),
                        file_name=os.path.basename(txt_filename),
                        mime="text/plain",
                        type="primary"
                    )
            
        except Exception as e:
            st.warning(f"⚠️ Could not export TXT file: {str(e)}")
        
        # PASS 3: Rubric assessment based on video narrative with final product image
        status_text.text("PASS 3: Applying rubric assessment to video narrative with final product image...")
        
        # Extract final product image for verification with robust error handling
        final_product_image = None
        try:
            status_text.text("Extracting final product image...")
            progress_bar.progress(0.85)
            
            from surgical_report_generator import SurgicalVOPReportGenerator
            report_gen = SurgicalVOPReportGenerator()
            # Use the actual temp video path that exists, not original filename
            actual_video_path = temp_video_path if 'temp_video_path' in locals() else video_path
            assessment_data_for_image = {'video_path': actual_video_path, 'api_key': api_key}
            report_gen._return_pil_for_html = True  # Flag to return PIL Image
            
            # Retry logic for image extraction with timeout
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    status_text.text(f"Extracting final product image (attempt {attempt + 1}/{max_retries})...")
                    final_product_image = report_gen._extract_final_product_image_enhanced_full(assessment_data_for_image, 400)
                    if final_product_image:
                        print("✅ Final product image extracted for Pass 3 assessment")
                        st.session_state.final_product_image = final_product_image
                        break
                    else:
                        print(f"⚠️ Image extraction attempt {attempt + 1} returned None")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                except Exception as img_error:
                    print(f"⚠️ Image extraction attempt {attempt + 1} failed: {str(img_error)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        raise img_error
            
            if not final_product_image:
                print("⚠️ Could not extract final product image after all retries, proceeding without it")
                st.session_state.final_product_image = None
                
        except Exception as e:
            print(f"Warning: Final product image extraction failed completely: {str(e)}")
            st.warning(f"⚠️ Could not extract final product image: {str(e)[:100]}...")
            st.session_state.final_product_image = None
            # Continue processing even without image
        
        # For subcuticular assessments, disable final image usage in Pass 3
        image_for_pass3 = None if current_pattern == 'subcuticular' else final_product_image
        
        enhanced_narrative = gpt5_client.pass3_rubric_assessment(
            current_pattern, rubric_engine, image_for_pass3,
            model=st.session_state.get('gpt5_model_pass3', 'gpt-5'),
            reasoning_level=st.session_state.get('reasoning_level_pass3', 'high'),
            verbosity_level=st.session_state.get('verbosity_level_pass3', 'medium')
        )
        
        if not enhanced_narrative.get('success', False):
            st.error(f"❌ PASS 3 failed: {enhanced_narrative.get('error', 'Unknown error')}")
            return
        
        st.success("✅ PASS 3 complete: Rubric assessment generated")
        
        # Get complete analysis data
        full_transcript = gpt5_client.get_full_transcript()
        event_timeline = gpt5_client.get_event_timeline()
        
        # Extract scores from enhanced narrative
        if enhanced_narrative.get('success', False):
            extracted_scores = enhanced_narrative.get('rubric_scores', {})
            st.session_state.rubric_scores = extracted_scores
            st.session_state.summative_feedback = enhanced_narrative.get('summative_assessment', '')
            st.success(f"✅ Extracted {len(extracted_scores)} rubric scores from GPT-5 assessment")
        else:
            st.error(f"Assessment failed: {enhanced_narrative.get('error', 'Unknown error')}")
            return
        
        # Store results in the proven format
        # Derive original filename (strip temp_ prefix if present)
        original_filename = os.path.basename(video_path)
        if original_filename.startswith("temp_"):
            original_filename = original_filename[len("temp_"):]

        st.session_state.assessment_results = {
            "full_transcript": full_transcript,  # Keep for debugging if needed
            "video_narrative": video_narrative,  # PASS 2: Video narrative
            "enhanced_narrative": enhanced_narrative,  # PASS 3: Final assessment
            "event_timeline": event_timeline,
            "extracted_scores": extracted_scores,  # Store the extracted scores
            "video_path": video_path,  # Add video path for final product image extraction
            "video_info": {
                "filename": original_filename,
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
        
        # AUTOMATIC TXT AND HTML REPORT GENERATION
        try:
            # Ensure directories exist
            os.makedirs("narratives", exist_ok=True)
            os.makedirs("html_reports", exist_ok=True)
            
            # Use original filename without temp_ prefix
            clean_filename = original_filename
            if clean_filename.startswith("temp_"):
                clean_filename = clean_filename[5:]
            
            base_filename = f"VOP_Assessment_{clean_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            txt_filename = os.path.join("narratives", f"{base_filename}.txt")
            html_filename = os.path.join("html_reports", f"{base_filename}.html")
            
            # Generate TXT report with FULL Pass 2 narrative
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(f"SURGICAL VOP ASSESSMENT - PASS 2 NARRATIVE\n")
                f.write(f"Video: {original_filename}\n")
                f.write(f"Pattern: {current_pattern.replace('_', ' ').title()}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*60}\n\n")
                f.write("COMPLETE VIDEO NARRATIVE:\n")
                f.write(f"{'='*60}\n")
                f.write(video_narrative)  # Write the FULL Pass 2 narrative
                f.write(f"\n\n{'='*60}\n")
                f.write("END OF NARRATIVE")
            
            # Generate HTML report with properly stacked images
            with open(html_filename, 'w', encoding='utf-8') as f:
                f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                f.write("<title>Surgical VOP Assessment Report</title>\n")
                f.write("<style>\n")
                f.write("body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }\n")
                f.write("h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }\n")
                f.write("h2 { color: #34495e; margin-top: 30px; }\n")
                f.write(".rubric-point { margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #007bff; }\n")
                f.write(".summative { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }\n")
                f.write(".average-score { background-color: #fff3cd; padding: 10px; border-radius: 5px; text-align: center; margin: 20px 0; }\n")
                f.write(".image-section { margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }\n")
                f.write(".image-section h3 { margin-top: 0; color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 10px; }\n")
                f.write(".image-section img { max-width: 100%; max-height: 500px; width: auto; height: auto; border: 2px solid #ddd; border-radius: 8px; margin: 10px 0; }\n")
                f.write("</style>\n</head>\n<body>\n")
                
                f.write(f"<h1>Surgical VOP Assessment Report</h1>\n")
                f.write(f"<p><strong>Video:</strong> {original_filename}</p>\n")
                f.write(f"<p><strong>Pattern:</strong> {current_pattern.replace('_', ' ').title()}</p>\n")
                
                # Get the assessment results data
                assessment_results = st.session_state.get('assessment_results', {})
                enhanced_narrative = assessment_results.get('enhanced_narrative', {})
                extracted_scores = assessment_results.get('extracted_scores', {})
                
                # Extract assessment data - handle both structured and unstructured formats
                rubric_comments = {}
                summative_assessment = ""
                
                if enhanced_narrative and isinstance(enhanced_narrative, dict):
                    # Try to get structured data first
                    rubric_comments = enhanced_narrative.get('rubric_comments', {})
                    summative_assessment = enhanced_narrative.get('summative_assessment', '')
                    
                    # If no structured data, try to parse from full_response
                    if not rubric_comments and 'full_response' in enhanced_narrative:
                        full_response = enhanced_narrative['full_response']
                        # Parse rubric points from full response
                        for point_num in range(1, 8):
                            point_pattern = f"RUBRIC_POINT_{point_num}:"
                            if point_pattern in full_response:
                                start_idx = full_response.find(point_pattern)
                                if start_idx != -1:
                                    next_point_idx = full_response.find(f"RUBRIC_POINT_{point_num+1}:", start_idx)
                                    summative_idx = full_response.find("SUMMATIVE_ASSESSMENT:", start_idx)
                                    end_idx = len(full_response)
                                    if next_point_idx != -1:
                                        end_idx = min(end_idx, next_point_idx)
                                    if summative_idx != -1:
                                        end_idx = min(end_idx, summative_idx)
                                    
                                    point_section = full_response[start_idx:end_idx]
                                    comment_match = re.search(r'Comment:\s*(.+?)(?=Score:|$)', point_section, re.DOTALL)
                                    if comment_match:
                                        rubric_comments[point_num] = comment_match.group(1).strip()
                        
                        # Parse summative assessment from full response
                        if not summative_assessment:
                            if "SUMMATIVE_ASSESSMENT:" in full_response:
                                summative_section = full_response.split("SUMMATIVE_ASSESSMENT:")[1].strip()
                                summative_assessment = summative_section.split('\n')[0].strip() if summative_section else ""
                
                # Get rubric point titles for proper labeling
                pattern_data = rubric_engine.get_pattern_rubric(current_pattern)
                rubric_titles = {}
                if pattern_data and 'points' in pattern_data:
                    for point in pattern_data['points']:
                        rubric_titles[point['pid']] = point['title']
                
                # Display rubric assessment points
                if extracted_scores:
                    f.write("<h2>Rubric Assessment</h2>\n")
                    # Display rubric points in order
                    for point_num in range(1, 8):  # Points 1-7
                        if point_num in extracted_scores:
                            comment = rubric_comments.get(point_num, "Assessment comment not available")
                            score = extracted_scores[point_num]
                            title = rubric_titles.get(point_num, f"Rubric Point {point_num}")
                            
                            # Determine Likert scale level
                            if score >= 5.0:
                                competency = "Expert"
                            elif score >= 4.0:
                                competency = "Proficient"
                            elif score >= 3.0:
                                competency = "Competent"
                            elif score >= 2.0:
                                competency = "Novice"
                            else:
                                competency = "Remediate"
                            
                            f.write(f"<div class='rubric-point'><strong>{point_num}. {title}</strong><br>{comment}<br><strong>Score: {score}/5 - {competency}</strong></div>\n")
                    
                    # Add average score
                    avg_score = sum(extracted_scores.values()) / len(extracted_scores)
                    
                    f.write(f"<div class='average-score'><strong>Average Score: {avg_score:.1f}/5</strong></div>\n")
                    
                    # Add summative comment if available
                    if summative_assessment:
                        f.write(f"<div class='summative'><strong>Summative Assessment:</strong><br>{summative_assessment}</div>\n")
                else:
                    f.write("<p><strong>Assessment scores not available</strong></p>\n")
                
                # Learner Final Product Image (first)
                f.write("<h2>Final Product Comparison</h2>\n")
                f.write("<div class='image-section'>\n")
                f.write("<h3>Learner Final Product</h3>\n")
                try:
                    # Use the final product image from session state
                    final_product_image = st.session_state.get('final_product_image', None)
                    if final_product_image is not None and hasattr(final_product_image, 'save'):  # Check if it's a PIL Image
                        import base64
                        from io import BytesIO
                        buffered = BytesIO()
                        final_product_image.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        f.write(f"<img src='data:image/jpeg;base64,{img_str}' alt='Learner Final Product' />\n")
                    else:
                        f.write("<p><em>No suitable final product image found</em></p>\n")
                except Exception as e:
                    f.write(f"<p><em>Error loading final product image: {str(e)}</em></p>\n")
                f.write("</div>\n")
                
                # Gold Standard Image (below)
                f.write("<div class='image-section'>\n")
                f.write("<h3>Gold Standard Reference</h3>\n")
                try:
                    # Use correct gold standard image filenames
                    gold_standard_mapping = {
                        'simple_interrupted': 'surgical-vop-assessment/Simple_Interrupted_Suture_example.png',
                        'vertical_mattress': 'surgical-vop-assessment/Vertical_Mattress_Suture_example.png',
                        'subcuticular': 'surgical-vop-assessment/subcuticular_example.png'
                    }
                    gold_standard_path = gold_standard_mapping.get(current_pattern, f"surgical-vop-assessment/gold_standard_{current_pattern}.jpg")
                    if os.path.exists(gold_standard_path):
                        with open(gold_standard_path, "rb") as img_file:
                            import base64 as b64
                            img_data = b64.b64encode(img_file.read()).decode()
                            f.write(f"<img src='data:image/jpeg;base64,{img_data}' alt='Gold Standard Reference' />\n")
                    else:
                        f.write(f"<p><em>Gold standard image not found: {gold_standard_path}</em></p>\n")
                except Exception as e:
                    f.write(f"<p><em>Error loading gold standard image: {str(e)}</em></p>\n")
                f.write("</div>\n")
                
                f.write("</body>\n</html>\n")
            
            st.success(f"✅ TXT and HTML reports auto-generated:")
            
            # Immediate download section for Pass 3 completion (only if not in batch mode)
            if not st.session_state.get('suppress_individual_downloads', False):
                st.markdown("### 📥 Save Final Reports to Your Local Machine")
                
                # HTML Report Download with local path reminder
                st.info("💾 **HTML Report save to:** `C:\\CursorAI_folders\\AI_video_watcher\\html_reports`")
                with open(html_filename, "rb") as f:
                    st.download_button(
                        label="🌐 Download HTML Final Report",
                        data=f.read(),
                        file_name=os.path.basename(html_filename),
                        mime="text/html",
                        type="primary"
                    )
                
                # TXT Report Download with local path reminder  
                st.info("💾 **TXT Report save to:** `C:\\CursorAI_folders\\AI_video_watcher\\narratives`")
                with open(txt_filename, "rb") as f:
                    st.download_button(
                        label="📄 Download TXT Report",
                        data=f.read(),
                        file_name=os.path.basename(txt_filename),
                        mime="text/plain"
                    )
            
            # Also show file paths for reference
            st.info(f"📁 **Files created**: {os.path.basename(txt_filename)}, {os.path.basename(html_filename)}")
        except Exception as report_error:
            st.error(f"❌ **AUTOMATIC REPORT GENERATION FAILED**: {str(report_error)}")
            import traceback
            st.code(traceback.format_exc())
            print(f"AUTOMATIC GENERATION ERROR: {report_error}")
            print(f"TRACEBACK: {traceback.format_exc()}")

        # Cleanup temp video file
        try:
            if os.path.basename(video_path).startswith("temp_") and os.path.exists(video_path):
                os.remove(video_path)
        except Exception as cleanup_err:
            print(f"Warning: failed to remove temp file {video_path}: {cleanup_err}")
        
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
    
    # Display video narrative (PASS 2)
    st.subheader("📹 Video Narrative Analysis")
    if results.get("video_narrative"):
        st.text_area("Complete Video Narrative", results["video_narrative"], height=200, disabled=True)
    
    # Display GPT-5 enhanced narrative (PASS 3 - primary assessment)
    st.subheader("🏥 Final Surgical Assessment")
    if results.get("enhanced_narrative"):
        # Handle enhanced narrative as dictionary
        enhanced_data = results["enhanced_narrative"]
        
        if isinstance(enhanced_data, dict):
            # Display individual rubric point assessments with titles
            rubric_comments = enhanced_data.get('rubric_comments', {})
            extracted_scores = results.get("extracted_scores", {})
            
            if rubric_comments and extracted_scores:
                st.subheader("📋 Individual Rubric Point Assessments")
                
                # Get rubric point titles for proper labeling
                pattern_data = rubric_engine.get_pattern_rubric(st.session_state.selected_pattern)
                rubric_titles = {}
                if pattern_data and 'points' in pattern_data:
                    for point in pattern_data['points']:
                        rubric_titles[point['pid']] = point['title']
                
                # Display rubric points in order
                for point_num in range(1, 8):  # Points 1-7
                    if point_num in rubric_comments and point_num in extracted_scores:
                        comment = rubric_comments[point_num]
                        score = extracted_scores[point_num]
                        title = rubric_titles.get(point_num, f"Rubric Point {point_num}")
                        
                        # Determine adjective
                        if score >= 4.0:
                            adj = "exemplary"
                        elif score >= 3.0:
                            adj = "proficient"
                        elif score >= 2.0:
                            adj = "competent"
                        elif score >= 1.0:
                            adj = "developing"
                        else:
                            adj = "inadequate"
                        
                        st.markdown(f"**{point_num}. {title}** ({score}/5 {adj})")
                        st.markdown(f"*{comment}*")
                        st.markdown("---")
            
            # Extract summative assessment from dictionary
            narrative = enhanced_data.get("summative_assessment", enhanced_data.get("full_response", ""))
        else:
            # Fallback for string format
            narrative = enhanced_data
        
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
                Average Score: {overall_result['average_score']:.1f}/5.0
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
        
        # HTML report generation and download button
        if st.button("🌐 Generate & Download HTML Report"):
            try:
                # Generate HTML filename based on current assessment
                html_filename = f"html_reports/VOP_Assessment_{results['video_info']['pattern']}_{results['video_info']['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                
                # Ensure html_reports directory exists
                os.makedirs("html_reports", exist_ok=True)
                
                # Generate HTML report content
                with open(html_filename, 'w', encoding='utf-8') as f:
                    f.write("<!DOCTYPE html>\n<html>\n<head>\n")
                    f.write("<title>Surgical VOP Assessment Report</title>\n")
                    f.write("<style>\n")
                    f.write("body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }\n")
                    f.write("h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }\n")
                    f.write("h2 { color: #34495e; margin-top: 30px; }\n")
                    f.write(".rubric-point { margin: 15px 0; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #007bff; }\n")
                    f.write(".summative { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; }\n")
                    f.write(".average-score { background-color: #fff3cd; padding: 10px; border-radius: 5px; text-align: center; margin: 20px 0; }\n")
                    f.write(".image-section { margin: 30px 0; padding: 20px; background-color: #f8f9fa; border-radius: 8px; }\n")
                    f.write(".image-section h3 { margin-top: 0; color: #2c3e50; border-bottom: 1px solid #ddd; padding-bottom: 10px; }\n")
                    f.write(".image-section img { max-width: 100%; max-height: 500px; width: auto; height: auto; border: 2px solid #ddd; border-radius: 8px; margin: 10px 0; }\n")
                    f.write("</style>\n</head>\n<body>\n")
                    
                    f.write(f"<h1>Surgical VOP Assessment Report</h1>\n")
                    f.write(f"<p><strong>Video:</strong> {results['video_info']['filename']}</p>\n")
                    f.write(f"<p><strong>Pattern:</strong> {results['video_info']['pattern'].replace('_', ' ').title()}</p>\n")
                    
                    # Add rubric assessment with GPT-5 generated content
                    enhanced_narrative = results.get('enhanced_narrative', {})
                    extracted_scores = results.get('extracted_scores', {})
                    
                    # Parse GPT-5 comments for each rubric point
                    rubric_comments = {}
                    summative_assessment = ""
                    
                    if enhanced_narrative and isinstance(enhanced_narrative, dict):
                        # Try to get structured data first
                        rubric_comments = enhanced_narrative.get('rubric_comments', {})
                        summative_assessment = enhanced_narrative.get('summative_assessment', '')
                        
                        # If no structured data, parse from full_response (GPT-5 Pass 3 output)
                        if not rubric_comments and 'full_response' in enhanced_narrative:
                            full_response = enhanced_narrative['full_response']
                            # Parse rubric points from full response
                            for point_num in range(1, 8):
                                point_pattern = f"RUBRIC_POINT_{point_num}:"
                                if point_pattern in full_response:
                                    start_idx = full_response.find(point_pattern)
                                    if start_idx != -1:
                                        next_point_idx = full_response.find(f"RUBRIC_POINT_{point_num+1}:", start_idx)
                                        summative_idx = full_response.find("SUMMATIVE_ASSESSMENT:", start_idx)
                                        end_idx = len(full_response)
                                        if next_point_idx != -1:
                                            end_idx = min(end_idx, next_point_idx)
                                        if summative_idx != -1:
                                            end_idx = min(end_idx, summative_idx)
                                        
                                        point_section = full_response[start_idx:end_idx]
                                        comment_match = re.search(r'Comment:\s*(.+?)(?=Score:|$)', point_section, re.DOTALL)
                                        if comment_match:
                                            rubric_comments[point_num] = comment_match.group(1).strip()
                            
                            # Parse summative assessment from full response
                            if not summative_assessment:
                                if "SUMMATIVE_ASSESSMENT:" in full_response:
                                    summative_section = full_response.split("SUMMATIVE_ASSESSMENT:")[1].strip()
                                    summative_assessment = summative_section.split('\n')[0].strip() if summative_section else ""
                    
                    # Get rubric point titles for proper labeling
                    rubric_data = st.session_state.get('current_rubric_data', {})
                    rubric_titles = {}
                    if 'points' in rubric_data:
                        for point in rubric_data['points']:
                            rubric_titles[point['pid']] = point['title']
                    
                    f.write("<h2>Rubric Assessment</h2>\n")
                    # Display rubric points with GPT-5 generated content
                    for point_num in range(1, 8):
                        if point_num in extracted_scores:
                            comment = rubric_comments.get(point_num, "Assessment comment not available")
                            ai_score = extracted_scores[point_num] 
                            title = rubric_titles.get(point_num, f"Rubric Point {point_num}")
                            
                            # Determine Likert scale level based on AI score
                            if ai_score >= 5.0:
                                competency = "Expert"
                            elif ai_score >= 4.0:
                                competency = "Proficient"
                            elif ai_score >= 3.0:
                                competency = "Competent"
                            elif ai_score >= 2.0:
                                competency = "Novice"
                            else:
                                competency = "Remediate"
                            
                            f.write(f"<div class='rubric-point'><strong>{point_num}. {title}</strong><br>{comment}<br><strong>Score: {ai_score}/5 - {competency}</strong></div>\n")
                    
                    # Add AI-generated average score and summative assessment
                    if extracted_scores:
                        avg_score = sum(extracted_scores.values()) / len(extracted_scores)
                        if avg_score >= 5.0:
                            overall_competency = "Expert"
                        elif avg_score >= 4.0:
                            overall_competency = "Proficient"
                        elif avg_score >= 3.0:
                            overall_competency = "Competent"
                        elif avg_score >= 2.0:
                            overall_competency = "Novice"
                        else:
                            overall_competency = "Remediate"
                        
                        f.write(f"<div class='average-score'><strong>Average Score: {avg_score:.1f}/5</strong></div>\n")
                        
                        # Add summative comment if available
                        if summative_assessment:
                            f.write(f"<div class='summative'><strong>Summative Assessment:</strong><br>{summative_assessment}</div>\n")
                    else:
                        f.write("<p><strong>AI Assessment scores not available - using manual scores</strong></p>\n")
                        f.write(f"<div class='average-score'><strong>Manual Average Score: {overall_result['average_score']:.1f}/5 ")
                        if overall_result['pass']:
                            f.write("</strong></div>\n")
                        else:
                            f.write("</strong></div>\n")
                    
                    # Add final product image if available
                    f.write("<h2>Final Product Comparison</h2>\n")
                    f.write("<div class='image-section'>\n")
                    f.write("<h3>Learner Final Product</h3>\n")
                    try:
                        final_product_image = st.session_state.get('final_product_image', None)
                        if final_product_image is not None and hasattr(final_product_image, 'save'):
                            import base64
                            from io import BytesIO
                            buffered = BytesIO()
                            final_product_image.save(buffered, format="JPEG")
                            img_str = base64.b64encode(buffered.getvalue()).decode()
                            f.write(f"<img src='data:image/jpeg;base64,{img_str}' alt='Learner Final Product' />\n")
                        else:
                            f.write("<p><em>No final product image available</em></p>\n")
                    except Exception as e:
                        f.write(f"<p><em>Error loading final product image: {str(e)}</em></p>\n")
                    f.write("</div>\n")
                    
                    # Add gold standard image
                    f.write("<div class='image-section'>\n")
                    f.write("<h3>Gold Standard Reference</h3>\n")
                    try:
                        gold_standard_mapping = {
                            'simple_interrupted': 'surgical-vop-assessment/Simple_Interrupted_Suture_example.png',
                            'vertical_mattress': 'surgical-vop-assessment/Vertical_Mattress_Suture_example.png',
                            'subcuticular': 'surgical-vop-assessment/subcuticular_example.png'
                        }
                        current_pattern = results['video_info']['pattern']
                        gold_standard_path = gold_standard_mapping.get(current_pattern, f"surgical-vop-assessment/gold_standard_{current_pattern}.jpg")
                        if os.path.exists(gold_standard_path):
                            with open(gold_standard_path, "rb") as img_file:
                                import base64 as b64
                                img_data = b64.b64encode(img_file.read()).decode()
                                f.write(f"<img src='data:image/jpeg;base64,{img_data}' alt='Gold Standard Reference' />\n")
                        else:
                            f.write(f"<p><em>Gold standard image not found: {gold_standard_path}</em></p>\n")
                    except Exception as e:
                        f.write(f"<p><em>Error loading gold standard image: {str(e)}</em></p>\n")
                    f.write("</div>\n")
                    
                    f.write("</body>\n</html>\n")
                
                st.success(f"✅ HTML report generated: {os.path.basename(html_filename)}")
                
                # Offer download
                with open(html_filename, "rb") as html_file:
                    st.download_button(
                        label="📥 Download HTML Report",
                        data=html_file.read(),
                        file_name=os.path.basename(html_filename),
                        mime="text/html"
                    )
                    
            except Exception as e:
                st.error(f"Error generating HTML report: {e}")

if __name__ == "__main__":
    main()
