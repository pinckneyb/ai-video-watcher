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
    """
    Stage 2: Narrative Synthesis using GPT-5
    This function implements the proper two-stage AI pipeline as specified:
    - Stage 1 (GPT-4o): Already completed - frame-by-frame batch analysis
    - Stage 2 (GPT-5): Synthesize ALL batch outputs into comprehensive narrative
    """
    try:
        # Get pattern and rubric information
        pattern_data = rubric_engine.get_pattern_rubric(pattern_id)
        assessment_points = rubric_engine.get_assessment_criteria(pattern_id)
        
        # Build rubric criteria for reference
        rubric_criteria = "\n".join([
            f"{p['pid']}. {p['title']}: {p['what_you_assess']} (Ideal: {p['ideal_result']})"
            for p in assessment_points
        ])
        
        # Create OpenAI client
        gpt5_client = GPT4oClient(api_key=api_key)
        
        # CRITICAL: This is the system prompt that enforces the two-stage pipeline
        system_prompt = f"""You are a senior attending surgeon conducting a VOP assessment for {pattern_data['display_name']} suturing.

IMPORTANT: You are operating in Stage 2 (Narrative Synthesis) of a two-stage AI pipeline.

STAGE 1 (Vision, gpt-4o) has already completed:
- Analyzed batches of still frames from the video (every 5-10 frames)
- Described each batch locally in detail with frame ranges
- Identified surgical actions, tissue handling, needle angles, knot formation, and visible outcomes
- Output was structured observations tagged with frame ranges

STAGE 2 (Narrative Synthesis, gpt-5) - YOUR CRITICAL TASK:
- Input: the FULL set of batch descriptions from Stage 1, spanning the ENTIRE video in chronological order
- Task: construct a comprehensive chronological narrative of the ENTIRE procedure
- You must integrate ALL batches into one overarching account

CRITICAL REQUIREMENTS:
• Connect actions across batches into a continuous timeline
• Capture progress (setup, each stitch placement, knot tying, completion)
• Identify transitions (needle reloads, knot sequences, wound closure milestones)
• Do NOT overweight early batches; treat all batches as equally important
• Synthesize repeated actions into higher-level patterns (e.g., "four simple interrupted stitches placed in sequence")
• This stage is NOT a re-summarization of the first few inputs
• You must use ALL the batch data provided to create a complete picture

Your output will be used for rubric-based assessment, so be comprehensive and specific."""

        # User prompt with the complete transcript data
        user_prompt = f"""RAW ANALYSIS FROM STAGE 1 (ALL BATCHES):
{raw_transcript}

RUBRIC CRITERIA:
{rubric_criteria}

ASSESSMENT TASK:
Based on your comprehensive synthesis of ALL batch outputs above, assess each rubric point with specific evidence spanning the entire video.

CRITICAL GRADING GUIDELINES - BE EXTREMELY DEMANDING:
- Score 1 = Remediation/Unsafe (major errors, unsafe technique, closure unreliable)
- Score 2 = Minimal Pass/Basic Competent (safe but inefficient, rough technique, minor errors)
- Score 3 = Developing Pass/Generally Reliable (mostly correct, occasional flaws, reliable closure)
- Score 4 = Proficient (consistently correct, independent, efficient - genuinely excellent work)
- Score 5 = Exemplary/Model (near-ideal execution, precise, smooth, teachable example - EXTREMELY RARE)

YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE:
- Be relentlessly critical and unforgiving in your assessment
- Assume EVERY technique has flaws until proven otherwise
- Score 4-5 ONLY for truly exceptional performance that occurs maybe 1-2% of the time
- Most competent residents will score 1-2, with 3 being solid performance
- A score of 2 should be your STARTING POINT for safe, adequate technique
- A score of 3 means genuinely skilled, reliable performance worthy of independence
- A score of 4 means you would use this video to teach other attendings
- A score of 5 means this is among the best technique you've seen in your entire career

CRITICAL VISIBILITY RULES:
- NEVER penalize scores due to poor visibility, lighting, focus, blockage, or resolution
- If you cannot adequately assess a rubric point due to visibility issues, assign 3* (provisional score)
- Only mention visibility limitations when they genuinely prevent assessment (3* cases)
- For scoreable observations (1-5), focus on what you CAN see, don't mention visibility issues
- Make your best estimate from visible technique and score accordingly
- Use "3*" format only when visibility truly prevents assessment

BE RUTHLESSLY CRITICAL - DEMAND PERFECTION:
- Your default assumption is that technique is FLAWED until proven exceptional
- Start with score 1-2 and only move higher with compelling evidence of excellence
- Look for EVERY imperfection: hesitation, wasted motion, suboptimal angles, inconsistent spacing, loose knots
- ANY visible flaw should significantly lower the score
- Score 2 should be your DEFAULT for safe, functional technique
- Score 3 requires demonstration of genuine surgical skill and consistency
- Scores 4-5 are reserved for truly outstanding performance that you see rarely
- Remember: surgical excellence requires perfection - patients' lives depend on it

For each rubric point, write assessment in this EXACT format:

**[X]. [Rubric Point Title]:**
[Clean, specific observation of what was seen across the complete video - NO timestamps, NO generic advice]
Score: [X]/5

CRITICAL FORMATTING RULES:
- NO TIMESTAMPS in the main assessment (save for appendix)
- Use exactly "Score: X/5" format for each rubric point
- NO generic improvement suggestions like "practice more" or "focus on consistency" 
- NO lists of timestamp ranges like "(00:00:12–00:00:14; 00:00:15–00:00:17...)"
- Write clean, prose descriptions of what was consistently observed
- Only mention visibility issues if they genuinely prevent assessment (requiring 3* score)
- Be specific about technique patterns, not individual moments

Then write a comprehensive summative assessment that:
- Provides a holistic, gestalt review of the entire procedure (minimum 4-5 sentences)
- Synthesizes unique insights drawn from the interplay of different technical aspects
- Discusses the learner's overall surgical approach and decision-making patterns
- Identifies both strengths and areas for development with specific observations
- Offers a nuanced evaluation of surgical maturity and technique integration
- Avoids generic encouragement, timestamp lists, or simple recitation of individual rubric points
- STRUCTURE AS MEANINGFUL PARAGRAPHS: Break into 2-3 focused paragraphs for readability

EXAMPLE FORMAT:
**1. Perpendicular needle passes:**
Needle entries consistently demonstrated 90-degree angles throughout most of the procedure, with symmetric bites maintained across the majority of passes. Some minor angular deviations were noted but did not significantly compromise overall technique quality.
Score: 4/5

Format your response with clear sections for each rubric point, followed by the summative assessment."""

        # Validate input
        transcript_length = len(raw_transcript)
        if transcript_length < 500:
            st.error("❌ **STAGE 2 FAILED**: Insufficient transcript content for synthesis")
            st.warning(f"Transcript only {transcript_length} characters - need at least 500 for meaningful synthesis")
            return ""
        
        st.info(f"📊 **STAGE 2 Input**: Transcript: {transcript_length:,} characters")
        st.info(f"📝 **Content Preview**: {raw_transcript[:200]}...")
        
        # Make API call to GPT-5
        try:
            st.info(f"🔍 **Calling GPT-5 API** for narrative synthesis...")
            st.info(f"🔍 **Prompt lengths**: System: {len(system_prompt)}, User: {len(user_prompt)}")
            
            response = gpt5_client.client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=8000,
                reasoning_effort="low",
                verbosity="low"
            )
            
            enhanced_narrative = response.choices[0].message.content
            
            # Debug: Show what GPT-5 actually returned
            st.info(f"🔍 **GPT-5 Response Debug**: {type(enhanced_narrative)}, length: {len(enhanced_narrative) if enhanced_narrative else 0}")
            if enhanced_narrative:
                st.info(f"🔍 **First 200 chars**: {enhanced_narrative[:200]}")
            else:
                st.error(f"🔍 **GPT-5 returned None or empty string**")
                st.info(f"🔍 **Full response object**: {response}")
            
            if not enhanced_narrative or len(enhanced_narrative.strip()) < 100:
                st.error("❌ **GPT-5 returned empty or insufficient narrative**")
                st.info(f"🔍 **Content length**: {len(enhanced_narrative.strip()) if enhanced_narrative else 0}")
                return ""
            
            # Ensure proper encoding
            if isinstance(enhanced_narrative, str):
                enhanced_narrative = enhanced_narrative.encode('utf-8', errors='replace').decode('utf-8')
            
            st.success(f"✅ **Stage 2 Complete**: GPT-5 narrative synthesis successful - {len(enhanced_narrative):,} characters")
            st.info(f"📝 **First 300 chars**: {enhanced_narrative[:300]}...")
            return enhanced_narrative
            
        except Exception as api_error:
            st.error(f"❌ **GPT-5 API Error**: {api_error}")
            st.error(f"🔍 **Error Type**: {type(api_error).__name__}")
            st.error(f"🔍 **Error Details**: {str(api_error)}")
            
            # Fallback to GPT-4o
            st.warning("🔄 **Falling back to GPT-4o** for narrative synthesis...")
            try:
                response = gpt5_client.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_completion_tokens=4000
                )
                
                enhanced_narrative = response.choices[0].message.content
                if isinstance(enhanced_narrative, str):
                    enhanced_narrative = enhanced_narrative.encode('utf-8', errors='replace').decode('utf-8')
                
                st.success(f"✅ **Fallback Complete**: GPT-4o narrative synthesis successful - {len(enhanced_narrative):,} characters")
                return enhanced_narrative
                
            except Exception as fallback_error:
                st.error(f"❌ **Fallback GPT-4o also failed**: {fallback_error}")
                return ""
        
    except Exception as e:
        st.error(f"❌ **Unexpected Error in Stage 2**: {e}")
        st.info("🔍 **Debug Info**: Check console for detailed error messages")
        print(f"Stage 2 narrative synthesis error: {e}")
        return ""

def extract_rubric_scores_from_narrative(enhanced_narrative: str) -> Dict[int, int]:
    """Extract numerical scores from GPT-5 enhanced narrative using natural language patterns."""
    scores = {}
    
    if not enhanced_narrative:
        return scores
    
    try:
        import re
        
        # Look for the exact format: "Score: X/5" or "Score: X*/5" (provisional)
        score_patterns = [
            r'Score:\s*(\d+)(\*)?/5',                              # "Score: 3/5" or "Score: 3*/5"
            r'(\d+)\.\s+[^:]+Score:\s*(\d+)(\*)?/5',              # "1. Title Score: 3/5" or "3*/5"
            r'\*\*(\d+)\.[^:]*\*\*:.*?Score:\s*(\d+)(\*)?/5',     # "**1. Title**: ... Score: 3/5" or "3*/5"
        ]
        
        # First, try the primary format: "Score: X/5" or "Score: X*/5"
        primary_matches = re.findall(r'Score:\s*(\d+)(\*)?/5', enhanced_narrative, re.IGNORECASE)
        
        # If we have exactly 7 scores in order, use them
        if len(primary_matches) == 7:
            for i, (score_str, asterisk) in enumerate(primary_matches, 1):
                try:
                    score = int(score_str)
                    if 1 <= score <= 5:
                        scores[i] = score
                        # Store provisional flag if present
                        if asterisk:
                            if 'provisional_scores' not in globals():
                                globals()['provisional_scores'] = {}
                            globals()['provisional_scores'][i] = True
                except ValueError:
                    continue
        else:
            # Fallback: look for patterns with point numbers
            numbered_patterns = [
                r'(\d+)\.\s+[^:]*Score:\s*(\d+)/5',              # "1. Title Score: 3/5"
                r'\*\*(\d+)\.[^:]*\*\*:.*?Score:\s*(\d+)/5',     # "**1. Title**: ... Score: 3/5"
            ]
            
            for pattern in numbered_patterns:
                matches = re.findall(pattern, enhanced_narrative, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    try:
                        point_id = int(match[0])
                        score = int(match[1])
                        
                        # Validate score is in range
                        if 1 <= score <= 5 and 1 <= point_id <= 7:
                            scores[point_id] = score
                    except (ValueError, IndexError):
                        continue
        
        # If we don't have all 7 scores, try alternate patterns
        if len(scores) < 7:
            # Look for patterns in sections starting with numbers
            lines = enhanced_narrative.split('\n')
            current_point = None
            
            for line in lines:
                line = line.strip()
                
                # Check if line starts with rubric point number
                point_match = re.match(r'^(\d+)\.', line)
                if point_match:
                    current_point = int(point_match.group(1))
                    if current_point > 7:
                        current_point = None
                
                # Look for score in current section
                if current_point and current_point not in scores:
                    score_match = re.search(r'\b(\d+)(?:/5)?\b', line)
                    if score_match:
                        score = int(score_match.group(1))
                        if 1 <= score <= 5:
                            scores[current_point] = score
        
        # If still missing scores, default missing ones to 3 (Generally Reliable)
        for i in range(1, 8):
            if i not in scores:
                scores[i] = 3
                st.info(f"Defaulted rubric point {i} to score 3 (Generally Reliable)")
        
        st.info(f"✅ Extracted scores: {scores}")
        
        # Debug: Show what we're trying to extract from
        st.info(f"🔍 **Score extraction debug**: Looking for scores in {len(enhanced_narrative)} character narrative")
        if enhanced_narrative:
            st.info(f"📝 **Sample text**: {enhanced_narrative[:500]}...")
        
    except Exception as e:
        st.error(f"Error extracting scores from narrative: {e}")
        # Default all to 3 if extraction completely fails
        scores = {i: 3 for i in range(1, 8)}
    
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
            help="Upload a video of suturing technique for assessment (up to 2GB). MOV files supported."
        )
        
        # Simple file size check
        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.read()) / (1024 * 1024)
            uploaded_file.seek(0)  # Reset file pointer
            
            if file_size_mb > 2000:  # 2GB limit
                st.error("❌ **File too large**: Maximum size is 2GB. Please compress your video.")
                uploaded_file = None
        
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
                        st.markdown(f"**{point['pid']}. {point['title']}**")
                        st.markdown(f"*{point['what_you_assess']}*")
        
        # Analysis settings
        st.subheader("⚙️ Analysis Settings")
        fps = st.slider("Analysis FPS", 1.0, 5.0, 1.0, 0.5)
        batch_size = 6  # Fixed optimal value
        
        # Concurrency settings
        st.subheader("⚡ Concurrency Settings")
        max_concurrent_batches = st.slider(
            "Concurrent Batches", 
            1, 20, 
            20,  # Maximum performance setting
            step=1,
            help="Higher values = faster processing (requires OpenAI Tier 4+)"
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
            st.info(f"📊 **Batches processed**: {performance_metrics.get('successful_batches', 0)}/{total_batches}")
            
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
        
        # CRITICAL: Wait a moment for all async operations to complete
        time.sleep(2)
        
        # Verify that the transcript was populated
        st.info("🔍 **Verifying transcript collection...**")
        transcript_length = len(gpt4o_client.get_full_transcript())
        st.info(f"📝 **Transcript collected**: {transcript_length} characters")
        
        if transcript_length < 100:
            st.error("❌ **CRITICAL ERROR**: Transcript not populated by batch processing!")
            st.warning("This indicates the batch processing is not calling update_context() properly")
            
            # Try manual test to see if the client works at all
            st.info("🧪 **Testing client manually...**")
            try:
                test_narrative = "Test narrative for debugging"
                test_events = [{"type": "test", "timestamp": "00:00:00"}]
                gpt4o_client.update_context(test_narrative, test_events)
                test_length = len(gpt4o_client.get_full_transcript())
                st.info(f"🧪 **Manual test result**: {test_length} characters after manual update")
            except Exception as test_error:
                st.error(f"🧪 **Manual test failed**: {test_error}")
            
            return
        
        # If we get here, the transcript should be populated
        st.success("✅ **Transcript verification passed** - proceeding to Stage 2")
        
        # Get complete analysis using proven narrative building
        full_transcript = gpt4o_client.get_full_transcript()
        event_timeline = gpt4o_client.get_event_timeline()
        
        # DEBUG: Let's see exactly what we're getting
        st.info(f"🔍 **DEBUG**: Raw transcript type: {type(full_transcript)}")
        st.info(f"🔍 **DEBUG**: Raw transcript length: {len(full_transcript) if full_transcript else 0}")
        st.info(f"🔍 **DEBUG**: Transcript content: {repr(full_transcript[:500]) if full_transcript else 'None'}")
        
        # Also check the internal state of the client
        st.info(f"🔍 **DEBUG**: Client full_transcript list length: {len(gpt4o_client.full_transcript)}")
        if gpt4o_client.full_transcript:
            st.info(f"🔍 **DEBUG**: First transcript entry: {repr(gpt4o_client.full_transcript[0][:200]) if gpt4o_client.full_transcript[0] else 'None'}")
        
        # STAGE 1: GPT-4o has completed frame-by-frame analysis
        # STAGE 2: GPT-5 synthesizes ALL batch outputs into comprehensive narrative
        status_text.text("STAGE 2: Synthesizing complete video narrative with GPT-5...")
        
        # Ensure we have ALL batch outputs for synthesis
        if not full_transcript or len(full_transcript.strip()) < 100:
            st.error("❌ STAGE 1 FAILED: Insufficient GPT-4o batch analysis for synthesis")
            st.warning(f"Transcript content: {repr(full_transcript[:200]) if full_transcript else 'None'}")
            st.warning(f"Client internal state: {len(gpt4o_client.full_transcript)} transcript entries")
            return
        
        # Check that we have substantial content from multiple batches
        transcript_lines = full_transcript.split('\n')
        substantial_lines = [line for line in transcript_lines if len(line.strip()) > 50]
        
        if len(substantial_lines) < 5:
            st.error("❌ STAGE 1 FAILED: Insufficient batch analysis content for synthesis")
            st.warning(f"Only {len(substantial_lines)} substantial analysis lines found")
            st.warning(f"Sample lines: {substantial_lines[:3] if substantial_lines else 'None'}")
            return
        
        # Check for time diversity across batches
        time_indicators = ['00:', '01:', '02:', '03:', '04:', '05:', '06:', '07:', '08:', '09:', '10:']
        time_ranges_found = sum(1 for indicator in time_indicators if indicator in full_transcript)
        
        if time_ranges_found < 3:
            st.warning(f"⚠️ **Limited Time Coverage**: Only {time_ranges_found} time ranges detected - may indicate incomplete video analysis")
        
        st.info(f"📊 **STAGE 1 Complete**: {len(full_transcript):,} characters from {len(substantial_lines)} substantial analysis lines")
        st.info(f"📝 **Time Coverage**: {time_ranges_found} time ranges detected")
        st.info(f"📝 **Content Preview**: {full_transcript[:200]}...")
        
        enhanced_narrative = create_surgical_vop_narrative(
            full_transcript, 
            event_timeline, 
            api_key,
            st.session_state.selected_pattern,
            rubric_engine
        )
        
        # Debug: Check if enhanced narrative was generated
        if enhanced_narrative:
            st.success(f"✅ GPT-5 Enhanced Narrative Generated: {len(enhanced_narrative)} characters")
            st.info(f"First 200 chars: {enhanced_narrative[:200]}...")
        else:
            st.error("❌ GPT-5 Enhanced Narrative Generation FAILED")
            st.warning("This will cause PDF generation to fail")
        
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
                "pattern": st.session_state.selected_pattern,
                "fps": fps,
                "total_frames": len(frames),
                "duration": video_processor.duration
            },
            "performance_metrics": performance_metrics if max_concurrent_batches > 1 else None,
            "timestamp": datetime.now().isoformat(),
            "video_path": video_path  # Add video path for final image extraction
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
        
        # PDF report generation and download button
        if st.button("📄 Generate & Download PDF Report", type="primary"):
            try:
                from surgical_report_generator import SurgicalVOPReportGenerator
                
                report_generator = SurgicalVOPReportGenerator()
                report_filename = f"VOP_Assessment_{results['video_info']['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                
                with st.spinner("Generating PDF report..."):
                    report_path = report_generator.generate_vop_report(
                        results,
                        st.session_state.rubric_scores,
                        overall_result,
                        report_filename
                    )
                
                st.success(f"✅ PDF report generated successfully!")
                
                # Automatically trigger download
                with open(report_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=report_filename,
                        mime="application/pdf",
                        key="pdf_download"
                    )
                    
            except Exception as e:
                st.error(f"Error generating PDF report: {e}")
                st.error("Check console for detailed error information")

if __name__ == "__main__":
    main()
