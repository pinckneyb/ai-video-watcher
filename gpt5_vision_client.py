#!/usr/bin/env python3
"""
GPT-5 Vision Client for Surgical VOP Assessment
Handles both image analysis and flow interpretation using GPT-5
"""

import base64
import json
from typing import List, Dict, Any, Tuple
from openai import OpenAI

class GPT5VisionClient:
    """Client for GPT-5 vision capabilities in surgical assessment"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.frame_descriptions = []
        self.video_narrative = ""
        self.context_state = ""
    
    def pass1_surgical_description(self, frames: List[Dict], context_state: str = "") -> Tuple[str, List[Dict]]:
        """
        PASS 1: Describe frames in surgical terms relevant to suturing procedure
        Pure observation without assessment bias
        """
        base64_frames = [frame['base64'] for frame in frames]
        
        # Create surgical description prompt
        surgical_prompt = self._build_surgical_description_prompt(frames, context_state)
        
        # Prepare messages for GPT-5
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": surgical_prompt}
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame_data}", "detail": "high"}}
                    for frame_data in base64_frames
                ]
            }
        ]
        
        print(f"DEBUG: Message structure - text length: {len(surgical_prompt)}, images: {len(base64_frames)}")
        print(f"DEBUG: First image URL preview: data:image/jpeg;base64,{base64_frames[0][:50]}..." if base64_frames else "No images")
        
        try:
            print(f"DEBUG: Sending {len(frames)} frames to GPT-5 for surgical description")
            print(f"DEBUG: Frame timestamps: {frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s")
            print(f"DEBUG: Base64 data length: {len(base64_frames[0]) if base64_frames else 'None'}")
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=4000,
                reasoning_effort="medium"
            )
            
            frame_analysis = response.choices[0].message.content
            print(f"DEBUG: GPT-5 response length: {len(frame_analysis)}")
            print(f"DEBUG: GPT-5 response preview: {frame_analysis[:200]}...")
            
            # Limit individual description length to prevent overwhelming PASS 2
            max_description_length = 1500  # Reasonable limit per batch
            if len(frame_analysis) > max_description_length:
                print(f"WARNING: Description too long ({len(frame_analysis)} chars), truncating to {max_description_length}")
                frame_analysis = frame_analysis[:max_description_length] + "\n\n[TRUNCATED - too long for processing]"
            
            self.frame_descriptions.append({
                'timestamp_range': f"{frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s",
                'frame_count': len(frames),
                'analysis': frame_analysis
            })
            
            # Update context for next batch
            self.context_state = self._condense_context(frame_analysis, context_state)
            
            return frame_analysis, self.frame_descriptions
            
        except Exception as e:
            error_msg = f"Error in GPT-5 frame analysis: {e}"
            print(f"ERROR: {error_msg}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Frame count: {len(frames)}")
            print(f"ERROR: Base64 frames count: {len(base64_frames)}")
            
            # Add error description to track
            self.frame_descriptions.append({
                'timestamp_range': f"{frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s",
                'frame_count': len(frames),
                'analysis': f"ERROR: {error_msg}"
            })
            
            return error_msg, self.frame_descriptions
    
    def pass2_video_narrative(self, all_descriptions: List[Dict]) -> str:
        """
        PASS 2: Assemble descriptions into video narrative with flow analysis
        Focus on motion, connections, and temporal relationships
        """
        # Format all frame descriptions
        descriptions_text = "\n\n".join([
            f"TIMESTAMP: {desc['timestamp_range']}\nFRAMES: {desc['frame_count']}\nSURGICAL DESCRIPTION: {desc['analysis']}"
            for desc in all_descriptions
        ])
        
        # Check if descriptions are too long and truncate if necessary
        max_descriptions_length = 30000  # More reasonable limit for GPT-5 with 1500 char descriptions
        if len(descriptions_text) > max_descriptions_length:
            print(f"WARNING: Descriptions text too long ({len(descriptions_text)} chars), truncating to {max_descriptions_length}")
            descriptions_text = descriptions_text[:max_descriptions_length] + "\n\n[TRUNCATED - too long for processing]"
        
        narrative_prompt = self._build_video_narrative_prompt(descriptions_text)
        
        messages = [
            {
                "role": "user",
                "content": narrative_prompt
            }
        ]
        
        try:
            print(f"DEBUG: Creating video narrative from {len(all_descriptions)} frame descriptions")
            print(f"DEBUG: Descriptions preview: {descriptions_text[:500]}...")
            print(f"DEBUG: Full descriptions text length: {len(descriptions_text)}")
            print(f"DEBUG: Prompt length: {len(narrative_prompt)}")
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=6000,
                reasoning_effort="medium"  # Changed from "high" to "medium"
            )
            
            self.video_narrative = response.choices[0].message.content
            print(f"DEBUG: Video narrative length: {len(self.video_narrative)}")
            print(f"DEBUG: Video narrative preview: {self.video_narrative[:200]}...")
            
            if not self.video_narrative or len(self.video_narrative.strip()) == 0:
                print("ERROR: Video narrative is empty!")
                print(f"DEBUG: Raw response: {response}")
                return "Error: Video narrative generation returned empty response"
            
            return self.video_narrative
            
        except Exception as e:
            error_msg = f"Error in GPT-5 video narrative: {e}"
            print(f"ERROR: {error_msg}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Descriptions count: {len(all_descriptions)}")
            print(f"ERROR: Descriptions text length: {len(descriptions_text)}")
            return error_msg
    
    def pass3_rubric_assessment(self, pattern_id: str, rubric_engine) -> Dict[str, Any]:
        """
        PASS 3: Compare video narrative to rubric for final assessment
        Focus on scoring against specific criteria
        """
        if not self.video_narrative:
            return {"error": "No video narrative available for assessment"}
        
        assessment_prompt = self._build_rubric_assessment_prompt(self.video_narrative, pattern_id, rubric_engine)
        
        messages = [
            {
                "role": "user",
                "content": assessment_prompt
            }
        ]
        
        try:
            print(f"DEBUG: Starting rubric assessment for pattern: {pattern_id}")
            print(f"DEBUG: Video narrative length: {len(self.video_narrative)}")
            print(f"DEBUG: Video narrative preview: {self.video_narrative[:200]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=8000,
                reasoning_effort="high"
            )
            
            assessment_response = response.choices[0].message.content
            print(f"DEBUG: Assessment response length: {len(assessment_response)}")
            print(f"DEBUG: Assessment response preview: {assessment_response[:200]}...")
            
            parsed_response = self._parse_assessment_response(assessment_response, rubric_engine)
            print(f"DEBUG: Parsed assessment success: {parsed_response.get('success', False)}")
            
            return parsed_response
            
        except Exception as e:
            error_msg = f"Error in GPT-5 rubric assessment: {e}"
            print(f"ERROR: {error_msg}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Pattern ID: {pattern_id}")
            print(f"ERROR: Video narrative available: {bool(self.video_narrative)}")
            return {"error": error_msg, "success": False}
    
    def analyze_flow_and_technique(self, all_descriptions: List[Dict], profile: Dict[str, Any], 
                                 pattern_id: str, rubric_engine) -> Dict[str, Any]:
        """
        Use GPT-5 to analyze video flow and apply rubric assessment
        This is the second GPT-5 call for holistic video assessment
        """
        # Create flow analysis prompt
        flow_prompt = self._build_flow_analysis_prompt(all_descriptions, profile, pattern_id, rubric_engine)
        
        messages = [
            {
                "role": "user",
                "content": flow_prompt
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=8000,
                reasoning_effort="high"
            )
            
            return self._parse_flow_response(response.choices[0].message.content, rubric_engine)
            
        except Exception as e:
            print(f"Error in GPT-5 flow analysis: {e}")
            return {"error": f"Flow analysis failed: {e}"}
    
    def _build_surgical_description_prompt(self, frames: List[Dict], context_state: str) -> str:
        """Build prompt for PASS 1: Surgical description of frames"""
        
        timestamp_range = f"{frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s"
        
        prompt = f"""You are analyzing still images from a surgical suturing procedure. These are {len(frames)} consecutive frames from {timestamp_range}.

CONTEXT SO FAR: {context_state if context_state else 'Beginning of video analysis'}

ANALYSIS REQUIREMENTS:
Focus ONLY on surgically critical details:

1. HAND POSITIONS: Where are hands positioned? What instruments are held?
2. NEEDLE HANDLING: How is the needle grasped and positioned?
3. TISSUE INTERACTION: What tissue is being manipulated? How are wound edges handled?
4. SPATIAL RELATIONSHIPS: How are hands positioned relative to each other and the field?

OUTPUT: Concise, factual observations only. No color commentary. 150-300 words maximum.

Analyze these {len(frames)} frames:"""
        
        return prompt
    
    def _build_video_narrative_prompt(self, descriptions_text: str) -> str:
        """Build prompt for PASS 2: Video narrative assembly"""
        
        prompt = f"""You are analyzing surgical frame descriptions to create a video narrative. These are still images from a suturing procedure.

FRAME DESCRIPTIONS:
{descriptions_text}

TASK: Create a comprehensive narrative describing the complete surgical procedure.

Focus on:
1. How the procedure progresses over time
2. What actions occurred between frames  
3. How hand positions and techniques evolved
4. The overall flow and rhythm of the suturing

OUTPUT: Write a detailed chronological narrative that connects the frame observations into a coherent story of the surgical technique.

Create the video narrative:"""
        
        return prompt
    
    def _build_rubric_assessment_prompt(self, video_narrative: str, pattern_id: str, rubric_engine) -> str:
        """Build prompt for PASS 3: Rubric assessment"""
        
        # Get rubric data
        rubric_data = rubric_engine.get_pattern_rubric(pattern_id)
        if not rubric_data:
            return "Error: Could not load rubric data"
        
        prompt = f"""YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE. You are training surgeons who will operate on real patients. Assume EVERY technique has flaws until proven otherwise.

SURGICAL ASSESSMENT: {rubric_data['display_name']} Suturing Technique

You are assessing a surgical video based on this comprehensive narrative:

VIDEO NARRATIVE:
{video_narrative}

ASSESSMENT TASK:
Evaluate this surgical performance against the specific rubric criteria. Base your assessment on the complete video narrative, not individual observations.

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

RUBRIC ASSESSMENT:
For each rubric point, write ONE OR TWO SENTENCES that:
- Describe what you observed in the technique based on the video narrative
- Explain why the performance earned its score
- NO timestamps, NO references to inability to judge
- The AI must judge from the evidence available

RUBRIC POINTS:
{json.dumps(rubric_data['points'], indent=2)}

SUMMATIVE ASSESSMENT:
Write a summative paragraph that:
- Provides entirely actionable critiques from holistic review
- NO reprise of individual rubric assessments
- NO timestamps, NO uncertainty about visibility
- Must be useful observations about technique execution
- Focus on what the video narrative reveals about surgical skill

MANDATORY SCORING:
RUBRIC_SCORES_START
1: X
2: X
3: X
4: X
5: X
6: X
7: X
RUBRIC_SCORES_END

Provide your complete assessment:"""
        
        return prompt
    
    def _build_flow_analysis_prompt(self, all_descriptions: List[Dict], profile: Dict[str, Any], 
                                   pattern_id: str, rubric_engine) -> str:
        """Build prompt for holistic flow and technique analysis"""
        
        # Get rubric data
        rubric_data = rubric_engine.get_pattern_rubric(pattern_id)
        if not rubric_data:
            return "Error: Could not load rubric data"
        
        # Format all frame descriptions
        descriptions_text = "\n\n".join([
            f"TIMESTAMP: {desc['timestamp_range']}\nFRAMES: {desc['frame_count']}\nDESCRIPTION: {desc['analysis']}"
            for desc in all_descriptions
        ])
        
        prompt = f"""YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE. You are training surgeons who will operate on real patients. Assume EVERY technique has flaws until proven otherwise.

SURGICAL ASSESSMENT: {rubric_data['display_name']} Suturing Technique

You are now assessing a complete surgical video based on detailed frame-by-frame descriptions. These descriptions are from still images captured throughout the procedure. Your task is to:

1. ANALYZE THE VIDEO FLOW: Use the frame descriptions to understand the overall flow and motion of the surgical procedure
2. INTUIT MISSING ACTIONS: Fill in the gaps between frames to understand the complete technique
3. ASSESS TECHNIQUE QUALITY: Apply the rubric criteria to evaluate the surgical performance

FRAME-BY-FRAME DESCRIPTIONS:
{descriptions_text}

VIDEO ASSESSMENT INSTRUCTIONS:
- Analyze the sequence of descriptions to understand the flow of the procedure
- Intuit what actions occurred between the captured frames
- Assess the overall technique quality and consistency
- Look for patterns, improvements, or deteriorations over time
- Consider the spatial and temporal relationships between actions

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

RUBRIC ASSESSMENT:
For each rubric point, write ONE OR TWO SENTENCES that:
- Describe what you observed in the technique and flow
- Explain why the performance earned its score
- NO timestamps, NO references to inability to judge
- The AI must judge from the evidence available

RUBRIC POINTS:
{json.dumps(rubric_data['points'], indent=2)}

SUMMATIVE ASSESSMENT:
Write a summative paragraph that:
- Provides entirely actionable critiques from holistic review
- NO reprise of individual rubric assessments
- NO timestamps, NO uncertainty about visibility
- Must be useful observations about technique flow and execution
- Focus on what the video analysis reveals about surgical skill

MANDATORY SCORING:
RUBRIC_SCORES_START
1: X
2: X
3: X
4: X
5: X
6: X
7: X
RUBRIC_SCORES_END

Provide your complete assessment:"""
        
        return prompt
    
    def _condense_context(self, new_analysis: str, existing_context: str) -> str:
        """Condense context for next batch analysis"""
        if not existing_context:
            return new_analysis[:500] + "..." if len(new_analysis) > 500 else new_analysis
        
        # Combine and condense
        combined = f"{existing_context}\n\n{new_analysis}"
        if len(combined) > 1000:
            return combined[-1000:] + "..."
        return combined
    
    def _parse_flow_response(self, response: str, rubric_engine) -> Dict[str, Any]:
        """Parse GPT-5 flow analysis response"""
        try:
            # Extract rubric scores
            scores = {}
            if "RUBRIC_SCORES_START" in response and "RUBRIC_SCORES_END" in response:
                score_section = response.split("RUBRIC_SCORES_START")[1].split("RUBRIC_SCORES_END")[0]
                for line in score_section.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                rubric_id = int(parts[0].strip())
                                score = int(parts[1].strip())
                                scores[rubric_id] = score
                            except ValueError:
                                continue
            
            # Extract summative assessment
            summative_start = response.find("SUMMATIVE ASSESSMENT:") if "SUMMATIVE ASSESSMENT:" in response else response.find("Summative Assessment:")
            if summative_start != -1:
                summative = response[summative_start:].replace("SUMMATIVE ASSESSMENT:", "").replace("Summative Assessment:", "").strip()
            else:
                summative = response
            
            return {
                "rubric_scores": scores,
                "summative_assessment": summative,
                "full_response": response,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Failed to parse response: {e}",
                "full_response": response,
                "success": False
            }
    
    def get_full_transcript(self) -> str:
        """Get complete analysis transcript"""
        return "\n\n".join([desc['analysis'] for desc in self.frame_descriptions])
    
    def get_event_timeline(self) -> List[Dict]:
        """Get timeline of analysis events"""
        return [
            {
                "timestamp": desc['timestamp_range'],
                "event": f"Frame analysis ({desc['frame_count']} frames)",
                "description": desc['analysis'][:200] + "..." if len(desc['analysis']) > 200 else desc['analysis']
            }
            for desc in self.frame_descriptions
        ]
    
    def _parse_assessment_response(self, response: str, rubric_engine) -> Dict[str, Any]:
        """Parse GPT-5 assessment response"""
        try:
            # Extract rubric scores
            scores = {}
            if "RUBRIC_SCORES_START" in response and "RUBRIC_SCORES_END" in response:
                score_section = response.split("RUBRIC_SCORES_START")[1].split("RUBRIC_SCORES_END")[0]
                for line in score_section.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 1)
                        if len(parts) == 2:
                            try:
                                rubric_id = int(parts[0].strip())
                                score = int(parts[1].strip())
                                scores[rubric_id] = score
                            except ValueError:
                                continue
            
            # Extract summative assessment
            summative_start = response.find("SUMMATIVE ASSESSMENT:") if "SUMMATIVE ASSESSMENT:" in response else response.find("Summative Assessment:")
            if summative_start != -1:
                summative = response[summative_start:].replace("SUMMATIVE ASSESSMENT:", "").replace("Summative Assessment:", "").strip()
            else:
                summative = response
            
            return {
                "rubric_scores": scores,
                "summative_assessment": summative,
                "full_response": response,
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Failed to parse response: {e}",
                "full_response": response,
                "success": False
            }
    
    def clear_analysis(self):
        """Clear current analysis data"""
        self.frame_descriptions = []
        self.video_narrative = ""
        self.context_state = ""
