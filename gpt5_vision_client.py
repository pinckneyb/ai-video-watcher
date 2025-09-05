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
        
        # Error tracking and monitoring
        self.error_stats = {
            'pass1_truncations': 0,
            'pass1_api_errors': 0,
            'pass1_total_batches': 0,
            'pass1_successful_batches': 0,
            'pass1_char_stats': [],  # Track character lengths
            'pass2_truncations': 0,
            'pass2_api_errors': 0,
            'pass2_total_attempts': 0,
            'pass2_successful_attempts': 0,
            'pass2_input_char_stats': [],
            'pass2_output_char_stats': [],
            'pass3_api_errors': 0,
            'pass3_total_attempts': 0,
            'pass3_successful_attempts': 0,
            'pass3_input_char_stats': [],
            'pass3_output_char_stats': [],
            'context_length_errors': 0,
            'rate_limit_errors': 0,
            'quota_exceeded_errors': 0,
            'other_api_errors': 0
        }
    
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
        
        # Track batch processing
        self.error_stats['pass1_total_batches'] += 1
        
        try:
            print(f"DEBUG: Sending {len(frames)} frames to GPT-5 for surgical description")
            print(f"DEBUG: Frame timestamps: {frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s")
            print(f"DEBUG: Base64 data length: {len(base64_frames[0]) if base64_frames else 'None'}")
            print(f"📊 PASS 1 BATCH {self.error_stats['pass1_total_batches']}: Processing {len(frames)} frames")
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=4000,
                reasoning_effort="medium"
            )
            
            frame_analysis = response.choices[0].message.content
            print(f"DEBUG: GPT-5 response length: {len(frame_analysis)}")
            print(f"DEBUG: GPT-5 response preview: {frame_analysis[:200]}...")
            
            # INCREASED LIMIT: Allow longer descriptions for better quality
            max_description_length = 3000  # Increased from 1500 for better quality
            
            # Track character statistics
            original_length = len(frame_analysis)
            self.error_stats['pass1_char_stats'].append(original_length)
            
            # Check for truncation
            if len(frame_analysis) > max_description_length:
                self.error_stats['pass1_truncations'] += 1
                print(f"⚠️ PASS 1 TRUNCATION: Description {original_length} chars, truncating to {max_description_length}")
                print(f"📊 TRUNCATION STATS: {self.error_stats['pass1_truncations']} of {self.error_stats['pass1_total_batches']} batches truncated")
                frame_analysis = frame_analysis[:max_description_length] + "\n\n[TRUNCATED - exceeded 3000 char limit for batch processing]"
            
            # Track successful batch
            self.error_stats['pass1_successful_batches'] += 1
            
            self.frame_descriptions.append({
                'timestamp_range': f"{frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s",
                'frame_count': len(frames),
                'analysis': frame_analysis,
                'original_char_length': original_length,
                'final_char_length': len(frame_analysis),
                'was_truncated': original_length > max_description_length
            })
            
            # Update context for next batch
            self.context_state = self._condense_context(frame_analysis, context_state)
            
            print(f"✅ PASS 1 BATCH {self.error_stats['pass1_total_batches']} SUCCESS: {len(frame_analysis)} chars generated")
            
            return frame_analysis, self.frame_descriptions
            
        except Exception as e:
            # Track API errors with detailed categorization
            self.error_stats['pass1_api_errors'] += 1
            error_msg = f"Error in GPT-5 frame analysis: {e}"
            
            # Categorize error types
            error_str = str(e).lower()
            if "context_length_exceeded" in error_str or "maximum context length" in error_str:
                self.error_stats['context_length_errors'] += 1
                print(f"🚨 PASS 1 CONTEXT LENGTH ERROR: {e}")
                print(f"📊 Context length errors: {self.error_stats['context_length_errors']}")
            elif "rate_limit" in error_str:
                self.error_stats['rate_limit_errors'] += 1
                print(f"⏰ PASS 1 RATE LIMIT ERROR: {e}")
                print(f"📊 Rate limit errors: {self.error_stats['rate_limit_errors']}")
            elif "quota_exceeded" in error_str or "insufficient_quota" in error_str:
                self.error_stats['quota_exceeded_errors'] += 1
                print(f"💳 PASS 1 QUOTA ERROR: {e}")
                print(f"📊 Quota errors: {self.error_stats['quota_exceeded_errors']}")
            else:
                self.error_stats['other_api_errors'] += 1
                print(f"❌ PASS 1 OTHER ERROR: {e}")
                print(f"📊 Other API errors: {self.error_stats['other_api_errors']}")
            
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Frame count: {len(frames)}")
            print(f"ERROR: Base64 frames count: {len(base64_frames)}")
            print(f"📊 PASS 1 ERROR SUMMARY: {self.error_stats['pass1_api_errors']} errors of {self.error_stats['pass1_total_batches']} total batches")
            
            # Add error description to track
            self.frame_descriptions.append({
                'timestamp_range': f"{frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s",
                'frame_count': len(frames),
                'analysis': f"ERROR: {error_msg}",
                'original_char_length': 0,
                'final_char_length': len(error_msg),
                'was_truncated': False
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
        
        # INCREASED LIMIT: Allow much longer input for Pass 2
        max_descriptions_length = 100000  # Increased from 30000 for better quality
        
        # Track Pass 2 statistics
        self.error_stats['pass2_total_attempts'] += 1
        original_descriptions_length = len(descriptions_text)
        self.error_stats['pass2_input_char_stats'].append(original_descriptions_length)
        
        # Check for truncation
        if len(descriptions_text) > max_descriptions_length:
            self.error_stats['pass2_truncations'] += 1
            print(f"⚠️ PASS 2 TRUNCATION: Input {original_descriptions_length:,} chars, truncating to {max_descriptions_length:,}")
            print(f"📊 PASS 2 TRUNCATION STATS: {self.error_stats['pass2_truncations']} of {self.error_stats['pass2_total_attempts']} attempts truncated")
            descriptions_text = descriptions_text[:max_descriptions_length] + "\n\n[TRUNCATED - exceeded 100,000 char limit for narrative processing]"
        
        print(f"📊 PASS 2 INPUT ANALYSIS: {original_descriptions_length:,} chars from {len(all_descriptions)} batch descriptions")
        
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
            narrative_length = len(self.video_narrative)
            
            # Track Pass 2 success and output statistics
            self.error_stats['pass2_successful_attempts'] += 1
            self.error_stats['pass2_output_char_stats'].append(narrative_length)
            
            print(f"✅ PASS 2 SUCCESS: Generated {narrative_length:,} character narrative")
            print(f"DEBUG: Video narrative preview: {self.video_narrative[:200]}...")
            print(f"📊 PASS 2 STATS: Success rate {self.error_stats['pass2_successful_attempts']}/{self.error_stats['pass2_total_attempts']}")
            
            if not self.video_narrative or len(self.video_narrative.strip()) == 0:
                print("ERROR: Video narrative is empty!")
                print(f"DEBUG: Raw response: {response}")
                return "Error: Video narrative generation returned empty response"
            
            return self.video_narrative
            
        except Exception as e:
            # Track Pass 2 API errors with detailed categorization
            self.error_stats['pass2_api_errors'] += 1
            error_msg = f"Error in GPT-5 video narrative: {e}"
            
            # Categorize error types
            error_str = str(e).lower()
            if "context_length_exceeded" in error_str or "maximum context length" in error_str:
                self.error_stats['context_length_errors'] += 1
                print(f"🚨 PASS 2 CONTEXT LENGTH ERROR: Input was {len(descriptions_text):,} chars")
                print(f"🚨 This suggests our 100,000 char limit may still be too high for GPT-5")
            elif "rate_limit" in error_str:
                self.error_stats['rate_limit_errors'] += 1
                print(f"⏰ PASS 2 RATE LIMIT ERROR: {e}")
            elif "quota_exceeded" in error_str or "insufficient_quota" in error_str:
                self.error_stats['quota_exceeded_errors'] += 1
                print(f"💳 PASS 2 QUOTA ERROR: {e}")
            else:
                self.error_stats['other_api_errors'] += 1
                print(f"❌ PASS 2 OTHER ERROR: {e}")
            
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Descriptions count: {len(all_descriptions)}")
            print(f"ERROR: Descriptions text length: {len(descriptions_text):,}")
            print(f"📊 PASS 2 ERROR SUMMARY: {self.error_stats['pass2_api_errors']} errors of {self.error_stats['pass2_total_attempts']} attempts")
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
        
        # Track Pass 3 processing
        self.error_stats['pass3_total_attempts'] += 1
        narrative_length = len(self.video_narrative)
        self.error_stats['pass3_input_char_stats'].append(narrative_length)
        
        try:
            print(f"DEBUG: Starting rubric assessment for pattern: {pattern_id}")
            print(f"📊 PASS 3 INPUT: {narrative_length:,} character narrative for assessment")
            print(f"DEBUG: Video narrative preview: {self.video_narrative[:200]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=8000,
                reasoning_effort="high"
            )
            
            assessment_response = response.choices[0].message.content
            response_length = len(assessment_response)
            
            # Track Pass 3 success and output statistics
            self.error_stats['pass3_successful_attempts'] += 1
            self.error_stats['pass3_output_char_stats'].append(response_length)
            
            print(f"✅ PASS 3 SUCCESS: Generated {response_length:,} character assessment")
            print(f"DEBUG: Assessment response preview: {assessment_response[:200]}...")
            
            parsed_response = self._parse_assessment_response(assessment_response, rubric_engine)
            parse_success = parsed_response.get('success', False)
            print(f"📊 PASS 3 PARSE SUCCESS: {parse_success}")
            print(f"📊 PASS 3 STATS: Success rate {self.error_stats['pass3_successful_attempts']}/{self.error_stats['pass3_total_attempts']}")
            
            return parsed_response
            
        except Exception as e:
            # Track Pass 3 API errors with detailed categorization
            self.error_stats['pass3_api_errors'] += 1
            error_msg = f"Error in GPT-5 rubric assessment: {e}"
            
            # Categorize error types
            error_str = str(e).lower()
            if "context_length_exceeded" in error_str or "maximum context length" in error_str:
                self.error_stats['context_length_errors'] += 1
                print(f"🚨 PASS 3 CONTEXT LENGTH ERROR: Input was {len(self.video_narrative):,} chars")
                print(f"🚨 This suggests the narrative from Pass 2 may be too long for Pass 3")
            elif "rate_limit" in error_str:
                self.error_stats['rate_limit_errors'] += 1
                print(f"⏰ PASS 3 RATE LIMIT ERROR: {e}")
            elif "quota_exceeded" in error_str or "insufficient_quota" in error_str:
                self.error_stats['quota_exceeded_errors'] += 1
                print(f"💳 PASS 3 QUOTA ERROR: {e}")
            else:
                self.error_stats['other_api_errors'] += 1
                print(f"❌ PASS 3 OTHER ERROR: {e}")
            
            print(f"ERROR: Exception type: {type(e).__name__}")
            print(f"ERROR: Pattern ID: {pattern_id}")
            print(f"ERROR: Video narrative available: {bool(self.video_narrative)}")
            print(f"ERROR: Video narrative length: {len(self.video_narrative):,} chars")
            print(f"📊 PASS 3 ERROR SUMMARY: {self.error_stats['pass3_api_errors']} errors of {self.error_stats['pass3_total_attempts']} attempts")
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

SUMMATIVE_ASSESSMENT_START
Write a concise, holistic assessment (2-3 paragraphs maximum) that:
- Focuses on MOTION and FLOW of the overall procedure
- Discusses rhythm, efficiency, and surgical confidence
- Comments on hand coordination and instrument management
- Addresses overall procedural competence and areas for improvement
- NEVER repeats individual rubric point assessments
- NO timestamps, NO uncertainty language
- Break into paragraphs if longer than 4 sentences
- Keep concise and actionable
SUMMATIVE_ASSESSMENT_END

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

SUMMATIVE_ASSESSMENT_START
Write a concise, holistic assessment (2-3 paragraphs maximum) that:
- Focuses on MOTION and FLOW of the overall procedure
- Discusses rhythm, efficiency, and surgical confidence
- Comments on hand coordination and instrument management
- Addresses overall procedural competence and areas for improvement
- NEVER repeats individual rubric point assessments
- NO timestamps, NO uncertainty language
- Break into paragraphs if longer than 4 sentences
- Keep concise and actionable
SUMMATIVE_ASSESSMENT_END

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
            
            # Extract summative assessment using new delimiters
            summative = ""
            if "SUMMATIVE_ASSESSMENT_START" in response and "SUMMATIVE_ASSESSMENT_END" in response:
                # Extract content between the new delimiters
                summative = response.split("SUMMATIVE_ASSESSMENT_START")[1].split("SUMMATIVE_ASSESSMENT_END")[0].strip()
            elif "RUBRIC_SCORES_END" in response:
                # Fallback: Look for content after scores
                after_scores = response.split("RUBRIC_SCORES_END")[1].strip()
                
                # Look for "Summative assessment:" or similar markers
                summative_markers = ["Summative assessment:", "SUMMATIVE ASSESSMENT:", "Summative Assessment:"]
                for marker in summative_markers:
                    if marker in after_scores:
                        summative = after_scores.split(marker)[1].strip()
                        break
                
                # If no marker found, but we have content after scores, use that
                if not summative and after_scores:
                    summative = after_scores
            else:
                # Final fallback
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
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics and analysis metrics"""
        stats = self.error_stats.copy()
        
        # Calculate success rates
        if stats['pass1_total_batches'] > 0:
            stats['pass1_success_rate'] = (stats['pass1_successful_batches'] / stats['pass1_total_batches']) * 100
            stats['pass1_truncation_rate'] = (stats['pass1_truncations'] / stats['pass1_total_batches']) * 100
        else:
            stats['pass1_success_rate'] = 0
            stats['pass1_truncation_rate'] = 0
            
        if stats['pass2_total_attempts'] > 0:
            stats['pass2_success_rate'] = (stats['pass2_successful_attempts'] / stats['pass2_total_attempts']) * 100
            stats['pass2_truncation_rate'] = (stats['pass2_truncations'] / stats['pass2_total_attempts']) * 100
        else:
            stats['pass2_success_rate'] = 0
            stats['pass2_truncation_rate'] = 0
            
        if stats['pass3_total_attempts'] > 0:
            stats['pass3_success_rate'] = (stats['pass3_successful_attempts'] / stats['pass3_total_attempts']) * 100
        else:
            stats['pass3_success_rate'] = 0
        
        # Calculate character statistics
        if stats['pass1_char_stats']:
            stats['pass1_avg_chars'] = sum(stats['pass1_char_stats']) / len(stats['pass1_char_stats'])
            stats['pass1_max_chars'] = max(stats['pass1_char_stats'])
            stats['pass1_min_chars'] = min(stats['pass1_char_stats'])
        
        if stats['pass2_input_char_stats']:
            stats['pass2_avg_input_chars'] = sum(stats['pass2_input_char_stats']) / len(stats['pass2_input_char_stats'])
            stats['pass2_max_input_chars'] = max(stats['pass2_input_char_stats'])
        
        if stats['pass2_output_char_stats']:
            stats['pass2_avg_output_chars'] = sum(stats['pass2_output_char_stats']) / len(stats['pass2_output_char_stats'])
        
        if stats['pass3_input_char_stats']:
            stats['pass3_avg_input_chars'] = sum(stats['pass3_input_char_stats']) / len(stats['pass3_input_char_stats'])
        
        if stats['pass3_output_char_stats']:
            stats['pass3_avg_output_chars'] = sum(stats['pass3_output_char_stats']) / len(stats['pass3_output_char_stats'])
        
        # Total error counts
        stats['total_api_errors'] = (
            stats['pass1_api_errors'] + 
            stats['pass2_api_errors'] + 
            stats['pass3_api_errors']
        )
        
        stats['total_truncations'] = stats['pass1_truncations'] + stats['pass2_truncations']
        
        return stats
    
    def reset_error_statistics(self):
        """Reset all error statistics - useful for testing different configurations"""
        self.error_stats = {
            'pass1_truncations': 0,
            'pass1_api_errors': 0,
            'pass1_total_batches': 0,
            'pass1_successful_batches': 0,
            'pass1_char_stats': [],
            'pass2_truncations': 0,
            'pass2_api_errors': 0,
            'pass2_total_attempts': 0,
            'pass2_successful_attempts': 0,
            'pass2_input_char_stats': [],
            'pass2_output_char_stats': [],
            'pass3_api_errors': 0,
            'pass3_total_attempts': 0,
            'pass3_successful_attempts': 0,
            'pass3_input_char_stats': [],
            'pass3_output_char_stats': [],
            'context_length_errors': 0,
            'rate_limit_errors': 0,
            'quota_exceeded_errors': 0,
            'other_api_errors': 0
        }
        print("🔄 Error statistics reset - ready for new analysis")
