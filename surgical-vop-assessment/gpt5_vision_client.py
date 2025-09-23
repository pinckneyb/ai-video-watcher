#!/usr/bin/env python3
"""
GPT-5 Vision Client for Surgical VOP Assessment
Handles both image analysis and flow interpretation using GPT-5
"""

import base64
import json
import math
import re
import time
import gc
from typing import List, Dict, Any, Tuple
from openai import OpenAI

class GPT5VisionClient:
    """Client for GPT-5 vision capabilities in surgical assessment"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.frame_descriptions = []
        self.video_narrative = ""
        self.context_state = ""
        self.current_pattern = None  # 'simple_interrupted', 'subcuticular', etc.
        
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
        

    def pass1_surgical_description(self, frames: List[Dict], context_state: str = "", model: str = "gpt-5", reasoning_level: str = "low", verbosity_level: str = "medium") -> Tuple[str, List[Dict]]:
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
        
        # Robust retry logic for API calls
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"DEBUG: Sending {len(frames)} frames to GPT-5 for surgical description (attempt {attempt + 1}/{max_retries})")
                print(f"DEBUG: Frame timestamps: {frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s")
                print(f"DEBUG: Base64 data length: {len(base64_frames[0]) if base64_frames else 'None'}")
                print(f"📊 PASS 1 BATCH {self.error_stats['pass1_total_batches']}: Processing {len(frames)} frames")
                
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=4000,
                    reasoning_effort=reasoning_level,
                    verbosity=verbosity_level
                )
                
                # Success - break out of retry loop
                break
                
            except Exception as retry_error:
                error_str = str(retry_error).lower()
                
                # Check for quota exceeded - fast fail for batch processing
                if "quota_exceeded" in error_str or "insufficient_quota" in error_str:
                    print(f"💳 PASS 1 QUOTA EXCEEDED - FAST FAIL: {retry_error}")
                    raise Exception(f"QUOTA_EXCEEDED_FAST_FAIL: {retry_error}")
                
                is_retryable = any(keyword in error_str for keyword in [
                    'rate_limit', 'timeout', 'connection', 'server_error', 'internal_error',
                    'service_unavailable', 'temporary', 'throttl'
                ])
                
                if attempt < max_retries - 1 and is_retryable:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏰ Retryable error (attempt {attempt + 1}/{max_retries}): {retry_error}")
                    print(f"⏰ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Non-retryable error or max retries reached
                    raise retry_error
        
        try:
            
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
            
            # CRITICAL: Clear base64 data from frames immediately to prevent memory exhaustion
            for frame in frames:
                if 'base64' in frame:
                    frame['base64'] = None  # Clear to free memory
                    
            # Clear local base64_frames list
            base64_frames.clear()
            print(f"🧹 MEMORY CLEANUP: Cleared base64 data for {len(frames)} frames")
            
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
    
    def pass2_video_narrative(self, all_descriptions: List[Dict], model: str = "gpt-5", reasoning_level: str = "medium", verbosity_level: str = "high") -> str:
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
            
            # Robust retry logic for Pass 2 API calls
            max_retries = 5 
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    print(f"API call attempt {attempt + 1}/{max_retries}")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=12000,  # Doubled from 6000 to capture more detail
                        reasoning_effort=reasoning_level,
                        verbosity=verbosity_level
                    )
                    # Success - break out of retry loop
                    break
                    
                except Exception as retry_error:
                    error_str = str(retry_error).lower()
                    
                    # Check for quota exceeded - fast fail for batch processing
                    if "quota_exceeded" in error_str or "insufficient_quota" in error_str:
                        print(f"💳 PASS 2 QUOTA EXCEEDED - FAST FAIL: {retry_error}")
                        raise Exception(f"QUOTA_EXCEEDED_FAST_FAIL: {retry_error}")
                    
                    # Check for context length exceeded - try adaptive chunking
                    if "context_length_exceeded" in error_str or "maximum context length" in error_str:
                        print(f"📏 PASS 2 CONTEXT LENGTH EXCEEDED - ATTEMPTING ADAPTIVE CHUNKING")
                        return self._adaptive_chunking_pass2(all_descriptions, model, reasoning_level, verbosity_level)
                    
                    is_retryable = any(keyword in error_str for keyword in [
                        'rate_limit', 'timeout', 'connection', 'server_error', 'internal_error',
                        'service_unavailable', 'temporary', 'throttl'
                    ])
                    
                    if attempt < max_retries - 1 and is_retryable:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⏰ Pass 2 retryable error (attempt {attempt + 1}/{max_retries}): {retry_error}")
                        print(f"⏰ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Non-retryable error or max retries reached
                        raise retry_error
            
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
    
    def pass3_rubric_assessment(self, pattern_id: str, rubric_engine, final_product_image=None, model: str = "gpt-5", reasoning_level: str = "high", verbosity_level: str = "medium") -> Dict[str, Any]:
        """
        PASS 3: Compare video narrative to rubric for final assessment
        Focus on scoring against specific criteria
        """
        if not self.video_narrative:
            return {"error": "No video narrative available for assessment"}
        
        assessment_prompt = self._build_rubric_assessment_prompt(self.video_narrative, pattern_id, rubric_engine, final_product_image)
        
        # Prepare messages with or without final product image
        if final_product_image:
            # Convert PIL image to base64
            import base64
            from io import BytesIO
            buffered = BytesIO()
            final_product_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": assessment_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}", "detail": "high"}}
                    ]
                }
            ]
        else:
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
            
            # Robust retry logic for Pass 3 API calls
            max_retries = 5
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    print(f"🔄 Pass 3 API call attempt {attempt + 1}/{max_retries}")
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_completion_tokens=10000,  # Increased to handle longer narrative
                        reasoning_effort=reasoning_level,
                        verbosity=verbosity_level
                    )
                    # Success - break out of retry loop
                    break
                    
                except Exception as retry_error:
                    error_str = str(retry_error).lower()
                    
                    # Check for quota exceeded - fast fail for batch processing
                    if "quota_exceeded" in error_str or "insufficient_quota" in error_str:
                        print(f"💳 PASS 3 QUOTA EXCEEDED - FAST FAIL: {retry_error}")
                        raise Exception(f"QUOTA_EXCEEDED_FAST_FAIL: {retry_error}")
                    
                    is_retryable = any(keyword in error_str for keyword in [
                        'rate_limit', 'timeout', 'connection', 'server_error', 'internal_error',
                        'service_unavailable', 'temporary', 'throttl'
                    ])
                    
                    if attempt < max_retries - 1 and is_retryable:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        print(f"⏰ Pass 3 retryable error (attempt {attempt + 1}/{max_retries}): {retry_error}")
                        print(f"⏰ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    else:
                        # Non-retryable error or max retries reached
                        raise retry_error
            
            assessment_response = response.choices[0].message.content
            response_length = len(assessment_response)
            
            # Track Pass 3 success and output statistics
            self.error_stats['pass3_successful_attempts'] += 1
            self.error_stats['pass3_output_char_stats'].append(response_length)
            
            print(f"✅ PASS 3 SUCCESS: Generated {response_length:,} character assessment")
            print(f"DEBUG: Assessment response preview: {assessment_response[:200]}...")
            
            # Sanitize output to remove disallowed phrases and process references
            assessment_response = self._sanitize_assessment_output(assessment_response)
            
            parsed_response = self._parse_assessment_response(assessment_response, rubric_engine, pattern_id)
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
                max_completion_tokens=10000,  # Increased to handle longer narrative
                reasoning_effort="high"  # Increased for better assessment quality
            )
            
            return self._parse_flow_response(response.choices[0].message.content, rubric_engine)
            
        except Exception as e:
            print(f"Error in GPT-5 flow analysis: {e}")
            return {"error": f"Flow analysis failed: {e}"}
    
    def _build_surgical_description_prompt(self, frames: List[Dict], context_state: str) -> str:
        """Build prompt for PASS 1: Surgical description of frames"""
        
        timestamp_range = f"{frames[0]['timestamp']:.1f}s - {frames[-1]['timestamp']:.1f}s"
        pattern = (self.current_pattern or "").lower()
        
        if pattern == "subcuticular":
            prompt = f"""You are analyzing still images from a surgical suturing procedure. These are {len(frames)} consecutive frames from {timestamp_range}.

ASSUMPTION: The suturing technique label in the app is authoritative. Assume this procedure is subcuticular; do NOT reclassify the pattern. If observations conflict, note deficiencies but continue to assess as subcuticular.

CONTEXT SO FAR: {context_state if context_state else 'Beginning of video analysis'}

CRITICAL FOCUS: SUBCUTICULAR (INTRADERMAL) CLOSURE ON THE ACTIVE LINE
Each video has exactly ONE suture line being worked on. This subcuticular procedure demonstrates continuous intradermal technique throughout. Focus on technique execution, hand positioning, and bite quality.

ANALYSIS REQUIREMENTS (Subcuticular):
1. ACTIVE LINE: Identify the incision being closed with intradermal technique.
2. HAND/INSTRUMENTS: Positions and coordination relative to the suture line.
3. NEEDLE HANDLING: Needle grasp, orientation, and control for intradermal bites.
4. BITE TECHNIQUE: Sequential intradermal bite placement and progression.
5. SYMMETRY: Consistent bite depth, length, and spacing across the closure.
6. FLOW: Smoothness of technique progression and instrument handling.

OUTPUT: Concise, factual observations only (150–300 words). Do not mention 'active line' explicitly in output; describe naturally.

At the end, include these lines exactly:
TECHNIQUE_QUALITY: [Describe execution quality - smooth, controlled, consistent]
BITES_OBSERVED: [integer estimate within these frames]
OCCLUSION: [LOW/MEDIUM/HIGH]
HAND_COORDINATION: [Describe hand positioning and instrument control]
BITE_PROGRESSION: [Describe sequential placement and spacing]"""
        else:
            label_text = pattern.replace('_', ' ') if pattern else 'the selected technique'
            prompt = f"""You are analyzing still images from a surgical suturing procedure. These are {len(frames)} consecutive frames from {timestamp_range}.

ASSUMPTION: The suturing technique label in the app is authoritative. Assume this procedure is {label_text}; do NOT reclassify the pattern. If observations conflict, note deficiencies but continue to assess as {label_text}.

CONTEXT SO FAR: {context_state if context_state else 'Beginning of video analysis'}

CRITICAL FOCUS: IDENTIFY THE ACTIVE SUTURE LINE
Each video has exactly ONE suture line being worked on throughout the entire procedure. You must identify and focus ONLY on this single active incision - the one with hands and instruments nearby, where sutures are being added or completed. Ignore any other incision lines that are not being worked on.

ANALYSIS REQUIREMENTS:
Focus ONLY on surgically critical details of the ACTIVE suture line:

1. ACTIVE SUTURE IDENTIFICATION: Which incision line is currently being worked on? Describe its location and characteristics.
2. HAND POSITIONS: Where are hands positioned relative to the ACTIVE suture line? What instruments are held?
3. NEEDLE HANDLING: How is the needle grasped and positioned for the ACTIVE suture?
4. TISSUE INTERACTION: What tissue is being manipulated at the ACTIVE suture line? How are wound edges handled?
5. SUTURE PROGRESSION: Are new sutures being added at the ACTIVE line?
6. SPATIAL RELATIONSHIPS: How are hands positioned relative to the ACTIVE suture line and each other?

OUTPUT: Concise, factual observations only. No color commentary. 150-300 words maximum.
Focus internally on the active suture line, but describe the procedure naturally without mentioning "active suture line" or "focus" in your output.

Analyze these {len(frames)} frames:"""
        
        return prompt
    
    def _build_video_narrative_prompt(self, descriptions_text: str) -> str:
        """Build prompt for PASS 2: Video narrative assembly"""
        pattern = (self.current_pattern or "").lower()
        
        if pattern == "subcuticular":
            prompt = f"""You are analyzing surgical frame descriptions to create a video narrative. These are still images from a subcuticular (intradermal) closure.

ASSUMPTION: The suturing technique label in the app is authoritative. Assume this procedure is subcuticular; do NOT reclassify the pattern. If observations conflict, note deficiencies but continue to assess as subcuticular.

FRAME DESCRIPTIONS:
{descriptions_text}

TASK: Create a comprehensive narrative describing the complete procedure.

CRITICAL FOCUS: MAINTAIN ACTIVE SUTURE LINE CONTINUITY
Track the single incision being closed throughout. This subcuticular procedure demonstrates continuous intradermal technique.

Focus on:
1. TECHNIQUE EXECUTION: Quality of intradermal bite placement and progression.
2. BITE PROGRESSION: Sequential intradermal bites advancing the closure.
3. SYMMETRY/CONSISTENCY: Consistent bite depth, length, and spacing.
4. HAND EVOLUTION: How hand positions and instruments support smooth technique.
5. FLOW: Smoothness and efficiency of the suturing process.
6. FINAL STATE: Describe the completed closure quality and appearance.

OUTPUT: Comprehensive chronological narrative. Describe naturally without mentioning internal focusing processes.

NARRATIVE LENGTH REQUIREMENT: Aim for 8,000–12,000 characters.

At the end, include these lines exactly:
TECHNIQUE_EXECUTION: [Describe overall quality of intradermal technique]
BITES_TOTAL_ESTIMATE: [integer]
SYMMETRY_ASSESSMENT: [Consistent/Variable - describe bite spacing and depth]
HAND_COORDINATION: [Smooth/Adequate/Hesitant - describe instrument handling]
FLOW_QUALITY: [Efficient/Standard/Slow - describe technique progression]
CONFIDENCE: [HIGH/MEDIUM/LOW] based on evidence quantity/occlusion."""
        else:
            label_text = pattern.replace('_', ' ') if pattern else 'the selected technique'
            prompt = f"""You are analyzing surgical frame descriptions to create a video narrative. These are still images from a suturing procedure.

ASSUMPTION: The suturing technique label in the app is authoritative. Assume this procedure is {label_text}; do NOT reclassify the pattern. If observations conflict, note deficiencies but continue to assess as {label_text}.

FRAME DESCRIPTIONS:
{descriptions_text}

TASK: Create a comprehensive narrative describing the complete surgical procedure.

CRITICAL FOCUS: MAINTAIN ACTIVE SUTURE LINE CONTINUITY
Throughout the narrative, you must track the single suture line being worked on. Each video has exactly ONE suture line being worked on throughout the entire procedure. Track the progression of sutures being added to this single active line.

Focus on:
1. ACTIVE SUTURE LINE TRACKING: Which incision line is being worked on throughout the procedure? (This will be the same line throughout the entire video)
2. SUTURE PROGRESSION: What is the pattern of suture placement and technique development on the active line?
3. TECHNIQUE PROGRESSION: Track how the suturing technique develops throughout the video.
4. HAND POSITION EVOLUTION: How do hand positions change relative to the active suture line over time?
5. TECHNIQUE CONSISTENCY: How does the suturing technique evolve on the active line?
6. SPATIAL RELATIONSHIPS: How do hands and instruments move relative to the active suture line?
7. FINAL STATE: Describe the final state of the suture line closure.

CRITICAL SUTURE COUNTING REQUIREMENT:
Focus on describing the progression and completion of the suturing technique without emphasizing specific counts.

OUTPUT: Write a comprehensive, detailed chronological narrative that connects the frame observations into a coherent story of the surgical technique. Include ALL important details from the frame descriptions. Be thorough and detailed - it's better to be comprehensive than to miss critical elements. Describe the procedure naturally without mentioning "active suture line" or internal focusing processes.

NARRATIVE LENGTH REQUIREMENT: Generate a substantial narrative that captures the full complexity of the procedure. Aim for 8,000-12,000 characters to ensure no important details are lost."""
        
        return prompt
    
    def _build_rubric_assessment_prompt(self, video_narrative: str, pattern_id: str, rubric_engine, final_product_image=None) -> str:
        """Build prompt for PASS 3: Rubric assessment"""
        
        # Get rubric data
        rubric_data = rubric_engine.get_pattern_rubric(pattern_id)
        if not rubric_data:
            return "Error: Could not load rubric data"
        
        pattern = (pattern_id or "").lower()
        
        if pattern == "subcuticular":
            prompt = f"""YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE. You are training surgeons who will operate on real patients. Assume EVERY technique has flaws until proven otherwise.

SURGICAL ASSESSMENT: {rubric_data['display_name']} Suturing Technique

ASSUMPTION: The suturing technique label in the app is authoritative. Assume this procedure is subcuticular; do NOT reclassify the pattern. If observations conflict, note deficiencies but continue to assess as subcuticular.

You have just observed a complete surgical video of suturing technique. The following is your detailed observation record from watching the procedure:

OBSERVATION RECORD:
{video_narrative}

ASSESSMENT TASK:
Evaluate this surgical performance against the specific rubric criteria. Base your assessment on what you directly observed in the video, not on any written description.

ASSESSMENT FOUNDATION:
This subcuticular procedure demonstrates continuous intradermal technique throughout. Assess based on execution quality, not layer identification.

STRICT SCORING CALIBRATION (Subcuticular):
- Default score for a criterion is 3 (competent, standard technique) unless strong evidence supports higher or lower.
- Confidence scaling: if CONFIDENCE = LOW → cap all points at 3; if MEDIUM → allow up to 4; only HIGH confidence may reach 5.
- Award 4 for smooth, consistent technique with good symmetry and flow.
- Award 5 ONLY for exemplary execution: flawless symmetry, efficient flow, excellent hand coordination, and consistent bite quality.

CRITICAL FOCUS: ACTIVE SUTURE LINE ASSESSMENT
Evaluate ONLY the single incision that was worked on throughout.

PATTERN ASSESSMENT RULES (Subcuticular):
- Assess continuous intradermal bite technique and progression
- Evaluate bite consistency: symmetric depth, length, and spacing
- Check hand coordination and instrument control
- Judge technique flow: smooth progression and efficient execution  
- Assess bite quality: consistent tension and placement
- Evaluate overall procedural competence and surgical skill

LANGUAGE AND OUTPUT RULES (MANDATORY):
- Do NOT mention images/frames/narratives in output
- Do NOT say "no full passes were recorded"; draw conclusions from observed evidence
- Eyewitness, concise clinical language only
- If making any failing claim, include a brief parenthetical with cited evidence times (e.g., "(breaches at ~11:32:55 and ~11:33:13)").

RUBRIC ASSESSMENT FORMAT:
For each rubric point, provide a 1-2 sentence comment and a 1-5 score. Apply the caps above before writing the final number.

IMPORTANT: Assess only against {rubric_data['display_name']} criteria. Do not mention other suture patterns.

Format each rubric point exactly like this:

{self._generate_rubric_format_example(rubric_data['points'])}

SUMMATIVE_ASSESSMENT:
[Write a holistic 2-3 paragraph assessment of the entire procedure focusing on flow, hand coordination, plane integrity, and overall competence. Do NOT mention images, frames, or narratives.]

RUBRIC POINTS TO ASSESS:
{json.dumps(rubric_data['points'], indent=2)}

MANDATORY SCORING OUTPUT:
{self._generate_rubric_scores_format(rubric_data['points'])}

Provide your complete assessment:"""
            return prompt
        
        prompt = f"""YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE. You are training surgeons who will operate on real patients. Assume EVERY technique has flaws until proven otherwise.

SURGICAL ASSESSMENT: {rubric_data['display_name']} Suturing Technique

ASSUMPTION: The suturing technique label in the app is authoritative. Assume this procedure is {rubric_data['display_name']}; do NOT reclassify the pattern. If observations conflict, note deficiencies but continue to assess as {rubric_data['display_name']}.

You have just observed a complete surgical video of suturing technique. The following is your detailed observation record from watching the procedure:

OBSERVATION RECORD:
{video_narrative}

{"FINAL PRODUCT IMAGE VERIFICATION: You have access to the final product image showing the completed sutures. You MUST use this image to verify your assessment. Look at this image carefully and count the actual number of completed sutures visible. Do NOT rely solely on the narrative - use this visual evidence to confirm suture counts and spacing. If the narrative says one thing but the image shows another, trust the image over the narrative." if final_product_image else ""}

ASSESSMENT TASK:
Evaluate this surgical performance against the specific rubric criteria. Base your assessment on what you directly observed in the video, not on any written description.

CRITICAL FOCUS: ACTIVE SUTURE LINE ASSESSMENT
Your assessment must focus ONLY on the single suture line that was worked on throughout the procedure. Each video has exactly ONE suture line being worked on. You must evaluate only this one line where hands and instruments were actively working, where sutures were being added progressively. Do not assess any other incision lines. However, do not mention "active suture line" or "focus" in your final assessment output - describe the technique naturally.

PATTERN ASSESSMENT RULES:
- You are assessing a {rubric_data['display_name']} suturing technique
- Do NOT speculate about what pattern was being attempted or rename the technique
- Assess the performance against the {rubric_data['display_name']} criteria only
- If the technique doesn't match {rubric_data['display_name']} standards, give low scores but don't suggest alternative patterns
- Assume the surgeon intended to perform {rubric_data['display_name']} and evaluate accordingly
- FOCUS ON TECHNIQUE QUALITY: Evaluate the overall technique execution and closure quality rather than specific counts
- ASSESS SPACING AND DISTRIBUTION: Look for even spacing and appropriate distribution without emphasizing exact numbers
- FINAL PRODUCT IMAGE VERIFICATION: If you have access to the final product image, use it to assess closure quality and technique execution

LANGUAGE AND OUTPUT RULES (MANDATORY):
- Do NOT say "no full passes were recorded/captured". Draw conclusions from visible evidence.
- Do NOT mention images/frames/narratives in output.
- Use direct eyewitness language; concise clinical conclusions.

RUBRIC ASSESSMENT FORMAT:
For each rubric point (1-7), provide:
1. A brief comment (1-2 sentences) describing what you observed
2. A score (1-5) based on the strict guidelines above

IMPORTANT: Assess only against {rubric_data['display_name']} criteria. Do not mention other suture patterns or speculate about what was intended.

CRITICAL INSTRUCTION: Before assessing any rubric point, look at the final product image (if available) and count the actual number of completed sutures visible. Use this visual evidence to verify your assessment. Do not make assumptions based on the narrative alone.

Format each rubric point exactly like this:

RUBRIC_POINT_1:
Comment: [Your 1-2 sentence assessment of this specific point]
Score: [1-5]

RUBRIC_POINT_2:
Comment: [Your 1-2 sentence assessment of this specific point]
Score: [1-5]

[Continue for all 7 points...]

RUBRIC_POINT_7:
Comment: [Your 1-2 sentence assessment of this specific point]
Score: [1-5]

SUMMATIVE_ASSESSMENT:
[Write a comprehensive, holistic assessment (2-3 paragraphs) that evaluates the ENTIRE PROCEDURE as a whole. Focus on flow, hand coordination, and overall competence. Do NOT mention images, frames, or narratives.]

RUBRIC POINTS TO ASSESS:
{json.dumps(rubric_data['points'], indent=2)}

MANDATORY SCORING OUTPUT:
{self._generate_rubric_scores_format(rubric_data['points'])}

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

ASSUMPTION: The suturing technique label in the app is authoritative. Assume this procedure is {rubric_data['display_name']}; do NOT reclassify the pattern. If observations conflict, note deficiencies but continue to assess as {rubric_data['display_name']}.

You have just observed a complete surgical video of suturing technique. The following is your detailed observation record from watching the procedure:

OBSERVATION RECORD:
{descriptions_text}

ASSESSMENT TASK:
Evaluate this surgical performance against the specific rubric criteria. Base your assessment on what you directly observed in the video, not on any written description.

CRITICAL FOCUS: ACTIVE SUTURE LINE ASSESSMENT
Your assessment must focus ONLY on the single suture line that was worked on throughout the procedure. Each video has exactly ONE suture line being worked on. You must evaluate only this one line where hands and instruments were actively working, where sutures were being added progressively. Do not assess any other incision lines. However, do not mention "active suture line" or "focus" in your final assessment output - describe the technique naturally.

PATTERN ASSESSMENT RULES:
- You are assessing a {rubric_data['display_name']} suturing technique
- Do NOT speculate about what pattern was being attempted or rename the technique
- Assess the performance against the {rubric_data['display_name']} criteria only
- If the technique doesn't match {rubric_data['display_name']} standards, give low scores but don't suggest alternative patterns
- Assume the surgeon intended to perform {rubric_data['display_name']} and evaluate accordingly
- FOCUS ON TECHNIQUE QUALITY: Evaluate the overall technique execution and closure quality rather than specific counts
- ASSESS SPACING AND DISTRIBUTION: Look for even spacing and appropriate distribution without emphasizing exact numbers
- FINAL PRODUCT IMAGE VERIFICATION: If you have access to the final product image, use it to assess closure quality and technique execution

STRICT SCORING GUIDELINES:
- Score 1 = Major deficiencies - technique significantly below standard
- Score 2 = Some deficiencies - technique below standard with notable issues  
- Score 3 = Meets standard - technique is adequate and competent
- Score 4 = Exceeds standard - technique is consistently good with minor areas for improvement
- Score 5 = Exemplary - technique demonstrates mastery and serves as a model

CRITICAL SCORING PHILOSOPHY:
- Score 3 should be your DEFAULT for competent, standard technique
- Score 3 means the technique is competent but has room for improvement
- Score 4 means you would use this video to teach other attendings - RARE
- Score 5 means this is among the best technique you've seen in your entire career - EXTREMELY RARE
- Assume EVERY technique has flaws until proven otherwise
- You are training surgeons who will operate on real patients
- BE CONSERVATIVE: Most learners should score 2-3, not 3-4
- Only give 4+ if the technique is genuinely impressive and teachable
- Only give 5 if it's truly exemplary and could serve as a gold standard

RUBRIC ASSESSMENT FORMAT:
For each rubric point (1-7), provide:
1. A brief comment (1-2 sentences) describing what you observed
2. A score (1-5) based on the strict guidelines above

IMPORTANT: Assess only against {rubric_data['display_name']} criteria. Do not mention other suture patterns or speculate about what was intended.

CRITICAL INSTRUCTION: Before assessing any rubric point, look at the final product image (if available) and count the actual number of completed sutures visible. Use this visual evidence to verify your assessment. Do not make assumptions based on the narrative alone.

Format each rubric point exactly like this:

RUBRIC_POINT_1:
Comment: [Your 1-2 sentence assessment of this specific point]
Score: [1-5]

RUBRIC_POINT_2:
Comment: [Your 1-2 sentence assessment of this specific point]
Score: [1-5]

[Continue for all 7 points...]

RUBRIC_POINT_7:
Comment: [Your 1-2 sentence assessment of this specific point]
Score: [1-5]

SUMMATIVE_ASSESSMENT:
[Write a comprehensive, holistic assessment (2-3 paragraphs) that evaluates the ENTIRE PROCEDURE as a whole. Focus on flow, hand coordination, and overall competence. Do NOT mention images, frames, or narratives.]

RUBRIC POINTS TO ASSESS:
{json.dumps(rubric_data['points'], indent=2)}

MANDATORY SCORING OUTPUT:
{self._generate_rubric_scores_format(rubric_data['points'])}

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
    
    def _generate_rubric_format_example(self, rubric_points) -> str:
        """Generate dynamic rubric format example based on actual rubric points."""
        examples = []
        for i, point in enumerate(rubric_points[:2]):  # Show first 2 as examples
            examples.append(f"RUBRIC_POINT_{point['pid']}:\nComment: [Your 1-2 sentence assessment]\nScore: [1-5]\n")
        examples.append("...")
        return "\n".join(examples)
    
    def _generate_rubric_scores_format(self, rubric_points) -> str:
        """Generate dynamic RUBRIC_SCORES format based on actual rubric points."""
        score_lines = ["RUBRIC_SCORES_START"]
        for point in rubric_points:
            score_lines.append(f"{point['pid']}: X")
        score_lines.append("RUBRIC_SCORES_END")
        return "\n".join(score_lines)
    
    def _parse_assessment_response(self, response: str, rubric_engine, pattern_id: str = None) -> Dict[str, Any]:
        """Parse GPT-5 assessment response"""
        try:
            # Extract rubric scores and comments from structured format
            scores = {}
            rubric_comments = {}
            
            # Get expected rubric points from rubric engine using pattern_id
            expected_points = []
            if rubric_engine and pattern_id:
                try:
                    pattern_data = rubric_engine.get_pattern_rubric(pattern_id)
                    if pattern_data and 'points' in pattern_data:
                        expected_points = [point['pid'] for point in pattern_data['points']]
                        print(f"📊 Dynamic scoring: Found {len(expected_points)} rubric points for {pattern_id}: {expected_points}")
                except Exception as e:
                    print(f"⚠️ Error getting rubric points for {pattern_id}: {e}")
            
            # Fallback to 1-7 only as last resort with warning
            if not expected_points:
                expected_points = list(range(1, 8))
                print(f"⚠️ FALLBACK: Using default 1-7 rubric points (pattern_id={pattern_id})")
            
            # Parse each RUBRIC_POINT_X section
            for i in expected_points:
                point_pattern = f"RUBRIC_POINT_{i}:"
                if point_pattern in response:
                    # Extract the section for this rubric point
                    start_idx = response.find(point_pattern)
                    if start_idx != -1:
                        # Find the next RUBRIC_POINT or SUMMATIVE_ASSESSMENT
                        next_point_idx = response.find(f"RUBRIC_POINT_{i+1}:", start_idx)
                        summative_idx = response.find("SUMMATIVE_ASSESSMENT:", start_idx)
                        
                        end_idx = len(response)
                        if next_point_idx != -1:
                            end_idx = min(end_idx, next_point_idx)
                        if summative_idx != -1:
                            end_idx = min(end_idx, summative_idx)
                        
                        point_section = response[start_idx:end_idx]
                        
                        # Extract comment
                        comment_match = re.search(r'Comment:\s*(.+?)(?=Score:|$)', point_section, re.DOTALL)
                        if comment_match:
                            rubric_comments[i] = comment_match.group(1).strip()
                        
                        # Extract score
                        score_match = re.search(r'Score:\s*(\d+)', point_section)
                        if score_match:
                            try:
                                raw_score = int(score_match.group(1))
                                # Apply 0.65 multiplier and round up to correct scoring
                                adjusted_score = math.ceil(raw_score * 0.65)
                                # Ensure score stays within 1-5 range
                                scores[i] = max(1, min(5, adjusted_score))
                            except ValueError:
                                continue
            
            # Extract summative assessment
            summative = ""
            if "SUMMATIVE_ASSESSMENT:" in response:
                summative_section = response.split("SUMMATIVE_ASSESSMENT:")[1].strip()
                # Clean up any remaining prefixes or formatting
                summative = summative_section.split('\n')[0].strip() if summative_section else ""
            else:
                # Fallback - look for summative anywhere in response
                summative_start = response.find("Summative assessment:")
                if summative_start != -1:
                    summative = response[summative_start + len("Summative assessment:"):].strip()
            
            # Validate scores - ensure all expected points have scores
            missing_points = [pid for pid in expected_points if pid not in scores]
            invalid_scores = {pid: score for pid, score in scores.items() if not (1 <= score <= 5)}
            
            # Auto-repair: set missing scores to default of 3
            for pid in missing_points:
                scores[pid] = 3
                print(f"⚠️ Missing score for rubric point {pid}, defaulting to 3")
            
            # Auto-repair: clamp invalid scores to 1-5 range
            for pid, invalid_score in invalid_scores.items():
                if invalid_score < 1:
                    scores[pid] = 1
                elif invalid_score > 5:
                    scores[pid] = 5
                else:
                    scores[pid] = 3
                print(f"⚠️ Invalid score {invalid_score} for rubric point {pid}, corrected to {scores[pid]}")
            
            return {
                'success': True,
                'rubric_scores': scores,
                'rubric_comments': rubric_comments,
                'summative_assessment': summative,
                'full_response': response,
                'validation_warnings': {
                    'missing_points': missing_points,
                    'invalid_scores': invalid_scores,
                    'expected_points': expected_points
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to parse response: {e}",
                "full_response": response,
                "success": False
            }
    
    def _adaptive_chunking_pass2(self, all_descriptions: List[Dict], model: str, reasoning_level: str, verbosity_level: str) -> str:
        """
        ADAPTIVE CHUNKING: Handle context length errors by processing descriptions in chunks
        """
        print(f"🔄 ADAPTIVE CHUNKING: Processing {len(all_descriptions)} descriptions in chunks")
        
        # Calculate chunk size - start with half the descriptions
        chunk_size = max(1, len(all_descriptions) // 2)
        chunk_narratives = []
        
        for i in range(0, len(all_descriptions), chunk_size):
            chunk = all_descriptions[i:i + chunk_size]
            print(f"📦 Processing chunk {i//chunk_size + 1}: descriptions {i} to {min(i+chunk_size-1, len(all_descriptions)-1)}")
            
            # Format chunk descriptions
            descriptions_text = "\n\n".join([
                f"TIMESTAMP: {desc['timestamp_range']}\nFRAMES: {desc['frame_count']}\nSURGICAL DESCRIPTION: {desc['analysis']}"
                for desc in chunk
            ])
            
            # Use shorter character limit for chunks
            max_chunk_length = 50000  # Reduced from 100000
            if len(descriptions_text) > max_chunk_length:
                print(f"⚠️ CHUNK TRUNCATION: {len(descriptions_text):,} chars, truncating to {max_chunk_length:,}")
                descriptions_text = descriptions_text[:max_chunk_length] + "\n\n[CHUNK TRUNCATED]"
            
            narrative_prompt = self._build_video_narrative_prompt(descriptions_text)
            
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": narrative_prompt}],
                    max_completion_tokens=8000,  # Reduced for chunks
                    reasoning_effort=reasoning_level,
                    verbosity=verbosity_level
                )
                
                chunk_narrative = response.choices[0].message.content
                chunk_narratives.append(chunk_narrative)
                print(f"✅ CHUNK {i//chunk_size + 1} SUCCESS: {len(chunk_narrative)} chars")
                
            except Exception as chunk_error:
                error_str = str(chunk_error).lower()
                if "quota_exceeded" in error_str or "insufficient_quota" in error_str:
                    print(f"💳 CHUNK QUOTA EXCEEDED - FAST FAIL: {chunk_error}")
                    raise Exception(f"QUOTA_EXCEEDED_FAST_FAIL: {chunk_error}")
                
                print(f"❌ CHUNK {i//chunk_size + 1} FAILED: {chunk_error}")
                chunk_narratives.append(f"[CHUNK FAILED: {chunk_error}]")
        
        # Combine all chunk narratives
        combined_narrative = "\n\n=== SURGICAL VIDEO NARRATIVE (COMBINED FROM CHUNKS) ===\n\n" + \
                           "\n\n".join(chunk_narratives)
        
        # Final pass to create cohesive narrative from chunks
        try:
            final_prompt = f"""Combine these surgical video analysis chunks into one cohesive narrative:

{combined_narrative[:80000]}  # Limit input for final pass

Create a single flowing narrative that captures all key surgical observations."""
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": final_prompt}],
                max_completion_tokens=12000,
                reasoning_effort=reasoning_level,
                verbosity=verbosity_level
            )
            
            self.video_narrative = response.choices[0].message.content
            print(f"✅ ADAPTIVE CHUNKING SUCCESS: Combined into {len(self.video_narrative):,} char narrative")
            
        except Exception as final_error:
            print(f"⚠️ FINAL COMBINATION FAILED: {final_error} - Using chunk combination")
            self.video_narrative = combined_narrative[:50000]  # Fallback with limit
        
        return self.video_narrative

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

    def _sanitize_assessment_output(self, text: str) -> str:
        """Remove disallowed phrases and process references from model output."""
        try:
            # Remove sentences like "Final image confirmation: ..."
            cleaned = re.sub(r"(?i)final\s+image\s+confirmation:\s*.*?(?=(\n|$))", "", text)
            # Remove sentences claiming capture issues like "no full passes were recorded/captured..."
            cleaned = re.sub(r"(?i)no\s+full\s+passes\s+were\s+(recorded|captured)[^.]*\.\s*", "", cleaned)
            # Avoid explicit process references if present
            cleaned = re.sub(r"(?i)\b(final\s+image|image|photo|narrative|description|synopsis|frames)\b", "", cleaned)
            # Normalize whitespace
            cleaned = re.sub(r"\s{2,}", " ", cleaned)
            cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
            return cleaned.strip()
        except Exception:
            return text
