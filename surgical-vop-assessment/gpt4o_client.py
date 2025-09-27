"""
GPT-4o client module for video analysis and context management.
"""

import openai
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GPT4oClient:
    """Client for GPT-4o API integration with video analysis capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT-4o client.
        
        Args:
            api_key: OpenAI API key (will use env var if not provided)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Context management
        self.context_state = ""
        self.full_transcript = []
        self.event_log = []
    
    def analyze_frames(self, frames: List[Dict], profile: Dict[str, Any], 
                      context_state: str = "") -> Tuple[str, List[Dict]]:
        """
        Analyze a batch of frames using GPT-4o.
        
        Args:
            frames: List of frame metadata dictionaries
            profile: Profile configuration for prompting
            context_state: Condensed context from previous analysis
            
        Returns:
            Tuple of (narrative_text, event_log)
        """
        try:
            # Prepare the prompt
            prompt = self._build_frame_analysis_prompt(frames, profile, context_state)
            
            # Convert frames to base64 for API
            base64_frames = self._frames_to_base64(frames)
            
            # Create messages for GPT-4o
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_data}",
                                "detail": "high"
                            }
                        } for frame_data in base64_frames
                    ]
                }
            ]
            
            # Make API call using Responses API
            response = self.client.responses.create(
                model="gpt-4o",
                input=messages,
                max_output_tokens=2000,
                reasoning={"effort": "low"}
            )
            
            # Parse response using Responses API format
            narrative_text = response.output_text
            
            # Ensure narrative text is properly encoded
            if isinstance(narrative_text, str):
                narrative_text = narrative_text.encode('utf-8', errors='replace').decode('utf-8')
            
            # Extract JSON events if present
            event_log = self._extract_events_from_response(narrative_text)
            
            return narrative_text, event_log
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error analyzing frames: {error_msg}")
            return f"Error during analysis: {error_msg}", []
    
    def rescan_segment(self, start_time: str, end_time: str, frames: List[Dict], 
                       profile: Dict[str, Any]) -> Tuple[str, List[Dict]]:
        """
        Rescan a video segment at higher detail.
        
        Args:
            start_time: Start time in HH:MM:SS format
            end_time: End time in HH:MM:SS format
            frames: High-fps frames for detailed analysis
            profile: Profile configuration
            
        Returns:
            Tuple of (detailed_narrative, event_log)
        """
        try:
            # Use rescan prompt from profile
            rescan_prompt = profile["rescan_prompt"].format(
                start_time=start_time, 
                end_time=end_time
            )
            
            # Add context about what we're rescanning
            full_prompt = f"""You are performing a detailed rescan of a video segment.

{rescan_prompt}

Previous analysis context: {self.context_state}

Analyze these frames with much higher detail, focusing on subtle movements and events that may have been missed in the initial scan.

{rescan_prompt}"""
            
            # Convert frames to base64
            base64_frames = self._frames_to_base64(frames)
            
            # Create messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": full_prompt
                        }
                    ] + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{frame_data}",
                                "detail": "high"
                            }
                        } for frame_data in base64_frames
                    ]
                }
            ]
            
            # Make API call using Responses API
            response = self.client.responses.create(
                model="gpt-4o",
                input=messages,
                max_output_tokens=2500,
                reasoning={"effort": "low"}
            )
            
            narrative_text = response.output_text
            
            # Ensure narrative text is properly encoded
            if isinstance(narrative_text, str):
                narrative_text = narrative_text.encode('utf-8', errors='replace').decode('utf-8')
            
            event_log = self._extract_events_from_response(narrative_text)
            
            return narrative_text, event_log
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error rescanning segment: {error_msg}")
            return f"Error during rescan: {error_msg}", []
    
    def condense_context(self, full_transcript: List[str]) -> str:
        """
        Condense full transcript into context state using GPT-4o.
        
        Args:
            full_transcript: List of narrative texts from all batches
            
        Returns:
            Condensed context state
        """
        try:
            # Join transcript with timestamps
            transcript_text = "\n\n".join(full_transcript)
            
            # Use context condensation prompt
            condensation_prompt = """You are maintaining a running summary of a video as it is being narrated. 
Your task is to take the current full transcript so far and compress it into a concise "state summary" that captures: 
- Key events (with approximate timestamps) 
- Main actors or objects of focus 
- Current scene status (who/what is present, what is happening) 
- Any unresolved actions (e.g., "Person is reaching toward the door, action incomplete") 

Guidelines: 
- Use no more than 150 words. 
- Preserve chronological flow. 
- Keep timestamps coarse (to the nearest ~10–15 seconds). 
- Do not include stylistic narration, just a factual condensed state. 
- Focus on continuity—what should be remembered for interpreting the next frames. 

Transcript to condense:
{transcript_text}

Output format: [Condensed State Summary]"""
            
            # Make API call for condensation using Responses API
            response = self.client.responses.create(
                model="gpt-4o",
                input=condensation_prompt.format(transcript_text=transcript_text),
                max_output_tokens=300,
                reasoning={"effort": "low"}
            )
            
            condensed_state = response.output_text
            
            # Ensure condensed state is properly encoded
            if isinstance(condensed_state, str):
                condensed_state = condensed_state.encode('utf-8', errors='replace').decode('utf-8')
            
            return condensed_state
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error condensing context: {error_msg}")
            # Fallback to simple truncation
            return "Context condensation failed. Using fallback summary."
    
    def _build_frame_analysis_prompt(self, frames: List[Dict], profile: Dict[str, Any], 
                                   context_state: str) -> str:
        """Build the prompt for frame analysis."""
        # Get timestamps for context
        timestamps = [f"{frame['timestamp_formatted']}" for frame in frames]
        timestamp_range = f"{timestamps[0]} to {timestamps[-1]}"
        
        prompt = f"""{profile['base_prompt']}

Context so far: {context_state if context_state else 'Beginning of video'}

Analyze the following {len(frames)} frames from {timestamp_range}:

"""
        
        return prompt
    
    def _frames_to_base64(self, frames: List[Dict]) -> List[str]:
        """Convert frames to base64 strings."""
        import base64
        import io
        
        base64_frames = []
        for i, frame_data in enumerate(frames):
            try:
                # Convert PIL image to base64
                img_buffer = io.BytesIO()
                frame_data['frame_pil'].save(img_buffer, format='JPEG', quality=85)
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                base64_frames.append(img_str)
            except Exception as e:
                error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
                print(f"Error converting frame {i} to base64: {error_msg}")
                # Skip this frame and continue
                continue
        
        return base64_frames
    
    def _extract_events_from_response(self, response_text: str) -> List[Dict]:
        """Extract JSON events from GPT-4o response."""
        try:
            # Look for JSON blocks in the response
            json_pattern = r'```json\s*(\[.*?\])\s*```'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            
            if matches:
                # Parse the first JSON match
                events = json.loads(matches[0])
                return events if isinstance(events, list) else []
            
            # Fallback: look for any JSON-like structure
            json_pattern2 = r'\[.*?\]'
            matches2 = re.findall(json_pattern2, response_text, re.DOTALL)
            
            for match in matches2:
                try:
                    events = json.loads(match)
                    if isinstance(events, list) and events:
                        return events
                except:
                    continue
            
            return []
            
        except Exception as e:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error extracting events: {error_msg}")
            return []
    
    def update_context(self, narrative: str, events: List[Dict]):
        """Update internal context with new analysis results."""
        self.full_transcript.append(narrative)
        self.event_log.extend(events)
        
        # Condense context for next batch
        if len(self.full_transcript) > 1:  # Only condense after first batch
            self.context_state = self.condense_context(self.full_transcript)
        else:
            self.context_state = "Beginning of video analysis"
    
    def get_full_transcript(self) -> str:
        """Get the complete transcript as markdown."""
        return "\n\n".join(self.full_transcript)
    
    def get_event_timeline(self) -> List[Dict]:
        """Get the complete event timeline."""
        return sorted(self.event_log, key=lambda x: x.get('timestamp', '00:00:00'))
    
    def reset_context(self):
        """Reset all context and start fresh."""
        self.context_state = ""
        self.full_transcript = []
        self.event_log = []
