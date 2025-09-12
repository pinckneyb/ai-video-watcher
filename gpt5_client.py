"""
GPT-5 client module for video analysis and context management.
"""

import openai
import json
import re
from typing import List, Dict, Any, Optional, Tuple
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GPT5Client:
    """Client for GPT-5 API integration with video analysis capabilities."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize GPT-5 client.
        
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
        Analyze a batch of frames using GPT-5.
        
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
            
            # Create messages for GPT-5
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
            
            # Call GPT-5 API
            response = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                max_completion_tokens=2000,
                reasoning_effort="low"
            )
            
            # Extract response
            analysis_text = response.choices[0].message.content
            
            # Parse response for events
            events = self._extract_events_from_response(analysis_text, frames)
            
            # Update context
            self.context_state = self._update_context(analysis_text, context_state)
            
            return analysis_text, events
            
        except Exception as e:
            error_msg = f"GPT-5 API error: {str(e)}"
            print(error_msg)
            # Return error as narrative with empty events
            return error_msg, []
    
    def _build_frame_analysis_prompt(self, frames: List[Dict], profile: Dict[str, Any], 
                                   context_state: str) -> str:
        """Build the analysis prompt for GPT-5."""
        
        # Get profile information
        focus_areas = profile.get('focus_areas', ['General video content'])
        tone = profile.get('tone', 'informative')
        detail_level = profile.get('detail_level', 'moderate')
        
        # Build frame timestamps info
        frame_info = []
        for i, frame in enumerate(frames):
            timestamp = frame.get('timestamp', i)
            frame_info.append(f"Frame {i+1}: {timestamp:.1f}s")
        
        frame_timestamps = ", ".join(frame_info)
        
        # Build the prompt
        prompt = f"""You are analyzing video frames for content understanding. 

ANALYSIS CONTEXT:
{f"Previous context: {context_state}" if context_state else "This is the beginning of the video."}

CURRENT BATCH:
Analyzing {len(frames)} frames at timestamps: {frame_timestamps}

PROFILE SETTINGS:
- Focus areas: {', '.join(focus_areas)}
- Tone: {tone}
- Detail level: {detail_level}

INSTRUCTIONS:
1. Analyze the provided video frames in sequence
2. Focus on the specified areas: {', '.join(focus_areas)}
3. Provide a {detail_level} level narrative in a {tone} tone
4. Note any significant events, changes, or actions
5. Maintain continuity with previous context if provided

Provide your analysis as a flowing narrative that captures what's happening in these frames."""

        return prompt
    
    def _frames_to_base64(self, frames: List[Dict]) -> List[str]:
        """Convert frame data to base64 strings for API."""
        base64_frames = []
        
        for frame in frames:
            if 'base64' in frame:
                # Frame already has base64 data
                base64_frames.append(frame['base64'])
            elif 'image_data' in frame:
                # Convert image data to base64
                import base64
                encoded = base64.b64encode(frame['image_data']).decode('utf-8')
                base64_frames.append(encoded)
            else:
                print(f"Warning: Frame missing image data: {frame}")
                
        return base64_frames
    
    def _extract_events_from_response(self, analysis_text: str, frames: List[Dict]) -> List[Dict]:
        """Extract structured events from GPT-5 response."""
        events = []
        
        # Simple event extraction based on keywords
        event_keywords = [
            'appears', 'shows', 'enters', 'exits', 'moves', 'changes', 
            'begins', 'ends', 'starts', 'stops', 'opens', 'closes'
        ]
        
        # Look for sentences containing event keywords
        sentences = re.split(r'[.!?]+', analysis_text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in event_keywords):
                # Create event entry
                event = {
                    'timestamp': frames[0]['timestamp'] if frames else 0.0,
                    'description': sentence,
                    'type': 'action'
                }
                events.append(event)
        
        return events
    
    def _update_context(self, new_analysis: str, previous_context: str) -> str:
        """Update context state with new analysis."""
        # Summarize the new analysis for context
        context_summary = new_analysis[:200] + "..." if len(new_analysis) > 200 else new_analysis
        
        if previous_context:
            # Combine with previous context, keeping it concise
            combined_context = f"{previous_context} | {context_summary}"
            # Limit context length
            if len(combined_context) > 500:
                combined_context = combined_context[-500:]
        else:
            combined_context = context_summary
            
        return combined_context
    
    def update_context(self, new_context: str):
        """Update the context state."""
        self.context_state = new_context
    
    def reset_context(self):
        """Reset context state for new video."""
        self.context_state = ""
        self.full_transcript = []
        self.event_log = []
    
    def get_context_state(self) -> str:
        """Get current context state."""
        return self.context_state