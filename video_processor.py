"""
Video processing module for frame extraction and video handling.
"""

import cv2
import ffmpeg
import numpy as np
from typing import List, Tuple, Optional, Union
import os
import tempfile
import requests
from PIL import Image
import io

class VideoProcessor:
    """Handles video processing, frame extraction, and metadata management."""
    
    def __init__(self):
        self.video_path: Optional[str] = None
        self.fps: float = 1.0
        self.total_frames: int = 0
        self.duration: float = 0.0
        self.frame_metadata: List[dict] = []
        
    def load_video(self, source: Union[str, bytes], fps: float = 1.0) -> bool:
        """
        Load video from file path, URL, or bytes.
        
        Args:
            source: Video source (file path, URL, or bytes)
            fps: Frames per second to extract
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.fps = fps
            
            if isinstance(source, str):
                if source.startswith(('http://', 'https://')):
                    # Download from URL
                    response = requests.get(source)
                    response.raise_for_status()
                    video_bytes = response.content
                    self.video_path = self._save_temp_video(video_bytes)
                else:
                    # Local file
                    if not os.path.exists(source):
                        raise FileNotFoundError(f"Video file not found: {source}")
                    self.video_path = source
            else:
                # Bytes input
                self.video_path = self._save_temp_video(source)
            
            # Get video properties
            self._get_video_properties()
            return True
            
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def _save_temp_video(self, video_bytes: bytes) -> str:
        """Save video bytes to temporary file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(video_bytes)
        temp_file.close()
        return temp_file.name
    
    def _get_video_properties(self):
        """Extract video properties using ffmpeg."""
        try:
            probe = ffmpeg.probe(self.video_path)
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            
            self.duration = float(probe['format']['duration'])
            self.total_frames = int(video_info['nb_frames'])
            print(f"FFmpeg: Duration={self.duration}, Frames={self.total_frames}")
            
        except Exception as e:
            print(f"Error getting video properties with FFmpeg: {e}")
            # Fallback to OpenCV
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_orig = cap.get(cv2.CAP_PROP_FPS)
                self.duration = self.total_frames / fps_orig if fps_orig > 0 else 0
                print(f"OpenCV fallback: Duration={self.duration}, Frames={self.total_frames}, FPS={fps_orig}")
            else:
                print("Failed to open video with OpenCV")
                self.duration = 0
                self.total_frames = 0
            cap.release()
    
    def extract_frames(self, start_time: float = 0.0, end_time: Optional[float] = None, 
                      custom_fps: Optional[float] = None) -> List[dict]:
        """
        Extract frames from video with metadata.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds (None for end of video)
            custom_fps: Override default fps for this extraction
            
        Returns:
            List of frame metadata dictionaries
        """
        if not self.video_path:
            raise ValueError("No video loaded")
        
        if not self.duration or self.duration <= 0:
            raise ValueError(f"Invalid video duration: {self.duration}")
        
        fps = custom_fps if custom_fps else self.fps
        if fps <= 0:
            raise ValueError(f"Invalid FPS: {fps}")
        
        end_time = end_time if end_time else self.duration
        
        print(f"Extracting frames: start={start_time}, end={end_time}, fps={fps}, duration={self.duration}")
        
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError("Failed to open video for frame extraction")
        
        # Calculate frame intervals
        frame_interval = 1.0 / fps
        current_time = start_time
        
        while current_time < end_time:
            # Set position to current time
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame at time {current_time}")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create frame metadata
            frame_data = {
                'frame_id': len(frames),
                'timestamp': current_time,
                'timestamp_formatted': self._format_timestamp(current_time),
                'frame': frame_rgb,
                'frame_pil': Image.fromarray(frame_rgb)
            }
            
            frames.append(frame_data)
            current_time += frame_interval
        
        cap.release()
        print(f"Extracted {len(frames)} frames")
        return frames
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_frame_at_time(self, timestamp: float) -> Optional[dict]:
        """Get a single frame at specific timestamp."""
        frames = self.extract_frames(start_time=timestamp, end_time=timestamp + 0.1, custom_fps=1)
        return frames[0] if frames else None
    
    def cleanup(self):
        """Clean up temporary files."""
        if self.video_path and os.path.exists(self.video_path):
            try:
                os.unlink(self.video_path)
            except:
                pass
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

class FrameBatchProcessor:
    """Handles batching of frames for API calls."""
    
    def __init__(self, batch_size: int = 5):
        self.batch_size = batch_size
    
    def create_batches(self, frames: List[dict]) -> List[List[dict]]:
        """
        Create batches of frames for processing.
        
        Args:
            frames: List of frame metadata dictionaries
            
        Returns:
            List of frame batches
        """
        batches = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def frames_to_base64(self, frames: List[dict]) -> List[str]:
        """
        Convert frames to base64 strings for API transmission.
        
        Args:
            frames: List of frame metadata dictionaries
            
        Returns:
            List of base64 encoded frame strings
        """
        import base64
        
        base64_frames = []
        for frame_data in frames:
            # Convert PIL image to base64
            img_buffer = io.BytesIO()
            frame_data['frame_pil'].save(img_buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            base64_frames.append(img_str)
        
        return base64_frames
