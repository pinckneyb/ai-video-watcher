"""
Video processing module for frame extraction and video handling.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import os
import tempfile
import requests
from PIL import Image
import io
import subprocess

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
                    # Local file - try multiple path variations if original doesn't exist
                    if os.path.exists(source):
                        self.video_path = source
                    else:
                        # Try temp_videos with temp_ prefix
                        basename = os.path.basename(source)
                        temp_path = os.path.join("temp_videos", f"temp_{basename}")
                        if os.path.exists(temp_path):
                            self.video_path = temp_path
                        else:
                            # Try temp_videos without temp_ prefix
                            temp_path2 = os.path.join("temp_videos", basename)
                            if os.path.exists(temp_path2):
                                self.video_path = temp_path2
                            else:
                                raise FileNotFoundError(f"Video file not found: {source} (also tried {temp_path}, {temp_path2})")
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
        """Extract video properties using OpenCV."""
        try:
            if not self.video_path:
                raise ValueError("No video path available")
            
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_orig = cap.get(cv2.CAP_PROP_FPS)
                self.duration = self.total_frames / fps_orig if fps_orig > 0 else 0
                print(f"OpenCV: Duration={self.duration}, Frames={self.total_frames}, FPS={fps_orig}")
            else:
                print("Failed to open video with OpenCV")
                self.duration = 0
                self.total_frames = 0
            cap.release()
            
        except Exception as e:
            print(f"Error getting video properties with OpenCV: {e}")
            self.duration = 0
            self.total_frames = 0
    
    def extract_frames(self, start_time: float = 0.0, end_time: Optional[float] = None,
                      custom_fps: Optional[float] = None) -> List[dict]:
        """
        Extract frames from video with metadata using FFmpeg (primary) or OpenCV (fallback).
        
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
        
        end_time = end_time if end_time else max(0, self.duration - 1.0)  # Use 1 second before end as final frame
        
        print(f"Extracting frames: start={start_time}, end={end_time}, fps={fps}, duration={self.duration}")
        
        # Try FFmpeg first (much faster), fall back to OpenCV if FFmpeg fails
        try:
            frames = self._extract_frames_ffmpeg(start_time, end_time, fps)
            if frames:
                print(f"Extracted {len(frames)} frames using FFmpeg (fast extraction)")
                return frames
        except Exception as e:
            print(f"FFmpeg extraction failed: {e}, falling back to OpenCV")
        
        # Fallback to OpenCV for frame extraction
        frames = []
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError("Failed to open video for frame extraction")
        
        # Get total frame count and FPS for more accurate positioning
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} total frames at {video_fps} FPS")
        
        # Calculate frame intervals
        frame_interval = 1.0 / fps
        current_time = start_time
        consecutive_failures = 0
        max_failures = 10  # Allow some failures but not endless loop
        
        while current_time < end_time and consecutive_failures < max_failures:
            try:
                # Try frame-based positioning as alternative to time-based
                frame_number = int(current_time * video_fps)
                
                # Method 1: Time-based
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()
                
                if not ret and frame_number < total_frames:
                    # Method 2: Frame-based
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                    ret, frame = cap.read()
                
                if ret and frame is not None:
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
                    consecutive_failures = 0  # Reset failure counter
                else:
                    consecutive_failures += 1
                    if consecutive_failures <= 3:  # Only log first few failures
                        print(f"Failed to read frame at time {current_time} (failure {consecutive_failures})")
                
                current_time += frame_interval
                
            except Exception as e:
                print(f"Error extracting frame at {current_time}: {e}")
                consecutive_failures += 1
                current_time += frame_interval
        
        cap.release()
        
        if consecutive_failures >= max_failures:
            print(f"WARNING: Stopped extraction due to {max_failures} consecutive failures")
        
        print(f"Extracted {len(frames)} frames using OpenCV (requested {int((end_time - start_time) * fps)} frames)")
        
        if len(frames) == 0:
            raise ValueError("No frames could be extracted from video. Video may be corrupted or in unsupported format.")
        
        return frames
    
    
    def _extract_frames_ffmpeg(self, start_time: float, end_time: float, fps: float) -> List[dict]:
        """
        Extract frames using FFmpeg subprocess (faster than OpenCV seeking).
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds  
            fps: Frames per second to extract
            
        Returns:
            List of frame metadata dictionaries
        """
        import shutil
        
        # Check if FFmpeg is available
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg not found in PATH")
        
        # Build FFmpeg command for streaming JPEG frames
        # -ss before -i for fast seeking to start position
        # -to limits extraction duration
        # -vf fps= sets sampling rate
        # -f image2pipe -vcodec mjpeg outputs JPEG frames to stdout
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),
            '-i', self.video_path,
            '-to', str(end_time - start_time),  # Duration from start
            '-vf', f'fps={fps}',
            '-vsync', 'vfr',
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-q:v', '3',  # Quality (2-5 is good, lower = better)
            '-'
        ]
        
        frames = []
        try:
            # Run FFmpeg and capture stdout
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            # Read JPEG frames from stdout
            # JPEG starts with FFD8 and ends with FFD9
            jpeg_start = b'\xff\xd8'
            jpeg_end = b'\xff\xd9'
            
            buffer = b''
            frame_count = 0
            current_time = start_time
            time_increment = 1.0 / fps
            
            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                    
                buffer += chunk
                
                # Find complete JPEG frames in buffer
                while True:
                    start_idx = buffer.find(jpeg_start)
                    if start_idx == -1:
                        break
                    
                    end_idx = buffer.find(jpeg_end, start_idx + 2)
                    if end_idx == -1:
                        # Incomplete frame, keep in buffer
                        break
                    
                    # Extract complete JPEG frame
                    jpeg_data = buffer[start_idx:end_idx + 2]
                    buffer = buffer[end_idx + 2:]
                    
                    try:
                        # Convert JPEG bytes to PIL Image
                        img = Image.open(io.BytesIO(jpeg_data))
                        img_rgb = img.convert('RGB')
                        
                        # Create frame metadata
                        frame_data = {
                            'frame_id': frame_count,
                            'timestamp': current_time,
                            'timestamp_formatted': self._format_timestamp(current_time),
                            'frame': np.array(img_rgb),
                            'frame_pil': img_rgb
                        }
                        
                        frames.append(frame_data)
                        frame_count += 1
                        current_time += time_increment
                        
                    except Exception as e:
                        print(f"Error decoding JPEG frame {frame_count}: {e}")
                        continue
            
            # Wait for process to complete
            return_code = process.wait(timeout=30)
            
            if return_code != 0:
                stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                raise RuntimeError(f"FFmpeg failed with code {return_code}: {stderr_output}")
            
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError("FFmpeg extraction timed out")
        except Exception as e:
            if process.poll() is None:
                process.kill()
            raise
        
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
    
    def frames_to_base64(self, frames: List[dict]) -> List[str]:
        """
        Convert frames to base64 strings for API transmission.
        
        Args:
            frames: List of frame metadata dictionaries
            
        Returns:
            List of base64 encoded frame strings
        """
        import base64
        import io
        
        base64_frames = []
        for frame_data in frames:
            # Convert PIL image to base64
            img_buffer = io.BytesIO()
            frame_data['frame_pil'].save(img_buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(img_buffer.getvalue()).decode()
            base64_frames.append(img_str)
        
        return base64_frames
    
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
    
