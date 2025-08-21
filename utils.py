"""
Utility functions for the video analysis app.
"""

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

def parse_timestamp(timestamp_str: str) -> float:
    """
    Parse timestamp string (HH:MM:SS) to seconds.
    
    Args:
        timestamp_str: Timestamp in HH:MM:SS format
        
    Returns:
        float: Time in seconds
    """
    try:
        # Handle HH:MM:SS format
        if ':' in timestamp_str:
            parts = timestamp_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:
                minutes, seconds = map(int, parts)
                return minutes * 60 + seconds
        
        # Handle seconds as float
        return float(timestamp_str)
        
    except (ValueError, TypeError):
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")

def format_timestamp(seconds: float) -> str:
    """
    Format seconds to HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted timestamp
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def validate_time_range(start_time: str, end_time: str, video_duration: float) -> bool:
    """
    Validate time range for rescan operations.
    
    Args:
        start_time: Start time string
        end_time: End time string
        video_duration: Total video duration in seconds
        
    Returns:
        bool: True if valid range
    """
    try:
        start_sec = parse_timestamp(start_time)
        end_sec = parse_timestamp(end_time)
        
        if start_sec < 0 or end_sec > video_duration:
            return False
        
        if start_sec >= end_sec:
            return False
        
        return True
        
    except ValueError:
        return False

def save_transcript(transcript: str, filename: str = None, prefix: str = "transcript") -> str:
    """
    Save transcript to markdown file.
    
    Args:
        transcript: Transcript text
        filename: Optional filename (will generate if not provided)
        prefix: Prefix for generated filename
        
    Returns:
        str: Saved filename
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.md"
    
    # Ensure .md extension
    if not filename.endswith('.md'):
        filename += '.md'
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(transcript)
    
    return filename

def save_timeline_json(events: List[Dict], filename: str = None) -> str:
    """
    Save event timeline to JSON file.
    
    Args:
        events: List of event dictionaries
        filename: Optional filename (will generate if not provided)
        
    Returns:
        str: Saved filename
    """
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"timeline_{timestamp}.json"
    
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Sort events by timestamp
    sorted_events = sorted(events, key=lambda x: x.get('timestamp', '00:00:00'))
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sorted_events, f, indent=2, ensure_ascii=False)
    
    return filename

def create_download_link(file_path: str, link_text: str = "Download") -> str:
    """
    Create a download link for Streamlit.
    
    Args:
        file_path: Path to the file
        link_text: Text to display for the link
        
    Returns:
        str: HTML download link
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = f.read()
    
    # For markdown files
    if file_path.endswith('.md'):
        b64 = data.encode()
        return f'<a href="data:file/md;base64,{b64.decode()}" download="{file_path}">{link_text}</a>'
    
    # For JSON files
    elif file_path.endswith('.json'):
        b64 = data.encode()
        return f'<a href="data:file/json;base64,{b64.decode()}" download="{file_path}">{link_text}</a>'
    
    return f'<a href="{file_path}" download>{link_text}</a>'

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.
    
    Args:
        filename: Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Limit length
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:100-len(ext)] + ext
    
    return filename

def extract_audio_from_video(video_path: str, audio_path: str = None) -> str:
    """
    Extract audio from video file using FFmpeg.
    
    Args:
        video_path: Path to video file
        audio_path: Optional path for output audio file
        
    Returns:
        str: Path to extracted audio file
    """
    if not audio_path:
        # Generate audio filename
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = f"{base_name}_audio.wav"
    
    try:
        import ffmpeg
        
        # Extract audio using FFmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16000')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        return audio_path
        
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

def transcribe_audio_with_whisper(audio_path: str, api_key: str) -> str:
    """
    Transcribe audio using OpenAI Whisper API.
    
    Args:
        audio_path: Path to audio file
        api_key: OpenAI API key
        
    Returns:
        str: Transcribed text
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        return transcript
        
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def estimate_processing_time(video_duration: float, fps: float, batch_size: int) -> str:
    """
    Estimate processing time for video analysis.
    
    Args:
        video_duration: Video duration in seconds
        fps: Frames per second to extract
        batch_size: Number of frames per batch
        
    Returns:
        str: Estimated time string
    """
    total_frames = int(video_duration * fps)
    total_batches = (total_frames + batch_size - 1) // batch_size
    
    # Rough estimate: 10-15 seconds per batch (API calls + processing)
    estimated_seconds = total_batches * 12
    
    if estimated_seconds < 60:
        return f"~{estimated_seconds} seconds"
    elif estimated_seconds < 3600:
        minutes = estimated_seconds // 60
        return f"~{minutes} minutes"
    else:
        hours = estimated_seconds // 3600
        minutes = (estimated_seconds % 3600) // 60
        return f"~{hours}h {minutes}m"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size string
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def extract_video_info(video_path: str) -> Dict[str, Any]:
    """
    Extract basic video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dict: Video information
    """
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Get file size
        file_size = os.path.getsize(video_path)
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "file_size": file_size,
            "file_size_formatted": format_file_size(file_size)
        }
        
    except Exception as e:
        return {"error": str(e)}
