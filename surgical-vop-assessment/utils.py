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
        
        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist")
            return None
        
        # Check if video has audio stream
        probe = ffmpeg.probe(video_path)
        audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
        
        if not audio_streams:
            print(f"Warning: No audio stream found in {video_path}")
            return None
        
        print(f"Found {len(audio_streams)} audio stream(s) in {video_path}")
        
        # Extract audio using FFmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path, acodec='pcm_s16le', ac=1, ar='16000')
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # Verify audio file was created and has content
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
            print(f"Audio extracted successfully to {audio_path} ({os.path.getsize(audio_path)} bytes)")
            return audio_path
        else:
            print(f"Error: Audio file {audio_path} was not created or is empty")
            return None
        
    except ImportError:
        print("Error: ffmpeg-python not installed. Install with: pip install ffmpeg-python")
        return None
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
        # Check if audio file exists and has content
        if not os.path.exists(audio_path):
            print(f"Error: Audio file {audio_path} does not exist")
            return None
        
        if os.path.getsize(audio_path) == 0:
            print(f"Error: Audio file {audio_path} is empty")
            return None
        
        print(f"Transcribing {audio_path} ({os.path.getsize(audio_path)} bytes) with Whisper...")
        
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key)
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        if transcript and len(transcript.strip()) > 0:
            print(f"Whisper transcription successful: {len(transcript)} characters")
            return transcript
        else:
            print("Warning: Whisper returned empty transcript")
            return None
        
    except ImportError:
        print("Error: openai library not installed. Install with: pip install openai")
        return None
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return None

def transcribe_audio_with_diarization(audio_path: str, api_key: str) -> str:
    """
    Transcribe audio with speaker diarization using OpenAI Whisper + Pyannote.
    
    Args:
        audio_path: Path to audio file
        api_key: OpenAI API key
        
    Returns:
        str: Transcribed text with speaker labels
    """
    try:
        # First, get the basic transcript
        basic_transcript = transcribe_audio_with_whisper(audio_path, api_key)
        if not basic_transcript:
            return None
        
        # Then add diarization
        try:
            from pyannote.audio import Pipeline
            from pyannote.audio.pipelines.utils.hook import ProgressHook
            
            # Initialize diarization pipeline
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token="YOUR_HF_TOKEN"  # User needs to provide this
            )
            
            # Perform diarization
            diarization = pipeline(audio_path)
            
            # Format the output with speaker labels
            # This is a simplified version - in practice, you'd want more sophisticated
            # alignment between transcript and diarization timestamps
            
            formatted_transcript = f"TRANSCRIPT WITH SPEAKER IDENTIFICATION:\n{basic_transcript}\n\nSPEAKER SEGMENTS:\n"
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                formatted_transcript += f"Speaker {speaker}: {turn.start:.1f}s - {turn.end:.1f}s\n"
            
            return formatted_transcript
            
        except ImportError:
            # Fallback to basic transcript if diarization not available
            return f"TRANSCRIPT:\n{basic_transcript}\n\nNote: Speaker diarization not available. Install pyannote.audio for speaker identification."
        except Exception as e:
            # Fallback to basic transcript if diarization fails
            return f"TRANSCRIPT:\n{basic_transcript}\n\nNote: Speaker diarization failed: {e}"
        
    except Exception as e:
        print(f"Error in diarization transcription: {e}")
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

def analyze_audio_with_yamnet(audio_path: str) -> dict:
    """
    Analyze audio using Google's YAMNet model to classify non-speech sounds.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        dict: Audio classification results with timestamps and sound events
    """
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
        import numpy as np
        import soundfile as sf
        import csv
        
        print(f"Loading YAMNet model...")
        
        # Load YAMNet model from TensorFlow Hub
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load class map
        class_map = {}
        try:
            with open('yamnet_class_map.csv', 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    class_map[int(row['index'])] = row['display_name']
        except FileNotFoundError:
            print("Warning: yamnet_class_map.csv not found, using class indices")
            class_map = {i: f"Class_{i}" for i in range(521)}
        
        # Load and preprocess audio
        print(f"Processing audio file: {audio_path}")
        wav_data, sample_rate = sf.read(audio_path)
        
        # Ensure mono audio
        if wav_data.ndim > 1:
            wav_data = wav_data.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import resampy
            wav_data = resampy.resample(wav_data, sample_rate, 16000)
            sample_rate = 16000
        
        # Normalize audio to [-1.0, 1.0]
        wav_data = wav_data.astype(np.float32)
        if wav_data.max() > 1.0 or wav_data.min() < -1.0:
            wav_data = wav_data / np.max(np.abs(wav_data))
        
        print(f"Audio processed: {len(wav_data)/sample_rate:.2f} seconds at {sample_rate}Hz")
        
        # Run YAMNet inference
        print("Running YAMNet inference...")
        scores, embeddings, spectrogram = model(wav_data)
        
        # Convert scores to numpy
        scores = scores.numpy()
        
        # Process results - YAMNet outputs scores every 0.48 seconds
        hop_duration = 0.48  # seconds per frame
        results = []
        
        for i, frame_scores in enumerate(scores):
            timestamp = i * hop_duration
            
            # Get top 5 predictions for this frame
            top_indices = np.argsort(frame_scores)[-5:][::-1]
            top_scores = frame_scores[top_indices]
            
            frame_predictions = []
            for idx, score in zip(top_indices, top_scores):
                if score > 0.1:  # Only include predictions with confidence > 0.1
                    class_name = class_map.get(idx, f"Unknown_{idx}")
                    frame_predictions.append({
                        'class': class_name,
                        'confidence': float(score),
                        'class_id': int(idx)
                    })
            
            if frame_predictions:  # Only add if we have confident predictions
                results.append({
                    'timestamp': timestamp,
                    'predictions': frame_predictions
                })
        
        # Aggregate results to find dominant sound events
        all_classes = {}
        for result in results:
            for pred in result['predictions']:
                class_name = pred['class']
                if class_name not in all_classes:
                    all_classes[class_name] = []
                all_classes[class_name].append(pred['confidence'])
        
        # Calculate average confidence for each class
        summary = []
        for class_name, confidences in all_classes.items():
            avg_confidence = np.mean(confidences)
            max_confidence = np.max(confidences)
            detection_count = len(confidences)
            
            summary.append({
                'class': class_name,
                'avg_confidence': float(avg_confidence),
                'max_confidence': float(max_confidence),
                'detection_count': detection_count,
                'presence_ratio': detection_count / len(results) if results else 0
            })
        
        # Sort by average confidence
        summary.sort(key=lambda x: x['avg_confidence'], reverse=True)
        
        print(f"YAMNet analysis complete: {len(results)} frames analyzed, {len(summary)} sound classes detected")
        
        return {
            'frame_results': results,
            'summary': summary[:20],  # Top 20 classes
            'total_frames': len(results),
            'duration': len(wav_data) / sample_rate,
            'sample_rate': sample_rate
        }
        
    except ImportError as e:
        print(f"Error: Missing required libraries for YAMNet: {e}")
        print("Please install: tensorflow, tensorflow-hub, soundfile, resampy")
        return None
    except Exception as e:
        print(f"Error analyzing audio with YAMNet: {e}")
        return None
