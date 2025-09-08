#!/usr/bin/env python3
"""
Test app for debugging final product image selection
Allows testing the AI frame selection logic in isolation
"""

import streamlit as st
import cv2
import numpy as np
import base64
import os
from openai import OpenAI
from PIL import Image
import io

st.set_page_config(
    page_title="Final Product Image Selector Test",
    page_icon="🔍",
    layout="wide"
)

def extract_candidate_frames(video_path, search_duration=5.0, sample_interval=0.2):
    """Extract candidate frames from the end of video."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        candidate_frames = []
        
        # Search backward from last frame
        for i, time_offset in enumerate(np.arange(0, search_duration, sample_interval)):
            timestamp = duration - time_offset
            if timestamp >= 0:
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    candidate_frames.append({
                        'index': i,
                        'timestamp': timestamp,
                        'frame': frame,
                        'frame_number': frame_number
                    })
        
        cap.release()
        return candidate_frames
        
    except Exception as e:
        st.error(f"Error extracting frames: {e}")
        return []

def ai_select_frame(candidate_frames, api_key):
    """Use AI to select the best final product frame."""
    try:
        client = OpenAI(api_key=api_key)
        
        # Convert frames to base64
        frame_data = []
        for frame_info in candidate_frames:
            _, buffer = cv2.imencode('.jpg', frame_info['frame'])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            frame_data.append({
                'index': frame_info['index'],
                'timestamp': frame_info['timestamp'],
                'base64': frame_b64
            })
        
        # Create prompt
        prompt = f"""You are searching backward from the end of a surgical video to find the perfect final product image.

CRITICAL TASK: Find a frame showing ONLY the practice pad with completed sutures and NOTHING ELSE.

THE FRAME MUST CONTAIN:
- The suturing practice pad (synthetic skin pad on board/surface)
- Completed sutures visible on the pad
- NOTHING ELSE

THE FRAME MUST NOT CONTAIN:
- Hands, fingers, or gloves (any color - blue, green, latex, etc.)
- Any part of a person's head, face, or body
- Surgical instruments (scissors, forceps, needle drivers, etc.)
- Needles or suture material being manipulated
- Any human presence whatsoever

SEARCH STRATEGY: These {len(frame_data)} frames are from the last 5 seconds, ordered from most recent backward.
Find the first frame that shows ONLY the practice pad with completed sutures.

FRAMES (newest to oldest):
{[f"Frame {f['index']}: {f['timestamp']:.1f}s" for f in frame_data]}

Examine each frame carefully. Respond with ONLY the frame number (0-{len(frame_data)-1}) that shows just the practice pad.
If NO frame shows only the practice pad, respond with "NONE".

Frame number:"""

        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{f['base64']}",
                            "detail": "high"
                        }
                    } for f in frame_data
                ]
            }
        ]
        
        # Make API call - GPT-5 specific parameters
        response = client.chat.completions.create(
            model="gpt-5",
            messages=messages,
            max_completion_tokens=50,
            reasoning_effort="low"
            # Note: GPT-5 only supports default temperature=1
        )
        
        ai_response = response.choices[0].message.content.strip()
        return ai_response, frame_data
        
    except Exception as e:
        st.error(f"Error in AI selection: {e}")
        return None, frame_data

def main():
    st.title("🔍 Final Product Image Selector Test")
    st.markdown("*Debug tool for testing AI frame selection logic*")
    
    # API key input
    api_key = st.text_input("OpenAI API Key", type="password", help="Required for GPT-5 frame selection")
    
    # Video upload
    uploaded_file = st.file_uploader(
        "Upload Surgical Video",
        type=['mp4', 'avi', 'mov', 'mkv', 'm4v'],
        help="Upload a surgical video to test final product image selection"
    )
    
    if uploaded_file and api_key:
        # Save video temporarily in organized folder
        os.makedirs("temp_videos", exist_ok=True)
        temp_path = os.path.join("temp_videos", f"temp_test_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        st.success(f"Video uploaded: {uploaded_file.name}")
        
        # Extract candidate frames
        st.subheader("🎬 Extracting Candidate Frames")
        with st.spinner("Extracting frames from last 5 seconds..."):
            candidate_frames = extract_candidate_frames(temp_path)
        
        if candidate_frames:
            st.success(f"Extracted {len(candidate_frames)} candidate frames")
            
            # Show all candidate frames
            st.subheader("📸 Candidate Frames (Newest to Oldest)")
            
            cols = st.columns(5)
            for i, frame_info in enumerate(candidate_frames[:20]):  # Show first 20
                col_idx = i % 5
                with cols[col_idx]:
                    # Convert frame to PIL for display
                    pil_img = Image.fromarray(cv2.cvtColor(frame_info['frame'], cv2.COLOR_BGR2RGB))
                    st.image(pil_img, caption=f"Frame {i}: {frame_info['timestamp']:.1f}s", use_container_width=True)
            
            # AI selection
            if st.button("🤖 Run AI Selection", type="primary"):
                st.subheader("🧠 AI Frame Selection")
                with st.spinner("GPT-5 analyzing frames..."):
                    ai_response, frame_data = ai_select_frame(candidate_frames, api_key)
                
                if ai_response:
                    st.write(f"**AI Response**: '{ai_response}'")
                    
                    if ai_response == "NONE":
                        st.warning("⚠️ AI found no optimal frames - all contain hands/instruments/heads")
                        st.info("Using most recent frame as fallback")
                        selected_frame = candidate_frames[0]
                    else:
                        try:
                            selected_index = int(ai_response)
                            if 0 <= selected_index < len(candidate_frames):
                                selected_frame = candidate_frames[selected_index]
                                st.success(f"✅ AI selected frame {selected_index}")
                            else:
                                st.error(f"Invalid index {selected_index}")
                                selected_frame = candidate_frames[0]
                        except ValueError:
                            st.error(f"Non-numeric response: {ai_response}")
                            selected_frame = candidate_frames[0]
                    
                    # Display selected frame
                    st.subheader("🎯 Selected Final Product Image")
                    selected_pil = Image.fromarray(cv2.cvtColor(selected_frame['frame'], cv2.COLOR_BGR2RGB))
                    st.image(selected_pil, caption=f"Selected: Frame {selected_frame['index']} at {selected_frame['timestamp']:.1f}s")
                    
                    # Analysis
                    st.subheader("🔍 Analysis")
                    if ai_response == "NONE":
                        st.warning("**Issue**: No clean frames found in last 5 seconds")
                        st.info("**Recommendation**: Check if video ends with hands/instruments still in frame")
                    else:
                        st.success("**Result**: AI found a clean final product frame")
                        st.info(f"**Timing**: Selected frame from {selected_frame['timestamp']:.1f}s ({5-(selected_frame['timestamp']-(max(candidate_frames, key=lambda x: x['timestamp'])['timestamp']-5)):.1f}s from end)")
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
    
    else:
        st.info("👆 Upload a video and enter API key to test frame selection")

if __name__ == "__main__":
    main()
