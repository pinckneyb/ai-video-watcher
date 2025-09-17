# COMPREHENSIVE RECOVERY GUIDE
**Date**: January 16, 2025  
**Status**: FULLY FUNCTIONAL - All features working  
**GitHub Commit**: c9dec5d (latest)

## 🚨 CRITICAL WARNING
**NEVER REVERT TO OLD VERSIONS** - This document contains the complete working state. If you lose functionality, use this guide to restore it exactly.

---

## 📋 CURRENT WORKING STATE SUMMARY

### ✅ **FULLY FUNCTIONAL FEATURES**
1. **Enhanced Final Product Image Selection** - Multi-tier sampling with practice pad detection
2. **Intelligent Square Cropping** - Around practice pad boundaries starting from center
3. **Professional Image Placement** - Final product at top, gold standard below
4. **Concise Rubric Assessments** - 1-2 sentences, no timestamps/uncertainty
5. **Actionable Summative Comments** - Holistic critiques only
6. **User-Controlled Batch Size** - Slider 5-15, default 10
7. **High Performance Concurrency** - Max 150, default 100
8. **Clean Interface** - No verbose file listings
9. **STOP Buttons** - Graceful app control
10. **Multiple Video Support** - Individual pattern detection per file

---

## 🏗️ ARCHITECTURE OVERVIEW

### **Core Files (DO NOT DELETE)**
- `surgical_vop_app.py` - Main Streamlit application
- `surgical_report_generator.py` - PDF report generation with enhanced image processing
- `video_processor.py` - Video frame extraction (FFmpeg + OpenCV fallback)
- `gpt4o_client.py` - OpenAI API integration
- `unified_rubric.JSON` - Assessment criteria
- `profiles.py` - User profiles and narrative guides

### **Two-Stage AI Pipeline**
1. **Stage 1 (GPT-4o Vision)**: Analyzes video in configurable batches
2. **Stage 2 (GPT-5 Synthesis)**: Combines outputs into comprehensive narrative + scores

---

## 🔧 CRITICAL CODE SECTIONS

### **1. Enhanced Final Product Image Selection**
**Location**: `surgical_report_generator.py` lines 603-672

```python
def _extract_final_product_image_with_pad_detection(self, assessment_data: Dict[str, Any], target_width: float):
    """Extract and crop final product image with intelligent suturing pad detection."""
    # Multi-tier sampling strategy for best final product image
    duration = processor.duration
    candidate_frames = []
    
    # Tier 1: Last 3% of video (most likely to show final product without hands)
    self._sample_video_segment_enhanced(processor, duration * 0.97, duration, 10, candidate_frames, "final_3pct")
    
    # Tier 2: Last 7% of video (backup)
    self._sample_video_segment_enhanced(processor, duration * 0.93, duration * 0.97, 8, candidate_frames, "final_7pct")
    
    # Tier 3: Last 15% of video (final fallback)
    self._sample_video_segment_enhanced(processor, duration * 0.85, duration * 0.93, 6, candidate_frames, "final_15pct")
```

### **2. Intelligent Square Cropping**
**Location**: `surgical_report_generator.py` lines 843-949

```python
def _intelligent_crop_suturing_area(self, image) -> Image.Image:
    """Intelligently crop to focus on the suturing area with practice pad detection."""
    # Start search from center of image
    center_x, center_y = w // 2, h // 2
    
    # Detect practice pad boundaries using edge detection and contour analysis
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Use adaptive thresholding to find practice pad boundaries
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Find contours to identify practice pad
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

### **3. Strict Scoring System**
**Location**: `surgical_vop_app.py` lines 209-250

```python
enhancement_prompt = f"""YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE. You are training surgeons who will operate on real patients. Assume EVERY technique has flaws until proven otherwise.

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

For each rubric point, write ONE OR TWO SENTENCES that:
- Describe what you observed in the technique
- Explain why the performance earned its score
- NO timestamps, NO references to inability to judge
- The AI must judge from the evidence available

Then write a summative paragraph that:
- Provides entirely actionable critiques from holistic review
- NO reprise of individual rubric assessments
- NO timestamps, NO uncertainty about visibility
- Must be useful observations and nothing else
```

### **4. Image Placement (Vertical Layout)**
**Location**: `surgical_report_generator.py` lines 432-489

```python
def _create_image_comparison_section(self, assessment_data: Dict[str, Any]) -> List:
    """Create vertical comparison with final product at top, gold standard below."""
    # Extract final product image first (at top)
    story.append(Paragraph("<b>Learner Performance</b>", self.styles['Normal']))
    story.append(Spacer(1, 6))
    
    final_product_img = self._extract_final_product_image_with_pad_detection(assessment_data, 3*inch)
    story.append(final_product_img)
    story.append(Spacer(1, 20))
    
    # Add gold standard image below
    if gold_standard_path and os.path.exists(gold_standard_path):
        story.append(Paragraph("<b>Gold Standard Reference</b>", self.styles['Normal']))
        story.append(Spacer(1, 6))
```

### **5. User Controls**
**Location**: `surgical_vop_app.py` lines 518-531

```python
# Analysis settings
st.subheader("⚙️ Analysis Settings")
fps = st.slider("Analysis FPS", 1.0, 5.0, 5.0, 0.5)
batch_size = st.slider("Batch Size", 5, 15, 10, 1, help="Number of frames processed together in each batch")

# Concurrency settings
st.subheader("⚡ Concurrency Settings")
max_concurrent_batches = st.slider(
    "Concurrent Batches", 
    1, 150, 
    100,  # Default high performance setting
    step=1,
    help="Higher values = faster processing (requires OpenAI Tier 4+). Use 100-150 for maximum speed."
)
```

### **6. STOP Buttons**
**Location**: `surgical_vop_app.py` lines 568-571, 630-633, 611-614

```python
# STOP button in sidebar
st.subheader("🛑 App Control")
if st.button("🛑 STOP Application", type="secondary", help="Gracefully stop the application"):
    st.stop()

# Emergency STOP button in main area
if st.button("🛑 STOP Analysis", type="secondary", help="Stop current analysis process"):
    st.warning("🛑 Analysis stopped by user")
    st.stop()

# STOP button for batch processing
if st.button("🛑 STOP Batch", type="secondary", help="Stop batch processing"):
    st.warning("🛑 Batch processing stopped by user")
    st.stop()
```

---

## 📦 REQUIRED IMPORTS

### **surgical_report_generator.py**
```python
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import os
import io
import re
import numpy as np
import cv2
from PIL import Image
```

### **surgical_vop_app.py**
```python
import streamlit as st
import os
import tempfile
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from video_processor import VideoProcessor, FrameBatchProcessor
from gpt4o_client import GPT4oClient
from surgical_report_generator import SurgicalVOPReportGenerator
from profiles import SuturePatternDetector, RubricEngine
```

---

## 🎯 KEY FUNCTIONALITY TESTS

### **Test 1: Enhanced Image Selection**
1. Upload a surgical video
2. Run assessment
3. Generate PDF report
4. **Verify**: Final product image is intelligently cropped square around practice pad
5. **Verify**: Image is placed at top with "Learner Performance" label

### **Test 2: Strict Scoring**
1. Run assessment on any video
2. **Verify**: Scores are realistic (mostly 2-3, few 4-5)
3. **Verify**: Rubric responses are 1-2 sentences
4. **Verify**: No timestamps or uncertainty language

### **Test 3: User Controls**
1. **Verify**: Batch size slider (5-15, default 10)
2. **Verify**: Concurrency slider (1-150, default 100)
3. **Verify**: FPS slider (1.0-5.0, default 5.0)

### **Test 4: STOP Buttons**
1. **Verify**: STOP Application button in sidebar
2. **Verify**: STOP Analysis button in main area
3. **Verify**: STOP Batch button for multiple files

### **Test 5: Multiple Video Support**
1. Upload multiple videos with different patterns
2. **Verify**: Each file gets individual pattern detection
3. **Verify**: Mixed patterns are handled correctly

---

## 🚨 RECOVERY PROCEDURES

### **If You Lose Enhanced Image Processing**
1. Restore `_extract_final_product_image_with_pad_detection()` function
2. Restore `_intelligent_crop_suturing_area()` function
3. Restore `_sample_video_segment_enhanced()` function
4. Restore `_score_frame_quality_enhanced()` function
5. Restore `_calculate_center_penalty()` function
6. Restore `_detect_skin_tones()` function
7. Restore `_detect_metallic_objects()` function

### **If You Lose Strict Scoring**
1. Restore GPT-5 prompt in `create_surgical_vop_narrative()` function
2. Ensure "YOU ARE A STRICT ATTENDING SURGEON" language
3. Ensure "Score 2 should be your DEFAULT" philosophy
4. Ensure 1-2 sentence limit for rubric responses

### **If You Lose User Controls**
1. Restore batch size slider: `st.slider("Batch Size", 5, 15, 10, 1)`
2. Restore concurrency slider: `st.slider("Concurrent Batches", 1, 150, 100, step=1)`
3. Restore STOP buttons in all locations

### **If You Lose Image Placement**
1. Restore vertical layout in `_create_image_comparison_section()`
2. Ensure "Learner Performance" at top
3. Ensure "Gold Standard Reference" below
4. Ensure proper spacing and sizing

---

## 📁 FILE STRUCTURE (DO NOT MODIFY)

```
AI_video_watcher/
├── surgical_vop_app.py              # Main application (CRITICAL)
├── surgical_report_generator.py     # PDF generation (CRITICAL)
├── video_processor.py               # Video processing
├── gpt4o_client.py                  # OpenAI integration
├── profiles.py                      # Pattern detection
├── unified_rubric.JSON              # Assessment criteria
├── utils.py                         # Utilities
├── requirements.txt                 # Dependencies
├── surgical_requirements.txt        # Surgical app dependencies
├── COMPREHENSIVE_RECOVERY_GUIDE.md  # This document
└── [PDF reports and temp files]
```

---

## 🔄 GIT RECOVERY COMMANDS

### **If You Need to Restore from Git**
```bash
# Check current status
git status

# See recent commits
git log --oneline -10

# Restore to working commit
git checkout c9dec5d

# Or restore specific files
git checkout c9dec5d -- surgical_vop_app.py
git checkout c9dec5d -- surgical_report_generator.py
```

### **If You Need to Force Restore**
```bash
# Hard reset to working commit (DESTRUCTIVE - only if necessary)
git reset --hard c9dec5d

# Force push (DESTRUCTIVE - only if necessary)
git push origin master --force
```

---

## 🎛️ CURRENT SETTINGS (DO NOT CHANGE)

### **Performance Settings**
- **FPS**: 5.0 (default)
- **Batch Size**: 10 (default, range 5-15)
- **Concurrency**: 100 (default, range 1-150)
- **Upload Limit**: 2GB

### **Scoring Philosophy**
- **Default Score**: 2 (safe, functional technique)
- **Score 4**: Video good enough to teach attendings
- **Score 5**: Among best technique seen in career
- **Response Length**: 1-2 sentences maximum

### **Image Processing**
- **Sampling**: Last 3%, 7%, 15% of video
- **Quality Scoring**: Sharpness, contrast, brightness, edge density
- **Hand Detection**: HSV color space analysis
- **Instrument Detection**: Edge detection and line analysis
- **Cropping**: Square around practice pad boundaries

---

## 🚨 EMERGENCY CONTACTS

### **If All Else Fails**
1. **Check GitHub**: https://github.com/pinckneyb/ai-video_watcher
2. **Latest Working Commit**: c9dec5d
3. **Restore from Backup**: Use git commands above
4. **Test Core Functions**: Use test procedures above

---

## 📝 CHANGE LOG

### **Latest Changes (c9dec5d)**
- ✅ Enhanced final product image selection with practice pad detection
- ✅ Intelligent square cropping around practice pad boundaries
- ✅ Vertical image placement (final product top, gold standard bottom)
- ✅ Shortened rubric responses to 1-2 sentences
- ✅ Eliminated timestamps and uncertainty language
- ✅ Actionable summative comments from holistic review
- ✅ Batch size slider (5-15, default 10)
- ✅ Increased max concurrency to 150 (default 100)
- ✅ Removed verbose file listings
- ✅ Added STOP buttons for graceful control
- ✅ Fixed import conflicts in PDF generation

### **Previous Working State (6ddf5af)**
- ✅ Strict scoring system
- ✅ Enhanced final product image selection
- ✅ Paragraph-formatted summative comments
- ✅ GUI cleanup

---

## 🎯 SUCCESS CRITERIA

**The app is working correctly when:**
1. ✅ Final product images are intelligently cropped squares around practice pads
2. ✅ Images are placed vertically with proper labels
3. ✅ Rubric responses are 1-2 sentences with no timestamps
4. ✅ Summative comments are actionable holistic critiques
5. ✅ Batch size slider works (5-15, default 10)
6. ✅ Concurrency slider works (1-150, default 100)
7. ✅ STOP buttons work in all locations
8. ✅ Multiple videos get individual pattern detection
9. ✅ PDF reports generate without errors
10. ✅ All changes are saved to GitHub

---

**🚨 REMEMBER: This document is your safety net. If you lose functionality, use this guide to restore it exactly. Never revert to old versions without consulting this document first.**
