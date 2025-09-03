# CursorAI Session Prep Document
**Date**: January 16, 2025  
**Project**: AI Video Watcher - Surgical VOP Assessment App

## PROJECT OVERVIEW
This is a Streamlit-based surgical assessment application that analyzes suturing videos using AI (GPT-4o for vision, GPT-5 for narrative synthesis) and generates professional PDF reports with VOP-aligned scoring.

## CURRENT APP STATUS
✅ **FULLY FUNCTIONAL** - All major features working, recently enhanced per user requirements
✅ **PUSHED TO GITHUB** - All changes safely stored at: `https://github.com/pinckneyb/ai-video-watcher.git`

## KEY ARCHITECTURE COMPONENTS

### Core Files
- **`surgical_vop_app.py`** - Main Streamlit application
- **`surgical_report_generator.py`** - PDF report generation 
- **`video_processor.py`** - Video frame extraction (FFmpeg + OpenCV fallback)
- **`gpt4o_client.py`** - OpenAI API integration
- **`unified_rubric.JSON`** - Assessment criteria
- **`profiles.py`** - User profiles and narrative guides

### Two-Stage AI Pipeline
1. **Stage 1 (GPT-4o Vision)**: Analyzes video in 6-frame batches, describes each batch locally
2. **Stage 2 (GPT-5 Synthesis)**: Combines ALL batch outputs into comprehensive narrative, generates scores

### Current Performance Settings
- **Analysis FPS**: 1.0 (default)
- **Batch Size**: 6 frames (fixed, optimal)
- **Concurrency**: 20 batches (maximum performance)
- **Upload Limit**: 2GB

## RECENT ENHANCEMENTS (ALL COMPLETED)

### 1. Much Stricter Scoring System
**Problem**: AI was inflating scores (too many 4s and 5s)
**Solution**: Completely rewrote GPT-5 prompt in `surgical_vop_app.py` lines 255-281:
- "YOU ARE A STRICT ATTENDING SURGEON WHO DEMANDS EXCELLENCE"
- "Assume EVERY technique has flaws until proven otherwise"
- "Score 2 should be your DEFAULT for safe, functional technique"
- "Score 4 means you would use this video to teach other attendings"
- "Score 5 means this is among the best technique you've seen in your entire career"

### 2. Enhanced Final Product Image Selection
**Problem**: Poor final product images with hands/instruments visible
**Solution**: Completely rewrote `_extract_final_product_image()` in `surgical_report_generator.py`:
- Multi-tier sampling (last 3%, 7%, 15% of video)
- 24 candidate frames analyzed for quality
- Hand/instrument detection and avoidance using HSV color space and edge detection
- Quality scoring based on sharpness, contrast, brightness, and edge density

### 3. Paragraph-Formatted Summative Comments
**Problem**: Summative comments were wall-of-text, hard to read
**Solution**: Added `_format_summative_paragraphs()` function:
- Intelligent sentence parsing
- 2-3 focused paragraphs with transition word detection
- HTML line breaks for PDF formatting

### 4. GUI Cleanup
**Problem**: Too much explanatory clutter
**Solution**: Removed all explanatory sections:
- Batch size explanation and slider (now fixed at 6)
- Upload troubleshooting section
- Concurrency guidance messages
- Performance history displays

## USER PREFERENCES & CONTEXT

### User Profile (Medical Professional)
- **Role**: Experienced attending surgeon training residents
- **Standards**: Extremely demanding, wants realistic/critical assessment
- **Communication Style**: Direct, no-nonsense, hates generic advice
- **Technical Level**: Limited coding knowledge, relies on AI for implementation

### Assessment Philosophy
- **VOP-Aligned Scoring**: 1-Remediation, 2-Minimal Pass, 3-Developing Pass, 4-Proficient, 5-Exemplary
- **Pass Rule**: All rubric points ≥ 2, otherwise Remediation
- **Critical Evaluation**: "You are training surgeons who will operate on real patients"
- **No Generic Advice**: Specific observations only, no "practice more" statements

### PDF Report Requirements
- **First Page**: Rubric assessments with scores, summative comment in paragraphs
- **Second Page**: Side-by-side gold standard vs learner final product images
- **NO Appendix**: Removed entirely per user request
- **Encoding**: Fixed character issues (0.5-1.0 cm spacing)

## TECHNICAL IMPLEMENTATION DETAILS

### Scoring Extraction
**Function**: `extract_rubric_scores_from_narrative()` in `surgical_vop_app.py`
- Uses regex to find "Score: X/5" patterns in GPT-5 output
- Handles provisional scores (3*) for visibility issues
- Defaults missing scores to 3

### Final Image Quality Scoring
**Functions**: `_score_frame_quality_enhanced()`, `_detect_skin_tones()`, `_detect_metallic_objects()`
- Combines sharpness (Laplacian variance), contrast (std dev), brightness analysis
- Penalizes skin tones and metallic objects in center region
- Tier bonuses prefer latest frames (final_3pct > final_7pct > final_15pct)

### Narrative Synthesis (Critical)
**Location**: `create_surgical_vop_narrative()` in `surgical_vop_app.py` lines 196-365
- **System Prompt**: Emphasizes Stage 2 synthesis, comprehensive timeline integration
- **User Prompt**: Contains raw transcript + rubric criteria + strict grading guidelines
- **API Call**: `model="gpt-5"`, `max_completion_tokens=8000`, `reasoning_effort="low"`
- **Fallback**: GPT-4o if GPT-5 fails

## CRITICAL SUCCESS FACTORS

### What Works Well
1. **Two-stage pipeline**: GPT-4o vision → GPT-5 synthesis prevents early batch bias
2. **Strict scoring prompts**: Produces realistic, critical assessments
3. **Robust video processing**: FFmpeg primary, OpenCV fallback handles various formats
4. **Error handling**: Multiple fallbacks and encoding strategies

### Common Failure Points
1. **Empty GPT-5 responses**: Usually token limit issues, solved with max_completion_tokens=8000
2. **Video processing**: Some formats fail FFmpeg, OpenCV fallback usually works
3. **Encoding issues**: Narrative files need utf-8, latin1, cp1252 fallbacks

## ENVIRONMENT & DEPLOYMENT

### Dependencies
- **Python Environment**: Virtual environment in `venv/`
- **Key Packages**: streamlit, openai, opencv-python, reportlab, pillow, ffmpeg-python
- **API Access**: OpenAI Tier 4+ required for GPT-5 and high concurrency

### Configuration
- **API Key**: Stored in `.env` file (gitignored)
- **Upload Config**: `.streamlit/config.toml` sets maxUploadSize=2048
- **Git**: API keys previously caused push protection, now resolved

## USER WORKFLOW EXPECTATIONS
1. **Upload video** → automatic pattern detection from filename
2. **Adjust FPS if needed** (1.0 default works well)
3. **Enter API key** → validates and saves to session
4. **Start Assessment** → two-stage analysis with progress tracking
5. **Review results** → scores displayed in GUI
6. **Generate PDF** → professional report with images
7. **Download** → timestamped filename

## NEXT SESSION PREPARATION
- **App is fully functional** - should work immediately
- **All enhancements complete** - scoring is strict, images are better, formatting is clean
- **GitHub is current** - latest commit `6ddf5af` contains all improvements
- **No known issues** - ready for user testing and further enhancements

## USER COMMUNICATION STYLE
- **Be direct and efficient** - user dislikes verbose explanations
- **Focus on specific technical details** when problems arise
- **Don't speculate** - user wants concrete solutions
- **Test immediately** after changes - user expects working code
- **Follow surgical terminology** when discussing assessments

