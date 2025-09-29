# Surgical VOP Assessment Application

## Overview

This is a Surgical Verification of Proficiency (VOP) Assessment System that analyzes video recordings of suturing procedures and generates professional assessment reports. The application uses AI to evaluate surgical technique against standardized rubrics, providing detailed feedback for medical education and resident training.

The system processes uploaded surgical videos through a two-stage AI pipeline: GPT-4o Vision for frame-by-frame analysis and GPT-5 for comprehensive narrative synthesis. It automatically detects suturing patterns (simple interrupted, subcuticular, vertical mattress) and generates professional PDF reports with detailed rubric assessments and comparative imagery.

## Recent Changes

**September 29, 2025** - Fixed critical frame extraction regression
- Restored FFmpeg-based frame extraction (primary method) after recent switch to OpenCV-only caused freezing issues
- Implemented streaming JPEG extraction via subprocess for much faster performance
- OpenCV retained as fallback for compatibility
- Frame extraction now uses `ffmpeg -ss {start} -i {input} -to {duration} -vf fps={fps} -f image2pipe -vcodec mjpeg` for efficient streaming

## User Preferences

Preferred communication style: Simple, everyday language.
- **AI Model Preference**: gpt-5-mini as default (other GPT-5 variants available in UI)
- **Performance Settings**: Batch size 7 frames, 50 concurrent batches, 1.0 FPS analysis

## System Architecture

### Core Application Structure
- **Main Application**: Streamlit-based web interface (`surgical_vop_app.py`) with video upload, processing controls, and report generation
- **Video Processing Pipeline**: FFmpeg-based frame extraction with OpenCV fallback (`video_processor.py`)
- **AI Analysis Engine**: Two-stage processing using OpenAI's GPT models (`gpt4o_client.py`, `gpt5_vision_client.py`)
- **Report Generation**: Professional PDF creation with embedded images and structured assessments (`surgical_report_generator.py`)

### Two-Stage AI Processing Pipeline
The system employs a sophisticated two-pass analysis approach:
1. **Stage 1 (GPT-4o Vision)**: Analyzes video frames in configurable batches (default 6 frames) with surgical-specific prompting focused on technique observation
2. **Stage 2 (GPT-5 Synthesis)**: Combines all frame observations into coherent narrative assessments with numerical scoring against rubric criteria

### Pattern Detection and Assessment
- **Automatic Pattern Recognition**: Detects suturing patterns from filenames and folder structure
- **Rubric-Based Evaluation**: Uses `unified_rubric.JSON` containing 7-point assessment criteria for different suturing techniques
- **Narrative Templates**: Pre-defined assessment frameworks for simple interrupted, subcuticular, and vertical mattress patterns

### Image Processing and Selection
- **Final Product Extraction**: Multi-tier sampling from video end (last 3%, 7%, 15%) to find optimal closure images
- **Practice Pad Detection**: Computer vision algorithms to identify suturing practice pads and exclude hands/instruments
- **Intelligent Cropping**: Square cropping around detected practice pad boundaries starting from frame center
- **Quality Assessment**: Automated scoring based on sharpness, contrast, brightness, and edge density

### Performance and Concurrency
- **Configurable Processing**: User-controlled FPS (0.5-5.0), batch sizes (3-15 frames), and concurrency levels (1-150 batches)
- **Resource Management**: Built-in rate limiting, retry logic, and memory optimization for large video files
- **Progress Tracking**: Real-time processing status with batch completion monitoring

### Assessment Framework
- **Structured Scoring**: 1-5 point scale with defined competency levels (1=Novice, 2=Competent, 3=Proficient, 4=Advanced, 5=Expert)
- **Holistic Evaluation**: Covers technical execution, tissue handling, instrument economy, and final product quality
- **Contextual Feedback**: Pattern-specific assessment criteria adapted to different suturing techniques

## External Dependencies

### AI Services
- **OpenAI API**: GPT-4o Vision for frame analysis and GPT-5 for narrative synthesis
- **API Tier Requirements**: OpenAI Tier 4+ recommended for optimal concurrent processing performance

### Video Processing
- **FFmpeg**: Primary video processing library for frame extraction and format conversion
- **OpenCV**: Fallback video processing and computer vision operations
- **PIL/Pillow**: Image manipulation and format conversion

### Web Framework and UI
- **Streamlit**: Web application framework with file upload, progress tracking, and interactive controls
- **ReportLab**: Professional PDF generation with embedded images and structured layouts

### Data Processing
- **NumPy**: Numerical computations for image analysis and quality scoring
- **JSON**: Structured data storage for rubrics, assessments, and configuration

### File Handling
- **Pathlib**: Modern file path operations and temporary file management
- **Base64**: Image encoding for HTML reports and API transmission