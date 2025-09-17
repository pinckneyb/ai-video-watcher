# Surgical VOP (Verification of Proficiency) Assessment

A specialized video analysis application for evaluating surgical suturing technique using AI-powered assessment and structured rubrics.

## Overview

This application analyzes surgical procedure videos to assess suturing techniques including Simple Interrupted, Vertical Mattress, and Subcuticular patterns. It generates detailed PDF reports with scoring based on established surgical assessment criteria.

## Features

- **Specialized Surgical Assessment**: Evaluate suturing videos using medical-grade rubrics
- **Pattern Recognition**: Automatic detection of Simple Interrupted, Vertical Mattress, and Subcuticular techniques  
- **Professional Reporting**: Generate comprehensive PDF reports with scoring and feedback
- **Gold Standard Comparison**: Compare techniques against reference examples
- **Frame-by-frame Analysis**: Detailed analysis using GPT-5 Vision capabilities

## Installation

### Prerequisites

Make sure you have FFmpeg installed on your system:
- **Ubuntu/Debian**: `sudo apt install ffmpeg`
- **MacOS**: `brew install ffmpeg`  
- **Windows**: Download from https://ffmpeg.org/

### Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your OpenAI API key in one of these ways:
   - Set environment variable: `export OPENAI_API_KEY=your_key_here`
   - Create `config.json` with: `{"openai_api_key": "your_key_here"}`
   - Enter it through the app interface

## Usage

Launch the application:
```bash
streamlit run surgical_vop_app.py
```

Then:
1. Upload a surgical procedure video
2. Select or detect the suturing pattern
3. Configure analysis settings
4. Run the assessment
5. Download the generated PDF report

## Assessment Criteria

The app evaluates surgical techniques using structured rubrics for:
- **Technique execution** - Proper suture placement and spacing
- **Safety protocols** - Adherence to sterile technique
- **Efficiency** - Time management and smooth workflow  
- **Outcomes** - Final suture quality and tissue alignment

## File Structure

- `surgical_vop_app.py` - Main application
- `surgical_report_generator.py` - PDF report generation
- `gpt5_vision_client.py` - AI vision analysis
- `unified_rubric.JSON` - Assessment scoring criteria
- `*_narrative.txt` - Reference guides for each suture pattern
- `*_example.png` - Gold standard reference images

## Documentation

See the included documentation files for detailed information:
- `VOP_APP_ARCHITECTURE.md` - Technical architecture
- `COMPREHENSIVE_RECOVERY_GUIDE.md` - Recovery and troubleshooting
- `CURSOR_SESSION_PREP.md` - Development session guide