# 🎥 AI Video Watcher

A powerful video analysis application that uses GPT-4o to intelligently watch and narrate video content. Built with Streamlit, OpenCV, and OpenAI's GPT-4o model, this app provides human-like narrative descriptions of video content with temporal awareness and detailed event tracking.

## ✨ Features

### 🧠 Intelligent Video Analysis
- **Frame-by-frame analysis** using GPT-4o vision capabilities
- **Temporal awareness** with precise timestamp anchoring
- **Context continuity** across video segments
- **Batch processing** for efficient API usage

### 📊 Multiple Analysis Profiles
- **Generic**: Human-like narrative description
- **Surgical**: Rubric-aware surgical assessment
- **Sports**: Play-by-play sports commentary style

### 🔍 Advanced Rescan Capabilities
- **Segment rescanning** at higher FPS for detail
- **Time range selection** (HH:MM:SS format)
- **Enhanced analysis** of specific video segments

### 💾 Comprehensive Output
- **Markdown transcripts** with full narrative
- **JSON event timelines** with structured data
- **Downloadable results** in multiple formats

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key with GPT-4o access
- FFmpeg installed on your system

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI_video_watcher
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your OpenAI API key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

4. **Launch the application**
   ```bash
   streamlit run app.py
   ```

## 📖 Usage Guide

### 1. Video Input
- **Upload**: Drag and drop video files (MP4, AVI, MOV, MKV, WMV)
- **URL**: Provide direct links to video files
- **Supported formats**: Most common video formats

### 2. Configuration
- **Profile Selection**: Choose analysis style (Generic, Surgical, Sports)
- **FPS Settings**: Adjust frame extraction rate (0.5 - 5.0 fps)
- **Batch Size**: Configure frames per API call (3-15 frames)

### 3. Analysis Process
1. Load your video
2. Select analysis profile
3. Click "Start Analysis"
4. Monitor progress in real-time
5. Review results and download outputs

### 4. Rescan Segments
- Select time range (e.g., 00:01:30 to 00:02:00)
- Choose higher FPS for detailed analysis
- Get enhanced narrative of specific segments

## 🏗️ Architecture

### Core Modules

#### `video_processor.py`
- Video loading and frame extraction
- Frame metadata management
- Batch processing utilities

#### `gpt4o_client.py`
- OpenAI API integration
- Context management and condensation
- Response parsing and event extraction

#### `profiles.py`
- Profile system for different narration styles
- Customizable prompting templates
- Domain-specific analysis modes

#### `utils.py`
- Utility functions for file operations
- Timestamp parsing and validation
- Output formatting and saving

#### `app.py`
- Streamlit web interface
- User interaction and state management
- Real-time progress tracking

### Data Flow

```
Video Input → Frame Extraction → Batch Processing → GPT-4o Analysis → Context Condensation → Output Generation
```

## ⚙️ Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Custom Profiles
Add new analysis profiles by extending the `ProfileManager` class in `profiles.py`:

```python
def _get_custom_profile(self) -> Dict[str, Any]:
    return {
        "name": "Custom",
        "description": "Custom analysis profile",
        "base_prompt": "Your custom prompt here...",
        "rescan_prompt": "Your custom rescan prompt...",
        "context_condensation_prompt": "Your custom condensation prompt..."
    }
```

## 📊 Output Formats

### Transcript (Markdown)
```markdown
At 00:00:10, a person enters the room holding a box.

At 00:00:15, they set the box on the table and open the lid.

At 00:00:22, another person appears in the doorway.
```

### Event Timeline (JSON)
```json
[
  {
    "timestamp": "00:00:10",
    "event": "Person enters room with box",
    "confidence": 0.85
  },
  {
    "timestamp": "00:00:15",
    "event": "Person places box on table",
    "confidence": 0.9
  }
]
```

## 🔧 Advanced Features

### Context Condensation
The app automatically maintains a rolling context summary to ensure narrative continuity across video segments without exhausting API context limits.

### Batch Processing
Frames are processed in configurable batches to optimize API usage and maintain context across video segments.

### Error Handling
Comprehensive error handling for:
- Video loading failures
- API rate limits
- Invalid time ranges
- Network connectivity issues

## 🚨 Troubleshooting

### Common Issues

1. **FFmpeg not found**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   
   # Windows
   # Download from https://ffmpeg.org/download.html
   ```

2. **OpenAI API errors**
   - Verify API key is correct
   - Check API quota and billing
   - Ensure GPT-4o access is enabled

3. **Video loading issues**
   - Check video format compatibility
   - Verify file integrity
   - Ensure sufficient disk space

### Performance Tips

- **Lower FPS** for longer videos to reduce processing time
- **Smaller batch sizes** for more frequent context updates
- **Use rescan** for detailed analysis of specific segments only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4o API access
- **Streamlit** for the web framework
- **OpenCV** for video processing capabilities
- **FFmpeg** for video format support

## 📞 Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

---

**Built with ❤️ using Streamlit, OpenCV, and OpenAI GPT-4o**
