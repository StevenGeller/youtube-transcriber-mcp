# YouTube Transcriber MCP

A Model Context Protocol (MCP) server that enables intelligent transcription of YouTube videos with automatic optimization for any video length. This tool integrates with desktop applications to provide high-quality, local transcription capabilities using OpenAI Whisper with smart processing strategies.

## Features

- **Automatic Strategy Selection**: Intelligently chooses optimal processing method based on video duration
- **Long Video Support**: Efficiently handles videos from minutes to hours with smart sampling
- **Local Processing**: All transcription happens on your machine - no external APIs required
- **Speaker Identification**: Automatically detects and labels different speakers in videos using local diarization
- **High Accuracy**: Leverages OpenAI Whisper for state-of-the-art transcription quality
- **MCP Integration**: Seamlessly works with MCP-compatible applications
- **Automatic Cleanup**: Downloaded files are automatically removed after processing
- **Multiple Model Sizes**: Choose from tiny to large models based on your accuracy/speed needs

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
- MCP-compatible application (e.g., Claude Desktop)

### Install FFmpeg

**macOS:**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows:**
Download from [FFmpeg website](https://ffmpeg.org/download.html)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/StevenGeller/youtube-transcriber-mcp.git
cd youtube-transcriber-mcp
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### For Claude Desktop

1. Open Claude Desktop settings
2. Navigate to the "Developer" section
3. Under "Edit Config", add the YouTube transcriber to your MCP servers:

```json
{
  "mcpServers": {
    "youtube-transcriber": {
      "command": "/path/to/youtube-transcriber-mcp/venv/bin/python",
      "args": ["/path/to/youtube-transcriber-mcp/youtube_mcp_server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

**Important:** Replace `/path/to/youtube-transcriber-mcp` with the actual path where you cloned the repository.

**Example for macOS:**
```json
{
  "mcpServers": {
    "youtube-transcriber": {
      "command": "/Users/yourusername/youtube-transcriber-mcp/venv/bin/python",
      "args": ["/Users/yourusername/youtube-transcriber-mcp/youtube_mcp_server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

4. Save the configuration
5. Restart Claude Desktop

### For Other MCP Clients

The server follows the MCP standard and can be used with any MCP-compatible client. The key configuration elements are:

- **Command:** Path to the Python interpreter in your virtual environment
- **Arguments:** Path to `youtube_mcp_server.py`
- **Environment:** Set `PYTHONUNBUFFERED=1` for proper output handling

## Usage

Once configured, you can transcribe YouTube videos by asking:

- "Transcribe this YouTube video: [URL]"
- "Get the transcript from: [URL]"
- "Transcribe [URL] without timestamps"

The server automatically optimizes processing based on video length:

### Automatic Strategy Selection

| Video Duration | Strategy | Description |
|----------------|----------|-------------|
| **≤ 10 minutes** | Full Transcription | Complete word-for-word transcription with base model |
| **10-60 minutes** | Chunked Processing | Parallel processing of 5-minute segments for faster results |
| **> 60 minutes** | Smart Sampling | Transcribes key sections (intro, conclusion, quarter points) for quick overview |

### Model Sizes

- **tiny**: Fastest, least accurate (~39M parameters)
- **base**: Good balance (default for short videos, ~74M parameters)
- **small**: Better accuracy (~244M parameters)
- **medium**: High accuracy (~769M parameters)
- **large**: Best accuracy (~1550M parameters)

**Note:** The server automatically selects appropriate model sizes based on video duration to optimize performance.

## Advanced Features

### Long Video Optimization

The transcriber automatically handles long videos efficiently:

- **Automatic Detection**: Analyzes video duration and selects optimal strategy
- **Chunked Processing**: For medium videos (10-60 min), splits into chunks for parallel processing
- **Smart Sampling**: For long videos (>60 min), intelligently samples key sections:
  - Introduction (first 2 minutes)
  - Key points at 25%, 50%, 75% marks
  - Conclusion (last 2 minutes)
- **Performance**: ~90% time savings on long videos while capturing essential content

### Speaker Diarization

The transcriber includes built-in local speaker diarization that works completely offline:

- Detects the number of speakers in the video
- Segments the audio by speaker
- Labels each transcript segment with the appropriate speaker
- Uses MFCC features and clustering for voice identification

## Project Structure

```
youtube-transcriber-mcp/
├── youtube_mcp_server.py     # Main MCP server
├── transcriber.py            # WhisperX transcription engine
├── local_diarization.py      # Local speaker diarization
├── quiet_transcriber.py      # Fallback transcriber
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Troubleshooting

### "Server disconnected" error
- Ensure FFmpeg is installed and in your PATH
- Check that all Python dependencies are installed
- Verify the file paths in your MCP configuration

### Memory issues
- Try using a smaller model size
- Ensure you have sufficient RAM (4GB+ recommended)

### Speaker identification issues
- The local diarization should work automatically
- If speaker detection fails, all speech will be labeled as SPEAKER_00
- Check the logs for any error messages

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is released into the public domain under The Unlicense - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [WhisperX](https://github.com/m-bain/whisperX) for enhanced transcription
- Uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) for reliable YouTube downloads
- Implements the [Model Context Protocol](https://modelcontextprotocol.io/) specification