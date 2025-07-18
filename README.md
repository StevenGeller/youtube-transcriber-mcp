# YouTube Transcriber MCP

A Model Context Protocol (MCP) server that enables transcription of YouTube videos with speaker identification. This tool integrates with desktop applications to provide high-quality, local transcription capabilities using OpenAI Whisper.

## Features

- **Local Processing**: All transcription happens on your machine - no external APIs required
- **Speaker Identification**: Automatically detects and labels different speakers in videos using local diarization (no HuggingFace token needed)
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

### Model Sizes

- **tiny**: Fastest, least accurate (~39M parameters)
- **base**: Good balance (default, ~74M parameters)
- **small**: Better accuracy (~244M parameters)
- **medium**: High accuracy (~769M parameters)
- **large**: Best accuracy (~1550M parameters)

## Advanced Features

### Speaker Diarization

The transcriber now includes built-in local speaker diarization that works without any external APIs or tokens. It automatically:

- Detects the number of speakers in the video
- Segments the audio by speaker
- Labels each transcript segment with the appropriate speaker

The local diarization uses:
- MFCC feature extraction for voice characteristics
- Clustering algorithms to group similar voices
- Energy-based voice activity detection

For enhanced diarization using HuggingFace models (optional):

```bash
export HF_TOKEN="your-huggingface-token"
```

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

### No speaker identification
- This is normal without a HuggingFace token
- The transcription will still work with a single speaker label

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is released into the public domain under The Unlicense - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [WhisperX](https://github.com/m-bain/whisperX) for enhanced transcription
- Uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) for reliable YouTube downloads
- Implements the [Model Context Protocol](https://modelcontextprotocol.io/) specification