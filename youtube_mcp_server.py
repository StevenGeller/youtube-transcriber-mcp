#!/usr/bin/env python3
"""
MCP Server for YouTube Transcription - Fixed version
"""

import os
import sys
import ssl
import certifi

# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Redirect all output to stderr during imports
import contextlib
with contextlib.redirect_stdout(sys.stderr):
    import asyncio
    from typing import Any, Dict, List, Sequence
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    import mcp.server.stdio
    import mcp.types as types

# Initialize the server
server = Server("youtube-transcriber")

@server.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """List available YouTube transcription tools"""
    return [
        types.Tool(
            name="transcribe_youtube",
            description="Transcribe a YouTube video with timestamps",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "YouTube video URL to transcribe"
                    },
                    "model_size": {
                        "type": "string",
                        "enum": ["tiny", "base", "small", "medium", "large"],
                        "default": "base",
                        "description": "Whisper model size"
                    },
                    "include_timestamps": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include timestamps in the transcript"
                    }
                },
                "required": ["url"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
    """Handle tool execution"""
    
    if name == "transcribe_youtube":
        url = arguments.get("url", "")
        model_size = arguments.get("model_size", "base")
        include_timestamps = arguments.get("include_timestamps", True)
        
        try:
            # Run transcription in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                transcribe_video_quiet,
                url,
                model_size
            )
            
            if not result['success']:
                return [types.TextContent(
                    type="text",
                    text=f"Error transcribing video: {result['error']}"
                )]
            
            # Format the response
            lines = []
            analysis = result.get('analysis', {})
            
            lines.append(f"# YouTube Video Transcript\n")
            lines.append(f"**Title:** {analysis.get('title', 'Unknown')}")
            lines.append(f"**Duration:** {analysis.get('duration', 0) // 60} minutes\n")
            
            lines.append("## Transcript\n")
            
            segments = result.get('formatted_segments', [])
            for segment in segments:
                if include_timestamps:
                    timestamp = f"[{int(segment['start']//60):02d}:{int(segment['start']%60):02d}]"
                    lines.append(f"{timestamp} {segment['text']}\n")
                else:
                    lines.append(f"{segment['text']}\n")
            
            return [types.TextContent(
                type="text",
                text="\n".join(lines)
            )]
            
        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]
    
    return [types.TextContent(
        type="text",
        text=f"Unknown tool: {name}"
    )]

def transcribe_video_quiet(url: str, model_size: str) -> Dict[str, Any]:
    """Transcribe video with output suppressed"""
    # Suppress all output
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        # Use the quiet transcriber
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from quiet_transcriber import QuietYouTubeTranscriber
        
        with QuietYouTubeTranscriber(model_size=model_size) as transcriber:
            result = transcriber.process_video(url)
            
        return result
        
    finally:
        # Restore output
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

async def main():
    """Run the server"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="youtube-transcriber",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())