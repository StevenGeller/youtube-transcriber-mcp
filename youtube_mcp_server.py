#!/usr/bin/env python3
"""
MCP Server for YouTube Transcription - Auto-optimized for video length
"""

import os
import sys
import ssl
import certifi
import tempfile
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

# Redirect all output to stderr during imports
import contextlib
with contextlib.redirect_stdout(sys.stderr):
    import asyncio
    from typing import Any, Dict, List, Sequence, Tuple
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
            # First, get video info to determine duration
            loop = asyncio.get_event_loop()
            video_info = await loop.run_in_executor(
                None,
                get_video_info_quiet,
                url
            )
            
            if not video_info:
                return [types.TextContent(
                    type="text",
                    text="Error: Could not retrieve video information"
                )]
            
            duration = video_info.get('duration', 0)
            duration_minutes = duration / 60
            
            # Select strategy based on duration
            if duration_minutes <= 10:
                # Short video: use full transcription
                strategy = 'full'
                selected_model = model_size
            elif duration_minutes <= 60:
                # Medium video: use chunked processing
                strategy = 'chunked'
                selected_model = 'tiny'  # Use faster model for chunks
            else:
                # Long video: use smart sampling
                strategy = 'smart_sample'
                selected_model = 'tiny'
            
            # Run appropriate transcription method
            if strategy == 'full':
                result = await loop.run_in_executor(
                    None,
                    transcribe_video_quiet,
                    url,
                    selected_model
                )
            elif strategy == 'chunked':
                result = await transcribe_video_chunked(
                    url,
                    video_info,
                    selected_model
                )
            else:  # smart_sample
                result = await transcribe_video_sampled(
                    url,
                    video_info,
                    selected_model
                )
            
            if not result['success']:
                return [types.TextContent(
                    type="text",
                    text=f"Error transcribing video: {result['error']}"
                )]
            
            # Format the response
            lines = []
            analysis = result.get('analysis', video_info)
            
            lines.append(f"# YouTube Video Transcript\n")
            lines.append(f"**Title:** {analysis.get('title', 'Unknown')}")
            lines.append(f"**Duration:** {analysis.get('duration', 0) // 60} minutes")
            lines.append(f"**Processing Strategy:** {result.get('strategy', strategy)}")
            if result.get('coverage'):
                lines.append(f"**Coverage:** {result['coverage']}")
            lines.append("")
            
            lines.append("## Transcript\n")
            
            segments = result.get('formatted_segments', result.get('segments', []))
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

def get_video_info_quiet(url: str) -> Dict[str, Any]:
    """Get video info quietly"""
    import yt_dlp
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'skip_download': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return {
                'title': info.get('title', 'Unknown'),
                'duration': info.get('duration', 0),
                'uploader': info.get('uploader', 'Unknown'),
                'view_count': info.get('view_count', 0),
                'upload_date': info.get('upload_date', 'Unknown')
            }
    except:
        return None

def transcribe_video_quiet(url: str, model_size: str) -> Dict[str, Any]:
    """Transcribe video with output suppressed - for short videos"""
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
            
        if result['success']:
            result['strategy'] = 'full'
            
        return result
        
    finally:
        # Restore output
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

async def transcribe_video_chunked(url: str, video_info: Dict, model_size: str) -> Dict[str, Any]:
    """Transcribe video in chunks for medium-length videos"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Download audio
        loop = asyncio.get_event_loop()
        audio_path = await loop.run_in_executor(
            None,
            download_audio_quiet,
            url,
            temp_dir
        )
        
        if not audio_path:
            return {'success': False, 'error': 'Failed to download audio'}
        
        duration = video_info.get('duration', 0)
        chunk_duration = 300  # 5 minutes
        
        # Split into chunks
        chunks = await split_audio_chunks(audio_path, duration, chunk_duration, temp_dir)
        
        # Process chunks in parallel
        max_workers = min(4, len(chunks))
        all_segments = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for chunk_info in chunks:
                future = loop.run_in_executor(
                    executor,
                    transcribe_chunk,
                    chunk_info,
                    model_size
                )
                futures.append(future)
            
            # Gather results
            results = await asyncio.gather(*futures)
            
            for result in results:
                if result['success']:
                    all_segments.extend(result['segments'])
        
        # Sort by timestamp
        all_segments.sort(key=lambda x: x['start'])
        
        # Format segments
        formatted_segments = []
        for seg in all_segments:
            formatted_segments.append({
                'speaker': 'SPEAKER_00',
                'text': seg['text'],
                'start': seg['start'],
                'end': seg['end']
            })
        
        return {
            'success': True,
            'formatted_segments': formatted_segments,
            'analysis': video_info,
            'strategy': 'chunked',
            'chunks_processed': len(chunks)
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

async def transcribe_video_sampled(url: str, video_info: Dict, model_size: str) -> Dict[str, Any]:
    """Smart sampling for long videos"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        duration = video_info.get('duration', 0)
        
        # Define sample points
        samples = [
            (0, min(120, duration)),  # First 2 minutes
            (max(0, duration//4 - 60), min(duration//4 + 60, duration)),  # Quarter
            (max(0, duration//2 - 60), min(duration//2 + 60, duration)),  # Middle
            (max(0, 3*duration//4 - 60), min(3*duration//4 + 60, duration)),  # 3/4
            (max(0, duration - 120), duration)  # Last 2 minutes
        ]
        
        # Remove duplicates
        samples = list(set(samples))
        samples.sort()
        
        # Download audio
        loop = asyncio.get_event_loop()
        audio_path = await loop.run_in_executor(
            None,
            download_audio_quiet,
            url,
            temp_dir
        )
        
        if not audio_path:
            return {'success': False, 'error': 'Failed to download audio'}
        
        # Process samples
        all_segments = []
        
        for i, (start, end) in enumerate(samples):
            if start < end:
                segment_path = os.path.join(temp_dir, f"sample_{i}.wav")
                
                # Extract segment
                await extract_audio_segment(audio_path, segment_path, start, end - start)
                
                # Transcribe
                result = await loop.run_in_executor(
                    None,
                    transcribe_segment,
                    segment_path,
                    model_size,
                    start
                )
                
                if result['success']:
                    all_segments.extend(result['segments'])
        
        # Sort by timestamp
        all_segments.sort(key=lambda x: x['start'])
        
        # Format segments
        formatted_segments = []
        for seg in all_segments:
            formatted_segments.append({
                'speaker': 'SPEAKER_00',
                'text': seg['text'],
                'start': seg['start'],
                'end': seg['end']
            })
        
        # Calculate coverage
        total_sampled = sum(end - start for start, end in samples)
        coverage_percent = (total_sampled / duration) * 100 if duration > 0 else 0
        
        return {
            'success': True,
            'formatted_segments': formatted_segments,
            'analysis': video_info,
            'strategy': 'smart_sample',
            'coverage': f'{coverage_percent:.0f}% of video sampled',
            'samples': len(samples)
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def download_audio_quiet(url: str, temp_dir: str) -> str:
    """Download audio quietly"""
    import yt_dlp
    
    output_path = os.path.join(temp_dir, 'audio.%(ext)s')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
        return os.path.join(temp_dir, "audio.wav")
    except:
        return None

async def split_audio_chunks(audio_path: str, duration: int, chunk_duration: int, temp_dir: str) -> List[Tuple[int, str, int]]:
    """Split audio into chunks"""
    chunks = []
    num_chunks = (duration + chunk_duration - 1) // chunk_duration
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        chunk_path = os.path.join(temp_dir, f"chunk_{i:04d}.wav")
        
        await extract_audio_segment(audio_path, chunk_path, start_time, chunk_duration)
        chunks.append((i, chunk_path, start_time))
    
    return chunks

async def extract_audio_segment(input_path: str, output_path: str, start: float, duration: float):
    """Extract audio segment using ffmpeg"""
    cmd = [
        'ffmpeg', '-i', input_path,
        '-ss', str(start),
        '-t', str(duration),
        '-c', 'copy',
        output_path,
        '-y'
    ]
    
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL
    )
    await proc.communicate()

def transcribe_chunk(chunk_info: Tuple[int, str, int], model_size: str) -> Dict:
    """Transcribe a single chunk"""
    chunk_idx, chunk_path, start_offset = chunk_info
    
    try:
        import whisper
        
        model = whisper.load_model(model_size)
        result = model.transcribe(chunk_path, fp16=False, verbose=False)
        
        # Adjust timestamps
        for segment in result.get('segments', []):
            segment['start'] += start_offset
            segment['end'] += start_offset
        
        return {
            'success': True,
            'segments': result.get('segments', [])
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def transcribe_segment(audio_path: str, model_size: str, time_offset: float) -> Dict:
    """Transcribe an audio segment"""
    try:
        import whisper
        
        model = whisper.load_model(model_size)
        result = model.transcribe(audio_path, fp16=False, verbose=False)
        
        # Adjust timestamps
        for segment in result.get('segments', []):
            segment['start'] += time_offset
            segment['end'] += time_offset
        
        return {
            'success': True,
            'segments': result.get('segments', [])
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

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