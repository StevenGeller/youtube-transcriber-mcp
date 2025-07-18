#!/usr/bin/env python3
"""
Silent YouTube transcriber - no output to stdout
"""

import os
import sys
import tempfile
import shutil
import gc
import logging

# Configure logging to only use stderr
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Set environment to reduce verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

class QuietYouTubeTranscriber:
    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.device = "cpu"  # Use CPU to avoid CUDA output
        self.compute_type = "int8"
        self.temp_dir = tempfile.mkdtemp()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def download_audio(self, url):
        """Download audio from YouTube video"""
        import yt_dlp
        
        output_path = os.path.join(self.temp_dir, 'audio.%(ext)s')
        
        # Completely silent yt-dlp options
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
            'noprogress': True,
            'logger': type('', (), {'debug': lambda *a: None, 'warning': lambda *a: None, 'error': lambda *a: None})(),
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')
            audio_file = os.path.join(self.temp_dir, "audio.wav")
            return audio_file, title, info
    
    def transcribe(self, audio_path):
        """Transcribe audio using Whisper"""
        import whisper
        
        # Load model silently
        model = whisper.load_model(self.model_size, device=self.device)
        
        # Transcribe
        result = model.transcribe(
            audio_path,
            fp16=False,
            verbose=False,
            task='transcribe'
        )
        
        # Add speaker labels to segments
        for segment in result.get("segments", []):
            segment["speaker"] = "SPEAKER_00"
        
        # Clean up
        del model
        gc.collect()
        
        return result
    
    def process_video(self, url):
        """Process video and return results"""
        try:
            # Download audio
            audio_path, title, video_info = self.download_audio(url)
            
            # Verify file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError("Audio download failed")
            
            # Transcribe
            result = self.transcribe(audio_path)
            
            # Clean up audio file immediately
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # Format segments
            formatted_segments = []
            for segment in result.get("segments", []):
                formatted_segments.append({
                    "speaker": segment.get("speaker", "SPEAKER_00"),
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"]
                })
            
            # Create analysis
            analysis = {
                'title': video_info.get('title', 'Unknown'),
                'duration': video_info.get('duration', 0),
                'uploader': video_info.get('uploader', 'Unknown'),
                'view_count': video_info.get('view_count', 0),
                'total_speakers': 1,
                'speaker_stats': {
                    "SPEAKER_00": {
                        "word_count": sum(len(seg["text"].split()) for seg in formatted_segments),
                        "time_spoken": sum(seg["end"] - seg["start"] for seg in formatted_segments)
                    }
                },
                'key_points': []
            }
            
            return {
                'success': True,
                'formatted_segments': formatted_segments,
                'analysis': analysis
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }