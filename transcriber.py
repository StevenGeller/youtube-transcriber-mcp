#!/usr/bin/env python3

import os
import sys
import tempfile
import shutil
from pathlib import Path
import yt_dlp
import whisperx
import torch
import gc
from tqdm import tqdm
import logging

# Set up logging
logger = logging.getLogger(__name__)

class YouTubeTranscriberWithDiarization:
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up temporary directory on exit"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Successfully cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                # Log but don't raise - we still want to exit cleanly
                logger.warning(f"Could not remove temp directory {self.temp_dir}: {e}")
        
    def download_audio(self, url):
        """Download audio from YouTube video"""
        # Use a simple filename to avoid issues with special characters
        output_path = os.path.join(self.temp_dir, 'audio.%(ext)s')
        
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
            'logger': logger,  # Use our logger instead of stdout
            'progress_hooks': [],  # Disable progress output
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info['title']
            # Get the actual downloaded file path - using simple filename
            audio_file = os.path.join(self.temp_dir, "audio.wav")
            return audio_file, title, info
            
    def transcribe_with_diarization(self, audio_path):
        """Transcribe audio using WhisperX with speaker diarization"""
        # Suppress stdout during entire transcription process
        import contextlib
        import io
        
        # Redirect stdout to stderr to avoid interfering with JSON protocol
        with contextlib.redirect_stdout(sys.stderr):
            # Load WhisperX model
            model = whisperx.load_model(self.model_size, self.device, compute_type=self.compute_type)
            
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe with word-level timestamps
            result = model.transcribe(audio, batch_size=16)
            
            # Align whisper output
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
            
            # Free up GPU memory
            del model
            del model_a
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Local Speaker Diarization (no HuggingFace required)
            try:
                from local_diarization import LocalSpeakerDiarizer, assign_speakers_to_transcript
                
                logger.info("Performing local speaker diarization...")
                diarizer = LocalSpeakerDiarizer(n_speakers=None)  # Auto-detect speakers
                diarization_segments = diarizer.diarize(audio_path)
                
                # Assign speakers to transcript segments
                result["segments"] = assign_speakers_to_transcript(
                    result["segments"], 
                    diarization_segments
                )
                
                logger.info(f"Diarization complete. Found {len(set(seg['speaker'] for seg in diarization_segments))} speakers.")
                
            except ImportError:
                # If local diarization not available, try HuggingFace as backup
                logger.warning("Local diarization not available, trying HuggingFace...")
                try:
                    import os
                    hf_token = os.environ.get("HF_TOKEN", None)
                    if hf_token:
                        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=self.device)
                        diarize_segments = diarize_model(audio)
                        result = whisperx.assign_word_speakers(diarize_segments, result)
                    else:
                        # Fallback: assign default speaker to all segments
                        for i, segment in enumerate(result["segments"]):
                            segment["speaker"] = "SPEAKER_00"
                        logger.info("Note: HF_TOKEN not set. Using single speaker.")
                except Exception as e:
                    # Final fallback
                    for i, segment in enumerate(result["segments"]):
                        segment["speaker"] = "SPEAKER_00"
                    logger.info(f"Note: Diarization unavailable ({str(e)}). Using single speaker.")
                    
            except Exception as e:
                # Fallback: assign default speaker to all segments
                for i, segment in enumerate(result["segments"]):
                    segment["speaker"] = "SPEAKER_00"
                logger.error(f"Diarization failed: {str(e)}. Using single speaker.")
            
            # Clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return result
    
    def format_transcript_with_speakers(self, result):
        """Format transcript with speaker labels and timestamps"""
        formatted_segments = []
        current_speaker = None
        speaker_text = []
        
        for segment in result["segments"]:
            speaker = segment.get("speaker", "Unknown")
            
            if speaker != current_speaker:
                if current_speaker is not None and speaker_text:
                    formatted_segments.append({
                        "speaker": current_speaker,
                        "text": " ".join(speaker_text),
                        "start": segment_start,
                        "end": last_end
                    })
                current_speaker = speaker
                speaker_text = [segment["text"]]
                segment_start = segment["start"]
            else:
                speaker_text.append(segment["text"])
            
            last_end = segment["end"]
        
        # Add the last segment
        if current_speaker is not None and speaker_text:
            formatted_segments.append({
                "speaker": current_speaker,
                "text": " ".join(speaker_text),
                "start": segment_start,
                "end": last_end
            })
        
        return formatted_segments
    
    def analyze_transcript(self, formatted_segments, video_info):
        """Analyze transcript for key takeaways and controversial points"""
        full_text = " ".join([seg["text"] for seg in formatted_segments])
        
        # Count speaker contributions
        speaker_stats = {}
        for segment in formatted_segments:
            speaker = segment["speaker"]
            if speaker not in speaker_stats:
                speaker_stats[speaker] = {"word_count": 0, "time_spoken": 0}
            speaker_stats[speaker]["word_count"] += len(segment["text"].split())
            speaker_stats[speaker]["time_spoken"] += segment["end"] - segment["start"]
        
        # Extract key points and controversial statements
        key_points = []
        controversial_indicators = [
            'controversial', 'debate', 'argue', 'disagree', 'problem',
            'issue', 'concern', 'mistake', 'wrong', 'fail', 'but',
            'however', 'actually', 'truth', 'reality', 'myth', 'criticism',
            'challenge', 'dispute', 'question', 'doubt'
        ]
        
        for segment in formatted_segments:
            text_lower = segment["text"].lower()
            
            # Look for potential key points (longer segments often contain main ideas)
            if len(segment["text"].split()) > 20:
                key_points.append({
                    "speaker": segment["speaker"],
                    "text": segment["text"],
                    "timestamp": f"[{int(segment['start']//60):02d}:{int(segment['start']%60):02d}]",
                    "type": "key_point"
                })
            
            # Look for controversial language
            for indicator in controversial_indicators:
                if indicator in text_lower:
                    key_points.append({
                        "speaker": segment["speaker"],
                        "text": segment["text"],
                        "timestamp": f"[{int(segment['start']//60):02d}:{int(segment['start']%60):02d}]",
                        "type": "controversial"
                    })
                    break
        
        return {
            'title': video_info.get('title', 'Unknown'),
            'duration': video_info.get('duration', 0),
            'uploader': video_info.get('uploader', 'Unknown'),
            'view_count': video_info.get('view_count', 0),
            'speaker_stats': speaker_stats,
            'key_points': key_points[:15],  # Top 15 key points
            'total_speakers': len(speaker_stats)
        }
    
    def process_video(self, url):
        """Main processing pipeline"""
        audio_path = None
        try:
            # Download audio
            audio_path, title, video_info = self.download_audio(url)
            
            # Verify the file exists
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Downloaded audio file not found at {audio_path}")
            
            # Transcribe with diarization
            result = self.transcribe_with_diarization(audio_path)
            
            # Format transcript
            formatted_segments = self.format_transcript_with_speakers(result)
            
            # Analyze
            analysis = self.analyze_transcript(formatted_segments, video_info)
            
            # Clean up the audio file immediately after processing
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass  # Ignore errors during cleanup
            
            return {
                'success': True,
                'formatted_segments': formatted_segments,
                'analysis': analysis,
                'raw_result': result
            }
            
        except Exception as e:
            # Ensure cleanup even on error
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            return {
                'success': False,
                'error': str(e)
            }