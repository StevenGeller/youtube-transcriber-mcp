#!/usr/bin/env python3
"""
Modified YouTube transcriber that uses local diarization without HuggingFace authentication
"""

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
from diarization_alternatives import WhisperXCompatibleDiarizer, SimpleDiarizer

# Set up logging
logger = logging.getLogger(__name__)


class YouTubeTranscriberWithLocalDiarization:
    def __init__(self, model_size="base", device="cuda" if torch.cuda.is_available() else "cpu", 
                 compute_type="float16" if torch.cuda.is_available() else "int8",
                 use_simple_diarizer=False, n_speakers=None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.temp_dir = tempfile.mkdtemp()
        self.use_simple_diarizer = use_simple_diarizer
        self.n_speakers = n_speakers
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
        """Transcribe audio using WhisperX with local speaker diarization"""
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
            
            # Local Diarization (no HuggingFace token required)
            logger.info("Performing local speaker diarization...")
            
            try:
                if self.use_simple_diarizer:
                    # Use the simple librosa/sklearn based diarizer
                    logger.info("Using SimpleDiarizer for speaker diarization")
                    diarizer = SimpleDiarizer(
                        n_speakers=self.n_speakers,
                        clustering_method='spectral'
                    )
                    diarization_segments = diarizer.diarize(audio_path)
                    
                    # Convert to WhisperX format
                    diarization_result = {'segments': diarization_segments}
                    
                    # Assign speakers to transcript segments
                    for trans_seg in result.get('segments', []):
                        trans_start = trans_seg['start']
                        trans_end = trans_seg['end']
                        trans_mid = (trans_start + trans_end) / 2
                        
                        # Find overlapping speaker segment
                        best_speaker = "SPEAKER_00"
                        best_overlap = 0
                        
                        for diar_seg in diarization_segments:
                            diar_start = diar_seg['start']
                            diar_end = diar_seg['end']
                            
                            # Calculate overlap
                            overlap_start = max(trans_start, diar_start)
                            overlap_end = min(trans_end, diar_end)
                            overlap = max(0, overlap_end - overlap_start)
                            
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_speaker = diar_seg['speaker']
                            elif overlap == 0 and diar_start <= trans_mid <= diar_end:
                                # Fallback: use midpoint
                                best_speaker = diar_seg['speaker']
                        
                        trans_seg['speaker'] = best_speaker
                else:
                    # Try to use simple-diarizer library if available
                    try:
                        from simple_diarizer.diarizer import Diarizer
                        logger.info("Using simple-diarizer library for speaker diarization")
                        
                        diarizer = Diarizer(
                            embed_model='xvec',  # or 'ecapa'
                            cluster_method='sc'  # spectral clustering
                        )
                        
                        # Perform diarization
                        segments = diarizer.diarize(
                            audio_path, 
                            num_speakers=self.n_speakers if self.n_speakers else None
                        )
                        
                        # Assign speakers to WhisperX segments
                        for trans_seg in result.get('segments', []):
                            trans_time = (trans_seg['start'] + trans_seg['end']) / 2
                            
                            # Find the speaker at this time
                            speaker_found = False
                            for diar_seg in segments:
                                if diar_seg['start'] <= trans_time <= diar_seg['end']:
                                    trans_seg['speaker'] = f"SPEAKER_{diar_seg['label']:02d}"
                                    speaker_found = True
                                    break
                            
                            if not speaker_found:
                                trans_seg['speaker'] = "SPEAKER_00"
                                
                    except ImportError:
                        logger.warning("simple-diarizer not installed, falling back to basic diarization")
                        # Fallback to our custom implementation
                        diarizer = WhisperXCompatibleDiarizer(device=self.device)
                        diarization_result = diarizer(audio_path)
                        result = diarizer.assign_speakers_to_transcript(diarization_result, result)
                        
            except Exception as e:
                logger.error(f"Diarization failed: {str(e)}")
                # Fallback: assign default speaker to all segments
                for i, segment in enumerate(result.get("segments", [])):
                    segment["speaker"] = "SPEAKER_00"
                logger.info("Using single speaker fallback")
            
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
        segment_start = 0
        last_end = 0
        
        for segment in result.get("segments", []):
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


# Example usage
if __name__ == "__main__":
    # Example 1: Using the simple built-in diarizer
    with YouTubeTranscriberWithLocalDiarization(
        use_simple_diarizer=True,
        n_speakers=2  # Optional: specify number of speakers if known
    ) as transcriber:
        result = transcriber.process_video("https://youtube.com/watch?v=...")
        
        if result['success']:
            print("Transcription successful!")
            for segment in result['formatted_segments'][:5]:
                print(f"{segment['speaker']}: {segment['text'][:100]}...")
        else:
            print(f"Error: {result['error']}")
    
    # Example 2: Using simple-diarizer library (if installed)
    # pip install simple-diarizer
    with YouTubeTranscriberWithLocalDiarization(
        use_simple_diarizer=False  # Will try to use simple-diarizer library
    ) as transcriber:
        result = transcriber.process_video("https://youtube.com/watch?v=...")