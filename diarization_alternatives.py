#!/usr/bin/env python3
"""
Alternative speaker diarization implementations that don't require HuggingFace authentication
"""

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from scipy.ndimage import median_filter
import scipy.signal
from typing import List, Dict, Tuple, Optional


class SimpleDiarizer:
    """
    A simple speaker diarization implementation using librosa and sklearn
    No authentication required, works completely offline
    """
    
    def __init__(self, n_speakers: Optional[int] = None, clustering_method: str = 'kmeans'):
        """
        Initialize the diarizer
        
        Args:
            n_speakers: Number of speakers (if known). If None, will try to estimate
            clustering_method: 'kmeans', 'spectral', or 'gmm'
        """
        self.n_speakers = n_speakers
        self.clustering_method = clustering_method
        
    def extract_features(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract acoustic features for speaker diarization
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Feature matrix (n_frames x n_features)
        """
        # Frame parameters
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)     # 10ms
        
        # Extract MFCCs (excluding 0th coefficient)
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=20,
            n_fft=frame_length,
            hop_length=hop_length
        )[1:, :]  # Exclude energy
        
        # Add delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        
        return features.T
    
    def voice_activity_detection(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Simple energy-based voice activity detection
        
        Args:
            audio: Audio signal
            sr: Sample rate
            
        Returns:
            Boolean array indicating speech frames
        """
        hop_length = int(0.010 * sr)
        
        # Compute energy
        energy = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Dynamic threshold based on energy distribution
        threshold = np.percentile(energy, 20)
        
        # Apply threshold
        speech_frames = energy > threshold
        
        # Apply median filter to remove short segments
        speech_frames = median_filter(speech_frames, size=5)
        
        return speech_frames
    
    def estimate_n_speakers(self, features: np.ndarray) -> int:
        """
        Estimate number of speakers using elbow method
        
        Args:
            features: Feature matrix
            
        Returns:
            Estimated number of speakers
        """
        max_speakers = min(6, len(features) // 100)  # Reasonable upper bound
        
        if max_speakers < 2:
            return 2
        
        inertias = []
        for k in range(2, max_speakers + 1):
            kmeans = KMeans(n_clusters=k, n_init=3, random_state=42)
            kmeans.fit(features)
            inertias.append(kmeans.inertia_)
        
        # Simple elbow detection
        if len(inertias) > 1:
            deltas = np.diff(inertias)
            elbow = np.argmax(deltas) + 2
            return elbow
        
        return 2
    
    def cluster_speakers(self, features: np.ndarray) -> np.ndarray:
        """
        Cluster features into speaker groups
        
        Args:
            features: Feature matrix
            
        Returns:
            Speaker labels
        """
        # Normalize features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        
        # Estimate number of speakers if not provided
        n_speakers = self.n_speakers
        if n_speakers is None:
            n_speakers = self.estimate_n_speakers(features_norm)
        
        # Apply clustering
        if self.clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=n_speakers, n_init=10, random_state=42)
        elif self.clustering_method == 'spectral':
            clusterer = SpectralClustering(
                n_clusters=n_speakers, 
                affinity='nearest_neighbors',
                n_neighbors=10,
                random_state=42
            )
        elif self.clustering_method == 'gmm':
            clusterer = GaussianMixture(
                n_components=n_speakers, 
                covariance_type='diag',
                random_state=42
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")
        
        labels = clusterer.fit_predict(features_norm)
        
        return labels
    
    def smooth_labels(self, labels: np.ndarray, window_size: int = 50) -> np.ndarray:
        """
        Smooth speaker labels to remove short segments
        
        Args:
            labels: Raw speaker labels
            window_size: Median filter window size
            
        Returns:
            Smoothed labels
        """
        # Apply median filter
        smoothed = median_filter(labels.astype(float), size=window_size)
        
        return smoothed.astype(int)
    
    def labels_to_segments(self, labels: np.ndarray, frame_times: np.ndarray) -> List[Dict]:
        """
        Convert frame-level labels to time segments
        
        Args:
            labels: Speaker labels for each frame
            frame_times: Time stamp for each frame
            
        Returns:
            List of speaker segments
        """
        segments = []
        
        if len(labels) == 0:
            return segments
        
        current_speaker = labels[0]
        start_time = frame_times[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                # End current segment
                segments.append({
                    'speaker': f'SPEAKER_{current_speaker:02d}',
                    'start': float(start_time),
                    'end': float(frame_times[i])
                })
                
                # Start new segment
                current_speaker = labels[i]
                start_time = frame_times[i]
        
        # Add final segment
        segments.append({
            'speaker': f'SPEAKER_{current_speaker:02d}',
            'start': float(start_time),
            'end': float(frame_times[-1])
        })
        
        return segments
    
    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Perform complete speaker diarization
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of speaker segments with start/end times
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Extract features
        features = self.extract_features(audio, sr)
        
        # Voice activity detection
        vad_frames = self.voice_activity_detection(audio, sr)
        
        # Ensure alignment
        min_len = min(len(features), len(vad_frames))
        features = features[:min_len]
        vad_frames = vad_frames[:min_len]
        
        # Filter features by VAD
        speech_features = features[vad_frames]
        
        if len(speech_features) < 10:
            # Not enough speech
            return [{
                'speaker': 'SPEAKER_00',
                'start': 0.0,
                'end': len(audio) / sr
            }]
        
        # Cluster speakers
        speech_labels = self.cluster_speakers(speech_features)
        
        # Map back to all frames
        all_labels = np.zeros(len(features), dtype=int)
        all_labels[vad_frames] = speech_labels + 1  # 0 reserved for non-speech
        
        # Smooth labels
        smoothed_labels = self.smooth_labels(all_labels)
        
        # Convert to time stamps
        hop_length = int(0.010 * sr)
        frame_times = librosa.frames_to_time(
            np.arange(len(smoothed_labels)), 
            sr=sr, 
            hop_length=hop_length
        )
        
        # Filter out non-speech segments
        speech_mask = smoothed_labels > 0
        speech_labels_final = smoothed_labels[speech_mask] - 1
        speech_times = frame_times[speech_mask]
        
        # Convert to segments
        segments = self.labels_to_segments(speech_labels_final, speech_times)
        
        return segments


class WhisperXCompatibleDiarizer:
    """
    A wrapper that makes SimpleDiarizer compatible with WhisperX output format
    """
    
    def __init__(self, device="cpu"):
        self.device = device
        self.diarizer = SimpleDiarizer(clustering_method='spectral')
    
    def __call__(self, audio_path: str) -> Dict:
        """
        Perform diarization and return in WhisperX-compatible format
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with 'segments' key containing diarization results
        """
        segments = self.diarizer.diarize(audio_path)
        
        # Convert to WhisperX format
        whisperx_segments = []
        for seg in segments:
            whisperx_segments.append({
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker']
            })
        
        return {'segments': whisperx_segments}
    
    def assign_speakers_to_transcript(self, diarization: Dict, transcript: Dict) -> Dict:
        """
        Assign speakers to transcript segments based on time overlap
        
        Args:
            diarization: Diarization results
            transcript: WhisperX transcript with segments
            
        Returns:
            Transcript with speaker assignments
        """
        diar_segments = diarization['segments']
        
        for trans_seg in transcript.get('segments', []):
            trans_start = trans_seg['start']
            trans_end = trans_seg['end']
            trans_mid = (trans_start + trans_end) / 2
            
            # Find overlapping speaker segment
            best_speaker = "SPEAKER_00"
            best_overlap = 0
            
            for diar_seg in diar_segments:
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
        
        return transcript


# Example usage functions
def demo_simple_diarization():
    """Demo of simple diarization"""
    diarizer = SimpleDiarizer(n_speakers=2, clustering_method='spectral')
    segments = diarizer.diarize("path/to/audio.wav")
    
    print("Speaker segments:")
    for seg in segments:
        print(f"{seg['speaker']}: {seg['start']:.2f}s - {seg['end']:.2f}s")


def demo_whisperx_integration():
    """Demo of WhisperX integration"""
    # This would replace the pyannote diarization in transcriber.py
    diarizer = WhisperXCompatibleDiarizer()
    
    # Perform diarization
    audio_path = "path/to/audio.wav"
    diarization_result = diarizer(audio_path)
    
    # Mock WhisperX transcript
    transcript = {
        'segments': [
            {'start': 0.0, 'end': 2.5, 'text': 'Hello, how are you?'},
            {'start': 2.5, 'end': 5.0, 'text': 'I am fine, thank you.'},
        ]
    }
    
    # Assign speakers
    transcript_with_speakers = diarizer.assign_speakers_to_transcript(
        diarization_result, 
        transcript
    )
    
    print("Transcript with speakers:")
    for seg in transcript_with_speakers['segments']:
        print(f"{seg.get('speaker', 'Unknown')}: {seg['text']}")


if __name__ == "__main__":
    # Run demos if executed directly
    print("Simple Diarization Demo:")
    print("-" * 50)
    # demo_simple_diarization()
    
    print("\nWhisperX Integration Demo:")
    print("-" * 50)
    # demo_whisperx_integration()