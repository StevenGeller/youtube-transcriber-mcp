#!/usr/bin/env python3
"""
Local speaker diarization using librosa and sklearn
No external APIs or authentication required
"""

import numpy as np
import librosa
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.signal import medfilt
import logging

logger = logging.getLogger(__name__)

class LocalSpeakerDiarizer:
    def __init__(self, n_speakers=None, min_speakers=2, max_speakers=5):
        """
        Initialize local diarizer
        
        Args:
            n_speakers: Fixed number of speakers (None for auto-detection)
            min_speakers: Minimum speakers for auto-detection
            max_speakers: Maximum speakers for auto-detection
        """
        self.n_speakers = n_speakers
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.sample_rate = 16000  # Standard for speech processing
        
    def extract_features(self, audio_path, frame_length=0.025, frame_shift=0.010):
        """Extract MFCC features from audio"""
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Calculate frame parameters
        n_fft = int(frame_length * sr)
        hop_length = int(frame_shift * sr)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=13,
            n_fft=n_fft, hop_length=hop_length
        )
        
        # Add delta features for better speaker discrimination
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Combine features
        features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        
        # Calculate energy for voice activity detection
        energy = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
        
        # Time stamps for each frame
        timestamps = librosa.frames_to_time(
            np.arange(features.shape[1]), 
            sr=sr, 
            hop_length=hop_length
        )
        
        return features.T, energy, timestamps
    
    def detect_speech_regions(self, energy, threshold_percentile=30):
        """Simple energy-based voice activity detection"""
        threshold = np.percentile(energy, threshold_percentile)
        speech_mask = energy > threshold
        
        # Apply median filter to smooth
        speech_mask = medfilt(speech_mask.astype(float), kernel_size=21) > 0.5
        
        return speech_mask
    
    def estimate_speakers(self, features, max_speakers=5):
        """Estimate optimal number of speakers using silhouette score"""
        if self.n_speakers is not None:
            return self.n_speakers
            
        scores = []
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        for n in range(self.min_speakers, min(max_speakers + 1, len(features))):
            if n >= len(features):
                break
                
            kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(features_scaled, labels)
                scores.append(score)
            else:
                scores.append(-1)
        
        # Return number of speakers with highest silhouette score
        if scores:
            return np.argmax(scores) + self.min_speakers
        else:
            return self.min_speakers
    
    def cluster_speakers(self, features, n_speakers):
        """Cluster features into speaker groups"""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Try spectral clustering first (often better for speaker diarization)
        try:
            clustering = SpectralClustering(
                n_clusters=n_speakers,
                affinity='nearest_neighbors',
                n_neighbors=10,
                random_state=42
            )
            labels = clustering.fit_predict(features_scaled)
        except:
            # Fallback to KMeans
            kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            
        return labels
    
    def smooth_labels(self, labels, window_size=50):
        """Smooth speaker labels using majority voting"""
        smoothed = np.copy(labels)
        
        for i in range(len(labels)):
            start = max(0, i - window_size // 2)
            end = min(len(labels), i + window_size // 2)
            
            if end > start:
                window_labels = labels[start:end]
                unique, counts = np.unique(window_labels, return_counts=True)
                smoothed[i] = unique[np.argmax(counts)]
                
        return smoothed
    
    def merge_segments(self, labels, timestamps, min_duration=0.5):
        """Merge consecutive segments from same speaker"""
        segments = []
        
        if len(labels) == 0:
            return segments
            
        current_speaker = labels[0]
        start_time = timestamps[0]
        
        for i in range(1, len(labels)):
            if labels[i] != current_speaker:
                # End current segment
                if timestamps[i-1] - start_time >= min_duration:
                    segments.append({
                        'start': float(start_time),
                        'end': float(timestamps[i-1]),
                        'speaker': f"SPEAKER_{current_speaker:02d}"
                    })
                
                # Start new segment
                current_speaker = labels[i]
                start_time = timestamps[i]
        
        # Add final segment
        if len(timestamps) > 0 and timestamps[-1] - start_time >= min_duration:
            segments.append({
                'start': float(start_time),
                'end': float(timestamps[-1]),
                'speaker': f"SPEAKER_{current_speaker:02d}"
            })
            
        return segments
    
    def diarize(self, audio_path):
        """Perform speaker diarization on audio file"""
        try:
            # Extract features
            features, energy, timestamps = self.extract_features(audio_path)
            
            # Detect speech regions
            speech_mask = self.detect_speech_regions(energy)
            
            # Filter features to only speech regions
            speech_features = features[speech_mask]
            speech_timestamps = timestamps[speech_mask]
            
            if len(speech_features) < 10:
                # Not enough speech for diarization
                logger.warning("Not enough speech detected for diarization")
                return [{
                    'start': 0.0,
                    'end': float(timestamps[-1]) if len(timestamps) > 0 else 1.0,
                    'speaker': 'SPEAKER_00'
                }]
            
            # Estimate or use provided number of speakers
            n_speakers = self.estimate_speakers(speech_features)
            logger.info(f"Detected {n_speakers} speakers")
            
            # Cluster speakers
            labels = self.cluster_speakers(speech_features, n_speakers)
            
            # Smooth labels
            labels = self.smooth_labels(labels)
            
            # Create full label array (including non-speech)
            full_labels = np.zeros(len(features), dtype=int)
            full_labels[speech_mask] = labels
            
            # Merge segments
            segments = self.merge_segments(full_labels, timestamps)
            
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {str(e)}")
            # Return single speaker as fallback
            return [{
                'start': 0.0,
                'end': float(timestamps[-1]) if 'timestamps' in locals() and len(timestamps) > 0 else 1.0,
                'speaker': 'SPEAKER_00'
            }]

def assign_speakers_to_transcript(transcript_segments, diarization_segments):
    """Assign speaker labels to transcript segments based on time overlap"""
    
    for transcript_seg in transcript_segments:
        t_start = transcript_seg['start']
        t_end = transcript_seg['end']
        t_mid = (t_start + t_end) / 2
        
        # Find diarization segment with maximum overlap
        max_overlap = 0
        assigned_speaker = "SPEAKER_00"
        
        for diar_seg in diarization_segments:
            d_start = diar_seg['start']
            d_end = diar_seg['end']
            
            # Calculate overlap
            overlap_start = max(t_start, d_start)
            overlap_end = min(t_end, d_end)
            overlap = max(0, overlap_end - overlap_start)
            
            # Also check if midpoint falls in segment
            if d_start <= t_mid <= d_end:
                overlap += (t_end - t_start) * 0.5  # Bonus for midpoint match
            
            if overlap > max_overlap:
                max_overlap = overlap
                assigned_speaker = diar_seg['speaker']
        
        transcript_seg['speaker'] = assigned_speaker
    
    return transcript_segments