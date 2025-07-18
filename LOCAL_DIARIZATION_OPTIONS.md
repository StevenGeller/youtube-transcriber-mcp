# Local Speaker Diarization Options (No HuggingFace Authentication Required)

This document outlines various speaker diarization options that work completely offline without requiring HuggingFace authentication tokens.

## Summary of Options

### 1. **Simple-Diarizer** (Recommended for Easy Setup)
- Uses pre-trained SpeechBrain models
- No authentication required after initial model download
- Good balance of accuracy and ease of use
- Installation: `pip install simple-diarizer`

### 2. **Custom Librosa + Sklearn Implementation** (Most Control)
- Completely offline, no external models needed
- Lightweight and fast
- Lower accuracy than pre-trained models
- Already implemented in `diarization_alternatives.py`

### 3. **Resemblyzer** (Good for Speaker Embeddings)
- Pre-trained speaker encoder
- Can be combined with custom clustering
- Installation: `pip install resemblyzer`

### 4. **WebRTCVAD + Clustering** (Lightweight VAD)
- Fast Voice Activity Detection
- Requires additional feature extraction
- Installation: `pip install webrtcvad`

### 5. **SpectralCluster** (Advanced Clustering)
- Google's spectral clustering implementation
- Installation: `pip install spectralcluster`

## Implementation Details

### Option 1: Simple-Diarizer

```python
from simple_diarizer.diarizer import Diarizer

diarizer = Diarizer(
    embed_model='xvec',  # or 'ecapa' 
    cluster_method='sc'  # spectral clustering
)

segments = diarizer.diarize("audio.wav", num_speakers=2)
```

**Pros:**
- Pre-trained models from SpeechBrain
- Good accuracy
- Simple API

**Cons:**
- Requires initial model download
- Larger dependency

### Option 2: Custom Implementation (Already Included)

The `diarization_alternatives.py` file includes a complete implementation using:
- **Librosa** for audio processing and MFCC extraction
- **Sklearn** for clustering (KMeans, Spectral, GMM)
- **Voice Activity Detection** using energy-based approach

```python
from diarization_alternatives import SimpleDiarizer

diarizer = SimpleDiarizer(n_speakers=2, clustering_method='spectral')
segments = diarizer.diarize("audio.wav")
```

**Features:**
- MFCC + delta features
- Multiple clustering algorithms
- Automatic speaker count estimation
- Label smoothing
- VAD integration

### Option 3: Resemblyzer-based Implementation

```python
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.cluster import SpectralClustering
import numpy as np

# Load pre-trained model
encoder = VoiceEncoder()

# Load and preprocess audio
wav = preprocess_wav("audio.wav")

# Create embeddings for segments
# (You need to segment the audio first)
embeddings = []
for segment in segments:
    embed = encoder.embed_utterance(segment)
    embeddings.append(embed)

# Cluster embeddings
clustering = SpectralClustering(n_clusters=n_speakers)
labels = clustering.fit_predict(embeddings)
```

### Option 4: WebRTCVAD + Feature Extraction

```python
import webrtcvad
import librosa
from sklearn.cluster import KMeans

vad = webrtcvad.Vad(3)  # Aggressiveness level 3

# Process audio in frames
frame_duration_ms = 30
frames = make_frames(audio, sample_rate, frame_duration_ms)

# Detect speech frames
speech_frames = []
for frame in frames:
    is_speech = vad.is_speech(frame.bytes, sample_rate)
    if is_speech:
        speech_frames.append(frame)

# Extract features from speech frames and cluster
```

## Integration with WhisperX

The `transcriber_with_local_diarization.py` file shows how to integrate these diarization methods with WhisperX:

1. **Transcribe with WhisperX** first to get word-level timestamps
2. **Run diarization** separately on the audio
3. **Assign speakers** to transcript segments based on time overlap

```python
from transcriber_with_local_diarization import YouTubeTranscriberWithLocalDiarization

# Use built-in simple diarizer
transcriber = YouTubeTranscriberWithLocalDiarization(
    use_simple_diarizer=True,
    n_speakers=2  # Optional
)

result = transcriber.process_video(youtube_url)
```

## Performance Comparison

| Method | Accuracy | Speed | Dependencies | Offline |
|--------|----------|-------|--------------|---------|
| Simple-Diarizer | High | Medium | SpeechBrain | ✓ |
| Custom Librosa | Medium | Fast | Minimal | ✓ |
| Resemblyzer | High | Medium | TensorFlow | ✓ |
| WebRTCVAD | Low | Very Fast | Minimal | ✓ |
| Pyannote (HF) | Very High | Slow | Many | Needs token |

## Recommendations

1. **For production use**: Simple-Diarizer or Resemblyzer
2. **For minimal dependencies**: Custom Librosa implementation
3. **For real-time applications**: WebRTCVAD + simple clustering
4. **For research/experimentation**: Try multiple methods and compare

## Installation

### Minimal Setup (Custom Implementation)
```bash
pip install librosa scikit-learn scipy
```

### Simple-Diarizer Setup
```bash
pip install simple-diarizer
```

### Full Setup (All Options)
```bash
pip install librosa scikit-learn scipy simple-diarizer resemblyzer webrtcvad spectralcluster
```

## Usage Examples

See the following files for complete examples:
- `diarization_alternatives.py` - Custom implementations
- `transcriber_with_local_diarization.py` - WhisperX integration
- Original `transcriber.py` - Pyannote-based implementation (requires HF token)

## Notes

- Speaker diarization quality depends heavily on audio quality
- Clear, distinct voices work best
- Background noise and overlapping speech reduce accuracy
- Most methods require at least 30 seconds of speech per speaker
- The number of speakers may need to be specified or estimated