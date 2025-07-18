# Automatic Strategy Selection for YouTube Transcriber MCP

The enhanced MCP server now automatically selects the optimal transcription strategy based on video duration. No configuration needed!

## How It Works

When you request a transcription, the server:
1. Fetches video metadata to determine duration
2. Automatically selects the best strategy
3. Applies optimized settings for that strategy

## Strategy Selection Logic

| Video Duration | Strategy | Model | Description |
|----------------|----------|-------|-------------|
| **≤ 10 minutes** | Full | base | Complete transcription with better accuracy |
| **10-60 minutes** | Chunked | tiny | Parallel processing of 5-minute chunks |
| **> 60 minutes** | Smart Sample | tiny | Transcribes key sections (~30% coverage) |

## What Each Strategy Does

### 1. Full Transcription (Short Videos)
- Complete word-for-word transcription
- Uses larger model for better accuracy
- Ideal for short clips, tutorials, music videos

### 2. Chunked Processing (Medium Videos)
- Splits video into 5-minute segments
- Processes up to 4 chunks in parallel
- Maintains 100% coverage with faster processing
- Perfect for lectures, podcasts, interviews

### 3. Smart Sampling (Long Videos)
- Transcribes strategic portions:
  - First 2 minutes (introduction)
  - Samples at 25%, 50%, 75% points
  - Last 2 minutes (conclusion)
- Provides ~30% coverage focusing on key content
- Ideal for conferences, livestreams, long-form content

## Usage

Simply use the MCP as before - no changes needed:

```
Transcribe this video: https://www.youtube.com/watch?v=VIDEO_ID
```

The server automatically handles everything!

## Benefits

1. **No Manual Configuration**: Works out of the box
2. **Optimized Performance**: Each strategy is tuned for its use case
3. **Balanced Trade-offs**: 
   - Short videos: Maximum quality
   - Medium videos: Full coverage with speed
   - Long videos: Key insights without hours of processing

## Examples

### Short Video (3 minutes)
- Strategy: Full transcription
- Processing time: ~30 seconds
- Coverage: 100%

### Medium Video (45 minutes)
- Strategy: Chunked (9 chunks)
- Processing time: ~5 minutes
- Coverage: 100%

### Long Video (3.5 hours)
- Strategy: Smart sampling
- Processing time: ~3-5 minutes
- Coverage: ~30% (key sections)

## Technical Details

The server determines strategy in `handle_call_tool()`:

```python
if duration_minutes <= 10:
    strategy = 'full'
elif duration_minutes <= 60:
    strategy = 'chunked'
else:
    strategy = 'smart_sample'
```

Each strategy has its own optimized implementation:
- `transcribe_video_quiet()` - Full transcription
- `transcribe_video_chunked()` - Parallel chunk processing
- `transcribe_video_sampled()` - Smart section sampling

## Troubleshooting

If you need to override the automatic selection, you can still specify model_size in the request, though the strategy will remain automatic based on duration.

The enhanced server provides optimal results for videos of any length without requiring manual intervention!