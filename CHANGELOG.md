# Changelog

## [2.0.0] - 2025-01-18

### Added
- Automatic video length detection and strategy selection
- Smart sampling for long videos (>60 minutes)
- Chunked parallel processing for medium videos (10-60 minutes)
- Intelligent optimization without user configuration
- Progress reporting with coverage statistics

### Changed
- MCP server now automatically selects processing strategy based on video duration
- Model selection is optimized per strategy (base for short, tiny for long videos)
- Enhanced performance for long videos (up to 90% time savings)

### Technical Details
- Short videos (≤10 min): Full transcription with base model
- Medium videos (10-60 min): 5-minute chunks processed in parallel
- Long videos (>60 min): Smart sampling of key sections (~30% coverage)

### Backwards Compatibility
- Fully compatible with existing MCP client configurations
- No changes required to existing usage patterns