# lip_sync.py - Rhubarb Lip Sync Service Analysis

## Overview

`lip_sync.py` integrates the Rhubarb Lip Sync tool for precise mouth animation timing. It analyzes audio files and produces viseme sequences with precise timestamps.

## File Location
`/home/user/cluster/src/features/display/lip_sync.py`

## Rhubarb Overview

Rhubarb Lip Sync is an external command-line tool that:
- Analyzes audio files (WAV format)
- Optionally uses text transcripts for improved accuracy
- Outputs JSON with viseme timing cues
- Uses Preston Blair's phoneme groups

## Enums

### RhubarbViseme

Preston Blair's animation viseme set:

| Shape | Description | Phonemes |
|-------|-------------|----------|
| A | Closed mouth | M, B, P |
| B | Slightly open, teeth | Most consonants, EE |
| C | Open mouth | EH, AE vowels |
| D | Wide open | AA vowel |
| E | Slightly rounded | AO, ER vowels |
| F | Puckered | OO, UW, W |
| G | Upper teeth on lower lip | F, V |
| H | Tongue visible | L |
| X | Idle/Silence | Rest position |

## Dataclasses

### VisemeCue
```python
@dataclass
class VisemeCue:
    start_time: float        # When viseme starts
    end_time: float          # When viseme ends
    viseme: RhubarbViseme    # Shape to display

    @property
    def duration(self) -> float
```

### LipSyncData
```python
@dataclass
class LipSyncData:
    audio_file: str
    text: Optional[str]
    duration: float
    cues: List[VisemeCue]
    metadata: Dict[str, Any]

    def get_viseme_at_time(time_seconds: float) -> Optional[VisemeCue]
```

## Class: RhubarbLipSyncService

### Initialization
```python
RhubarbLipSyncService(
    rhubarb_path: str = None,      # Auto-detected if None
    recognizer: str = "pocketSphinx",  # Or "phonetic"
    extended_shapes: bool = True    # Use G, H, X shapes
)
```

### Rhubarb Path Discovery

Search order:
1. `vendor/rhubarb/rhubarb` (bundled)
2. System PATH
3. `/usr/local/bin/rhubarb`
4. `/usr/bin/rhubarb`
5. `~/.local/bin/rhubarb`
6. `~/rhubarb/rhubarb`
7. `/opt/rhubarb/rhubarb`

### Key Methods

| Method | Purpose | Async |
|--------|---------|-------|
| `is_available()` | Check Rhubarb installation | No |
| `generate_lip_sync()` | Analyze audio file | No (blocks) |
| `generate_lip_sync_async()` | Async wrapper | Yes |
| `convert_to_internal_viseme()` | Map to internal names | No |
| `get_cue_schedule()` | Get playback schedule | No |
| `clear_cache()` | Clear cached results | No |

### Rhubarb Command

```bash
rhubarb \
    -f json \                    # JSON output format
    -r pocketSphinx \            # Recognizer type
    --extendedShapes GHX \       # Extended shape set
    -d transcript.txt \          # Optional dialog file
    audio.wav                    # Input audio
```

### Output Parsing

Rhubarb JSON output:
```json
{
    "metadata": {
        "soundFile": "audio.wav",
        "duration": 2.5
    },
    "mouthCues": [
        {"start": 0.0, "value": "X"},
        {"start": 0.15, "value": "C"},
        {"start": 0.25, "value": "D"},
        ...
    ]
}
```

Parsed to `LipSyncData` with:
- End times calculated from next cue's start
- Unknown shapes default to X
- Metadata preserved

## Internal Viseme Mapping

```python
RHUBARB_TO_INTERNAL_VISEME = {
    RhubarbViseme.A: "BMP",      # Bilabial
    RhubarbViseme.B: "LNT",      # Consonants
    RhubarbViseme.C: "AH",       # Open vowels
    RhubarbViseme.D: "EE",       # Wide vowels
    RhubarbViseme.E: "OH",       # Rounded vowels
    RhubarbViseme.F: "OO",       # Puckered vowels
    RhubarbViseme.G: "FV",       # Labiodental
    RhubarbViseme.H: "AH",       # L with tongue
    RhubarbViseme.X: "SILENCE",  # Rest
}
```

## Caching

Results are cached by `"{audio_file}:{text}"` key:
- Avoids re-processing same audio
- Cache persists for session lifetime
- `clear_cache()` to reset

## Improvements Suggested

### 1. Async Rhubarb Process
True async subprocess handling:
```python
async def _run_rhubarb_async(self, cmd: List[str]) -> str:
    """Run Rhubarb as async subprocess."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await proc.communicate()
    return stdout.decode()
```

### 2. Phonetic Fallback
When pocketSphinx fails:
```python
async def generate_lip_sync(self, audio_file: str, text: str):
    """Try pocketSphinx, fall back to phonetic."""
    result = await self._generate(audio_file, text, "pocketSphinx")
    if result is None:
        result = await self._generate(audio_file, text, "phonetic")
    return result
```

### 3. Quality Scoring
Assess lip sync quality:
```python
def score_lip_sync_quality(self, data: LipSyncData) -> float:
    """Score lip sync quality (0-1)."""
    # Check for:
    # - Too many X visemes (silence)
    # - Unrealistic transitions
    # - Duration mismatch
    return quality_score
```

### 4. Real-time Processing
Process audio chunks in real-time:
```python
class StreamingLipSync:
    """Real-time lip sync without pre-processing."""
    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[str]:
        # Use simpler phoneme detection for real-time
        pass
```

### 5. Persistent Cache
Cache to disk for reuse across sessions:
```python
def _get_cache_path(self, audio_file: str) -> Path:
    """Get cache file path for audio."""
    audio_hash = hashlib.md5(Path(audio_file).read_bytes()).hexdigest()
    return self.cache_dir / f"{audio_hash}.json"
```

### 6. Audio Format Conversion
Support more audio formats:
```python
def _ensure_wav(self, audio_file: str) -> str:
    """Convert to WAV if needed."""
    if not audio_file.endswith('.wav'):
        wav_path = tempfile.mktemp(suffix='.wav')
        subprocess.run(['ffmpeg', '-i', audio_file, wav_path])
        return wav_path
    return audio_file
```
