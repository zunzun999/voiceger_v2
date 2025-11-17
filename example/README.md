# Voiceger API

A minimal FastAPI implementation providing Text-to-Speech (TTS) and Voice Conversion (VC) functionality.

## Features

- **TTS**: Convert text to speech in multiple languages (Chinese, English, Japanese, Korean, Cantonese)
- **VC**: Convert audio to Zundamon voice style using RVC technology
- **Local-only**: All models loaded from local project directory, no downloads required

## Quick Start

### 1. Dependencies

If you have already installed the main project dependencies, no additional installation is needed. The API uses the same dependencies as the main project.

### 2. Start API Server

```bash
cd voiceger-webui/example
python voiceger_api.py
```

Server will start at `http://127.0.0.1:8000`

### 3. Test API

```bash
python test_api.py
```

## API Endpoints

### GET /

Returns API information.

```bash
curl http://127.0.0.1:8000/
```

### POST /tts

Convert text to speech.

**Request:**
```json
{
  "text": "Hello, world!",
  "text_language": "English"
}
```

**Response:**
```json
{
  "message": "success",
  "file_path": "<output_directory>/Hello, world_1700000000_abc12345.wav",
  "sampling_rate": 32000
}
```

**Supported Languages:**
- `Chinese`, `English`, `Japanese`, `Korean`, `Cantonese`
- `Chinese-English Mixed`, `Japanese-English Mixed`, `Korean-English Mixed`, `Cantonese-English Mixed`

### POST /vc/single

Convert voice to Zundamon style.

**Request:**
```json
{
  "input_audio_path": "/path/to/input.wav",
  "f0_method": "rmvpe"
}
```

**Response:**
```json
{
  "message": "success",
  "info": "Success",
  "file_path": "<output_directory>/input_1700000000_def67890.wav",
  "sampling_rate": 40000,
  "sid": "train-0814-2.pth",
  "file_index": "/path/to/index.file"
}
```

## Output Files

Generated audio files are saved in:

**Windows:**
```
%LOCALAPPDATA%\voiceger\output\
(typically: C:\Users\<username>\AppData\Local\voiceger\output\)
```

**Linux:**
```
$TEMP/voiceger_temp/output/
(typically: /tmp/voiceger_temp/output/)
```

### File Naming Format

```
{cleaned_content}_{timestamp}_{random_id}.wav
```

- **cleaned_content**: Original text or filename with invalid characters removed
- **timestamp**: Unix timestamp (seconds)
- **random_id**: 8-character UUID hex
- **extension**: `.wav`

Example: `Hello, world_1700000000_abc12345.wav`


## Python API Usage

### TTS Example

```python
import requests

url = "http://127.0.0.1:8000/tts"
response = requests.post(url, json={
    "text": "こんにちは、世界！",
    "text_language": "Japanese"
})

result = response.json()
print(f"Generated audio: {result['file_path']}")
```

### VC Example

```python
import requests

url = "http://127.0.0.1:8000/vc/single"
response = requests.post(url, json={
    "input_audio_path": "C:/path/to/input.wav",
    "f0_method": "rmvpe"
})

result = response.json()
print(f"Converted audio: {result['file_path']}")
```

## Testing

### Run All Tests

```bash
python test_api.py
```


### Test Coverage

- Root path
- TTS: Chinese, Japanese, English, Cantonese, Korean
- VC: Voice conversion

## Troubleshooting

### Model not found

Ensure model files are in correct locations:
- GPT: `GPT-SoVITS/GPT_weights_v2/zudamon_style_1-e15.ckpt`
- SoVITS: `GPT-SoVITS/SoVITS_weights_v2/zudamon_style_1_e8_s96.pth`
- Pretrained models in: `GPT-SoVITS/GPT_SoVITS/pretrained_models/`

### FFmpeg not found

Ensure `ffmpeg.exe` exists in `GPT-SoVITS/` directory.

### RVC not working

Ensure RVC model files are in:
- `Retrieval-based-Voice-Conversion-WebUI/assets/weights/*.pth`
- `Retrieval-based-Voice-Conversion-WebUI/assets/rmvpe/*.pth`

## Notes

- First API call may take 10-30 seconds for model loading
- All models are loaded from local project directory
- Output files are not automatically deleted