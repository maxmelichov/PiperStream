# StreamPiper - Hebrew Text-to-Speech with ONNX Streaming

A high-performance Hebrew text-to-speech system using ONNX models with real-time streaming capabilities.

## 🎯 Features

- **Hebrew TTS**: Native Hebrew text-to-speech synthesis with dual voice models (male/female)  
- **Voice Selection**: Choose between male voice (piper_medium_male.onnx) and female voice (custom trained)
- **Diacritization**: Automatic Hebrew diacritic addition using Phonikud
- **IPA Phonemization**: Converts Hebrew text to IPA phonemes for accurate pronunciation
- **Real-time Streaming**: Optimized for low-latency audio output
- **ONNX Inference**: Fast CPU-based inference using ONNX Runtime
- **REST API**: FastAPI server with Swagger documentation
- **Docker Support**: Containerized deployment with Docker and docker-compose
- **Custom Training**: Supports custom-trained models from PyTorch Lightning checkpoints

## 📁 Project Structure

```
StreamPiper/
├── README.md                    # This documentation  
├── api.py                      # FastAPI server with Swagger docs
├── piper_stream_onnx.py        # Command-line streaming TTS script
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Docker Compose setup
├── onnx.zip                    # ONNX models archive (you need to add this)
├── onnx/                       # ONNX models and configurations (extracted)
│   ├── model.config.json       # Model configuration and phoneme mappings
│   ├── piper_medium_male.onnx  # Male voice TTS model 
│   ├── female_model.onnx       # Female voice TTS model (custom trained)
│   └── phonikud-1.0.onnx       # Phonikud diacritization model
├── venv/                       # Python virtual environment
├── piper_train/               # Training utilities and scripts
├── phonikud/                  # Phonikud library source
└── phonikud_tts/              # TTS utilities
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

That's it! All dependencies (including PyTorch, ONNX Runtime, FastAPI, and Phonikud) will be installed automatically.

### Basic Usage

**Generate Hebrew speech with male voice:**
```bash
python piper_stream_onnx.py \
    --model onnx/piper_medium_male.onnx \
    --config onnx/model.config.json \
    --phonikud onnx/phonikud-1.0.onnx \
    --text "שלום עולם" \
    --out output_male.wav
```

**Generate Hebrew speech with female voice:**
```bash
python piper_stream_onnx.py \
    --model onnx/female_model.onnx \
    --config onnx/model.config.json \
    --phonikud onnx/phonikud-1.0.onnx \
    --text "שלום עולם" \
    --out output_female.wav
```

**Full command with custom parameters (female voice):**
```bash
python piper_stream_onnx.py \
    --model onnx/female_model.onnx \
    --config onnx/model.config.json \
    --phonikud onnx/phonikud-1.0.onnx \
    --text "זאת בדיקת מערכת, אני רוצה לראות אם זה עובד" \
    --out output_female.wav \
    --length_scale 1.0 \
    --noise_scale 0.64 \
    --noise_w 1.0
```

## 🐳 Docker Deployment

### Prerequisites for Docker
- Docker Engine 20.10+
- Docker Compose v2.0+
- `onnx.zip` file containing all model files

### Setup Steps

**1. Prepare the models archive:**
```bash
# Create onnx.zip with all model files
zip -r onnx.zip onnx/
# Should contain:
# - onnx/piper_medium_male.onnx   (Male voice model)
# - onnx/female_model.onnx        (Female voice model) 
# - onnx/model.config.json        (Model configuration)
# - onnx/phonikud-1.0.onnx        (Phonikud diacritization)
```

**2. Build and run with Docker Compose (Recommended):**
```bash
# Clone or download the project
# Ensure onnx.zip is in the project root

# Build and start the service
docker-compose up --build -d

# Check logs
docker-compose logs -f streampiper-api

# Stop the service
docker-compose down
```

**3. Alternative: Build and run with Docker directly:**
```bash
# Build the image
docker build -t streampiper-api .

# Run the container
docker run -d \
  --name streampiper \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  streampiper-api

# Check logs
docker logs -f streampiper
```

### Docker Configuration

**Environment Variables:**
- `PYTHONUNBUFFERED=1`: Real-time log output
- Custom port: Modify `docker-compose.yml` or use `-p <port>:8000`

**Volumes:**
- `./output:/app/output`: Mount output directory for generated audio files
- Optional: `./onnx:/app/onnx` for development (override built-in models)

## 🌐 REST API Usage

The FastAPI server provides a complete REST API with Swagger documentation.

### API Endpoints

**Base URL:** `http://localhost:8000`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |
| `/health` | GET | Health check |
| `/synthesize` | POST | Generate TTS (metadata only) |
| `/synthesize/audio` | POST | Generate TTS audio file |
| `/synthesize/stream` | POST | Streaming TTS audio |
| `/models` | GET | List available voice models |

### API Examples

**1. Health Check:**
```bash
curl http://localhost:8000/health
# Returns: {"status": "healthy", "models_available": true, "phonikud_available": true}
```

**2. Get Available Voice Models:**
```bash
curl http://localhost:8000/models
# Returns: {"available_models": ["male", "female"], "default_model": "male"}
```

**3. Synthesize Audio with Male Voice (Download WAV):**
```bash
curl -X POST "http://localhost:8000/synthesize/audio" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "שלום עולם",
    "model": "male",
    "length_scale": 1.0,
    "noise_scale": 0.667,
    "noise_w": 0.8
  }' \
  --output male_output.wav
```

**4. Synthesize Audio with Female Voice (Download WAV):**
```bash
curl -X POST "http://localhost:8000/synthesize/audio" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "שלום עולם",
    "model": "female", 
    "length_scale": 1.0,
    "noise_scale": 0.667,
    "noise_w": 0.8
  }' \
  --output female_output.wav
```

**5. Streaming Audio (Male Voice):**
```bash
curl -X POST "http://localhost:8000/synthesize/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "זאת בדיקת מערכת, אני רוצה לראות אם זה עובד",
    "model": "male",
    "length_scale": 1.2,
    "noise_scale": 0.8
  }' \
  --output streaming_male.wav
```

**6. Streaming Audio (Female Voice):**
```bash
curl -X POST "http://localhost:8000/synthesize/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "זאת בדיקת מערכת, אני רוצה לראות אם זה עובד", 
    "model": "female",
    "length_scale": 1.2,
    "noise_scale": 0.8
  }' \
  --output streaming_female.wav
```

**7. Get Metadata Only:**
```bash
curl -X POST "http://localhost:8000/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "שלום עולם",
    "model": "female"
  }'
```

### API Request Schema

```json
{
  "text": "שלום עולם",                 // Required: Hebrew text
  "length_scale": 1.0,              // Optional: Speech rate (0.1-3.0)
  "noise_scale": 0.667,             // Optional: Voice variation (0.1-2.0)  
  "noise_w": 0.8,                   // Optional: Pronunciation variation (0.1-2.0)
  "volume": 1.0,                    // Optional: Volume multiplier (0.1-2.0)
  "model": "male"                   // Optional: Voice model ("male" or "female", default: "male")
}
```

### API Response Headers

Audio endpoints include helpful headers:
- `X-Audio-Duration`: Audio length in seconds
- `X-Processing-Time`: Synthesis time in seconds  
- `X-RTF`: Real-time factor (lower is faster)

### Interactive API Documentation

**Swagger UI:** http://localhost:8000/docs
- Interactive API testing
- Request/response schemas
- Try all endpoints live

**ReDoc:** http://localhost:8000/redoc  
- Clean documentation format
- Detailed parameter descriptions

### Integration Examples

**Python:**
```python
import requests

# Synthesize and save audio
response = requests.post("http://localhost:8000/synthesize/audio", 
    json={"text": "שלום עולם", "length_scale": 1.0})

if response.status_code == 200:
    with open("output.wav", "wb") as f:
        f.write(response.content)
    print(f"Audio duration: {response.headers.get('X-Audio-Duration')}s")
```

**JavaScript:**
```javascript
const response = await fetch('http://localhost:8000/synthesize/audio', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'שלום עולם',
    length_scale: 1.0,
    noise_scale: 0.667
  })
});

if (response.ok) {
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.play();
}
```

**cURL with custom parameters:**
```bash
curl -X POST "http://localhost:8000/synthesize/audio" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "זה טקסט ארוך יותר לבדיקת המערכת",
    "length_scale": 0.8,
    "noise_scale": 0.9,
    "noise_w": 1.1,
    "volume": 1.2
  }' \
  -o custom_synthesis.wav
```

## 📖 Detailed Usage

### Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | ✅ | - | Path to ONNX model file |
| `--config` | ✅ | - | Path to model configuration JSON |
| `--phonikud` | ✅ | - | Path to Phonikud ONNX model |
| `--text` | ❌ | Hebrew sample text | Text to synthesize |
| `--out` | ❌ | `out.wav` | Output WAV file path |
| `--length_scale` | ❌ | `1.20` | Speech rate (1.0=normal, <1.0=faster, >1.0=slower) |
| `--noise_scale` | ❌ | `0.64` | Voice variation/expressiveness |
| `--noise_w` | ❌ | `1.0` | Pronunciation variation |
| `--volume` | ❌ | `1.0` | Output volume multiplier |
| `--chunk` | ❌ | `8192` | Audio chunk size for streaming |

### Performance Tuning

| Argument | Default | Description |
|----------|---------|-------------|
| `--ort_threads` | `4` | ONNX Runtime CPU threads |
| `--ort_inter_op` | `1` | Inter-operator parallelism |
| `--ort_exec_mode` | `parallel` | Execution mode (parallel/sequential) |
| `--ort_quiet` | `False` | Suppress ONNX Runtime logs |

## 🎵 Audio Parameters Guide

### Length Scale
- **0.5**: 2x faster speech
- **1.0**: Normal speech rate
- **1.5**: 1.5x slower speech
- **2.0**: 2x slower speech

### Noise Scale
- **0.1**: Very monotone, robotic
- **0.667**: Balanced expressiveness (recommended)
- **1.0**: High variation, more natural
- **1.5**: Very expressive, potentially unstable

### Noise W
- **0.1**: Very consistent pronunciation
- **0.8**: Balanced pronunciation variation
- **1.0**: Natural pronunciation variation
- **1.5**: High variation, potentially inconsistent

## 🔧 Technical Details

### Processing Pipeline

1. **Input Text**: Raw Hebrew text
2. **Diacritization**: Phonikud adds vowel markings (נקודות)
3. **Phonemization**: Converts to IPA phonemes
4. **ID Mapping**: Maps phonemes to model vocabulary IDs
5. **ONNX Inference**: Generates mel-spectrograms
6. **Vocoding**: Converts to audio waveform
7. **Streaming Output**: Real-time audio chunks

### Model Architecture

- **Type**: VITS (Variational Inference Text-to-Speech)
- **Vocoder**: Built-in neural vocoder
- **Sample Rate**: 22,050 Hz
- **Channels**: Mono (1 channel)
- **Bit Depth**: 16-bit PCM

### Performance Metrics

- **Real-time Factor**: ~0.05-0.1 (10-20x faster than real-time)
- **Latency**: <100ms for short phrases
- **Memory Usage**: ~200MB (model + runtime)
- **CPU Usage**: 1-4 cores, optimized for modern CPUs

## 🐛 Troubleshooting

### Common Issues

**1. Model not found:**
```bash
# Ensure files exist
ls -la onnx/*.onnx onnx/*.json
```

**2. Poor audio quality:**
```bash
# Try adjusting parameters
--length_scale 1.0 --noise_scale 0.667 --noise_w 0.8
```

**3. Slow inference:**
```bash
# Increase CPU threads
--ort_threads 8 --ort_inter_op 2
```

**4. Memory issues:**
```bash
# Reduce chunk size
--chunk 4096
```

**5. Docker container won't start:**
```bash
# Check if onnx.zip is present and valid
ls -la onnx.zip
unzip -t onnx.zip

# Check container logs
docker-compose logs streampiper-api

# Rebuild without cache
docker-compose build --no-cache
```

**6. API not responding:**
```bash
# Check health endpoint
curl http://localhost:8000/health

# Check if port is accessible
curl http://localhost:8000/

# Check container status
docker-compose ps
```

**7. Model files missing:**
```bash
# Verify onnx.zip contents
unzip -l onnx.zip

# Should show:
# onnx/piper_medium_male.onnx
# onnx/model.config.json  
# onnx/phonikud-1.0.onnx

# Manual extraction for testing
unzip -o onnx.zip
```

### Environment Issues

**Missing dependencies:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Docker dependencies:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin
```

**Python version:**
- Requires Python 3.8+
- Tested with Python 3.10
- Docker image uses Python 3.10-slim

## 📊 Example Outputs

### Performance Examples

| Text Length | Audio Duration | Processing Time | RTF |
|-------------|----------------|-----------------|-----|
| "שלום עולם" (2 words) | 0.65s | 0.035s | 0.055 |
| Long sentence (20+ words) | 5.55s | 0.202s | 0.036 |

### Quality Settings

| Setting | Length Scale | Noise Scale | Noise W | Use Case |
|---------|--------------|-------------|---------|----------|
| **Fast** | 0.8 | 0.5 | 0.6 | Quick previews |
| **Balanced** | 1.0 | 0.667 | 0.8 | General use |
| **High Quality** | 1.2 | 0.8 | 1.0 | Final output |

## 🔬 Model Training

The system supports custom model training using PyTorch Lightning:

1. **Checkpoint**: `epoch=49-step=192500.ckpt` (custom trained model)
2. **Export**: Use `piper_train/export_onnx.py` to convert checkpoints
3. **Config**: Model configuration in `model.config.json`

## 📝 File Formats

### Model Config (`model.config.json`)
```json
{
  "num_symbols": 185,
  "num_speakers": 1,
  "sample_rate": 22050,
  "phoneme_id_map": { ... },
  "inference": {
    "noise_scale": 0.667,
    "length_scale": 1.0,
    "noise_w": 0.8
  }
}
```

### Output Audio
- **Format**: WAV
- **Sample Rate**: 22.05 kHz
- **Bit Depth**: 16-bit PCM
- **Channels**: Mono

## 🚀 Advanced Usage

### Batch Processing
```bash
# Process multiple texts with male voice
python piper_stream_onnx.py --model onnx/piper_medium_male.onnx --config onnx/model.config.json --phonikud onnx/phonikud-1.0.onnx --text "שלום" --out hello_male.wav
python piper_stream_onnx.py --model onnx/piper_medium_male.onnx --config onnx/model.config.json --phonikud onnx/phonikud-1.0.onnx --text "להתראות" --out goodbye_male.wav

# Process with female voice
python piper_stream_onnx.py --model onnx/female_model.onnx --config onnx/model.config.json --phonikud onnx/phonikud-1.0.onnx --text "שלום" --out hello_female.wav
python piper_stream_onnx.py --model onnx/female_model.onnx --config onnx/model.config.json --phonikud onnx/phonikud-1.0.onnx --text "להתראות" --out goodbye_female.wav
```

### Integration Example
```python
import subprocess
import os

def synthesize_hebrew(text, output_file, voice="male"):
    """Synthesize Hebrew text with choice of voice model"""
    model_file = "onnx/piper_medium_male.onnx" if voice == "male" else "onnx/female_model.onnx"
    
    cmd = [
        "python", "piper_stream_onnx.py",
        "--model", model_file,
        "--config", "onnx/model.config.json", 
        "--phonikud", "onnx/phonikud-1.0.onnx",
        "--text", text,
        "--out", output_file
    ]
    subprocess.run(cmd, cwd="/mnt/data/StreamPiper")
    return os.path.exists(output_file)

# Usage examples
male_success = synthesize_hebrew("שלום עולם", "output_male.wav", voice="male")
female_success = synthesize_hebrew("שלום עולם", "output_female.wav", voice="female")
```

## ✅ Verification & Testing

After setup, verify that both voice models are working:

### Quick API Test
```bash
# Check available models
curl http://localhost:8000/models

# Test male voice
curl -X POST "http://localhost:8000/synthesize/audio" \
  -H "Content-Type: application/json" \
  -d '{"text": "בדיקת קול גברי", "model": "male"}' \
  --output test_male.wav

# Test female voice  
curl -X POST "http://localhost:8000/synthesize/audio" \
  -H "Content-Type: application/json" \
  -d '{"text": "בדיקת קול נשי", "model": "female"}' \
  --output test_female.wav
```

### Expected Results
- **Available models**: `["male", "female"]`
- **Audio files**: Both `test_male.wav` and `test_female.wav` should be generated successfully
- **Performance**: RTF (Real-time Factor) should be < 0.1 for optimal performance
- **Swagger UI**: Interactive testing available at `http://localhost:8000/docs`

## 📄 License

This project uses various open-source components. Please check individual component licenses:

- **Piper TTS**: MIT License
- **Phonikud**: See phonikud repository
- **ONNX Runtime**: MIT License

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Test with Hebrew text samples
4. Submit a pull request

## 📞 Support

For issues:
1. Check troubleshooting section
2. Verify all dependencies are installed
3. Test with simple Hebrew text first
4. Check system resources (CPU, memory)

---

## 🎯 Quick Setup Summary

### For API Users (Recommended):
```bash
# 1. Ensure onnx.zip is in project root
# 2. Start with Docker Compose
docker-compose up --build -d

# 3. Test the API
curl http://localhost:8000/health

# 4. Generate speech (Male voice)
curl -X POST "http://localhost:8000/synthesize/audio" \
  -H "Content-Type: application/json" \
  -d '{"text": "שלום עולם", "model": "male"}' \
  --output hello_male.wav

# 5. Generate speech (Female voice)
curl -X POST "http://localhost:8000/synthesize/audio" \
  -H "Content-Type: application/json" \
  -d '{"text": "שלום עולם", "model": "female"}' \
  --output hello_female.wav

# 6. Open Swagger docs
# Visit: http://localhost:8000/docs
```

### For Command Line Users:
```bash
# Setup (one-time)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run TTS
python piper_stream_onnx.py \
  --model onnx/piper_medium_male.onnx \
  --config onnx/model.config.json \
  --phonikud onnx/phonikud-1.0.onnx \
  --text "שלום עולם" \
  --out output.wav
```

---

**StreamPiper** - High-performance Hebrew TTS with REST API and Docker support 🎤🔊🐳
