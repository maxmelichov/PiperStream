# StreamPiper - Hebrew Text-to-Speech with ONNX Streaming

A high-performance Hebrew text-to-speech system using ONNX models with real-time streaming capabilities.

## 🎯 Features

- **Hebrew TTS**: Native Hebrew text-to-speech synthesis
- **Diacritization**: Automatic Hebrew diacritic addition using Phonikud
- **IPA Phonemization**: Converts Hebrew text to IPA phonemes for accurate pronunciation
- **Real-time Streaming**: Optimized for low-latency audio output
- **ONNX Inference**: Fast CPU-based inference using ONNX Runtime
- **Custom Training**: Supports custom-trained models from PyTorch Lightning checkpoints

## 📁 Project Structure

```
StreamPiper/
├── README.md                    # This documentation
├── piper_stream_onnx.py        # Main streaming TTS script
├── onnx/                       # ONNX models and configurations
│   ├── model.config.json       # Model configuration and phoneme mappings
│   ├── piper_medium_male.onnx  # Pre-trained Hebrew TTS model
│   └── phonikud-1.0.onnx       # Phonikud diacritization model
├── venv/                       # Python virtual environment
├── piper_train/               # Training utilities and scripts
├── phonikud/                  # Phonikud library source
└── phonikud_tts/              # TTS utilities
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Virtual environment activated

### Installation

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Install dependencies** (if not already installed):
   ```bash
   pip install torch onnxruntime numpy phonikud_tts
   ```

### Basic Usage

**Generate Hebrew speech with streaming:**
```bash
python piper_stream_onnx.py \
    --model onnx/piper_medium_male.onnx \
    --config onnx/model.config.json \
    --phonikud onnx/phonikud-1.0.onnx \
    --text "שלום עולם" \
    --out output.wav
```

**Full command with custom parameters:**
```bash
python piper_stream_onnx.py \
    --model onnx/piper_medium_male.onnx \
    --config onnx/model.config.json \
    --phonikud onnx/phonikud-1.0.onnx \
    --text "זאת בדיקת מערכת, אני רוצה לראות אם זה עובד" \
    --out output.wav \
    --length_scale 1.0 \
    --noise_scale 0.64 \
    --noise_w 1.0
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

### Environment Issues

**Missing dependencies:**
```bash
pip install torch onnxruntime numpy phonikud_tts
```

**Python version:**
- Requires Python 3.8+
- Tested with Python 3.10

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
# Process multiple texts
python piper_stream_onnx.py --model onnx/piper_medium_male.onnx --config onnx/model.config.json --phonikud onnx/phonikud-1.0.onnx --text "שלום" --out hello.wav
python piper_stream_onnx.py --model onnx/piper_medium_male.onnx --config onnx/model.config.json --phonikud onnx/phonikud-1.0.onnx --text "להתראות" --out goodbye.wav
```

### Integration Example
```python
import subprocess
import os

def synthesize_hebrew(text, output_file):
    cmd = [
        "python", "piper_stream_onnx.py",
        "--model", "onnx/piper_medium_male.onnx",
        "--config", "onnx/model.config.json", 
        "--phonikud", "onnx/phonikud-1.0.onnx",
        "--text", text,
        "--out", output_file
    ]
    subprocess.run(cmd, cwd="/mnt/data/StreamPiper")
    return os.path.exists(output_file)

# Usage
success = synthesize_hebrew("שלום עולם", "output.wav")
```

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

**StreamPiper** - High-performance Hebrew TTS with real-time streaming 🎤🔊
