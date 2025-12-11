#!/usr/bin/env python3
"""
Simple ONNX inference script for Hebrew TTS
Uses the same approach as piper_stream_onnx.py for compatibility
"""

import argparse
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
import wave
import time
import os
from phonikud_tts import Phonikud, phonemize

def main():
    parser = argparse.ArgumentParser(description="Simple Hebrew TTS inference")
    parser.add_argument("--text", default="שלום, אני כאן כדי לעזור לך עם כל שאלה שיש לך לגבי הפלטפורמה שלנו", help="Hebrew text to synthesize")
    parser.add_argument("--phonikud", default="onnx/phonikud-1.0.onnx", help="Path to Phonikud ONNX model")
    parser.add_argument("--model", default="onnx/piper_medium_male.onnx", help="Path to TTS ONNX model")
    parser.add_argument("--config", default="onnx/model.config.json", help="Path to model config")
    parser.add_argument("--output", default="output.wav", help="Output WAV file")
    parser.add_argument("--length-scale", type=float, default=0.65, help="Speech rate (1.0=normal, <1.0=faster, >1.0=slower)")
    parser.add_argument("--noise-scale", type=float, default=0.667, help="Voice variation (0.667=optimal)")
    parser.add_argument("--noise-w", type=float, default=0.8, help="Pronunciation variation (0.8=optimal)")
    parser.add_argument("--volume", type=float, default=1.0, help="Volume multiplier")
    parser.add_argument("--mode", choices=["text", "diacritics", "phonemes"], default="text", 
                       help="Input mode: text (add diacritics), diacritics (text with diacritics), phonemes (direct phonemes)")
    
    args = parser.parse_args()
    
    # Check if files exist
    phonikud_path = Path(args.phonikud)
    model_path = Path(args.model)
    config_path = Path(args.config)
    
    if not phonikud_path.exists():
        print(f"Phonikud model not found: {phonikud_path}")
        return
    
    if not model_path.exists():
        print(f"TTS model not found: {model_path}")
        return
        
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return
    
    print("Loading models...")
    num_threads = os.cpu_count() // 2
    # Encourage deterministic, stable behavior
    os.environ.setdefault('ORT_DISABLE_ALL_OPTIMIZATION', '1')
    # Load ONNX model (same as piper_stream_onnx.py)
    sess_options = ort.SessionOptions()
    sess_options.inter_op_num_threads = num_threads
    sess_options.intra_op_num_threads = 1
    # Deterministic execution and reduced allocator patterning can help avoid artifacts
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.enable_mem_pattern = False
    
    session = ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=['CPUExecutionProvider']
    )
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get input names
    input_names = [input_def.name for input_def in session.get_inputs()]
    print(f"Model input names: {input_names}")
    
    # Initialize Phonikud
    phonikud = Phonikud(str(phonikud_path))
    
    # Get commit info if available
    try:
        metadata = phonikud.get_metadata()
        commit = metadata.get("commit", "unknown")
        print(f"Phonikud commit: {commit}")
    except Exception:
        print("Phonikud commit: unknown")
    
    print(f"Input text: {args.text}")
    print(f"Mode: {args.mode}")
    
    # Process text based on mode
    if args.mode == "text":
        print("Adding diacritics...")
        with_diacritics = phonikud.add_diacritics(args.text)
        print(f"With diacritics: {with_diacritics}")
        
        print("Converting to phonemes...")
        phonemes = phonemize(with_diacritics)
        
    elif args.mode == "diacritics":
        with_diacritics = args.text
        print(f"With diacritics: {with_diacritics}")
        
        print("Converting to phonemes...")
        phonemes = phonemize(with_diacritics)
        
    else:  # phonemes mode
        phonemes = args.text
        with_diacritics = None
    
    print(f"Phonemes: {phonemes}")
    
    # Convert phonemes to IDs (same as piper_stream_onnx.py)
    phoneme_id_map = config['phoneme_id_map']
    phoneme_ids = []
    
    for char in phonemes:
        if char in phoneme_id_map:
            phoneme_ids.extend(phoneme_id_map[char])
        else:
            print(f"Warning: phoneme '{char}' not in phoneme map")
    
    # Add start/end tokens if available to stabilize prosody
    if '^' in phoneme_id_map:
        phoneme_ids = phoneme_id_map['^'] + phoneme_ids
    if '$' in phoneme_id_map:
        phoneme_ids = phoneme_ids + phoneme_id_map['$']
    
    print(f"Phoneme IDs: {phoneme_ids}")
    
    # Prepare inputs for ONNX model (same as piper_stream_onnx.py)
    input_ids = np.ascontiguousarray(np.array([phoneme_ids], dtype=np.int64))
    input_lengths = np.ascontiguousarray(np.array([len(phoneme_ids)], dtype=np.int64))
    scales = np.ascontiguousarray(np.array([args.noise_scale, args.length_scale, args.noise_w], dtype=np.float32))
    
    # Speaker ID
    speaker_id = None
    if config.get('num_speakers', 1) > 1:
        speaker_id = np.array([0], dtype=np.int64)
    
    # Prepare input dictionary
    inputs = {}
    for name in input_names:
        if name == "input":
            inputs[name] = input_ids
        elif name == "input_lengths":
            inputs[name] = input_lengths
        elif name == "scales":
            inputs[name] = scales
        elif name == "sid" and speaker_id is not None:
            inputs[name] = speaker_id
    
    print(f"Running inference...")
    print(f"Input shapes: {[(k, v.shape if v is not None else None) for k, v in inputs.items()]}")
    
    # Run inference
    start_time = time.perf_counter()
    outputs = session.run(None, inputs)
    inference_time = time.perf_counter() - start_time
    
    audio_data = outputs[0]  # First output should be audio
    print(f"Raw output shape: {audio_data.shape}")
    
    # Handle different output shapes (same as piper_stream_onnx.py)
    if len(audio_data.shape) == 4:  # [batch, channels, height, samples]
        audio_data = audio_data[0, 0, 0, :]
    elif len(audio_data.shape) == 3:  # [batch, channels, samples]
        audio_data = audio_data[0, 0, :]
    elif len(audio_data.shape) == 2:  # [batch, samples] 
        audio_data = audio_data[0, :]
    
    # Flatten if still multidimensional
    audio_data = audio_data.flatten().astype(np.float32)
    
    # Optional short fades to avoid clicks/jumps at boundaries
    fade_ms = 5
    sample_rate = 22050
    fade_samples = max(1, int(sample_rate * fade_ms / 1000))
    if len(audio_data) >= 2 * fade_samples:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        audio_data[:fade_samples] *= fade_in
        audio_data[-fade_samples:] *= fade_out
    
    # Apply volume
    if args.volume != 1.0:
        print(f"Applying volume factor: {args.volume}")
        audio_data = audio_data * args.volume
        audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure values are in valid range
    
    # Save audio using wave module (same as piper_stream_onnx.py)
    print(f"Saving audio to: {args.output}")
    
    # Convert to int16
    audio_int16 = np.clip(audio_data, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)
    
    with wave.open(args.output, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    # Print stats
    duration = len(audio_data) / sample_rate
    print(f"Audio generated successfully!")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(audio_data)}")
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Real-time factor: {inference_time / duration:.3f}")
    print(f"File: {args.output}")

if __name__ == "__main__":
    main()
