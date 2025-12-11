#!/usr/bin/env python3
import argparse
import json
import numpy as np
import onnxruntime as ort
from pathlib import Path
import wave
import time
from phonikud_tts import Phonikud, phonemize
import threading
import queue

# Optional sounddevice import for real-time playback
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: sounddevice not available ({e}). Real-time playback disabled.")
    SOUNDDEVICE_AVAILABLE = False
    sd = None

def float_to_int16(audio_float):
    """Convert float audio to int16."""
    audio_float = np.clip(audio_float, -1.0, 1.0)
    return (audio_float * 32767).astype(np.int16)

def write_wav(filename, sample_rate, audio_data):
    """Write audio data to WAV file."""
    audio_int16 = float_to_int16(audio_data)
    
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

class StreamingTTS:
    def __init__(self, model_path, config_path, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.playing = False
        
        # Load ONNX model
        print(f"Loading ONNX model: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 2
        sess_options.intra_op_num_threads = 2
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        print(f"Model loaded. Vocab size: {self.config['num_symbols']}")
        
        # Get input names
        self.input_names = [input_def.name for input_def in self.session.get_inputs()]
        print(f"Model input names: {self.input_names}")
    
    def synthesize_streaming(self, text, phonikud_path, length_scale=1.0, noise_scale=0.667, noise_w=0.8, chunk_size=8192):
        """Generate audio in chunks for streaming."""
        
        print(f"Processing text: {text}")
        
        # Process text through Phonikud and phonemize
        diac = Phonikud(phonikud_path)
        with_diac = diac.add_diacritics(text)
        print(f"With diacritics: {with_diac}")
        
        phons = phonemize(with_diac)
        print(f"Phonemes: {phons}")
        
        # Convert phonemes to IDs
        phoneme_id_map = self.config['phoneme_id_map']
        phoneme_ids = []
        
        for char in phons:
            if char in phoneme_id_map:
                phoneme_ids.extend(phoneme_id_map[char])
            else:
                print(f"Warning: phoneme '{char}' not in phoneme map")
        
        print(f"Phoneme IDs: {phoneme_ids}")
        
        # Prepare inputs for ONNX model
        input_ids = np.array([phoneme_ids], dtype=np.int64)
        input_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
        scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)
        
        # Speaker ID
        speaker_id = None
        if self.config.get('num_speakers', 1) > 1:
            speaker_id = np.array([0], dtype=np.int64)
        
        # Prepare input dictionary
        inputs = {}
        for name in self.input_names:
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
        outputs = self.session.run(None, inputs)
        inference_time = time.perf_counter() - start_time
        
        audio_data = outputs[0]  # First output should be audio
        print(f"Raw output shape: {audio_data.shape}")
        
        # Handle different output shapes
        if len(audio_data.shape) == 4:  # [batch, channels, height, samples]
            audio_data = audio_data[0, 0, 0, :]
        elif len(audio_data.shape) == 3:  # [batch, channels, samples]
            audio_data = audio_data[0, 0, :]
        elif len(audio_data.shape) == 2:  # [batch, samples] 
            audio_data = audio_data[0, :]
        
        # Flatten if still multidimensional
        audio_data = audio_data.flatten()
        
        print(f"Audio length: {len(audio_data)} samples")
        print(f"Audio duration: {len(audio_data) / self.sample_rate:.3f} seconds")
        print(f"Inference time: {inference_time:.3f} seconds")
        print(f"Real-time factor: {inference_time / (len(audio_data) / self.sample_rate):.3f}")
        
        # Yield audio in chunks for streaming
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            yield chunk.astype(np.float32)
    
    def play_audio_worker(self):
        """Worker thread for audio playback."""
        if not SOUNDDEVICE_AVAILABLE:
            return
            
        try:
            with sd.OutputStream(
                samplerate=self.sample_rate, 
                channels=1, 
                dtype='float32'
            ) as stream:
                while self.playing:
                    try:
                        # Get audio chunk from queue with timeout
                        chunk = self.audio_queue.get(timeout=0.1)
                        if chunk is None:  # Sentinel to stop
                            break
                        stream.write(chunk)
                        self.audio_queue.task_done()
                    except queue.Empty:
                        continue
        except Exception as e:
            print(f"Audio playback error: {e}")
    
    def synthesize_and_play(self, text, phonikud_path, output_file=None, 
                          length_scale=1.0, noise_scale=0.667, noise_w=0.8, 
                          real_time_playback=True):
        """Synthesize and optionally play audio in real-time."""
        
        all_audio_data = []
        
        # Start audio playback thread if requested
        playback_thread = None
        if real_time_playback and SOUNDDEVICE_AVAILABLE:
            self.playing = True
            playback_thread = threading.Thread(target=self.play_audio_worker)
            playback_thread.start()
            print("Started real-time audio playback...")
        
        try:
            # Generate audio chunks
            for chunk in self.synthesize_streaming(text, phonikud_path, length_scale, noise_scale, noise_w):
                all_audio_data.append(chunk)
                
                # Queue for real-time playback
                if real_time_playback and SOUNDDEVICE_AVAILABLE:
                    self.audio_queue.put(chunk)
                
                # Print progress
                total_samples = sum(len(c) for c in all_audio_data)
                duration = total_samples / self.sample_rate
                print(f"\rGenerated: {duration:.2f}s", end='', flush=True)
            
            print()  # New line
            
        finally:
            # Stop playback
            if playback_thread:
                self.playing = False
                self.audio_queue.put(None)  # Sentinel
                playback_thread.join()
                print("Audio playback finished.")
        
        # Combine all chunks
        if all_audio_data:
            full_audio = np.concatenate(all_audio_data)
            
            # Save to file if requested
            if output_file:
                print(f"Saving to: {output_file}")
                write_wav(output_file, self.sample_rate, full_audio)
                print(f"Audio saved to {output_file}")
            
            return full_audio
        else:
            return np.array([])

def main():
    parser = argparse.ArgumentParser(description="ONNX TTS with Streaming")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--config", required=True, help="Path to model config JSON")
    parser.add_argument("--phonikud", required=True, help="Path to Phonikud ONNX model")
    parser.add_argument("--text", default="אני כאן כדי לעזור לך עם כל שאלה שיש לך לגבי הפלטפורמה שלנו", help="Hebrew text to synthesize")
    parser.add_argument("--output", help="Output WAV file (optional)")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio, just save to file")
    parser.add_argument("--length-scale", type=float, default=1.0, help="Speech rate (1.0=optimal balance)")
    parser.add_argument("--noise-scale", type=float, default=0.667, help="Voice variation (0.667=optimal expressiveness)")
    parser.add_argument("--noise-w", type=float, default=0.8, help="Pronunciation variation (0.8=optimal consistency)")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Sample rate")
    parser.add_argument("--chunk-size", type=int, default=8192, help="Audio chunk size for streaming")
    
    args = parser.parse_args()
    
    # Create TTS instance
    tts = StreamingTTS(args.model, args.config, args.sample_rate)
    
    # Synthesize and play
    audio_data = tts.synthesize_and_play(
        text=args.text,
        phonikud_path=args.phonikud,
        output_file=args.output,
        length_scale=args.length_scale,
        noise_scale=args.noise_scale,
        noise_w=args.noise_w,
        real_time_playback=not args.no_play
    )
    
    print(f"Synthesis completed! Generated {len(audio_data)} samples.")

if __name__ == "__main__":
    main()
