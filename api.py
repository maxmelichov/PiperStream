#!/usr/bin/env python3
"""
StreamPiper API - Hebrew Text-to-Speech with ONNX Streaming
FastAPI server with Swagger documentation
"""

import os
import json
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import wave

from phonikud_tts import Phonikud, phonemize

# Configuration
MODEL_DIR = Path("onnx")
MALE_MODEL = MODEL_DIR / "piper_medium_male.onnx"
FEMALE_MODEL = MODEL_DIR / "female_model.onnx"
DEFAULT_CONFIG = MODEL_DIR / "model.config.json" 
DEFAULT_PHONIKUD = MODEL_DIR / "phonikud-1.0.onnx"
SAMPLE_RATE = 22050

# Available models
AVAILABLE_MODELS = {
    "male": MALE_MODEL,
    "female": FEMALE_MODEL
}

# Initialize FastAPI app
app = FastAPI(
    title="StreamPiper TTS API",
    description="Hebrew Text-to-Speech API with real-time streaming capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class TTSRequest(BaseModel):
    text: str = Field(..., description="Hebrew text to synthesize", example="שלום עולם")
    length_scale: float = Field(1.0, ge=0.1, le=3.0, description="Speech rate (1.0=optimal balance, <1.0=faster, >1.0=slower)")
    noise_scale: float = Field(0.667, ge=0.1, le=2.0, description="Voice variation (0.667=optimal expressiveness)")
    noise_w: float = Field(0.8, ge=0.1, le=2.0, description="Pronunciation variation (0.8=optimal consistency)")
    volume: float = Field(1.0, ge=0.1, le=2.0, description="Output volume multiplier")
    model: Optional[str] = Field("male", description="TTS model to use: 'male' or 'female' (default: 'male')")

class TTSResponse(BaseModel):
    success: bool
    message: str
    audio_duration: Optional[float] = None
    processing_time: Optional[float] = None
    rtf: Optional[float] = None  # Real-time factor

class HealthResponse(BaseModel):
    status: str
    models_available: bool
    phonikud_available: bool
    version: str

# Global TTS engine
class TTSEngine:
    def __init__(self):
        self.sessions = {}
        self.config = None
        self.phonikud = None
        self.load_models()
    
    def load_models(self):
        """Load ONNX models and configuration"""
        try:
            # Load all available TTS models
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 2
            sess_options.intra_op_num_threads = 2
            # Enable deterministic execution
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sess_options.enable_mem_pattern = False
            
            for model_name, model_path in AVAILABLE_MODELS.items():
                if model_path.exists():
                    session = ort.InferenceSession(
                        str(model_path),
                        sess_options=sess_options,
                        providers=['CPUExecutionProvider']
                    )
                    self.sessions[model_name] = session
                    print(f"✅ Loaded {model_name} model: {model_path}")
                else:
                    print(f"⚠️ Model not found: {model_name} at {model_path}")
            
            if not self.sessions:
                raise FileNotFoundError("No TTS models found")
            
            # Load config
            if not DEFAULT_CONFIG.exists():
                raise FileNotFoundError(f"Config not found: {DEFAULT_CONFIG}")
                
            with open(DEFAULT_CONFIG, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # Load Phonikud
            if not DEFAULT_PHONIKUD.exists():
                raise FileNotFoundError(f"Phonikud model not found: {DEFAULT_PHONIKUD}")
                
            self.phonikud = Phonikud(str(DEFAULT_PHONIKUD))
            
            print("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise

    def float_to_int16(self, audio_float):
        """Convert float audio to int16."""
        audio_float = np.clip(audio_float, -1.0, 1.0)
        return (audio_float * 32767).astype(np.int16)

    def audio_to_wav_bytes(self, audio_data):
        """Convert audio data to WAV bytes."""
        audio_int16 = self.float_to_int16(audio_data)
        
        # Create WAV file in memory
        import io
        wav_buffer = io.BytesIO()
        
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

    def synthesize(self, text: str, length_scale: float = 1.0, noise_scale: float = 0.667, 
                   noise_w: float = 0.8, volume: float = 1.0, model: str = "male", seed: int = 42) -> tuple[np.ndarray, dict]:
        """
        Synthesize Hebrew text to audio
        Returns: (audio_data, metadata)
        """
        start_time = time.perf_counter()
        
        # Set deterministic seed based on text content for consistent output
        import hashlib
        import random
        text_hash = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)
        deterministic_seed = (seed + text_hash) % (2**32)
        np.random.seed(deterministic_seed)
        random.seed(deterministic_seed)
        
        # Also try to set environment variable for ONNX Runtime
        import os
        os.environ['ORT_DISABLE_ALL_OPTIMIZATION'] = '1'
        
        try:
            # Check if model is available
            if model not in self.sessions:
                available = list(self.sessions.keys())
                raise ValueError(f"Model '{model}' not available. Available models: {available}")
            
            session = self.sessions[model]
            input_names = [input_def.name for input_def in session.get_inputs()]
            
            # Process text through Phonikud
            with_diac = self.phonikud.add_diacritics(text)
            phons = phonemize(with_diac)
            
            # Convert phonemes to IDs
            phoneme_id_map = self.config['phoneme_id_map']
            phoneme_ids = []
            
            for char in phons:
                if char in phoneme_id_map:
                    phoneme_ids.extend(phoneme_id_map[char])
            
            if not phoneme_ids:
                raise ValueError("No valid phonemes found in text")
            
            # Prepare inputs
            input_ids = np.array([phoneme_ids], dtype=np.int64)
            input_lengths = np.array([len(phoneme_ids)], dtype=np.int64)
            scales = np.array([noise_scale, length_scale, noise_w], dtype=np.float32)
            
            # Speaker ID (if needed)
            speaker_id = None
            if self.config.get('num_speakers', 1) > 1:
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
            
            # Run inference
            outputs = session.run(None, inputs)
            audio_data = outputs[0]
            
            # Handle different output shapes
            if len(audio_data.shape) == 4:
                audio_data = audio_data[0, 0, 0, :]
            elif len(audio_data.shape) == 3:
                audio_data = audio_data[0, 0, :]
            elif len(audio_data.shape) == 2:
                audio_data = audio_data[0, :]
            
            audio_data = audio_data.flatten().astype(np.float32)
            
            # Apply volume
            if volume != 1.0:
                audio_data = np.clip(audio_data * volume, -1.0, 1.0)
            
            # Calculate metadata
            processing_time = time.perf_counter() - start_time
            audio_duration = len(audio_data) / SAMPLE_RATE
            rtf = processing_time / max(audio_duration, 0.001)
            
            metadata = {
                'original_text': text,
                'diacritized_text': with_diac,
                'phonemes': phons,
                'phoneme_ids': phoneme_ids,
                'audio_duration': audio_duration,
                'processing_time': processing_time,
                'model_used': model,
                'rtf': rtf,
                'sample_rate': SAMPLE_RATE,
                'samples': len(audio_data)
            }
            
            return audio_data, metadata
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

    def audio_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert audio data to WAV bytes"""
        audio_int16 = self.float_to_int16(audio_data)
        
        # Create WAV in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        return wav_buffer.read()

# Initialize TTS engine
tts_engine = TTSEngine()

# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic information"""
    return {
        "service": "StreamPiper TTS API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/models")
async def get_available_models():
    """
    Get list of available TTS models
    """
    return {
        "available_models": list(tts_engine.sessions.keys()),
        "default_model": "male"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models_available = (
        len(tts_engine.sessions) > 0 and 
        tts_engine.config is not None
    )
    phonikud_available = tts_engine.phonikud is not None
    
    return HealthResponse(
        status="healthy" if (models_available and phonikud_available) else "unhealthy",
        models_available=models_available,
        phonikud_available=phonikud_available,
        version="1.0.0"
    )

@app.post("/synthesize", response_model=TTSResponse)
async def synthesize_text(request: TTSRequest):
    """
    Synthesize Hebrew text to speech and return metadata
    Returns JSON with synthesis information, use /synthesize/audio for audio file
    """
    try:
        audio_data, metadata = tts_engine.synthesize(
            text=request.text,
            length_scale=request.length_scale,
            noise_scale=request.noise_scale,
            noise_w=request.noise_w,
            volume=request.volume,
            model=request.model or "male"
        )
        
        return TTSResponse(
            success=True,
            message="Synthesis completed successfully",
            audio_duration=metadata['audio_duration'],
            processing_time=metadata['processing_time'],
            rtf=metadata['rtf']
        )
        
    except Exception as e:
        return TTSResponse(
            success=False,
            message=f"Synthesis failed: {str(e)}"
        )

@app.post("/synthesize/audio")
async def synthesize_audio(request: TTSRequest):
    """
    Synthesize Hebrew text to speech and return WAV audio file
    """
    try:
        audio_data, metadata = tts_engine.synthesize(
            text=request.text,
            length_scale=request.length_scale,
            noise_scale=request.noise_scale,
            noise_w=request.noise_w,
            volume=request.volume,
            model=request.model or "male"
        )
        
        # Convert to WAV bytes
        wav_bytes = tts_engine.audio_to_wav_bytes(audio_data)
        
        # Return as streaming response
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=synthesis.wav",
                "X-Audio-Duration": str(metadata['audio_duration']),
                "X-Processing-Time": str(metadata['processing_time']),
                "X-RTF": str(metadata['rtf'])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize/stream")
async def synthesize_stream(request: TTSRequest):
    """
    Synthesize Hebrew text to speech with streaming response
    Returns complete WAV file (chunking WAV breaks the header)
    """
    try:
        # Generate complete audio (same as regular endpoint)
        audio_data, metadata = tts_engine.synthesize(
            text=request.text,
            length_scale=request.length_scale,
            noise_scale=request.noise_scale,
            noise_w=request.noise_w,
            volume=request.volume,
            model=request.model or "male"
        )
        
        wav_bytes = tts_engine.audio_to_wav_bytes(audio_data)
        
        # Don't chunk WAV files as it breaks the WAV header structure
        # Return complete WAV file instead
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=streaming.wav",
                "X-Audio-Duration": str(metadata['audio_duration']),
                "X-Processing-Time": str(metadata['processing_time']),
                "X-RTF": str(metadata['rtf'])
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    models = []
    if DEFAULT_MODEL.exists():
        models.append({
            "name": "piper_medium_male",
            "path": str(DEFAULT_MODEL),
            "config": str(DEFAULT_CONFIG),
            "type": "Hebrew TTS"
        })
    
    return {"models": models}

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
