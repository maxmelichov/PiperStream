import json
import logging
import warnings
from pathlib import Path

import torch
import torchaudio
from pytorch_lightning.callbacks import Callback

# Suppress warnings in inference
warnings.filterwarnings("ignore")

_LOGGER = logging.getLogger(__name__)


class InferenceCallback(Callback):
    """Generate audio samples at the end of each epoch"""
    
    def __init__(self, test_sentence: str, config_path: Path, output_dir: Path):
        super().__init__()
        self.test_sentence = test_sentence
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config to get phoneme_id_map
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Convert IPA text to phoneme IDs
        self.phoneme_ids = self._text_to_phoneme_ids(test_sentence)
        _LOGGER.info(f"Test sentence: {test_sentence}")
        _LOGGER.info(f"Phoneme IDs: {self.phoneme_ids}")
    
    def _text_to_phoneme_ids(self, text: str):
        """Convert IPA text to phoneme IDs using config phoneme_id_map"""
        phoneme_id_map = self.config['phoneme_id_map']
        phoneme_ids = [1]  # BOS token
        
        for char in text:
            if char in phoneme_id_map:
                phoneme_ids.extend(phoneme_id_map[char])
            elif char == ' ':
                phoneme_ids.append(3)  # Word separator
            else:
                _LOGGER.warning(f"Unknown character: {char}")
        
        phoneme_ids.append(2)  # EOS token
        return phoneme_ids
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Generate audio at the end of each training epoch"""
        epoch = trainer.current_epoch
        _LOGGER.info(f"Generating inference sample for epoch {epoch}")
        
        # Set model to eval mode
        pl_module.eval()
        
        with torch.no_grad():
            # Prepare input
            text = torch.LongTensor(self.phoneme_ids).unsqueeze(0)
            text_lengths = torch.LongTensor([len(self.phoneme_ids)])
            
            # Move to device
            text = text.to(pl_module.device)
            text_lengths = text_lengths.to(pl_module.device)
            
            # Prepare speaker ID (None for single speaker)
            sid = None
            if pl_module.hparams.num_speakers > 1:
                sid = torch.LongTensor([0]).to(pl_module.device)
            
            # Generate audio
            try:
                # Use the generator model
                audio = pl_module.model_g.infer(
                    text,
                    text_lengths,
                    sid=sid,
                    noise_scale=0.667,
                    noise_scale_w=0.8,
                    length_scale=1.0
                )[0]
                
                # Save audio file
                output_path = self.output_dir / f"epoch_{epoch:04d}.wav"
                audio_np = audio[0, 0].cpu().float().numpy()
                
                # Normalize audio to prevent clipping
                max_val = abs(audio_np).max()
                if max_val > 0:
                    audio_np = audio_np / max_val * 0.95
                
                # Save as WAV file
                torchaudio.save(
                    str(output_path),
                    torch.from_numpy(audio_np).unsqueeze(0),
                    pl_module.hparams.sample_rate
                )
                
                _LOGGER.info(f"Saved inference sample to {output_path}")
            except Exception as e:
                _LOGGER.error(f"Failed to generate inference sample: {e}")
        
        # Set model back to train mode
        pl_module.train()

