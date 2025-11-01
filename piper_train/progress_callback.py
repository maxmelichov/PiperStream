import logging
from typing import Any, Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

_LOGGER = logging.getLogger(__name__)


class TqdmProgressCallback(Callback):
    """Custom progress callback using tqdm for better progress visualization"""
    
    def __init__(self):
        super().__init__()
        self.epoch_pbar: Optional[tqdm] = None
        self.train_pbar: Optional[tqdm] = None
        self.current_epoch = 0
        self.total_epochs = 0
        
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize epoch progress bar"""
        self.total_epochs = trainer.max_epochs
        self.epoch_pbar = tqdm(
            total=self.total_epochs,
            desc="🎯 Training Progress",
            unit="epoch",
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize batch progress bar for each epoch"""
        self.current_epoch = trainer.current_epoch
        
        # Get total number of batches
        if trainer.num_training_batches != float('inf'):
            total_batches = trainer.num_training_batches
        else:
            total_batches = None
            
        self.train_pbar = tqdm(
            total=total_batches,
            desc=f"📊 Epoch {self.current_epoch + 1:04d}",
            unit="batch",
            position=1,
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
    def on_train_batch_end(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule, 
        outputs: Any, 
        batch: Any, 
        batch_idx: int
    ) -> None:
        """Update batch progress bar"""
        if self.train_pbar:
            # Get current losses from outputs
            loss_info = ""
            if isinstance(outputs, dict):
                if 'loss' in outputs:
                    loss_info = f" Loss: {outputs['loss']:.4f}"
                elif 'loss_gen_all' in outputs:
                    gen_loss = outputs.get('loss_gen_all', 0)
                    disc_loss = outputs.get('loss_disc_all', 0)
                    loss_info = f" Gen: {gen_loss:.3f} Disc: {disc_loss:.3f}"
            
            self.train_pbar.set_postfix_str(loss_info)
            self.train_pbar.update(1)
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Update epoch progress bar and close batch progress bar"""
        if self.train_pbar:
            self.train_pbar.close()
            self.train_pbar = None
            
        if self.epoch_pbar:
            # Get epoch metrics
            metrics = trainer.callback_metrics
            postfix = {}
            
            if 'loss_gen_all' in metrics:
                postfix['Gen_Loss'] = f"{metrics['loss_gen_all']:.3f}"
            if 'loss_disc_all' in metrics:
                postfix['Disc_Loss'] = f"{metrics['loss_disc_all']:.3f}"
            if 'train_loss' in metrics:
                postfix['Loss'] = f"{metrics['train_loss']:.4f}"
                
            self.epoch_pbar.set_postfix(postfix)
            self.epoch_pbar.update(1)
            
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Close epoch progress bar"""
        if self.epoch_pbar:
            self.epoch_pbar.close()
            self.epoch_pbar = None
            
        if self.train_pbar:
            self.train_pbar.close()
            self.train_pbar = None
            
        print("🎉 Training completed!")


class SimpleProgressCallback(Callback):
    """Simple progress callback for basic epoch tracking"""
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Print epoch progress"""
        current_epoch = trainer.current_epoch + 1
        total_epochs = trainer.max_epochs
        
        # Get metrics
        metrics = trainer.callback_metrics
        loss_info = ""
        
        if 'loss_gen_all' in metrics and 'loss_disc_all' in metrics:
            gen_loss = metrics['loss_gen_all']
            disc_loss = metrics['loss_disc_all']
            loss_info = f" | Gen Loss: {gen_loss:.4f} | Disc Loss: {disc_loss:.4f}"
        elif 'train_loss' in metrics:
            loss_info = f" | Loss: {metrics['train_loss']:.4f}"
            
        progress_percent = (current_epoch / total_epochs) * 100
        
        print(f"🎯 Epoch {current_epoch:04d}/{total_epochs:04d} ({progress_percent:.1f}%){loss_info}")
