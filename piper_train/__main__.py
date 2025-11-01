import argparse
import json
import logging
import warnings
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress specific PyTorch warnings
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

# Set logging levels to reduce noise
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("numba").setLevel(logging.ERROR)

# Suppress specific warning categories
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=".*tensorboardX.*")
warnings.filterwarnings("ignore", message=".*weight_norm.*")
warnings.filterwarnings("ignore", message=".*DataLoader.*")
warnings.filterwarnings("ignore", message=".*weights_only.*")

from .vits.lightning import VitsModel
from .inference_callback import InferenceCallback
from .progress_callback import TqdmProgressCallback, SimpleProgressCallback

_LOGGER = logging.getLogger(__package__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir", default="/mnt/sda2/PiperTrain/male1", help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--progress", action="store_true", help="Show detailed progress bars"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        help="Save checkpoint every N epochs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    Trainer.add_argparse_args(parser)
    VitsModel.add_model_specific_args(parser)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug(args)

    args.dataset_dir = Path(args.dataset_dir)
    if not args.default_root_dir:
        args.default_root_dir = args.dataset_dir

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl"

    with open(config_path, "r", encoding="utf-8") as config_file:
        # See preprocess.py for format
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])

    # Save checkpoint path for manual loading later
    checkpoint_path_to_load = args.resume_from_checkpoint
    # Clear it from args so trainer doesn't try to load it
    args.resume_from_checkpoint = None
    
    trainer = Trainer.from_argparse_args(args)
    callbacks = []
    
    if args.checkpoint_epochs is not None:
        callbacks.append(ModelCheckpoint(every_n_epochs=args.checkpoint_epochs))
        _LOGGER.debug(
            "Checkpoints will be saved every %s epoch(s)", args.checkpoint_epochs
        )
    
    # Add progress callback
    if args.progress:
        try:
            callbacks.append(TqdmProgressCallback())
            _LOGGER.info("Using tqdm progress bars")
        except ImportError:
            callbacks.append(SimpleProgressCallback())
            _LOGGER.info("Using simple progress callback (tqdm not available)")
    else:
        callbacks.append(SimpleProgressCallback())
        _LOGGER.info("Using simple progress callback")
    
    # Add inference callback for model progress monitoring
    inference_output_dir = args.dataset_dir / "inference_samples"
    test_sentence = "ʃalˈom jinˈon kˈaχa niʃmˈa hamˈodel heχadˈaʃ mˈa daʔtχˈa ? lifʔamˈim hadˈeʁeχ lehavˈin ʔˈim ʁaʔjˈon beʔemˈet ʔovˈed hˈi paʃˈut lizʁˈok ʔotˈo laʔolˈam, liʁʔˈot ʔˈejχ ʔanaʃˈim meɡivˈim, lilmˈod mehatɡuvˈot, ullatˈeʃ ʃˈuv vaʃˈuv ʔˈad ʃehˈu mekabˈel ʔˈet hatsuʁˈa hamdujˈeket ʃematʔimˈa ɡˈam leχˈa veɡˈam lakahˈal ʃelχˈa."
    callbacks.append(InferenceCallback(
        test_sentence=test_sentence,
        config_path=config_path,
        output_dir=inference_output_dir
    ))
    _LOGGER.info("Inference samples will be saved to %s", inference_output_dir)
    
    trainer.callbacks = callbacks

    dict_args = vars(args)
    if args.quality == "x-low":
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        **dict_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Resuming from single-speaker checkpoint: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        model_single = VitsModel.load_from_checkpoint(
            args.resume_from_single_speaker_checkpoint,
            dataset=None,
        )
        g_dict = model_single.model_g.state_dict()
        for key in list(g_dict.keys()):
            # Remove keys that can't be copied over due to missing speaker embedding
            if (
                key.startswith("dec.cond")
                or key.startswith("dp.cond")
                or ("enc.cond_layer" in key)
            ):
                g_dict.pop(key, None)

        # Copy over the multi-speaker model, excluding keys related to the
        # speaker embedding (which is missing from the single-speaker model).
        load_state_dict(model.model_g, g_dict)
        load_state_dict(model.model_d, model_single.model_d.state_dict())
        _LOGGER.info(
            "Successfully converted single-speaker checkpoint to multi-speaker"
        )

    # If resuming from checkpoint with different architecture, load with strict=False
    if checkpoint_path_to_load:
        _LOGGER.info("Loading checkpoint with size mismatch handling")
        checkpoint = torch.load(checkpoint_path_to_load, weights_only=False)
        
        # Get current model state dict
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint["state_dict"]
        
        # Filter out parameters with size mismatches
        filtered_state_dict = {}
        skipped_keys = []
        
        for key, checkpoint_param in checkpoint_state_dict.items():
            if key in model_state_dict:
                model_param = model_state_dict[key]
                if checkpoint_param.shape == model_param.shape:
                    filtered_state_dict[key] = checkpoint_param
                else:
                    skipped_keys.append(f"{key} (checkpoint: {checkpoint_param.shape}, model: {model_param.shape})")
            else:
                # Key not in model, but we can still try to load it
                filtered_state_dict[key] = checkpoint_param
        
        if skipped_keys:
            _LOGGER.warning(f"Skipping {len(skipped_keys)} parameter(s) due to size mismatch:")
            for skipped in skipped_keys[:5]:
                _LOGGER.warning(f"  - {skipped}")
            if len(skipped_keys) > 5:
                _LOGGER.warning(f"  ... and {len(skipped_keys) - 5} more")
        
        # Load state dict, ignoring unexpected keys and missing keys
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        if missing_keys:
            _LOGGER.warning(f"Missing keys when loading checkpoint: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            _LOGGER.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        
        _LOGGER.info("✓ Checkpoint loaded successfully with size mismatch handling")
    
    trainer.fit(model)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
