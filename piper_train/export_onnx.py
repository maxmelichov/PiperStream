#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

# Add the parent directory to sys.path to allow absolute imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from piper_train.vits.lightning import VitsModel

_LOGGER = logging.getLogger("piper_train.export_onnx")

OPSET_VERSION = 20


def main() -> None:
    """Main entry point"""
    torch.manual_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to model checkpoint (.ckpt)", default="female_epoch=67-step=261800.ckpt")
    parser.add_argument("output", help="Path to output model (.onnx)", default="piper_medium_female.onnx")

    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    _LOGGER.debug(args)

    # -------------------------------------------------------------------------

    args.checkpoint = Path(args.checkpoint)
    args.output = Path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Force model to load on CPU to avoid CUDA memory issues
    model = VitsModel.load_from_checkpoint(args.checkpoint, dataset=None, map_location='cpu')
    model_g = model.model_g

    # Ensure model is on CPU to avoid device mismatch during export
    model_g = model_g.cpu()

    num_symbols = model_g.n_vocab
    num_speakers = model_g.n_speakers

    # Inference only
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

    def infer_forward(text, text_lengths, scales, sid=None):
        # Keep the same interface but handle scales more carefully
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        
        audio = model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sid=sid,
        )[0].unsqueeze(1)

        return audio

    model_g.forward = infer_forward

    # Use a more realistic phoneme sequence (similar to "שלום עולם")
    # This matches the actual usage pattern better
    dummy_phoneme_ids = [96, 24, 120, 27, 25, 3, 109, 27, 24, 120, 14, 25]  # "שלום עולם"
    dummy_input_length = len(dummy_phoneme_ids)
    sequences = torch.LongTensor([dummy_phoneme_ids]).cpu()
    sequence_lengths = torch.LongTensor([dummy_input_length]).cpu()

    sid: Optional[torch.LongTensor] = None
    if num_speakers > 1:
        sid = torch.LongTensor([0]).cpu()

    # Use more neutral scales for export to avoid baking in specific values
    scales = torch.FloatTensor([0.667, 1.0, 0.8]).cpu()
    dummy_input = (sequences, sequence_lengths, scales, sid)

    # Export using legacy mode to avoid torch.export issues
    torch.onnx.export(
        model=model_g,
        args=dummy_input,
        f=str(args.output),
        verbose=False,
        opset_version=OPSET_VERSION,
        input_names=["input", "input_lengths", "scales", "sid"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "phonemes"},
            "input_lengths": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time"},
        },
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        dynamo=False,  # Use legacy export path
    )

    _LOGGER.info("Exported model to %s", args.output)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
