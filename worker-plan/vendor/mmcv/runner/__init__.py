# Réimplémentation minimale de mmcv.runner.load_checkpoint (mmcv 1.x).
import torch


def load_checkpoint(model, filename, map_location=None, strict=False, logger=None):
    checkpoint = torch.load(filename, map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    # les checkpoints DataParallel préfixent les clés par "module."
    state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint
