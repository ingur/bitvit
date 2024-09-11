import os
import torch

from collections import defaultdict


def estimate_size(model):
    """Estimates model size of weights in MB"""
    torch.save(model.state_dict(), "temp.pth")
    size = os.path.getsize("temp.pth") / (1024**2)
    os.remove("temp.pth")
    return size


def parameter_stats(model):
    param_counts = defaultdict(lambda: {"count": 0, "params": 0})

    def count_params(module, name=""):
        if len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            param_counts[type(module).__name__]["count"] += 1
            param_counts[type(module).__name__]["params"] += param_count
        else:
            for child_name, child in module.named_children():
                child_name = f"{name}.{child_name}" if name else child_name
                count_params(child, child_name)

    count_params(model)
    return param_counts
