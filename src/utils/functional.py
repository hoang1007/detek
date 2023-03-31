import torch
from torch import nn


def init_weight(module: nn.Module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.01)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, torch.nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data)

    if isinstance(module, (nn.Linear, nn.Conv2d)) and module.bias is not None:
        module.bias.data.zero_()


def freeze_weight(model: nn.Module):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False


def random_choice(input: torch.Tensor, num_samples: int, replacement=False, auto_replace=False):
    """
    Args:
        input (Tensor): Shape: (N, ...)
        num_samples (int): Number of samples to be drawn
        replacement (boolean): Whether to draw with replacement
        auto_replace (boolean): Whether to replace the input with itself
                                if the number of samples is greater than the number of elements in the input

    Returns:
        ids (Tensor): Indices of the samples. Shape: (num_samples,)
    """

    if auto_replace:
        replacement = input.size(0) < num_samples

    dtype = torch.int64
    device = input.device

    if replacement:
        ids = torch.randint(input.size(0), size=(num_samples,), dtype=dtype, device=device)
    else:
        # Random permutation on CPU due to a bug in PyTorch
        ids = torch.randperm(input.size(0), dtype=dtype)[:num_samples].to(device)

    return ids
