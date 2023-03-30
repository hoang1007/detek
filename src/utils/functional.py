import torch
from torch import nn


def init_weight(model: nn.Module, std=0.01):
    model.weight.data.normal_(std=std)  # type: ignore

    try:
        model.bias.data.normal_(std=std)  # type: ignore
    except ValueError:
        print("Cannot init bias for model: ", model)

    return model


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
