"""Cross-entropy loss functions for modular arithmetic training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy loss evaluated at the final token position."""
    if logits.ndim == 3:
        logits = logits[:, -1]

    logits = logits.float()
    labels = labels.to(torch.long)
    num_classes = logits.shape[-1]

    ignore_index = -100
    targets = labels.clone()
    invalid_mask = labels >= num_classes

    if invalid_mask.all():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    targets[invalid_mask] = ignore_index
    return F.cross_entropy(logits, targets, reduction="mean", ignore_index=ignore_index)


def sequence_loss_fn(
    logits: torch.Tensor,
    labels_seq: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss over a full sequence [B, T, V] with label mask [B, T]."""
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [B, T, V], got {tuple(logits.shape)}")
    if labels_seq.ndim != 2:
        raise ValueError(f"Expected labels_seq with shape [B, T], got {tuple(labels_seq.shape)}")

    logits = logits.float()
    labels_seq = labels_seq.to(torch.long)
    num_classes = logits.shape[-1]

    targets = labels_seq.clone()
    invalid_mask = ((targets < 0) & (targets != ignore_index)) | (targets >= num_classes)
    targets[invalid_mask] = ignore_index

    flat_logits = logits.reshape(-1, num_classes)
    flat_targets = targets.reshape(-1)

    if (flat_targets != ignore_index).sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    return F.cross_entropy(flat_logits, flat_targets, reduction="mean", ignore_index=ignore_index)
