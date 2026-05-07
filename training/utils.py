import copy
import math
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from config.training import TrainingConfig


def make_dataloader(data: torch.Tensor, labels: torch.Tensor, batch_size: int,
                    shuffle: bool, pin_memory: bool, num_workers: int = 0) -> DataLoader:
    """Wrap tensors in a TensorDataset and return a DataLoader."""
    ds = TensorDataset(data, labels)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory,
                      num_workers=num_workers, persistent_workers=(num_workers > 0))


def make_op_dataloaders(
    dataset,
    label_target: str,
    effective_batch_size: int,
    pin_memory: bool,
    num_workers: int,
    filter_fn,
) -> Dict[str, DataLoader]:
    """Build per-operation DataLoaders with oversampling for smaller ops."""
    op_datasets = {}
    n_batches_per_op = {}
    for op in dataset.operations:
        op_data, op_labels = filter_fn(dataset, op, target=label_target)
        if len(op_data) > 0:
            op_ds = TensorDataset(op_data, op_labels)
            n_batches_per_op[op] = math.ceil(len(op_ds) / effective_batch_size)
            op_datasets[op] = op_ds

    if not op_datasets:
        return {}

    num_batches = max(n_batches_per_op.values())
    dataloaders = {}
    for op, op_ds in op_datasets.items():
        if n_batches_per_op[op] < num_batches:
            sampler = RandomSampler(op_ds, replacement=True,
                                    num_samples=num_batches * effective_batch_size)
            dataloaders[op] = DataLoader(op_ds, batch_size=effective_batch_size, sampler=sampler,
                                         pin_memory=pin_memory, num_workers=num_workers,
                                         persistent_workers=(num_workers > 0))
        else:
            dataloaders[op] = DataLoader(op_ds, batch_size=effective_batch_size, shuffle=True,
                                         pin_memory=pin_memory, num_workers=num_workers,
                                         persistent_workers=(num_workers > 0))
    return dataloaders


def setup_optimizer(tc: TrainingConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Setup AdamW optimizer from TrainingConfig."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=tc.lr,
        weight_decay=tc.wd,
        betas=tc.betas,
    )


def setup_scheduler(tc: TrainingConfig, optimizer: torch.optim.Optimizer):
    """Setup MultiStepLR scheduler from TrainingConfig."""
    milestones = list(range(tc.lr_decay_interval, tc.epochs + tc.lr_decay_interval,
                            tc.lr_decay_interval))
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=tc.lr_decay)


def get_sequence_positions(
    nterms: int,
    include_equals_token: bool = True,
) -> Dict[str, List[int] | int | None]:
    """Return index positions for [term, op, ..., term, (= optional)] layout."""
    if not isinstance(nterms, int):
        raise ValueError(f"nterms must be an int, got {type(nterms).__name__}")
    if nterms < 2:
        raise ValueError(f"nterms must be at least 2, got {nterms}")

    seq_len = (2 * nterms) if include_equals_token else (2 * nterms - 1)
    term_positions = list(range(0, seq_len, 2))
    op_positions = list(range(1, seq_len - 1, 2))
    equals_position = (seq_len - 1) if include_equals_token else None

    return {
        "term_positions": term_positions,
        "op_positions": op_positions,
        "equals_position": equals_position,
        "seq_len": seq_len,
    }


def align_running_targets_to_sequence(
    running_targets: torch.Tensor,
    include_equals_token: bool = True,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Map running targets [B, nterms] to aligned sequence targets [B, 2*nterms]."""
    if running_targets.ndim != 2:
        raise ValueError(
            f"running_targets must have shape [batch, nterms], got {tuple(running_targets.shape)}"
        )

    batch_size, nterms = running_targets.shape
    positions = get_sequence_positions(nterms, include_equals_token=include_equals_token)
    seq_len = int(positions["seq_len"])
    term_positions = positions["term_positions"]
    equals_position = positions["equals_position"]

    sequence_targets = torch.full(
        (batch_size, seq_len),
        fill_value=ignore_index,
        dtype=running_targets.dtype,
        device=running_targets.device,
    )

    sequence_targets[:, term_positions] = running_targets
    if equals_position is not None:
        sequence_targets[:, int(equals_position)] = running_targets[:, -1]
    return sequence_targets


def save_final_checkpoint(model, train_losses, train_accuracies, train_prefix_accuracies,
                          test_losses, test_accuracies, test_prefix_accuracies, save_dir: str):
    """Build and save final.pt checkpoint."""
    checkpoint = {
        'model': copy.deepcopy(model.state_dict()),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'train_prefix_accuracies': train_prefix_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'test_prefix_accuracies': test_prefix_accuracies,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'final_train_accuracy': train_accuracies[-1] if train_accuracies else None,
        'final_test_accuracy': test_accuracies[-1] if test_accuracies else None,
        'final_train_prefix_accuracy': train_prefix_accuracies[-1] if train_prefix_accuracies else None,
        'final_test_prefix_accuracy': test_prefix_accuracies[-1] if test_prefix_accuracies else None,
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoints', 'final.pt'))
    return checkpoint


def save_best_checkpoint(best_model_state, best_epoch_info, train_losses, train_accuracies,
                         train_prefix_accuracies, test_losses, test_accuracies,
                         test_prefix_accuracies, save_dir: str):
    """Build and save best.pt checkpoint if a best state was recorded."""
    if best_model_state is None or best_epoch_info is None:
        return
    idx = best_epoch_info['best_idx']
    checkpoint = {
        'model': best_model_state,
        **{k: v for k, v in best_epoch_info.items() if k != 'best_idx'},
        'train_losses': train_losses[:idx + 1],
        'train_accuracies': train_accuracies[:idx + 1],
        'test_losses': test_losses[:idx + 1],
        'test_accuracies': test_accuracies[:idx + 1],
        'train_prefix_accuracies': train_prefix_accuracies[:idx + 1],
        'test_prefix_accuracies': test_prefix_accuracies[:idx + 1],
    }
    torch.save(checkpoint, os.path.join(save_dir, 'checkpoints', 'best.pt'))


def update_best_checkpoint(
    epoch: int,
    lr_decay_interval: int,
    train_loss: float,
    train_accuracy: float,
    test_loss: float,
    test_accuracy: float,
    train_prefix_accuracy,
    test_prefix_accuracy,
    training_target: str,
    model,
    best_combined_loss: float,
    best_model_state,
    best_epoch_info,
):
    """Check if current epoch is a new best and update state if so.

    Returns (best_combined_loss, best_model_state, best_epoch_info).
    """
    is_seq_cot = (training_target == 'seq_cot')
    prefix_threshold_met = (
        not is_seq_cot or
        (train_prefix_accuracy is not None and train_prefix_accuracy >= 99.7 and
         test_prefix_accuracy is not None and test_prefix_accuracy >= 99.7)
    )
    if (
        train_accuracy >= 99.7
        and test_accuracy >= 99.7
        and prefix_threshold_met
    ):
        combined_loss = train_loss + test_loss
        if combined_loss < best_combined_loss:
            best_combined_loss = combined_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch_info = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'train_prefix_accuracy': train_prefix_accuracy,
                'test_prefix_accuracy': test_prefix_accuracy,
                'best_idx': epoch,
            }
    return best_combined_loss, best_model_state, best_epoch_info


def sequence_accuracy_metrics(logits: torch.Tensor, labels_seq: torch.Tensor) -> tuple[float, float]:
    """Return final-answer and prefix accuracies (%) for seq_cot labels."""
    predictions = logits.argmax(dim=-1)

    final_labels = labels_seq[:, -1]
    final_mask = final_labels >= 0
    final_total = int(final_mask.sum().item())
    final_correct = (predictions[:, -1][final_mask] == final_labels[final_mask]).sum().item()
    final_accuracy = (final_correct / final_total * 100) if final_total else 0.0

    prefix_mask = labels_seq >= 0
    prefix_mask[:, -1] = False
    prefix_total = int(prefix_mask.sum().item())
    prefix_correct = (predictions[prefix_mask] == labels_seq[prefix_mask]).sum().item()
    prefix_accuracy = (prefix_correct / prefix_total * 100) if prefix_total else 0.0

    return final_accuracy, prefix_accuracy
