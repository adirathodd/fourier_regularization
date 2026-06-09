import torch
from losses import loss_fn, sequence_loss_fn, fourier_regularization_term
from training.utils import sequence_accuracy_metrics


def train_epoch(
    model,
    dataloader,
    device,
    optimizer,
    fourier_mode=None,
    fourier_coeff=0.0,
    fourier_groups=2,
    training_target='last_token',
    dataset=None,
    use_non_blocking=False,
):
    """Perform one full training epoch over all batches in the dataloader."""
    total_loss = 0
    total_accuracy = 0
    total_prefix_accuracy = 0.0
    num_batches = 0

    model.train()
    for features, labels in dataloader:
        optimizer.zero_grad()

        features = features.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        if training_target == 'seq_cot':
            batch_logits = model(features, return_sequence_logits=True)
            base_loss = sequence_loss_fn(batch_logits, labels)
        else:
            batch_logits = model(features)
            base_loss = loss_fn(batch_logits, labels)

        batch_loss = base_loss
        if fourier_mode is not None and fourier_coeff > 0.0:
            is_multiplicative = all(op in ['multiplication', 'division'] for op in dataset.operations)
            has_mixed_ops = (
                any(op in ['addition', 'subtraction'] for op in dataset.operations)
                and any(op in ['multiplication', 'division'] for op in dataset.operations)
            )
            reg_term = fourier_regularization_term(
                model, fourier_mode, dataset.modulo, dataset.includes_zero,
                is_multiplicative, fourier_groups, has_mixed_ops=has_mixed_ops
            )
            batch_loss = batch_loss + float(fourier_coeff) * reg_term

        batch_loss.backward()
        optimizer.step()

        total_loss += batch_loss.item()

        with torch.no_grad():
            if training_target == 'seq_cot':
                batch_accuracy, batch_prefix_accuracy = sequence_accuracy_metrics(batch_logits, labels)
                total_prefix_accuracy += batch_prefix_accuracy
            else:
                predictions = batch_logits.argmax(dim=-1)
                correct = (predictions == labels).sum().item()
                total = len(labels)
                batch_accuracy = correct / total * 100
        total_accuracy += batch_accuracy
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, None
    if training_target == 'seq_cot':
        return total_loss / num_batches, total_accuracy / num_batches, total_prefix_accuracy / num_batches
    return total_loss / num_batches, total_accuracy / num_batches, None


def train_epoch_multi_length(
    model,
    dataloaders_by_nterms,
    device,
    optimizer,
    fourier_mode=None,
    fourier_coeff=0.0,
    fourier_groups=2,
    training_target='last_token',
    dataset=None,
    use_non_blocking=False,
):
    """Train on multiple exact sequence lengths without upcasting or padding."""
    total_loss = 0.0
    total_accuracy = 0.0
    total_prefix_accuracy = 0.0
    num_batches = 0

    model.train()
    for nterms in sorted(dataloaders_by_nterms.keys()):
        dataloader = dataloaders_by_nterms[nterms]
        for features, labels in dataloader:
            optimizer.zero_grad()

            features = features.to(device, non_blocking=use_non_blocking)
            labels = labels.to(device, non_blocking=use_non_blocking)

            if training_target == 'seq_cot':
                batch_logits = model(features, return_sequence_logits=True)
                base_loss = sequence_loss_fn(batch_logits, labels)
            else:
                batch_logits = model(features)
                base_loss = loss_fn(batch_logits, labels)

            batch_loss = base_loss
            if fourier_mode is not None and fourier_coeff > 0.0:
                is_multiplicative = all(op in ['multiplication', 'division'] for op in dataset.operations)
                has_mixed_ops = (
                    any(op in ['addition', 'subtraction'] for op in dataset.operations)
                    and any(op in ['multiplication', 'division'] for op in dataset.operations)
                )
                reg_term = fourier_regularization_term(
                    model, fourier_mode, dataset.modulo, dataset.includes_zero,
                    is_multiplicative, fourier_groups, has_mixed_ops=has_mixed_ops
                )
                batch_loss = batch_loss + float(fourier_coeff) * reg_term

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()
            with torch.no_grad():
                if training_target == 'seq_cot':
                    batch_accuracy, batch_prefix_accuracy = sequence_accuracy_metrics(batch_logits, labels)
                    total_prefix_accuracy += batch_prefix_accuracy
                else:
                    predictions = batch_logits.argmax(dim=-1)
                    correct = (predictions == labels).sum().item()
                    total = len(labels)
                    batch_accuracy = correct / total * 100
            total_accuracy += batch_accuracy
            num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, None
    if training_target == 'seq_cot':
        return total_loss / num_batches, total_accuracy / num_batches, total_prefix_accuracy / num_batches
    return total_loss / num_batches, total_accuracy / num_batches, None
