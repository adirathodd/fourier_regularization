import torch
from losses import loss_fn, sequence_loss_fn, fourier_regularization_term
from training.utils import sequence_accuracy_metrics


def train_epoch_operation_batch(
    model,
    dataloaders,
    device,
    optimizer,
    fourier_mode=None,
    fourier_coeff=0.0,
    fourier_groups=2,
    training_target='last_token',
    dataset=None,
    use_non_blocking=False,
):
    """Train each operation separately within each epoch."""
    operations = list(dataloaders.keys())
    total_loss = 0
    total_accuracy = 0
    total_prefix_accuracy = 0.0

    if not dataloaders:
        return 0.0, 0.0, None

    num_batches = max(len(dl) for dl in dataloaders.values())
    iterators = {op: iter(loader) for op, loader in dataloaders.items()}

    num_steps = 0

    model.train()
    for i in range(num_batches):
        # Iterate over each operation separately accumulating gradients
        optimizer.zero_grad()
        for op in operations:
            try:
                features, labels = next(iterators[op])
            except StopIteration:
                iterators[op] = iter(dataloaders[op])
                features, labels = next(iterators[op])

            features = features.to(device, non_blocking=use_non_blocking)
            labels = labels.to(device, non_blocking=use_non_blocking)

            if training_target == 'seq_cot':
                logits = model(features, return_sequence_logits=True)
                base_loss = sequence_loss_fn(logits, labels)
            else:
                logits = model(features)
                base_loss = loss_fn(logits, labels)

            batch_loss = base_loss
            if fourier_mode is not None and fourier_coeff > 0.0:
                is_multiplicative = op in ['multiplication', 'division']
                has_mixed_ops = (
                    any(o in ['addition', 'subtraction'] for o in operations)
                    and any(o in ['multiplication', 'division'] for o in operations)
                )
                reg_term = fourier_regularization_term(
                    model, fourier_mode, dataset.modulo, dataset.includes_zero,
                    is_multiplicative, fourier_groups, has_mixed_ops=has_mixed_ops
                )
                batch_loss = batch_loss + float(fourier_coeff) * reg_term

            batch_loss.backward()

            total_loss += batch_loss.item()

            with torch.no_grad():
                if training_target == 'seq_cot':
                    batch_accuracy, batch_prefix_accuracy = sequence_accuracy_metrics(logits, labels)
                    total_prefix_accuracy += batch_prefix_accuracy
                else:
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == labels).sum().item()
                    total = len(labels)
                    batch_accuracy = correct / total * 100

            total_accuracy += batch_accuracy
            num_steps += 1

        optimizer.step()

    if training_target == 'seq_cot':
        return total_loss / num_steps, total_accuracy / num_steps, total_prefix_accuracy / num_steps
    return total_loss / num_steps, total_accuracy / num_steps, None
