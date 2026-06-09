import torch
from losses import loss_fn, sequence_loss_fn
from training.utils import sequence_accuracy_metrics


def eval_step(model, test_data, test_labels, training_target='last_token'):
    """Perform one evaluation step."""
    model.eval()

    with torch.inference_mode():
        if training_target == 'seq_cot':
            test_logits = model(test_data, return_sequence_logits=True)
            test_loss = sequence_loss_fn(test_logits, test_labels)
            test_accuracy, test_prefix_accuracy = sequence_accuracy_metrics(test_logits, test_labels)
        else:
            test_logits = model(test_data)
            test_loss = loss_fn(test_logits, test_labels)
            predictions = test_logits.argmax(dim=-1)
            correct = (predictions == test_labels).sum().item()
            total = test_labels.numel()
            test_accuracy = correct / total * 100 if total else 0.0
            test_prefix_accuracy = None

    return test_loss.item(), test_accuracy, test_prefix_accuracy


def eval_step_multi_length(model, test_splits_by_nterms, training_target='last_token'):
    """Evaluate exact variable-length splits and aggregate metrics."""
    model.eval()

    total_loss_weighted = 0.0
    total_correct = 0
    total_count = 0
    total_prefix_correct = 0
    total_prefix_count = 0

    with torch.inference_mode():
        for nterms in sorted(test_splits_by_nterms.keys()):
            test_data, test_labels = test_splits_by_nterms[nterms]
            if test_data.numel() == 0:
                continue

            if training_target == 'seq_cot':
                logits = model(test_data, return_sequence_logits=True)
                loss = sequence_loss_fn(logits, test_labels)
                total_loss_weighted += loss.item() * test_data.size(0)

                predictions = logits.argmax(dim=-1)
                final_labels = test_labels[:, -1]
                final_mask = final_labels >= 0
                total_correct += (predictions[:, -1][final_mask] == final_labels[final_mask]).sum().item()
                total_count += int(final_mask.sum().item())

                prefix_mask = test_labels >= 0
                prefix_mask[:, -1] = False
                total_prefix_correct += (predictions[prefix_mask] == test_labels[prefix_mask]).sum().item()
                total_prefix_count += int(prefix_mask.sum().item())
            else:
                logits = model(test_data)
                loss = loss_fn(logits, test_labels)
                total_loss_weighted += loss.item() * test_data.size(0)

                predictions = logits.argmax(dim=-1)
                total_correct += (predictions == test_labels).sum().item()
                total_count += int(test_labels.numel())

    if total_count == 0:
        return 0.0, 0.0, (0.0 if training_target == 'seq_cot' else None)

    mean_loss = total_loss_weighted / total_count
    accuracy = total_correct / total_count * 100.0
    if training_target == 'seq_cot':
        prefix_accuracy = (total_prefix_correct / total_prefix_count * 100.0) if total_prefix_count else 0.0
        return mean_loss, accuracy, prefix_accuracy
    return mean_loss, accuracy, None
