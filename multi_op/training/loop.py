import copy
import os

import torch
import tqdm
from functools import partial

from config.training import TrainingConfig
from utils import filter_data_by_operation
from training.utils import (
    make_dataloader,
    make_op_dataloaders,
    setup_optimizer,
    setup_scheduler,
    update_best_checkpoint,
    save_final_checkpoint,
    save_best_checkpoint,
)
from training.epoch import train_epoch
from training.epoch import train_epoch_multi_length
from training.operation_batch import train_epoch_operation_batch
from training.evaluation import eval_step
from training.evaluation import eval_step_multi_length


def train(model, dataset, tc: TrainingConfig):
    """Main training loop."""
    batch_size = tc.batch_size if tc.batch_size is not None else len(dataset)

    if not os.path.isdir(os.path.join(tc.save_dir, 'checkpoints')):
        os.makedirs(os.path.join(tc.save_dir, 'checkpoints'))

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_non_blocking = device.type == 'cuda'
    pin_memory = device.type == 'cuda'

    # Pre-build dataloaders once (data is static)
    label_target = 'seq' if tc.training_target == 'seq_cot' else 'final'
    is_variable_length = bool(getattr(dataset, "is_variable_length", False))
    if is_variable_length:
        train_splits = dataset.get_train_splits(target=label_target)
        test_splits_raw = dataset.get_test_splits(target=label_target)

        train_dls_by_nterms = {}
        for nterms, (train_data, train_labels) in train_splits.items():
            effective_batch_size = batch_size if batch_size else len(train_data)
            train_dls_by_nterms[nterms] = make_dataloader(
                train_data, train_labels, effective_batch_size, shuffle=True,
                pin_memory=pin_memory, num_workers=tc.dataloader_num_workers,
            )

        test_splits_dev = {
            nterms: (data.to(device), labels.to(device))
            for nterms, (data, labels) in test_splits_raw.items()
        }
    else:
        train_data, train_labels = dataset.get_train_data(target=label_target)
        test_data_raw, test_labels_raw = dataset.get_test_data(target=label_target)

        # Move test data to device once
        test_data_dev = test_data_raw.to(device)
        test_labels_dev = test_labels_raw.to(device)

        effective_batch_size = batch_size if batch_size else len(train_data)

        if tc.batch_mode == "operation-batch":
            dataloaders = make_op_dataloaders(
                dataset, label_target, effective_batch_size, pin_memory, tc.dataloader_num_workers,
                filter_fn=partial(filter_data_by_operation, split='train'),
            )
        else:
            train_dl = make_dataloader(train_data, train_labels, effective_batch_size, shuffle=True,
                                       pin_memory=pin_memory, num_workers=tc.dataloader_num_workers)

    optimizer = setup_optimizer(tc, model)
    if tc.use_scheduler:
        scheduler = setup_scheduler(tc, optimizer)

    print(f"Starting training for {tc.epochs} epochs with {tc.batch_mode} batching...")
    if tc.batch_mode == "operation-batch":
        print(f"Operations: {dataset.operations}")

    train_losses = []
    train_accuracies = []
    train_prefix_accuracies = []
    test_losses = []
    test_accuracies = []
    test_prefix_accuracies = []
    best_combined_loss = float('inf')
    best_model_state = None
    best_epoch_info = None

    pbar = tqdm.tqdm(range(tc.epochs), desc="Training")

    for epoch in pbar:
        if is_variable_length:
            train_loss, train_accuracy, train_prefix_accuracy = train_epoch_multi_length(
                model, train_dls_by_nterms, device, optimizer,
                tc.fourier_reg_mode, tc.fourier_reg_coefficient, tc.fourier_reg_groups, tc.training_target,
                dataset=dataset, use_non_blocking=use_non_blocking,
            )
        elif tc.batch_mode == "operation-batch":
            train_loss, train_accuracy, train_prefix_accuracy = train_epoch_operation_batch(
                model, dataloaders, device, optimizer,
                tc.fourier_reg_mode, tc.fourier_reg_coefficient, tc.fourier_reg_groups, tc.training_target,
                dataset=dataset, use_non_blocking=use_non_blocking,
            )
        else:
            train_loss, train_accuracy, train_prefix_accuracy = train_epoch(
                model, train_dl, device, optimizer,
                tc.fourier_reg_mode, tc.fourier_reg_coefficient, tc.fourier_reg_groups, tc.training_target,
                dataset=dataset, use_non_blocking=use_non_blocking,
            )

        if is_variable_length:
            test_loss, test_accuracy, test_prefix_accuracy = eval_step_multi_length(
                model, test_splits_dev, tc.training_target
            )
        else:
            test_loss, test_accuracy, test_prefix_accuracy = eval_step(
                model, test_data_dev, test_labels_dev, tc.training_target
            )

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_prefix_accuracies.append(train_prefix_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_prefix_accuracies.append(test_prefix_accuracy)

        # Track best-only checkpoint when both accuracies are high and combined loss is minimal
        best_combined_loss, best_model_state, best_epoch_info = update_best_checkpoint(
            epoch, tc.lr_decay_interval,
            train_loss, train_accuracy, test_loss, test_accuracy,
            train_prefix_accuracy, test_prefix_accuracy,
            tc.training_target, model,
            best_combined_loss, best_model_state, best_epoch_info,
        )

        # Update progress bar with metrics
        postfix = {
            'TrL': f'{train_loss:.3f}',
            'TrA': f'{train_accuracy:.2f}%',
            'TsL': f'{test_loss:.3f}',
            'TsA': f'{test_accuracy:.2f}%'
        }
        if tc.training_target == 'seq_cot' and train_prefix_accuracy is not None and test_prefix_accuracy is not None:
            postfix['TrPA'] = f'{train_prefix_accuracy:.2f}%'
            postfix['TsPA'] = f'{test_prefix_accuracy:.2f}%'
        pbar.set_postfix(postfix)

        if tc.use_scheduler:
            scheduler.step()

        if tc.save_every > 0:
            if (epoch + 1) % tc.save_every == 0:
                checkpoint = {
                    'model': copy.deepcopy(model.state_dict()),
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy,
                    'train_prefix_accuracy': train_prefix_accuracy,
                    'test_prefix_accuracy': test_prefix_accuracy,
                }
                torch.save(checkpoint, os.path.join(tc.save_dir, f'checkpoints/{epoch+1}.pt'))

    pbar.close()

    final_checkpoint = save_final_checkpoint(
        model, train_losses, train_accuracies, train_prefix_accuracies,
        test_losses, test_accuracies, test_prefix_accuracies, tc.save_dir,
    )
    save_best_checkpoint(
        best_model_state, best_epoch_info,
        train_losses, train_accuracies, train_prefix_accuracies,
        test_losses, test_accuracies, test_prefix_accuracies, tc.save_dir,
    )

    print("Training completed!")

    return final_checkpoint
