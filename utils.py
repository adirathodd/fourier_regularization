import os
import glob
import torch
from models import RNNModel, TransformerModel

def create_model(model_cfg, device='cpu'):
    """Factory function to create and initialize a model."""
    model_type = model_cfg.get('model_type', 'rnn').lower()
    
    if model_type == 'rnn':
        hidden_dim = model_cfg.get("hidden_dim")
        num_layers = model_cfg.get("num_layers")
        vocab_size = model_cfg.get("vocab_size")

        missing = []

        if hidden_dim is None:
            missing.append("hidden_dim")
        if num_layers is None:
            missing.append("num_layers")
        if vocab_size is None:
            missing.append("vocab_size")

        if missing:
            raise ValueError(f"The following required parameter(s) for the RNN model are not provided in the config: {missing}")
        
        model = RNNModel(hidden_dim, num_layers, vocab_size)

    elif model_type == 'transformer':
        vocab_size = model_cfg.get('vocab_size')
        embedding_dim = model_cfg.get('embedding_dim', model_cfg.get('d_model', 256))
        n_heads = model_cfg.get('n_heads', model_cfg.get('num_heads', 8))
        n_layers = model_cfg.get('n_layers', model_cfg.get('num_layers', 4))
        dim_feedforward = model_cfg.get('dim_feedforward', model_cfg.get('d_ff', 512))
        dropout = model_cfg.get('dropout', 0.1)
        max_len = model_cfg.get('max_len', model_cfg.get('max_seq_len', 128))
        fixed_pos_enc = model_cfg.get('fixed_positional_encoding', True)
        use_rope = model_cfg.get('use_rope', False)
        rope_base = model_cfg.get('rope_base', 10000.0)
        activation = model_cfg.get('activation', 'relu')
        norm_first = model_cfg.get('norm_first', False)
        mask = model_cfg.get('mask', False)
        tie_weights = model_cfg.get('tie_weights', False)

        missing = []
        if vocab_size is None:
            missing.append('vocab_size')

        if missing:
            raise ValueError(f"The following required parameter(s) for the Transformer model are not provided in the config: {missing}")

        model = TransformerModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len,
            fixed_positional_encoding=fixed_pos_enc,
            use_rope=use_rope,
            rope_base=rope_base,
            activation=activation,
            norm_first=norm_first,
            mask=mask,
            tie_weights=tie_weights
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'rnn' or 'transformer'.")
    
    model = model.to(device)
    return model


def load_model(model_path, model_cfg, device='cpu'):
    """Load a trained model from checkpoint."""
    model = create_model(model_cfg, device=device)
    
    cached_data = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    model.load_state_dict(cached_data['model'])
    
    return model, cached_data


def save_model(checkpoint_data, save_path):
    """Save model and training data."""
    dir_name = os.path.dirname(save_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    torch.save(checkpoint_data, save_path)


def resolve_checkpoint(save_dir):
    """Return path to best.pt, falling back to final.pt.

    Raises FileNotFoundError with nearby run suggestions if neither exists.
    """
    best = os.path.join(save_dir, 'checkpoints', 'best.pt')
    final = os.path.join(save_dir, 'checkpoints', 'final.pt')
    if os.path.exists(best):
        return best
    if os.path.exists(final):
        print("best.pt not found. Falling back to final.pt for analysis.")
        return final

    parent_dir = os.path.dirname(os.path.normpath(save_dir))
    run_name = os.path.basename(os.path.normpath(save_dir))
    sibling_pattern = os.path.join(parent_dir, f"{run_name}*")
    sibling_candidates = sorted(glob.glob(sibling_pattern))
    sibling_with_checkpoints = []

    for sibling in sibling_candidates:
        sibling_best = os.path.join(sibling, 'checkpoints', 'best.pt')
        sibling_final = os.path.join(sibling, 'checkpoints', 'final.pt')
        if os.path.exists(sibling_best):
            sibling_with_checkpoints.append(sibling_best)
        elif os.path.exists(sibling_final):
            sibling_with_checkpoints.append(sibling_final)

    suggestions = ""
    if sibling_with_checkpoints:
        suggestions = (
            "\nFound nearby run directories with checkpoints:\n  - "
            + "\n  - ".join(sibling_with_checkpoints[:5])
        )

    raise FileNotFoundError(
        "No checkpoint found for analysis. "
        f"Checked: {best} and {final}.{suggestions}\n"
        "Run training first (`-t`) or pass `-p` to a directory that already has checkpoints."
    )


def extract_checkpoint_accuracies(checkpoint):
    """Extract train/test accuracy and epoch from a checkpoint dict, with fallbacks.

    Returns (train_acc, test_acc, epoch) — any value may be None.
    """
    train_acc = checkpoint.get('train_accuracy')
    if train_acc is None:
        train_acc = checkpoint.get('final_train_accuracy')
    if train_acc is None:
        train_accs = checkpoint.get('train_accuracies')
        if isinstance(train_accs, list) and train_accs:
            train_acc = train_accs[-1]

    test_acc = checkpoint.get('test_accuracy')
    if test_acc is None:
        test_acc = checkpoint.get('final_test_accuracy')
    if test_acc is None:
        test_accs = checkpoint.get('test_accuracies')
        if isinstance(test_accs, list) and test_accs:
            test_acc = test_accs[-1]

    epoch = checkpoint.get('epoch')
    if epoch is None:
        train_accs = checkpoint.get('train_accuracies')
        if isinstance(train_accs, list) and train_accs:
            epoch = len(train_accs)

    return train_acc, test_acc, epoch


def plot_training_curves(final_checkpoint, save_dir):
    """Save loss_curve.png and accuracy_curve.png to {save_dir}/figures/."""
    import matplotlib.pyplot as plt
    import pandas as pd

    figures_dir = os.path.join(save_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.DataFrame({
        'Train Loss': final_checkpoint['train_losses'],
        'Test Loss':  final_checkpoint['test_losses'],
    })
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(y=['Train Loss', 'Test Loss'], ax=ax)
    ax.set_title('Training and Test Loss Over Time')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    plt.savefig(os.path.join(figures_dir, 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    train_prefix_accs = final_checkpoint.get('train_prefix_accuracies', [])
    test_prefix_accs  = final_checkpoint.get('test_prefix_accuracies', [])
    is_seq_cot = train_prefix_accs and any(v is not None for v in train_prefix_accs)

    acc_data = {
        'Train Accuracy': final_checkpoint['train_accuracies'],
        'Test Accuracy':  final_checkpoint['test_accuracies'],
    }
    if is_seq_cot:
        acc_data['Train Prefix Accuracy'] = train_prefix_accs
        acc_data['Test Prefix Accuracy']  = test_prefix_accs

    acc_df = pd.DataFrame(acc_data)
    fig, ax = plt.subplots(figsize=(10, 6))
    acc_df.plot(ax=ax)
    ax.set_title('Accuracy Over Time')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Epochs')
    ax.set_ylim(0, 105)
    plt.savefig(os.path.join(figures_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


#edit later so undef doesnt count towards accuracy
def evaluate_model(model, data, labels, batch_size: int = 2048):  # analysis 1
    """Evaluate model accuracy.

    Supports:
    - fixed-length tensors: (data, labels)
    - variable-length shards: ([data_n], [labels_n])
    """
    if data is None or labels is None:
        raise ValueError("evaluate_model expected non-None data and labels")

    model.eval()
    # Move inputs to the same device as model parameters (if any)
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device('cpu')

    def _eval_single(data_tensor, label_tensor):
        total_local = len(label_tensor)
        if total_local == 0:
            return 0, 0

        batch_size_local = max(1, int(batch_size))
        correct_local = 0
        for start in range(0, total_local, batch_size_local):
            end = min(start + batch_size_local, total_local)
            batch_data = data_tensor[start:end].to(device)
            batch_labels = label_tensor[start:end].to(device)

            logits = model(batch_data)
            predictions = logits.argmax(dim=-1)
            correct_local += (predictions == batch_labels).sum().item()
        return correct_local, total_local

    with torch.no_grad():
        if isinstance(data, (list, tuple)) or isinstance(labels, (list, tuple)):
            if not (isinstance(data, (list, tuple)) and isinstance(labels, (list, tuple))):
                raise ValueError("data and labels must both be lists/tuples when evaluating variable-length shards")
            if len(data) != len(labels):
                raise ValueError(f"data/labels shard mismatch: {len(data)} vs {len(labels)}")

            correct = 0
            total = 0
            for shard_data, shard_labels in zip(data, labels):
                c, t = _eval_single(shard_data, shard_labels)
                correct += c
                total += t
            return (correct / total * 100) if total > 0 else 0.0

        correct, total = _eval_single(data, labels)
        return (correct / total * 100) if total > 0 else 0.0

def create_modular_addition_dataset(data_cfg, device='cpu'):
    """Factory function to create a modular addition dataset."""
    # Keep full dataset on CPU; move batches to device during training.
    from legacy.datasets import ModularAdditionDataset
    return ModularAdditionDataset(data_cfg)


def print_dataset_info(dataset):
    """Print information about the dataset."""
    info = dataset.get_data_info()
    print(f"Dataset Information:")
    print(f"  Modulo:        {info['modulo']}")
    print(f"  Vocab size:    {info['vocab_size']}")
    print(f"  Total samples: {info['total_samples']}")
    print(f"  Train samples: {info['train_samples']}  ({info['train_frac']:.0%})")
    print(f"  Test samples:  {info['test_samples']}")
    
    # Show sample equations
    print(f"\nSample equations:")
    import random
    n_samples = min(5, info['train_samples'])
    indices = random.sample(range(info['train_samples']), n_samples) if info['train_samples'] > 0 else []
    for i in indices:
        sample = dataset.get_sample(i, 'train')
        print(f"  {sample['equation']}")

def create_default_opt_config(save_dir=None):
    """Create default training configuration."""
    return {
        'epochs': 20000,
        'lr': 0.01,
        'wd': 5e-5,
        'betas': (0.9, 0.99),
        'use_scheduler': True,
        'lr_decay_interval': 5000,
        'lr_decay': 0.1,
        'save_every': 2000,
        'save_dir': save_dir if save_dir is not None else 'rnn_modular_addition',
        'batch_mode': 'full', # options - full, mini-batch, operation-batch (train in batches with equal number of samples from each op)
        'batch_size': None # None for full and operation-stratified, need to provide for mini-batch
    }

def create_default_data_config(operations: list, nterms: int = 2):
    """Create default training configuration."""
    includes_zero = any(op in ["addition", "subtraction"] for op in operations)
    base_term_vocab = 113 if includes_zero else 112
    return {
        'modulo': 113,
        'nterms': nterms,
        'train_frac': 0.3,
        'data_seed': 598,
        'include_equals_token': True,
        'includes_zero': includes_zero,
        'vocab_size': base_term_vocab + 1 + len(operations),
    }

def create_default_model_config(operations: list):
    """Create default training configuration."""
    includes_zero = any(op in ["addition", "subtraction"] for op in operations)
    base_term_vocab = 113 if includes_zero else 112
    return {
        'hidden_dim': 256,
        'num_layers': 1,
        'vocab_size': base_term_vocab + 1 + len(operations),
    }

def create_dataset(operations, data_cfg, device):
    from datasets import (
        AdditionModDataset,
        SubtractionModDataset,
        MultiplicationModDataset,
        DivisionModDataset,
        MultiOperationDataset,
        MixedSequenceOperationDataset,
        VariableLengthMixedSequenceDataset,
    )

    dataset_classes = {
        'addition': AdditionModDataset,
        'subtraction': SubtractionModDataset,
        'multiplication': MultiplicationModDataset,
        'division': DivisionModDataset
    }

    nterms_list = data_cfg.get('nterms_list')
    if nterms_list:
        return VariableLengthMixedSequenceDataset(data_cfg, operations)

    mixed_sequence = bool(data_cfg.get('mixed_sequence_operations', False))
    if mixed_sequence:
        return MixedSequenceOperationDataset(data_cfg, operations)

    if len(operations) == 1:
        return dataset_classes[operations[0]](data_cfg)

    return MultiOperationDataset(data_cfg, operations)


def _operation_filter_mask(data_tensor, dataset, target_op_token):
    """Build an operation mask that works for variable-length expressions.

    For nterms > 2, every operator position should match the same op token.
    """
    op_positions = getattr(dataset, 'op_positions', [1])
    if getattr(dataset, 'mixed_ops_within_sequence', False):
        mask = torch.zeros(data_tensor.size(0), dtype=torch.bool, device=data_tensor.device)
        for pos in op_positions:
            mask |= (data_tensor[:, pos] == target_op_token)
        return mask

    mask = torch.ones(data_tensor.size(0), dtype=torch.bool, device=data_tensor.device)
    for pos in op_positions:
        mask &= (data_tensor[:, pos] == target_op_token)
    return mask


def filter_data_by_operation(dataset, target_operation, split='all', target='final'):
    """Filter dataset by operation, optionally restricted to a data split.

    split: 'all' | 'train' | 'test'
    target: 'final' | 'seq'  (selects label variant)
    """
    if target not in ('final', 'seq'):
        raise ValueError(f"target must be 'final' or 'seq', got {target!r}")

    if split == 'all':
        data = dataset.dataset
        labels = dataset.labels if target == 'final' else dataset.labels_seq
    elif split == 'train':
        data = dataset.train_data
        labels = dataset.train_labels if target == 'final' else dataset.train_labels_seq
    elif split == 'test':
        data = dataset.test_data
        labels = dataset.test_labels if target == 'final' else dataset.test_labels_seq
    else:
        raise ValueError(f"split must be 'all', 'train', or 'test', got {split!r}")

    # Fast path for single-op datasets: every sample belongs to that operation.
    # This avoids brittle token-index assumptions across encoding variants.
    if len(dataset.operations) == 1:
        only_op = dataset.operations[0]
        if target_operation != only_op:
            raise ValueError(
                f"Operation {target_operation} not found in dataset operations: {dataset.operations}"
            )
        return data, labels

    base = dataset.modulo if dataset.includes_zero else dataset.modulo - 1
    include_equals_token = bool(getattr(dataset, "include_equals_token", True))
    first_op_token = base + (1 if include_equals_token else 0)
    try:
        op_index = dataset.operations.index(target_operation)
    except ValueError:
        raise ValueError(
            f"Operation {target_operation} not found in dataset operations: {dataset.operations}"
        )
    target_op_token = first_op_token + op_index

    if isinstance(data, (list, tuple)) or isinstance(labels, (list, tuple)):
        if not (isinstance(data, (list, tuple)) and isinstance(labels, (list, tuple))):
            raise ValueError("data and labels must both be lists/tuples for variable-length filtering")

        filtered_data = []
        filtered_labels = []
        for data_tensor, label_tensor in zip(data, labels):
            mask = _operation_filter_mask(data_tensor, dataset, target_op_token)
            filtered_data.append(data_tensor[mask])
            filtered_labels.append(label_tensor[mask])
        return filtered_data, filtered_labels

    mask = _operation_filter_mask(data, dataset, target_op_token)
    return data[mask], labels[mask]

def get_device(device, require_cuda: bool = False):
    if not device or device == "cuda":
        if torch.cuda.is_available():
            device = 'cuda'
            try:
                torch.ones(1).to(device)
            except Exception:
                found = False
                for i in range(torch.cuda.device_count()):
                    try:
                        device = f'cuda:{i}'
                        torch.ones(1).to(device)
                        found = True
                        break
                    except Exception:
                        continue
                if not found:
                    print("No available CUDA devices found (all busy or unavailable). Falling back to CPU.")
                    device = 'cpu'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'

    if require_cuda and device == 'cpu':
        raise RuntimeError('CUDA requested via --device but no CUDA device is available.')

    return device
