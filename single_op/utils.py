import torch
from models import RNNModel, S4Model, TransformerModel
from datasets import *

def create_model(model_cfg, device='cpu'):
    """Factory function to create and initialize a model."""
    model_type = model_cfg.get('model', 'rnn')
    
    if model_type == 'ssm':
        model = S4Model(
            d_input=model_cfg['vocab_size'],
            d_model=model_cfg['d_model'],
            d_output=model_cfg['vocab_size'],
            n_blocks=model_cfg['num_layers'],
            n=model_cfg['n'],
            l_max=model_cfg['l_max'],
            collapse=model_cfg.get('collapse', True),
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            vocab_size=model_cfg['vocab_size'],
            embedding_dim=model_cfg['d_model'],
            n_heads=model_cfg['attention_heads'],                            
            n_layers=1,
            dim_feedforward=512,
            dropout=0.0,
            max_len=128,
            fixed_positional_encoding=False,
            activation="relu",
            norm_first=False,
            tie=False
        )
    else:
        model = RNNModel(model_cfg['hidden_dim'], model_cfg['num_layers'], model_cfg['vocab_size'])
    
    model = model.to(device)
    return model


def load_model(model_path, model_cfg, device='cpu'):
    """Load a trained model from checkpoint."""
    model = create_model(model_cfg, device=device)
    
    cached_data = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(cached_data['model'])
    
    return model, cached_data


def save_model(checkpoint_data, save_path):
    """Save model and training data."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint_data, save_path)

#edit later so undef doesnt count towards accuracy
def evaluate_model(model, data, labels):  # analysis 1
    """Evaluate model accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(data)
        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        total = len(labels)
        accuracy = correct / total * 100
        return accuracy

def create_modular_addition_dataset(data_cfg, device='cpu'):
    """Factory function to create a modular addition dataset."""
    dataset = ModularAdditionDataset(data_cfg)
    dataset = dataset.to_device(device)
    return dataset


def print_dataset_info(dataset):
    """Print information about the dataset."""
    info = dataset.get_data_info()
    print(f"Dataset Information:")
    print(f"  Modulo: {info['modulo']}")
    print(f"  Total samples: {info['total_samples']}")
    print(f"  Train samples: {info['train_samples']}")
    print(f"  Test samples: {info['test_samples']}")
    print(f"  Train factor: {info['train_frac']}")
    print(f"  Vocabulary size: {info['vocab_size']}")
    
    # Show sample equations
    print(f"\nSample equations:")
    for i in range(min(5, info['train_samples'])):
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
        'save_dir': save_dir if save_dir is not None else 'rnn_modular_addition'
    }

def create_default_data_config():
    """Create default training configuration."""
    return {
        'modulo': 113,
        'train_frac': 0.3,
        'data_seed': 598,
        'seq_length': 12
    }

def create_default_model_config():
    """Create default training configuration."""
    return {
        'model': 'rnn',
        'hidden_dim': 256,
        'num_layers': 1,
        'vocab_size': 114,
        'd_model': 256,
        'd_output': 113,
        'n': 64,
        'l_max': 12,
        'collapse': True
    }