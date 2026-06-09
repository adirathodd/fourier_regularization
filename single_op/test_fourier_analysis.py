import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import einops
import torch.nn as nn

# --- Utility Functions from analysis_utils.py ---
def fourier_basis_constructor(p):
    fourier_basis = []
    fourier_basis_names = []
    fourier_basis.append(torch.ones(p) / np.sqrt(p))
    fourier_basis_names.append('Const')
    for i in range(1, p // 2 + 1):
        cos_term = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        cos_term /= cos_term.norm()
        fourier_basis.append(cos_term)
        fourier_basis_names.append(f'cos {i}')
        sin_term = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        sin_term /= sin_term.norm()
        fourier_basis.append(sin_term)
        fourier_basis_names.append(f'sin {i}')
    return fourier_basis, fourier_basis_names

def create_fourier_analysis_setup(modulo, includes_zero=True):
    dataset_size = modulo if includes_zero else modulo - 1
    fourier_basis, fourier_basis_names = fourier_basis_constructor(dataset_size)
    fourier_basis = torch.stack(fourier_basis).cpu()
    return fourier_basis, fourier_basis_names

def fft2d(mat, fourier_basis, dataset_size):
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=dataset_size, y=dataset_size)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', 
                              mat.to(torch.float), 
                              fourier_basis.to(torch.float), 
                              fourier_basis.to(torch.float))
    return einops.rearrange(fourier_mat, 'f F z -> (f F) z')

# --- Operation Config ---
OPERATION_CONFIGS = {
    'addition': {
        'freq_threshold': 1.5,
        'ip_elbow': 14,
        'hh_elbow': 42,
        'display_name': 'Addition',
        'symbol': '+',
        'trig_function': 'addition',
        'includes_zero': True
    },
    'subtraction': {
        'freq_threshold': 1.5,
        'ip_elbow': 14,
        'hh_elbow': 42,
        'display_name': 'Subtraction',
        'symbol': '-',
        'trig_function': 'subtraction',
        'includes_zero': True
    },
    'multiplication': {
        'freq_threshold': 3.5,
        'ip_elbow': 14,
        'hh_elbow': 42,
        'display_name': 'Multiplication',
        'symbol': '×',
        'trig_function': 'multiplication',
        'includes_zero' : False
    },
    'division': {
        'freq_threshold': 4.5,
        'ip_elbow': 14,
        'hh_elbow': 42,
        'display_name': 'Division',
        'symbol': '÷',
        'trig_function': 'division',
        'includes_zero' : False
    }
}
def get_operation_config(operation):
    if operation not in OPERATION_CONFIGS:
        raise ValueError(f"Unsupported operation: {operation}. Supported operations: {list(OPERATION_CONFIGS.keys())}")
    return OPERATION_CONFIGS[operation]

# --- Minimal Dataset Loader (user must adapt this to their dataset class) ---
class DummyDataset:
    def __init__(self, modulo, data_tensor):
        self.modulo = modulo
        self.dataset = data_tensor

# --- RNNModel from models.py ---
class RNNModel(nn.Module):
    """RNN model for modular arithmetic tasks."""
    def __init__(self, hidden_dim, n_layers, vocab_size):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = n_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, tokens):
        xs = self.embedding(tokens)
        hs, hn = self.rnn(xs)
        hs = self.bn(hs[:, -1, :])
        scores = self.fc(hs)
        return scores

    def get_hidden_states(self, tokens):
        xs = self.embedding(tokens)
        hs, hn = self.rnn(xs)
        return hs
    
    def get_embeddings(self, tokens):
        return self.embedding(tokens)

# --- Main Fourier Spectrum Analysis Function (copy from your code) ---
# (Paste the full fourier_spectrum_analysis_plotting function here)
# ... existing code ...

# --- Main script for testing ---
if __name__ == "__main__":
    # User: set these paths and parameters
    checkpoint_path = "./checkpoints/final.pt"  # Path to your .pt file
    operation = "addition"  # or 'subtraction', 'multiplication', 'division'
    save_dir = "."  # Where to save figures
    modulo = 13  # Set to your dataset's modulo

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # Load or create your dataset here. Replace DummyDataset with your real dataset class.
    # For demonstration, we use random data:
    dummy_data = torch.randint(0, modulo, (modulo*modulo, 3))
    dataset = DummyDataset(modulo, dummy_data)

    # Infer model parameters from checkpoint or set manually
    # You may need to adjust these if your checkpoint/config uses different names
    model_state = checkpoint['model'] if 'model' in checkpoint else checkpoint
    embedding_weight = model_state['embedding.weight']
    vocab_size, hidden_dim = embedding_weight.shape
    n_layers = 1  # Change if your model uses more layers

    # Instantiate and load the model
    model = RNNModel(hidden_dim=hidden_dim, n_layers=n_layers, vocab_size=vocab_size)
    model.load_state_dict(model_state)
    model.eval()

    # Run the analysis
    fourier_spectrum_analysis_plotting(model, checkpoint, dataset, save_dir, operation)
    print("Fourier spectrum analysis complete. Check the figures directory for output.") 