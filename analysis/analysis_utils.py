import torch
import numpy as np
import einops
import pandas as pd
from pandas import melt


def embed_to_cos_sin(fourier_embed):
    """Convert Fourier embedding to cos/sin components."""
    if len(fourier_embed.shape) == 1:
        return torch.stack([fourier_embed[1::2], fourier_embed[2::2]])
    else:
        return torch.stack([fourier_embed[:, 1::2], fourier_embed[:, 2::2]], dim=1)


def plot_embed_bars(fourier_embed, title='Norm of embedding of each Fourier Component', return_fig=False, **kwargs):
    """Plot bar chart of Fourier embedding components."""
    cos_sin_embed = embed_to_cos_sin(fourier_embed)
    cos_sin_embed_np = cos_sin_embed.detach().cpu().numpy()
    df = melt(pd.DataFrame(cos_sin_embed_np))
    return df.groupby('variable').mean().plot(kind='bar', title=title, **kwargs)


def fourier_basis_constructor(p):
    """Construct Fourier basis for modular arithmetic with p elements."""
    fourier_basis = []
    fourier_basis_names = []
    
    # Constant term
    fourier_basis.append(torch.ones(p) / np.sqrt(p))
    fourier_basis_names.append('Const')
    
    # Cos and sin terms
    for i in range(1, p // 2 + 1):
        # Cosine term
        cos_term = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        cos_term /= cos_term.norm()
        fourier_basis.append(cos_term)
        fourier_basis_names.append(f'cos {i}')
        
        # Sine term
        sin_term = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
        sin_term /= sin_term.norm()
        fourier_basis.append(sin_term)
        fourier_basis_names.append(f'sin {i}')
    
    return fourier_basis, fourier_basis_names


def fft_nd(mat, fourier_basis, dataset_size, nterms=2):
    """Convert flattened operand-grid activations into n-D Fourier coefficients.

    Args:
        mat: Tensor shaped [dataset_size**nterms, ...]
        fourier_basis: Tensor shaped [n_freqs, dataset_size]
        dataset_size: Number of values each operand can take
        nterms: Number of operands in expression (2 for pairwise tasks)

    Returns:
        Tensor shaped [n_freqs**nterms, features]
    """
    if nterms < 2:
        raise ValueError(f"nterms must be at least 2, got {nterms}")

    expected = dataset_size ** nterms
    if mat.shape[0] != expected:
        raise ValueError(
            f"fft_nd expected first dimension {expected} (= {dataset_size}^{nterms}), "
            f"got {mat.shape[0]}"
        )

    grid_shape = [dataset_size] * nterms
    reshaped = mat.reshape(*grid_shape, -1).to(torch.float)
    basis = fourier_basis.to(torch.float)

    # Apply the same 1D Fourier basis along each operand axis.
    transformed = reshaped
    for axis in range(nterms):
        transformed = torch.movedim(transformed, axis, 0)
        transformed = torch.tensordot(basis, transformed, dims=([1], [0]))
        transformed = torch.movedim(transformed, 0, axis)

    return transformed.reshape(-1, transformed.shape[-1])


def fft2d(mat, fourier_basis, dataset_size):
    """Convert matrix to 2D Fourier basis representation.

    Backward-compatible wrapper for pairwise tasks.
    """
    return fft_nd(mat, fourier_basis, dataset_size, nterms=2)


def apply_mixed_hidden_feature_split(hidden_state, operation, fourier_reg_mode=None, has_mixed_ops=False):
    """Apply mixed-op mode-1 split to hidden-state features for Fourier maps."""
    use_split = fourier_reg_mode == 1 and has_mixed_ops
    if not use_split:
        return hidden_state

    feature_dim = hidden_state.shape[-1]
    if feature_dim < 2:
        raise ValueError(
            f"fourier_reg_mode={fourier_reg_mode} requires hidden feature dim >= 2 for split analysis, got {feature_dim}."
        )

    split_idx = feature_dim // 2
    is_multiplicative = operation in ('multiplication', 'division')
    sliced = hidden_state[..., split_idx:] if is_multiplicative else hidden_state[..., :split_idx]

    if sliced.shape[-1] == 0:
        raise ValueError('Hidden-state split produced an empty feature slice.')
    return sliced

def get_prime_factors(n):
    """Get prime factors of a number."""
    factors = set()
    d = 2
    temp = n
    while d * d <= temp:
        while temp % d == 0:
            factors.add(d)
            temp //= d
        d += 1
    if temp > 1:
        factors.add(temp)
    return factors

def find_smallest_primitive_root(p):
    """Find the smallest primitive root modulo p."""
    if p == 2: return 1
    
    phi = p - 1
    factors = get_prime_factors(phi)
    
    for g in range(2, p):
        is_primitive = True
        for factor in factors:
            if pow(g, phi // factor, p) == 1:
                is_primitive = False
                break
        if is_primitive:
            return g
    return None

def unflatten_first(tensor, p, nterms=2):
    """Unflatten first dimension if it equals p**nterms."""
    if nterms < 2:
        raise ValueError(f"nterms must be at least 2, got {nterms}")

    if tensor.shape[0] == p ** nterms:
        return tensor.reshape(*([p] * nterms), *tensor.shape[1:])
    else:
        return tensor

#edited so it works for datsets size (modulo-1)^2 and not just modulo^2
def create_fourier_analysis_setup(modulo, includes_zero=True, device=None, dtype=None):
    """Create Fourier analysis setup for given modulo and operation type."""
    dataset_size = modulo if includes_zero else modulo - 1
    fourier_basis, fourier_basis_names = fourier_basis_constructor(dataset_size)
    fourier_basis = torch.stack(fourier_basis)
    if dtype is not None:
        fourier_basis = fourier_basis.to(dtype=dtype)
    if device is not None:
        fourier_basis = fourier_basis.to(device=device)
    else:
        fourier_basis = fourier_basis.cpu()
    
    return fourier_basis, fourier_basis_names
