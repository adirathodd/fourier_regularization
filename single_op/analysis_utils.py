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
    cos_sin_embed_np = cos_sin_embed.cpu().detach().numpy()
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
    R = p // 2 + 1
    for i in range(1, R):

        # Cosine term
        cos_term = torch.cos(2 * torch.pi * torch.arange(p) * i / p)
        cos_term /= cos_term.norm()
        fourier_basis.append(cos_term)
        fourier_basis_names.append(f'cos {i}')
        
        # Sin term - do not include last sin term if p is even
        if i < p // 2 or p % 2 == 1:
            sin_term = torch.sin(2 * torch.pi * torch.arange(p) * i / p)
            sin_term /= sin_term.norm()
            fourier_basis.append(sin_term)
            fourier_basis_names.append(f'sin {i}')
    
    return fourier_basis, fourier_basis_names



def fft2d(mat, fourier_basis, dataset_size):
    """Convert matrix to 2D Fourier basis representation."""
    # Reshape input to 2D grid
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=dataset_size, y=dataset_size)
    
    # Apply 2D Fourier transform
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', 
                              mat.to(torch.float), 
                              fourier_basis.to(torch.float), 
                              fourier_basis.to(torch.float))
    
    # Return in the 2D Fourier coefficient format (freq_pairs, features)
    return einops.rearrange(fourier_mat, 'f F z -> (f F) z')

def unflatten_first(tensor, p):
    """Unflatten first dimension if it equals p*p."""
    if tensor.shape[0] == p * p:
        return einops.rearrange(tensor, '(x y) ... -> x y ...', x=p, y=p)
    else:
        return tensor

#edited so it works for datsets size (modulo-1)^2 and not just modulo^2
def create_fourier_analysis_setup(modulo, includes_zero=True):
    """Create Fourier analysis setup for given modulo and operation type."""
    dataset_size = modulo if includes_zero else modulo - 1
    fourier_basis, fourier_basis_names = fourier_basis_constructor(dataset_size)
    fourier_basis = torch.stack(fourier_basis).cpu()
    
    return fourier_basis, fourier_basis_names
