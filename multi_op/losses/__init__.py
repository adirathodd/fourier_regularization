"""losses package — loss functions and regularization terms.

Adding a new module
--------------------
1. Create ``losses/<name>.py`` with your functions.
2. Import and re-export them here so callers use ``from losses import <fn>``.
"""

from losses.ce import loss_fn, sequence_loss_fn
from losses.fourier_reg import (
    fourier_regularization_term,
    clear_caches,
    get_cached_primitive_root,
    get_cached_permutation,
    get_cached_fourier_basis,
)

__all__ = [
    "loss_fn",
    "sequence_loss_fn",
    "fourier_regularization_term",
    "clear_caches",
    "get_cached_primitive_root",
    "get_cached_permutation",
    "get_cached_fourier_basis",
]
