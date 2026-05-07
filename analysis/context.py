"""Shared context and caching helpers for analysis modules."""

import torch

from analysis.analysis_utils import create_fourier_analysis_setup, find_smallest_primitive_root
from analysis.operation_config import get_operation_analysis_context


def apply_mixed_feature_split(W_E, W_fc, operation, fourier_reg_mode=None, has_mixed_ops=False):
    """For mode 1 with mixed ops, use first-half features for add/sub and second-half for mult/div."""
    use_split = fourier_reg_mode == 1 and has_mixed_ops
    if not use_split:
        return W_E, W_fc

    if W_E.shape[-1] < 2:
        raise ValueError(
            f"fourier_reg_mode={fourier_reg_mode} requires feature dim >= 2 for Fourier analysis, got {W_E.shape[-1]}."
        )

    if W_fc.shape[-1] != W_E.shape[-1]:
        raise ValueError(
            "Embedding and unembedding feature dimensions must match for split analysis, "
            f"got {W_E.shape[-1]} and {W_fc.shape[-1]}."
        )

    split_idx = W_E.shape[-1] // 2
    is_multiplicative = operation in ('multiplication', 'division')

    if is_multiplicative:
        W_E = W_E[:, split_idx:]
        W_fc = W_fc[:, split_idx:]
    else:
        W_E = W_E[:, :split_idx]
        W_fc = W_fc[:, :split_idx]

    if W_E.shape[-1] == 0 or W_fc.shape[-1] == 0:
        raise ValueError('Feature split produced an empty feature slice.')

    return W_E, W_fc


def get_fourier_context(checkpoint, dataset, operation, fourier_reg_mode=None, cache=None):
    """Build/cache shared Fourier context for an operation."""
    if operation in ('multiplication', 'division') and dataset.includes_zero:
        W_E_base = checkpoint['model']['embedding.weight'][1:dataset.modulo, :]
        W_fc_base = checkpoint['model']['fc.weight'][1:dataset.modulo, :]
    else:
        op_ctx_preview = get_operation_analysis_context(dataset, operation)
        W_E_base = checkpoint['model']['embedding.weight'][:op_ctx_preview['dataset_size'], :]
        W_fc_base = checkpoint['model']['fc.weight'][:op_ctx_preview['dataset_size'], :]

    cache_key = (
        'fourier_ctx',
        operation,
        fourier_reg_mode,
        str(W_E_base.device),
        str(W_E_base.dtype),
    )
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    op_ctx = get_operation_analysis_context(dataset, operation)
    is_multiplicative = op_ctx['is_multiplicative']
    modulo = op_ctx['modulo']
    dataset_size = op_ctx['dataset_size']

    fourier_basis, fourier_basis_names = create_fourier_analysis_setup(
        modulo,
        not is_multiplicative,
        device=W_E_base.device,
        dtype=W_E_base.dtype,
    )
    generator = find_smallest_primitive_root(modulo)

    if is_multiplicative and dataset.includes_zero:
        W_E_base = checkpoint['model']['embedding.weight'][1:modulo, :]
        W_fc_base = checkpoint['model']['fc.weight'][1:modulo, :]
    else:
        W_E_base = checkpoint['model']['embedding.weight'][:dataset_size, :]
        W_fc_base = checkpoint['model']['fc.weight'][:dataset_size, :]

    W_E = W_E_base
    W_fc = W_fc_base

    perm = None
    if is_multiplicative:
        perm = torch.tensor([generator**i % modulo for i in range(1, modulo)])
        W_E = W_E[perm - 1]
        W_fc = W_fc[perm - 1]

    W_E, W_fc = apply_mixed_feature_split(
        W_E,
        W_fc,
        operation,
        fourier_reg_mode=fourier_reg_mode,
        has_mixed_ops=op_ctx['has_mixed_ops'],
    )

    ctx = {
        'op_ctx': op_ctx,
        'fourier_basis': fourier_basis,
        'fourier_basis_names': fourier_basis_names,
        'generator': generator,
        'perm': perm,
        'W_E_unpermuted': W_E_base,
        'W_fc_unpermuted': W_fc_base,
        'W_E': W_E,
        'W_fc': W_fc,
    }

    if cache is not None:
        cache[cache_key] = ctx
    return ctx


def get_svd_factors(checkpoint, model_type, cache=None):
    """Build/cache SVD factorization for model weights used in analysis."""
    cache_key = ('svd_factors', model_type)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    weights = checkpoint['model']
    if model_type == 'transformer':
        names = ['embedding.weight', 'fc.weight']
    else:
        names = ['embedding.weight', 'fc.weight', 'rnn.weight_ih_l0', 'rnn.weight_hh_l0']

    factors = {}
    with torch.no_grad():
        for name in names:
            factors[name] = torch.linalg.svd(weights[name], full_matrices=False)

    if cache is not None:
        cache[cache_key] = factors
    return factors
