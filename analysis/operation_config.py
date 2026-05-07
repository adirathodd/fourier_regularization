"""Shared operation metadata and lightweight derived analysis context."""

import torch

OPERATION_CONFIGS = {
    'addition': {
        'display_name': 'Addition',
        'symbol': '+',
        'trig_function': 'addition',
    },
    'subtraction': {
        'display_name': 'Subtraction',
        'symbol': '-',
        'trig_function': 'subtraction',
    },
    'multiplication': {
        'display_name': 'Multiplication',
        'symbol': '×',
        'trig_function': 'multiplication',
    },
    'division': {
        'display_name': 'Division',
        'symbol': '÷',
        'trig_function': 'division',
    },
}


def get_operation_config(operation):
    """Get configuration for a specific operation."""
    if operation not in OPERATION_CONFIGS:
        raise ValueError(
            f"Unsupported operation: {operation}. Supported operations: {list(OPERATION_CONFIGS.keys())}"
        )
    return OPERATION_CONFIGS[operation]


def get_operation_analysis_context(dataset, operation):
    """Return shared per-operation analysis metadata to reduce repeated recomputation."""
    config = get_operation_config(operation)
    is_multiplicative = operation in ('multiplication', 'division')
    modulo = dataset.modulo
    nterms = int(getattr(dataset, 'nterms', 2))
    dataset_size = modulo - 1 if is_multiplicative else modulo
    sample_interval = dataset_size ** max(1, nterms - 1)
    expected_grid = dataset_size ** nterms
    has_mixed_ops = (
        any(op in ('addition', 'subtraction') for op in dataset.operations)
        and any(op in ('multiplication', 'division') for op in dataset.operations)
    )
    return {
        'config': config,
        'is_multiplicative': is_multiplicative,
        'includes_zero': dataset.includes_zero,
        'modulo': modulo,
        'nterms': nterms,
        'dataset_size': dataset_size,
        'sample_interval': sample_interval,
        'expected_grid': expected_grid,
        'has_mixed_ops': has_mixed_ops,
    }


def compute_frequency_threshold(coeffs, top_fraction=0.1, min_components=4, fixed_threshold=None):
    """Derive a data-dependent coefficient threshold from the current spectrum."""
    if fixed_threshold is not None:
        return float(fixed_threshold)

    coeffs = torch.as_tensor(coeffs)
    if coeffs.numel() <= 1:
        return float('inf')

    non_dc = coeffs[1:].abs()
    k = max(min_components, int(non_dc.numel() * top_fraction))
    k = min(k, non_dc.numel())
    if k <= 0:
        return float('inf')

    topk_vals = torch.topk(non_dc, k=k).values
    return float(topk_vals.min().item())


def compute_svd_energy_elbow(singular_values, energy_ratio=0.9, min_components=1):
    """Compute the smallest k such that cumulative singular-value mass reaches energy_ratio."""
    S = torch.as_tensor(singular_values)
    if S.numel() == 0:
        return 0

    mass = S.abs()
    total = mass.sum()
    if float(total) <= 0.0:
        return min_components

    cumsum = torch.cumsum(mass, dim=0) / total
    k = int(torch.searchsorted(cumsum, torch.tensor(energy_ratio, device=cumsum.device)).item()) + 1
    k = max(min_components, k)
    return min(k, S.numel())
