import torch

from utils import filter_data_by_operation

from analysis.analysis_utils import create_fourier_analysis_setup, fft2d
from analysis.context import get_svd_factors
from analysis.operation_config import compute_frequency_threshold


def _get_freq_pairs(
    coeffs_embed,
    modulo,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
):
    freq_threshold = compute_frequency_threshold(
        coeffs_embed,
        top_fraction=freq_threshold_top_fraction,
        min_components=freq_threshold_min_components,
        fixed_threshold=freq_threshold_fixed,
    )
    selected = set(torch.nonzero(coeffs_embed > freq_threshold).flatten().detach().cpu().numpy().tolist())
    freq_pairs = []
    for k in range(1, modulo // 2 + 1):
        cos_idx = 2 * k - 1
        sin_idx = 2 * k
        if cos_idx in selected and sin_idx in selected:
            freq_pairs.append((cos_idx, sin_idx))
    return freq_pairs


def _compute_plotting_coeffs(h_state, Vh_fc, fourier_basis, modulo):
    plotting_coeffs = torch.zeros(h_state.shape[0], 2, device=h_state.device, dtype=h_state.dtype)
    num_pairs = Vh_fc.shape[0] // 2
    for i in range(num_pairs):
        plotting_coeffs += fft2d(h_state @ Vh_fc[2 * i:2 * (i + 1)].t(), fourier_basis, modulo)
    return plotting_coeffs.reshape(modulo, modulo, 2)


def _print_identity_errors(plotting_coeffs, freq_pairs, fourier_basis_names, subtraction=False):
    header = 'Verifying trigonometric identities for subtraction:' if subtraction else 'Verifying trigonometric identities:'
    print(header)
    for cos_idx, sin_idx in freq_pairs:
        if subtraction:
            res_cos = 0.5 * (plotting_coeffs[cos_idx, cos_idx] - plotting_coeffs[sin_idx, sin_idx])
            res_sin = 0.5 * (plotting_coeffs[sin_idx, cos_idx] + plotting_coeffs[cos_idx, sin_idx])
            coeff_cos = 0.5 * (plotting_coeffs[cos_idx, cos_idx] + plotting_coeffs[sin_idx, sin_idx])
            coeff_sin = 0.5 * (plotting_coeffs[sin_idx, cos_idx] - plotting_coeffs[cos_idx, sin_idx])
        else:
            res_cos = 0.5 * (plotting_coeffs[cos_idx, cos_idx] + plotting_coeffs[sin_idx, sin_idx])
            res_sin = 0.5 * (plotting_coeffs[sin_idx, cos_idx] - plotting_coeffs[cos_idx, sin_idx])
            coeff_cos = 0.5 * (plotting_coeffs[cos_idx, cos_idx] - plotting_coeffs[sin_idx, sin_idx])
            coeff_sin = 0.5 * (plotting_coeffs[sin_idx, cos_idx] + plotting_coeffs[cos_idx, sin_idx])

        rel_error = ((res_cos.pow(2).sum() + res_sin.pow(2).sum()) / (coeff_cos.pow(2).sum() + coeff_sin.pow(2).sum()))
        freq_names = (fourier_basis_names[cos_idx], fourier_basis_names[sin_idx])
        print(f'\t\t Relative Error for components {freq_names}: {rel_error.sqrt():.2E}')


def _verify_identity(
    model,
    checkpoint,
    dataset,
    operation,
    subtraction,
    is_transformer,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    nterms = int(getattr(dataset, 'nterms', 2))
    if nterms != 2:
        prefix = 'transformer ' if is_transformer else ''
        print(
            f"Skipping {prefix}{operation} trig identity verification for nterms={nterms}. "
            'This check is currently defined for nterms=2.'
        )
        return

    param = model.embedding.weight
    model_device = param.device
    model_dtype = param.dtype
    modulo = dataset.modulo
    fourier_basis, fourier_basis_names = create_fourier_analysis_setup(
        modulo=modulo,
        device=model_device,
        dtype=model_dtype,
    )

    model.eval()
    with torch.no_grad():
        tokens = filter_data_by_operation(dataset, operation)[0]
        tokens = tokens.to(device=model_device)
        embedded_seq = model.get_embeddings(tokens)
        if is_transformer:
            hidden_states = model.get_hidden_states(embedded_seq=embedded_seq)
            h_state = hidden_states[-1][:, -1, :]
            svd_model_type = 'transformer'
        else:
            output, _ = model.rnn(embedded_seq)
            h_state = model.bn(output[:, 3, :])
            svd_model_type = 'rnn'

    factors = get_svd_factors(checkpoint, model_type=svd_model_type, cache=analysis_cache)
    _, _, Vh_fc = factors['fc.weight']
    Vh_fc = Vh_fc.to(device=h_state.device, dtype=h_state.dtype)

    coeffs_embed = fourier_basis @ model.embedding.weight[:dataset.modulo, :]
    coeffs_embed = coeffs_embed.norm(dim=1)
    freq_pairs = _get_freq_pairs(
        coeffs_embed,
        modulo,
        freq_threshold_top_fraction=freq_threshold_top_fraction,
        freq_threshold_min_components=freq_threshold_min_components,
        freq_threshold_fixed=freq_threshold_fixed,
    )
    if not freq_pairs:
        print('No complete cos/sin frequency pairs found above threshold.')
        return

    max_pairs = min(len(freq_pairs), Vh_fc.shape[0] // 2)
    plotting_coeffs = _compute_plotting_coeffs(h_state, Vh_fc[: 2 * max_pairs], fourier_basis, modulo)
    _print_identity_errors(plotting_coeffs, freq_pairs[:max_pairs], fourier_basis_names, subtraction=subtraction)


def verify_trigonometric_identity(
    model,
    checkpoint,
    dataset,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    """Verify that hidden states satisfy trigonometric identities. RNN ONLY!"""
    return _verify_identity(
        model,
        checkpoint,
        dataset,
        operation='addition',
        subtraction=False,
        is_transformer=False,
        freq_threshold_top_fraction=freq_threshold_top_fraction,
        freq_threshold_min_components=freq_threshold_min_components,
        freq_threshold_fixed=freq_threshold_fixed,
        analysis_cache=analysis_cache,
    )


def verify_trigonometric_identity_subtraction(
    model,
    checkpoint,
    dataset,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    """Verify that hidden states satisfy trigonometric identities for subtraction. RNN ONLY!"""
    return _verify_identity(
        model,
        checkpoint,
        dataset,
        operation='subtraction',
        subtraction=True,
        is_transformer=False,
        freq_threshold_top_fraction=freq_threshold_top_fraction,
        freq_threshold_min_components=freq_threshold_min_components,
        freq_threshold_fixed=freq_threshold_fixed,
        analysis_cache=analysis_cache,
    )


def verify_trigonometric_identity_transformer(
    model,
    checkpoint,
    dataset,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    """Verify that hidden states satisfy trigonometric identities. TRANSFORMER ONLY!"""
    return _verify_identity(
        model,
        checkpoint,
        dataset,
        operation='addition',
        subtraction=False,
        is_transformer=True,
        freq_threshold_top_fraction=freq_threshold_top_fraction,
        freq_threshold_min_components=freq_threshold_min_components,
        freq_threshold_fixed=freq_threshold_fixed,
        analysis_cache=analysis_cache,
    )


def verify_trigonometric_identity_transformer_subtraction(
    model,
    checkpoint,
    dataset,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    """Verify that hidden states satisfy trigonometric identities for subtraction. TRANSFORMER ONLY!"""
    return _verify_identity(
        model,
        checkpoint,
        dataset,
        operation='subtraction',
        subtraction=True,
        is_transformer=True,
        freq_threshold_top_fraction=freq_threshold_top_fraction,
        freq_threshold_min_components=freq_threshold_min_components,
        freq_threshold_fixed=freq_threshold_fixed,
        analysis_cache=analysis_cache,
    )
