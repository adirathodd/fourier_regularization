"""Transformer trigonometric identity analysis entrypoints."""

from analysis import common_trig


def verify_trigonometric_identity(
    model,
    checkpoint,
    dataset,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    if hasattr(model, 'rnn'):
        raise ValueError('Transformer analysis requested but model appears to be an RNN.')
    return common_trig.verify_trigonometric_identity_transformer(
        model,
        checkpoint,
        dataset,
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
    if hasattr(model, 'rnn'):
        raise ValueError('Transformer analysis requested but model appears to be an RNN.')
    return common_trig.verify_trigonometric_identity_transformer_subtraction(
        model,
        checkpoint,
        dataset,
        freq_threshold_top_fraction=freq_threshold_top_fraction,
        freq_threshold_min_components=freq_threshold_min_components,
        freq_threshold_fixed=freq_threshold_fixed,
        analysis_cache=analysis_cache,
    )


# Backward-compatible aliases
verify_trigonometric_identity_transformer = verify_trigonometric_identity
verify_trigonometric_identity_transformer_subtraction = verify_trigonometric_identity_subtraction
