"""Transformer SVD analysis entrypoints."""

from analysis import common_svd


def svd_spectrum_analysis_plotting(model, checkpoint, dataset, save_dir, operation, ip_elbow, analysis_cache=None):
    if hasattr(model, 'rnn'):
        raise ValueError('Transformer analysis requested but model appears to be an RNN.')
    return common_svd.svd_spectrum_analysis_plotting_transformer(
        model,
        checkpoint,
        dataset,
        save_dir,
        operation,
        ip_elbow,
        analysis_cache=analysis_cache,
    )


def svd_spectrum_analysis_plotting_model(
    model, checkpoint, dataset, save_dir, op_elbows, operations, analysis_cache=None
):
    if hasattr(model, 'rnn'):
        raise ValueError('Transformer analysis requested but model appears to be an RNN.')
    return common_svd.svd_spectrum_analysis_plotting_transformer_model(
        model,
        checkpoint,
        dataset,
        save_dir,
        op_elbows,
        operations,
        analysis_cache=analysis_cache,
    )


# Backward-compatible aliases
svd_spectrum_analysis_plotting_transformer = svd_spectrum_analysis_plotting
svd_spectrum_analysis_plotting_transformer_model = svd_spectrum_analysis_plotting_model
