"""RNN SVD analysis entrypoints."""

from analysis import common_svd


def svd_spectrum_analysis_plotting(model, checkpoint, dataset, save_dir, operation, ip_elbow, analysis_cache=None):
    if not hasattr(model, 'rnn'):
        raise ValueError('RNN analysis requested but model has no rnn module.')
    return common_svd.svd_spectrum_analysis_plotting(
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
    if not hasattr(model, 'rnn'):
        raise ValueError('RNN analysis requested but model has no rnn module.')
    return common_svd.svd_spectrum_analysis_plotting_model(
        model,
        checkpoint,
        dataset,
        save_dir,
        op_elbows,
        operations,
        analysis_cache=analysis_cache,
    )
