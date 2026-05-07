"""RNN Fourier analysis entrypoints."""

from analysis import common_fourier


def compute_ip_elbow(checkpoint, dataset, operation, norm_threshold=0.1, fourier_reg_mode=None, analysis_cache=None):
    return common_fourier.compute_ip_elbow(
        checkpoint,
        dataset,
        operation,
        norm_threshold=norm_threshold,
        fourier_reg_mode=fourier_reg_mode,
        analysis_cache=analysis_cache,
    )


def fourier_spectrum_analysis_plotting(
    model,
    checkpoint,
    dataset,
    save_dir,
    operation,
    fourier_reg_mode=None,
    norm_threshold=0.1,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    if not hasattr(model, 'rnn'):
        raise ValueError('RNN analysis requested but model has no rnn module.')
    return common_fourier.fourier_spectrum_analysis_plotting(
        model,
        checkpoint,
        dataset,
        save_dir,
        operation,
        fourier_reg_mode=fourier_reg_mode,
        norm_threshold=norm_threshold,
        freq_threshold_top_fraction=freq_threshold_top_fraction,
        freq_threshold_min_components=freq_threshold_min_components,
        freq_threshold_fixed=freq_threshold_fixed,
        analysis_cache=analysis_cache,
    )
