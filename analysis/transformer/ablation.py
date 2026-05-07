"""Transformer ablation analysis entrypoints."""

from analysis import common_ablation


def svd_ablation_analysis(
    model_cfg,
    checkpoint,
    dataset,
    operation,
    ip_elbow,
    fourier_reg_mode=None,
    analysis_cache=None,
):
    return common_ablation.svd_ablation_analysis_transformer(
        model_cfg,
        checkpoint,
        dataset,
        operation,
        ip_elbow,
        fourier_reg_mode=fourier_reg_mode,
        analysis_cache=analysis_cache,
    )


def fourier_ablation_analysis(
    model_cfg,
    checkpoint,
    dataset,
    operation,
    ip_elbow,
    fourier_reg_mode=None,
    analysis_cache=None,
):
    return common_ablation.fourier_ablation_analysis_transformer(
        model_cfg,
        checkpoint,
        dataset,
        operation,
        ip_elbow,
        fourier_reg_mode=fourier_reg_mode,
        analysis_cache=analysis_cache,
    )


# Backward-compatible aliases
svd_ablation_analysis_transformer = svd_ablation_analysis
fourier_ablation_analysis_transformer = fourier_ablation_analysis
