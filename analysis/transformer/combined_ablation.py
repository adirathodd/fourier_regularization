"""Transformer combined-ip-elbow ablation analysis entrypoint."""

from analysis import common_combined_ablation


def combined_ip_svd_ablation_analysis(
    model_cfg, checkpoint, dataset, op_elbows, operations, analysis_cache=None
):
    return common_combined_ablation.combined_ip_svd_ablation_analysis_transformer(
        model_cfg,
        checkpoint,
        dataset,
        op_elbows,
        operations,
        analysis_cache=analysis_cache,
    )


def combined_ip_svd_ablation_accuracy_curve(
    model_cfg, checkpoint, dataset, save_dir, op_elbows, operations, analysis_cache=None
):
    return common_combined_ablation.combined_ip_svd_ablation_accuracy_curve_transformer(
        model_cfg,
        checkpoint,
        dataset,
        save_dir,
        op_elbows,
        operations,
        analysis_cache=analysis_cache,
    )
