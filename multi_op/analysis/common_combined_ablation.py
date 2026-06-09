"""Combined-ip-elbow SVD ablation analysis shared by RNN and Transformer."""

import copy
import os

import matplotlib.pyplot as plt

from utils import create_model, evaluate_model

from analysis.context import get_svd_factors
from analysis.common_svd import compute_total_ip_elbow


def _resolve_combined_ip_elbow(op_elbows, operations):
    if isinstance(op_elbows, int):
        return op_elbows
    return compute_total_ip_elbow(operations, op_elbows)


def _build_truncated_model_weights(base_weights, U_E, S_E, Vh_E, U_fc, S_fc, Vh_fc, k):
    """Return checkpoint weights with embedding/unembedding truncated to top-k SVD components."""
    model_weights = copy.deepcopy(base_weights)
    k_embed = min(k, len(S_E))
    k_unembed = min(k, len(S_fc))

    model_weights['embedding.weight'] = U_E[:, :k_embed] @ S_E[:k_embed].diag() @ Vh_E[:k_embed, :]
    model_weights['fc.weight'] = U_fc[:, :k_unembed] @ S_fc[:k_unembed].diag() @ Vh_fc[:k_unembed, :]
    return model_weights


def combined_ip_svd_ablation_analysis_transformer(
    model_cfg, checkpoint, dataset, op_elbows, operations, analysis_cache=None
):
    """Keep top combined-ip-elbow components of embedding/unembedding and evaluate each op."""
    combined_ip_elbow = _resolve_combined_ip_elbow(op_elbows, operations)

    factors = get_svd_factors(checkpoint, model_type='transformer', cache=analysis_cache)
    U_E, S_E, Vh_E = factors['embedding.weight']
    U_fc, S_fc, Vh_fc = factors['fc.weight']

    model_weights = _build_truncated_model_weights(
        checkpoint['model'], U_E, S_E, Vh_E, U_fc, S_fc, Vh_fc, combined_ip_elbow
    )

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)

    print(
        f'Combined-ip-elbow ablation (Transformer): keeping top {combined_ip_elbow} '
        f'components in embedding/unembedding'
    )
    accuracies = {}
    for op in operations:
        op_data, op_labels = dataset.get_test_data(operation=op)
        acc = evaluate_model(model_trunc, op_data, op_labels)
        accuracies[op] = acc
        print(f'  -> Accuracy on {op}: {acc:.2f}%')
    return accuracies


def combined_ip_svd_ablation_analysis(
    model_cfg, checkpoint, dataset, op_elbows, operations, analysis_cache=None
):
    """Keep top combined-ip-elbow components of embedding/unembedding and evaluate each op."""
    combined_ip_elbow = _resolve_combined_ip_elbow(op_elbows, operations)

    factors = get_svd_factors(checkpoint, model_type='rnn', cache=analysis_cache)
    U_E, S_E, Vh_E = factors['embedding.weight']
    U_fc, S_fc, Vh_fc = factors['fc.weight']

    model_weights = _build_truncated_model_weights(
        checkpoint['model'], U_E, S_E, Vh_E, U_fc, S_fc, Vh_fc, combined_ip_elbow
    )

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)

    print(
        f'Combined-ip-elbow ablation (RNN): keeping top {combined_ip_elbow} '
        f'components in embedding/unembedding'
    )
    accuracies = {}
    for op in operations:
        op_data, op_labels = dataset.get_test_data(operation=op)
        acc = evaluate_model(model_trunc, op_data, op_labels)
        accuracies[op] = acc
        print(f'  -> Accuracy on {op}: {acc:.2f}%')
    return accuracies


def _combined_ip_svd_ablation_accuracy_curve(
    model_type,
    model_cfg,
    checkpoint,
    dataset,
    save_dir,
    op_elbows,
    operations,
    analysis_cache=None,
):
    combined_ip_elbow = _resolve_combined_ip_elbow(op_elbows, operations)
    factors = get_svd_factors(checkpoint, model_type=model_type, cache=analysis_cache)
    U_E, S_E, Vh_E = factors['embedding.weight']
    U_fc, S_fc, Vh_fc = factors['fc.weight']

    k_values = list(range(0, combined_ip_elbow + 6))
    op_accuracies = {op: [] for op in operations}

    print(
        f'Combined-ip-elbow accuracy sweep ({model_type}): evaluating k=0..{combined_ip_elbow + 5} '
        '(embedding/unembedding top-k components kept)'
    )
    for k in k_values:
        model_weights = _build_truncated_model_weights(
            checkpoint['model'], U_E, S_E, Vh_E, U_fc, S_fc, Vh_fc, k
        )
        model_trunc = create_model(model_cfg)
        model_trunc.load_state_dict(model_weights)

        for op in operations:
            op_data, op_labels = dataset.get_test_data(operation=op)
            acc = evaluate_model(model_trunc, op_data, op_labels)
            op_accuracies[op].append(acc)

    plt.rcParams.update({'font.size': 11})
    plt.figure(figsize=(7, 4.5))
    for op in operations:
        plt.plot(k_values, op_accuracies[op], marker='o', linewidth=1.8, label=op.capitalize())

    plt.axvline(
        combined_ip_elbow,
        color='k',
        linestyle='--',
        linewidth=1.2,
        label=f'combined_ip_elbow={combined_ip_elbow}',
    )
    plt.xlim(0, combined_ip_elbow + 5)
    plt.ylim(0, 101)
    plt.xlabel('Number of Significant Components Kept (top-k)')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Operation Accuracy vs Top-k SVD Components (Embedding + Unembedding)')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    fig_dir = os.path.join(save_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'combined_ip_svd_accuracy_curve.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Saved combined-ip accuracy curve: {fig_path}')

    return {
        'k_values': k_values,
        'combined_ip_elbow': combined_ip_elbow,
        'accuracies': op_accuracies,
        'figure_path': fig_path,
    }


def combined_ip_svd_ablation_accuracy_curve_transformer(
    model_cfg, checkpoint, dataset, save_dir, op_elbows, operations, analysis_cache=None
):
    return _combined_ip_svd_ablation_accuracy_curve(
        model_type='transformer',
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        dataset=dataset,
        save_dir=save_dir,
        op_elbows=op_elbows,
        operations=operations,
        analysis_cache=analysis_cache,
    )


def combined_ip_svd_ablation_accuracy_curve(
    model_cfg, checkpoint, dataset, save_dir, op_elbows, operations, analysis_cache=None
):
    return _combined_ip_svd_ablation_accuracy_curve(
        model_type='rnn',
        model_cfg=model_cfg,
        checkpoint=checkpoint,
        dataset=dataset,
        save_dir=save_dir,
        op_elbows=op_elbows,
        operations=operations,
        analysis_cache=analysis_cache,
    )
