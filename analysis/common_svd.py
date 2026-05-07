import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from analysis.context import get_svd_factors
from analysis.operation_config import compute_svd_energy_elbow, get_operation_analysis_context

ADDITIVE_OPS = {'addition', 'subtraction'}
MULTIPLICATIVE_OPS = {'multiplication', 'division'}


def compute_total_ip_elbow(operations, op_elbows):
    """Combine per-operation elbows into a single model-level elbow."""
    additive_ops = [op for op in operations if op in ADDITIVE_OPS]
    multiplicative_ops = [op for op in operations if op in MULTIPLICATIVE_OPS]

    additive_elbow = max((op_elbows[op] for op in additive_ops), default=0)
    multiplicative_elbow = max((op_elbows[op] for op in multiplicative_ops), default=0)
    return additive_elbow + multiplicative_elbow


def svd_spectrum_analysis_plotting_transformer(
    model, checkpoint, dataset, save_dir, operation, ip_elbow, analysis_cache=None
):
    """
    Perform SVD spectrum analysis plotting for any modular arithmetic operation.
    TRANSFORMERS ONLY!

    Args:
        model: The trained model
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        save_dir: Directory to save plots
        operation: String indicating the operation
        ip_elbow: Number of significant SVD components
    """

    op_ctx = get_operation_analysis_context(dataset, operation)
    config = op_ctx['config']
    
    factors = get_svd_factors(checkpoint, model_type='transformer', cache=analysis_cache)
    _, S_E_raw, _ = factors['embedding.weight']
    _, S_fc_raw, _ = factors['fc.weight']
    S_E = S_E_raw / S_E_raw.sum()
    S_fc = S_fc_raw / S_fc_raw.sum()

    # Determine expected dimensions based on operation type
    is_multiplicative = op_ctx['is_multiplicative']
    expected_vocab_size = op_ctx['dataset_size']
    
    # Create indices based on actual SVD matrix dimensions
    ip_idxs_E = np.arange(1, len(S_E) + 1)  # For embedding weights
    ip_idxs_fc = np.arange(1, len(S_fc) + 1)  # For unembedding weights
    
    # Sanity check: verify dimensions match expected vocabulary size
    print(f"Expected vocab size: {expected_vocab_size}, Actual embedding dim: {len(S_E)}")
    print(f"Operation: {config['display_name']}, Is multiplicative: {is_multiplicative}")
    
    # Adjust elbow points if they exceed matrix dimensions
    ip_elbow_E = min(ip_elbow, len(S_E))
    ip_elbow_fc = min(ip_elbow, len(S_fc))
    
    layers = ['Embedding', 'Unembedding']
    svd_spectra = [S_E, S_fc]
    n_comps = [ip_elbow_E, ip_elbow_fc]
    
    print(f'Fraction of Variance in significant components for {config["display_name"]}:')
    for layer, S, n in zip(layers, svd_spectra, n_comps):
        print(f'{layer} Weights, {n} components: {S[:n].sum():.4f}')
    
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 4))
    
    # Plot singular value spectra with correct indices
    plt.loglog(ip_idxs_E, S_E.detach().cpu().numpy(), 'b-')
    plt.loglog(ip_idxs_fc, S_fc.detach().cpu().numpy(), 'r-')
    
    # Highlight specific points (only if elbow points are within bounds)
    if ip_elbow_E <= len(S_E):
        plt.plot(ip_elbow_E, S_E[ip_elbow_E-1].detach().cpu().numpy(), marker='o', markersize=10,
                color='b', label='Embedding')
    if ip_elbow_fc <= len(S_fc):
        plt.plot(ip_elbow_fc, S_fc[ip_elbow_fc-1].detach().cpu().numpy(), marker='s', markersize=10,
                color='r', label='Unembedding')
    # Set x-ticks based on actual elbow points
    xtick_labels = []
    xtick_positions = []
    if ip_elbow_E <= len(S_E):
        xtick_positions.append(ip_elbow_E)
        xtick_labels.append(str(ip_elbow_E))
    
    if xtick_positions:
        plt.xticks(xtick_positions, labels=xtick_labels)
    
    plt.xlabel('Singular Value Index')
    plt.ylabel('Normalized Singular Value')
    plt.title(f'Singular Value Spectrum for Modular {config["display_name"]}')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'figures', f'singular_value_spectrum_{operation}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def svd_spectrum_analysis_plotting(
    model, checkpoint, dataset, save_dir, operation, ip_elbow, analysis_cache=None
):
    """
    Perform SVD spectrum analysis plotting for any modular arithmetic operation.

    Args:
        model: The trained model
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        save_dir: Directory to save plots
        operation: String indicating the operation
        ip_elbow: Number of significant SVD components
    """
    op_ctx = get_operation_analysis_context(dataset, operation)
    config = op_ctx['config']
    
    factors = get_svd_factors(checkpoint, model_type='rnn', cache=analysis_cache)
    _, S_E_raw, _ = factors['embedding.weight']
    _, S_fc_raw, _ = factors['fc.weight']
    _, S_ih_raw, _ = factors['rnn.weight_ih_l0']
    _, S_hh_raw, _ = factors['rnn.weight_hh_l0']
    S_E = S_E_raw / S_E_raw.sum()
    S_fc = S_fc_raw / S_fc_raw.sum()
    S_ih = S_ih_raw / S_ih_raw.sum()
    S_hh = S_hh_raw / S_hh_raw.sum()
    
    # Determine expected dimensions based on operation type
    is_multiplicative = op_ctx['is_multiplicative']
    expected_vocab_size = op_ctx['dataset_size']
    
    # Create indices based on actual SVD matrix dimensions
    ip_idxs_E = np.arange(1, len(S_E) + 1)  # For embedding weights
    ip_idxs_fc = np.arange(1, len(S_fc) + 1)  # For unembedding weights
    ip_idxs_ih = np.arange(1, len(S_ih) + 1)  # For input weights
    hh_idxs = np.arange(1, len(S_hh) + 1)  # For hidden weights
    
    # Sanity check: verify dimensions match expected vocabulary size
    print(f"Expected vocab size: {expected_vocab_size}, Actual embedding dim: {len(S_E)}")
    print(f"Operation: {config['display_name']}, Is multiplicative: {is_multiplicative}")
    
    # Adjust elbow points if they exceed matrix dimensions
    ip_elbow_E = min(ip_elbow, len(S_E))
    ip_elbow_fc = min(ip_elbow, len(S_fc))
    ip_elbow_ih = min(ip_elbow, len(S_ih))
    hh_elbow = compute_svd_energy_elbow(S_hh)
    hh_elbow_adj = min(hh_elbow, len(S_hh))
    
    layers = ['Embedding', 'Unembedding', 'Input', 'Hidden']
    svd_spectra = [S_E, S_fc, S_ih, S_hh]
    n_comps = [ip_elbow_E, ip_elbow_fc, ip_elbow_ih, hh_elbow_adj]
    
    print(f'Fraction of Variance in significant components for {config["display_name"]}:')
    for layer, S, n in zip(layers, svd_spectra, n_comps):
        print(f'{layer} Weights, {n} components: {S[:n].sum():.4f}')
    
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 4))
    
    # Plot singular value spectra with correct indices
    plt.loglog(ip_idxs_E, S_E.detach().cpu().numpy(), 'b-')
    plt.loglog(ip_idxs_fc, S_fc.detach().cpu().numpy(), 'r-')
    plt.loglog(ip_idxs_ih, S_ih.detach().cpu().numpy(), 'g-')
    plt.loglog(hh_idxs, S_hh.detach().cpu().numpy(), 'k-')
    
    # Highlight specific points (only if elbow points are within bounds)
    if ip_elbow_E <= len(S_E):
        plt.plot(ip_elbow_E, S_E[ip_elbow_E-1].detach().cpu().numpy(), marker='o', markersize=10,
                color='b', label='Embedding')
    if ip_elbow_fc <= len(S_fc):
        plt.plot(ip_elbow_fc, S_fc[ip_elbow_fc-1].detach().cpu().numpy(), marker='s', markersize=10,
                color='r', label='Unembedding')
    if ip_elbow_ih <= len(S_ih):
        plt.plot(ip_elbow_ih, S_ih[ip_elbow_ih-1].detach().cpu().numpy(), marker='^', markersize=10,
                color='g', label='Input')
    if hh_elbow_adj <= len(S_hh):
        plt.plot(hh_elbow_adj, S_hh[hh_elbow_adj-1].detach().cpu().numpy(), marker='d', markersize=10,
                color='k', label='Hidden')
    
    # Set x-ticks based on actual elbow points
    xtick_labels = []
    xtick_positions = []
    if ip_elbow_E <= len(S_E):
        xtick_positions.append(ip_elbow_E)
        xtick_labels.append(str(ip_elbow_E))
    if hh_elbow_adj <= len(S_hh) and hh_elbow_adj != ip_elbow_E:
        xtick_positions.append(hh_elbow_adj)
        xtick_labels.append(str(hh_elbow_adj))
    
    if xtick_positions:
        plt.xticks(xtick_positions, labels=xtick_labels)
    
    plt.xlabel('Singular Value Index')
    plt.ylabel('Normalized Singular Value')
    plt.title(f'Singular Value Spectrum for Modular {config["display_name"]}')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'figures', f'singular_value_spectrum_{operation}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def svd_spectrum_analysis_plotting_transformer_model(
    model, checkpoint, dataset, save_dir, op_elbows, operations, analysis_cache=None
):
    """
    Perform SVD spectrum analysis plotting for a full multi-operation transformer model.
    Plots the full embedding/unembedding weight spectra with a single combined ip_elbow.

    Args:
        model: The trained model
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        save_dir: Directory to save plots
        op_elbows: Per-operation elbows or a precomputed combined integer elbow
        operations: List of operations (for plot title)
    """
    ops_label = ' + '.join(op.capitalize() for op in operations)
    ip_elbow = (
        compute_total_ip_elbow(operations, op_elbows)
        if isinstance(op_elbows, dict)
        else int(op_elbows)
    )

    factors = get_svd_factors(checkpoint, model_type='transformer', cache=analysis_cache)
    _, S_E_raw, _ = factors['embedding.weight']
    _, S_fc_raw, _ = factors['fc.weight']
    S_E = S_E_raw / S_E_raw.sum()
    S_fc = S_fc_raw / S_fc_raw.sum()

    ip_idxs_E = np.arange(1, len(S_E) + 1)
    ip_idxs_fc = np.arange(1, len(S_fc) + 1)

    ip_elbow_E = min(ip_elbow, len(S_E))
    ip_elbow_fc = min(ip_elbow, len(S_fc))

    layers = ['Embedding', 'Unembedding']
    svd_spectra = [S_E, S_fc]
    n_comps = [ip_elbow_E, ip_elbow_fc]

    print(f'Fraction of Variance in significant components for {ops_label}:')
    for layer, S, n in zip(layers, svd_spectra, n_comps):
        print(f'{layer} Weights, {n} components: {S[:n].sum():.4f}')

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 4))

    plt.loglog(ip_idxs_E, S_E.detach().cpu().numpy(), 'b-')
    plt.loglog(ip_idxs_fc, S_fc.detach().cpu().numpy(), 'r-')

    if ip_elbow_E <= len(S_E):
        plt.plot(ip_elbow_E, S_E[ip_elbow_E - 1].detach().cpu().numpy(), marker='o', markersize=10,
                 color='b', label='Embedding')
    if ip_elbow_fc <= len(S_fc):
        plt.plot(ip_elbow_fc, S_fc[ip_elbow_fc - 1].detach().cpu().numpy(), marker='s', markersize=10,
                 color='r', label='Unembedding')

    xtick_positions = []
    xtick_labels = []
    if ip_elbow_E <= len(S_E):
        xtick_positions.append(ip_elbow_E)
        xtick_labels.append(str(ip_elbow_E))

    if xtick_positions:
        plt.xticks(xtick_positions, labels=xtick_labels)

    plt.xlabel('Singular Value Index')
    plt.ylabel('Normalized Singular Value')
    plt.title(f'Singular Value Spectrum for Modular {ops_label}')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'figures', 'singular_value_spectrum.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def svd_spectrum_analysis_plotting_model(
    model, checkpoint, dataset, save_dir, op_elbows, operations, analysis_cache=None
):
    """
    Perform SVD spectrum analysis plotting for a full multi-operation RNN model.
    Plots the full weight spectra with a single combined ip_elbow.

    Args:
        model: The trained model
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        save_dir: Directory to save plots
        op_elbows: Per-operation elbows or a precomputed combined integer elbow
        operations: List of operations (for plot title)
    """
    ops_label = ' + '.join(op.capitalize() for op in operations)
    ip_elbow = (
        compute_total_ip_elbow(operations, op_elbows)
        if isinstance(op_elbows, dict)
        else int(op_elbows)
    )
    factors = get_svd_factors(checkpoint, model_type='rnn', cache=analysis_cache)
    _, S_E_raw, _ = factors['embedding.weight']
    _, S_fc_raw, _ = factors['fc.weight']
    _, S_ih_raw, _ = factors['rnn.weight_ih_l0']
    _, S_hh_raw, _ = factors['rnn.weight_hh_l0']
    S_E = S_E_raw / S_E_raw.sum()
    S_fc = S_fc_raw / S_fc_raw.sum()
    S_ih = S_ih_raw / S_ih_raw.sum()
    S_hh = S_hh_raw / S_hh_raw.sum()

    ip_idxs_E = np.arange(1, len(S_E) + 1)
    ip_idxs_fc = np.arange(1, len(S_fc) + 1)
    ip_idxs_ih = np.arange(1, len(S_ih) + 1)
    hh_idxs = np.arange(1, len(S_hh) + 1)

    ip_elbow_E = min(ip_elbow, len(S_E))
    ip_elbow_fc = min(ip_elbow, len(S_fc))
    ip_elbow_ih = min(ip_elbow, len(S_ih))
    hh_elbow = compute_svd_energy_elbow(S_hh)
    hh_elbow_adj = min(hh_elbow, len(S_hh))

    layers = ['Embedding', 'Unembedding', 'Input', 'Hidden']
    svd_spectra = [S_E, S_fc, S_ih, S_hh]
    n_comps = [ip_elbow_E, ip_elbow_fc, ip_elbow_ih, hh_elbow_adj]

    print(f'Fraction of Variance in significant components for {ops_label}:')
    for layer, S, n in zip(layers, svd_spectra, n_comps):
        print(f'{layer} Weights, {n} components: {S[:n].sum():.4f}')

    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 4))

    plt.loglog(ip_idxs_E, S_E.detach().cpu().numpy(), 'b-')
    plt.loglog(ip_idxs_fc, S_fc.detach().cpu().numpy(), 'r-')
    plt.loglog(ip_idxs_ih, S_ih.detach().cpu().numpy(), 'g-')
    plt.loglog(hh_idxs, S_hh.detach().cpu().numpy(), 'k-')

    if ip_elbow_E <= len(S_E):
        plt.plot(ip_elbow_E, S_E[ip_elbow_E - 1].detach().cpu().numpy(), marker='o', markersize=10,
                 color='b', label='Embedding')
    if ip_elbow_fc <= len(S_fc):
        plt.plot(ip_elbow_fc, S_fc[ip_elbow_fc - 1].detach().cpu().numpy(), marker='s', markersize=10,
                 color='r', label='Unembedding')
    if ip_elbow_ih <= len(S_ih):
        plt.plot(ip_elbow_ih, S_ih[ip_elbow_ih - 1].detach().cpu().numpy(), marker='^', markersize=10,
                 color='g', label='Input')
    if hh_elbow_adj <= len(S_hh):
        plt.plot(hh_elbow_adj, S_hh[hh_elbow_adj - 1].detach().cpu().numpy(), marker='d', markersize=10,
                 color='k', label='Hidden')

    xtick_positions = []
    xtick_labels = []
    if ip_elbow_E <= len(S_E):
        xtick_positions.append(ip_elbow_E)
        xtick_labels.append(str(ip_elbow_E))
    if hh_elbow_adj <= len(S_hh) and hh_elbow_adj != ip_elbow_E:
        xtick_positions.append(hh_elbow_adj)
        xtick_labels.append(str(hh_elbow_adj))

    if xtick_positions:
        plt.xticks(xtick_positions, labels=xtick_labels)

    plt.xlabel('Singular Value Index')
    plt.ylabel('Normalized Singular Value')
    plt.title(f'Singular Value Spectrum for Modular {ops_label}')
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'figures', 'singular_value_spectrum.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
