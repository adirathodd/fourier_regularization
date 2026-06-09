import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import (
    AdditionModDataset,
    DivisionModDataset,
    MultiOperationDataset,
    MultiplicationModDataset,
    SubtractionModDataset,
)
from utils import *

from analysis.analysis_utils import apply_mixed_hidden_feature_split, fft_nd
from analysis.context import apply_mixed_feature_split, get_fourier_context
from analysis.operation_config import compute_frequency_threshold

def compute_ip_elbow(
    checkpoint,
    dataset,
    operation,
    norm_threshold=0.1,
    fourier_reg_mode=None,
    analysis_cache=None,
):
    """Compute ip_elbow (number of significant SVD components) from Fourier analysis.

    Counts frequency pairs where both the normalized embedding and unembedding
    Fourier coefficients exceed norm_threshold. Returns 2 * num_significant_pairs.

    Args:
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
        norm_threshold: Threshold for normalized Fourier coefficients
        fourier_reg_mode: Optional Fourier regularization mode (enables split analysis for mode 1 with mixed ops)

    Returns:
        int: The computed ip_elbow value
    """
    fctx = get_fourier_context(
        checkpoint,
        dataset,
        operation,
        fourier_reg_mode=fourier_reg_mode,
        cache=analysis_cache,
    )
    op_ctx = fctx['op_ctx']
    is_multiplicative = op_ctx['is_multiplicative']
    perm = fctx['perm']
    fourier_basis = fctx['fourier_basis']
    W_E = fctx['W_E_unpermuted']
    W_fc = fctx['W_fc_unpermuted']

    # Multiplicative operations must be permuted into primitive-root order
    # before computing Fourier coefficients.
    if is_multiplicative:
        if perm is None:
            raise ValueError('Missing multiplicative permutation for Fourier elbow computation.')
        W_E = W_E[perm - 1]
        W_fc = W_fc[perm - 1]

    W_E, W_fc = apply_mixed_feature_split(
        W_E,
        W_fc,
        operation,
        fourier_reg_mode=fourier_reg_mode,
        has_mixed_ops=op_ctx['has_mixed_ops'],
    )
    fourier_basis = fourier_basis.to(device=W_E.device, dtype=W_E.dtype)

    coeffs_embed = (fourier_basis @ W_E).norm(dim=1)
    coeffs_fc = (fourier_basis @ W_fc).norm(dim=1)

    # Normalize and skip DC component
    normalized_embed = (coeffs_embed / coeffs_embed.norm()).detach().cpu().numpy()[1:]
    normalized_fc = (coeffs_fc / coeffs_fc.norm()).detach().cpu().numpy()[1:]

    # Count frequency pairs where both cos terms exceed threshold
    num_significant = 0
    for i in range(0, len(normalized_embed) - 1, 2):
        if normalized_embed[i] > norm_threshold and normalized_fc[i] > norm_threshold:
            num_significant += 1

    ip_elbow = num_significant * 2
    if ip_elbow == 0 and len(normalized_embed) >= 2 and len(normalized_fc) >= 2:
        # Fall back to the strongest cosine pair from this run's spectra.
        cosine_scores = normalized_embed[::2] * normalized_fc[::2]
        if len(cosine_scores) > 0:
            ip_elbow = 2 * (int(np.argmax(cosine_scores)) + 1)

    return ip_elbow

def fourier_spectrum_analysis_plotting(
    model,
    checkpoint,
    dataset: AdditionModDataset | SubtractionModDataset | MultiplicationModDataset | DivisionModDataset | MultiOperationDataset,
    save_dir,
    operation,
    fourier_reg_mode=None,
    norm_threshold=0.1,
    freq_threshold_top_fraction=0.1,
    freq_threshold_min_components=4,
    freq_threshold_fixed=None,
    analysis_cache=None,
):
    fctx = get_fourier_context(
        checkpoint,
        dataset,
        operation,
        fourier_reg_mode=fourier_reg_mode,
        cache=analysis_cache,
    )
    op_ctx = fctx['op_ctx']
    config = op_ctx['config']
    includes_zero = op_ctx['includes_zero']
    is_multiplicative = op_ctx['is_multiplicative']
    modulo = op_ctx['modulo']  # used for reshaping in hidden-state coefficient maps
    nterms = op_ctx['nterms']
    dataset_size = op_ctx['dataset_size']
    sample_interval = op_ctx['sample_interval']
    expected_grid = op_ctx['expected_grid']
    fourier_basis = fctx['fourier_basis']
    fourier_basis_names = fctx['fourier_basis_names']
    generator = fctx['generator']
    perm = fctx['perm']
    W_E = fctx['W_E']
    W_fc = fctx['W_fc']
    fourier_basis = fourier_basis.to(device=W_E.device, dtype=W_E.dtype)

    is_variable_length = bool(getattr(dataset, 'is_variable_length', False))
    can_run_hidden_fourier = (nterms == 2) and (not is_variable_length)
    if not can_run_hidden_fourier:
        reason = (
            "variable-length datasets are not represented as a single dense sequence grid."
            if is_variable_length
            else f"nterms={nterms} (only implemented for nterms=2 pairwise grids)."
        )
        print(f"Skipping hidden-state Fourier coefficient heatmaps: {reason}")

    if can_run_hidden_fourier:
        op_dataset, _ = dataset.get_full_data(operation)
        device = next(model.parameters()).device
        op_dataset = op_dataset.to(device)

        model.eval()
        with torch.no_grad():
            embedded_seq = model.get_embeddings(op_dataset)
            if hasattr(model, 'rnn'):
                output, _ = model.rnn(embedded_seq)
            else: # transformer
                output = model.get_hidden_states(embedded_seq=embedded_seq)[-1]

        seq_length = output.shape[1]
        # Extract hidden states
        hidden_states = [output[::sample_interval, 0, :]]

        for pos in range(1, seq_length):
            hidden_states.append(output[:, pos, :])

        # Reorder hidden states for multiplicative operations
        if is_multiplicative and nterms == 2:
            reordered_hidden_states = [hidden_states[0][perm-1]]

            for pos in range(1, len(hidden_states)):
                curr_h = hidden_states[pos]
                h_reshaped = curr_h.view(sample_interval, sample_interval, -1)
                h_a_reordered = h_reshaped[perm-1]
                h_both_reordered = h_a_reordered.transpose(0, 1)[perm-1].transpose(0, 1).contiguous()
                reordered_h = h_both_reordered.view(-1, h_both_reordered.shape[-1])
            
                reordered_hidden_states.append(reordered_h)
            
            hidden_states = reordered_hidden_states
        elif is_multiplicative and nterms != 2:
            print(
                f"Skipping multiplicative hidden-state permutation for nterms={nterms}. "
                "Permutation is currently implemented for pairwise grids only."
            )

    fourier_embed = fourier_basis @ W_E
    fourier_fc = fourier_basis @ W_fc  
    
    coeffs_embed = fourier_embed.norm(dim=1)
    coeffs_fc = fourier_fc.norm(dim=1)
    freq_threshold = compute_frequency_threshold(
        coeffs_embed,
        top_fraction=freq_threshold_top_fraction,
        min_components=freq_threshold_min_components,
        fixed_threshold=freq_threshold_fixed,
    )
    
    '''
        Made hardcoded frequencies dynamic
        Isn't made in mind for division 
        Likely need to adjust as the research progresses
    '''

    # Apply operation-specific threshold to identify key frequencies
    key_freqs = torch.where(coeffs_embed > freq_threshold, 1, 0).nonzero()
    if key_freqs.numel() > 0:
        key_freqs = key_freqs.squeeze().detach().cpu().numpy()
        if key_freqs.ndim == 0:  # Handle single frequency case
            key_freqs = np.array([key_freqs.item()])
        key_freqs = key_freqs - 1  # Adjust for 0-indexing
        # Remove negative indices
        key_freqs = key_freqs[key_freqs >= 0]
    else:
        # If no key frequencies found, use a few low frequencies
        key_freqs = np.array([0, 1, 2, 3])

    ###########################################################################
    # Main Fourier coefficients plot
    ###########################################################################

    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(6, 4))
    
    # Plot normalized coefficients - skip DC component (index 0)
    coeffs_len = len(coeffs_embed) - 1
    plt.stem(range(coeffs_len), 
            (coeffs_embed/coeffs_embed.norm()).detach().cpu().numpy()[1:], 
            markerfmt='b^', linefmt='--', label='Embedding')
    plt.stem(range(coeffs_len), 
            (coeffs_fc/coeffs_fc.norm()).detach().cpu().numpy()[1:], 
            markerfmt='rs', linefmt='--', label='Unembedding')
            
    plt.legend(loc='upper right')
    
    # Find significant frequencies where BOTH cos AND sin are above threshold
    # for BOTH embedding AND unembedding
    # Use a separate normalized threshold (raw freq_threshold is calibrated for unnormalized values)
    normalized_embed = (coeffs_embed/coeffs_embed.norm()).detach().cpu().numpy()[1:]  # Skip DC
    normalized_fc = (coeffs_fc/coeffs_fc.norm()).detach().cpu().numpy()[1:]  # Skip DC
    valid_display_freqs = []
    for i in range(0, len(normalized_embed) - 1, 2):
        if (normalized_embed[i] > norm_threshold and
            normalized_fc[i] > norm_threshold):
            valid_display_freqs.append(i)
    
    valid_display_freqs = np.array(valid_display_freqs)
    
    if len(valid_display_freqs) == 0:
        valid_display_freqs = np.array([0, 2, 4, 6])  # Default to first few cos terms
    
    valid_display_freqs = valid_display_freqs[valid_display_freqs < len(fourier_basis_names) - 1]

    if len(valid_display_freqs) > 0:
        plt.xticks(valid_display_freqs, 
                   [fourier_basis_names[i+1] for i in valid_display_freqs], 
                   rotation=45, ha='right')
    
    plt.xlabel('Frequencies')
    plt.ylabel('Normalized Fourier Coefficient')
    title_suffix = f" (Generator: {generator})" if is_multiplicative else ""
    plt.title(f'Fourier Coefficients for Modular {config["display_name"]}{title_suffix} - Unembedding & Embedding')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'figures', f'fourier_coefficients_{operation}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    ###########################################################################
    # Plot hidden states - skip DC component
    ###########################################################################

    def _position_label(pos_idx, nterms_local, op_symbol_local):
        seq_len_local = 2 * nterms_local
        if pos_idx < 0 or pos_idx >= seq_len_local:
            return f"pos {pos_idx}"
        if pos_idx == seq_len_local - 1:
            return "="
        if pos_idx % 2 == 0:
            operand_idx = pos_idx // 2
            if operand_idx < 26:
                return chr(ord('a') + operand_idx)
            return f"x{operand_idx + 1}"
        return op_symbol_local

    if can_run_hidden_fourier:
        # Compute n-D Fourier transforms for 2nd hidden states and onwards.
        # For nterms > 2, project to 2D by aggregating higher frequency axes.
        coeffs_hidden = []
        basis_len = len(fourier_basis)
        for i in range(1, seq_length):
            if hidden_states[i].shape[0] != expected_grid:
                raise ValueError(
                    f"Expected hidden state first dim {expected_grid} (= {dataset_size}^{nterms}) "
                    f"for operation {operation}, got {hidden_states[i].shape[0]}"
                )

            hidden_state_i = apply_mixed_hidden_feature_split(
                hidden_states[i],
                operation,
                fourier_reg_mode=fourier_reg_mode,
                has_mixed_ops=op_ctx['has_mixed_ops'],
            )

            fourier_h = fft_nd(hidden_state_i, fourier_basis, dataset_size, nterms=nterms)
            coeffs_h = fourier_h.norm(dim=-1).detach().cpu().numpy()

            if nterms == 2:
                coeffs_h = coeffs_h.reshape(basis_len, basis_len)
            else:
                coeffs_h = coeffs_h.reshape(*([basis_len] * nterms))
                coeffs_h = coeffs_h.sum(axis=tuple(range(2, nterms)))

            coeffs_hidden.append(coeffs_h)

        num_hidden_plots = len(coeffs_hidden)
        if num_hidden_plots == 1:
            fig, axs = plt.subplots(1, 1, figsize=(6, 5))
            axs = np.array([axs])
        elif num_hidden_plots == 2:
            fig, axs = plt.subplots(1, num_hidden_plots, figsize=(6*num_hidden_plots, 5))
            axs = np.array(axs).reshape(-1)
        else:
            # For more than 3 plots, arrange in a 2-row grid
            cols = (num_hidden_plots + 1) // 2
            fig, axs = plt.subplots(2, cols, figsize=(6*cols, 10))
            axs = axs.flatten()
        
        for i in range(len(coeffs_hidden)):
            axs[i].matshow(coeffs_hidden[i][1:, 1:], cmap='Blues')
            axs[i].set_ylabel('a Component')
            axs[i].xaxis.set_label_position('bottom')
            axs[i].set_xlabel('b Component')
            axs[i].xaxis.tick_bottom()
            
            # Use the same significant frequencies for consistency
            matrix_size = coeffs_hidden[i].shape[1] - 1  # Excluding DC component
            tick_positions = valid_display_freqs[valid_display_freqs < matrix_size]
                    
            if len(tick_positions) > 0:
                axs[i].set_xticks(tick_positions, 
                                [fourier_basis_names[i+1] for i in tick_positions], 
                                rotation=45, fontsize=8, ha='right')
                axs[i].set_yticks(tick_positions, 
                                [fourier_basis_names[i+1] for i in tick_positions], 
                                rotation=0, fontsize=8)
            
            title_suffix_hidden = f" (Generator: {generator})" if is_multiplicative else ""
            token_pos = i + 1
            token_label = _position_label(token_pos, nterms, config["symbol"])
            axs[i].set_title(
                f'Fourier Coefficients at {token_label} [pos {token_pos}] '
                f'({config["display_name"]}{title_suffix_hidden})'
            )

        for j in range(len(coeffs_hidden), len(axs)):
            axs[j].axis('off')
            
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'figures', f'fourier_coefficients_hidden_{operation}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    ###########################################################################
    # SVD analysis of Unembedding Matrix
    ###########################################################################
    
    U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc)
    colors = ['b', 'r', 'g', 'm', 'k', 'y', 'c', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'darkblue', 'teal']
    markers = ['o', '^', 's', '*', 'x', '.', 'v', 'D', 'p', 'h', '+', '1', '2', '3', '4', 'd']
    plt.figure(figsize=(10,4))
    
    max_components = len(valid_display_freqs)
    
    for i in range(max_components):
        if 2*(i+1) <= U_fc.shape[1]:  # Ensure we have enough components
            W_fc_trunc = U_fc[:,2*i:2*(i+1)] @ torch.diag(S_fc[2*i:2*(i+1)]) @ Vh_fc[2*i:2*(i+1),:]
            fourier_transformed_fc_trunc = fourier_basis @ W_fc_trunc
            coeffs_trunc_fc = fourier_transformed_fc_trunc.norm(dim=1)

            markerline, stemlines, baseline = plt.stem(range(coeffs_len), 
            (coeffs_trunc_fc/coeffs_trunc_fc.norm()).detach().cpu().numpy()[1:], 
            linefmt=':', basefmt='k-', 
            label=f'Components {2*i+1}, {2*i+2}')
            markerline.set_marker(markers[i % len(markers)])
            markerline.set_color(colors[i % len(colors)])

    # FIXED: Use the same tick labeling logic for SVD plot
    if len(valid_display_freqs) > 0:
        plt.xticks(valid_display_freqs, 
                   [fourier_basis_names[i+1] for i in valid_display_freqs], 
                   rotation=45, ha='right')
    
    plt.xlabel('Frequencies')
    plt.ylabel('Normalized Fourier Coefficient')
    plt.title(f'Fourier Coefficients of Components of $W_{{fc}}$ ({config["display_name"]}{title_suffix})')
    plt.legend(loc='best', bbox_to_anchor=(0.6, 0.3))
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'figures', f'fourier_components_fc_{operation}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
