import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

from utils import *
from analysis_utils import *


# Global configuration for different operations
OPERATION_CONFIGS = {
    'addition': {
        'freq_threshold': 2, # the hardcoded freq from before
        'ip_elbow': 14,
        'hh_elbow': 42,
        'display_name': 'Addition',
        'symbol': '+',
        'trig_function': 'addition',
        'includes_zero': True
    },
    'subtraction': {
        'freq_threshold': 0.5,  # may need tuning based on empirical results
        'ip_elbow': 14,
        'hh_elbow': 42,
        'display_name': 'Subtraction', 
        'symbol': '-',
        'trig_function': 'subtraction',
        'includes_zero': True
    },
    'multiplication': {
        'freq_threshold': 1.5,  # arbitary numbers, may need different threshold for multiplication
        'ip_elbow': 12,
        'hh_elbow': 42,
        'display_name': 'Multiplication',
        'symbol': '×',
        'trig_function': 'multiplication',
        'includes_zero' : False
    },
    'division': {
        'freq_threshold': 1.5,  # arbitary numbers, may need higher threshold for division
        'ip_elbow': 14,
        'hh_elbow': 42,
        'display_name': 'Division',
        'symbol': '÷',
        'trig_function': 'division',
        'includes_zero' : False
    }
}

def get_operation_config(operation):
    """Get configuration for a specific operation."""
    if operation not in OPERATION_CONFIGS:
        raise ValueError(f"Unsupported operation: {operation}. Supported operations: {list(OPERATION_CONFIGS.keys())}")
    return OPERATION_CONFIGS[operation]

def fourier_spectrum_analysis_plotting(model, weights, dataset, save_dir, operation, model_name):

    W_E = weights['Embedding']['weight']
    W_fc = weights['Unembedding']['weight']
    output = weights['Hidden States']['output']
    hidden_states = weights['Hidden States']['hs']

    # To make analysis adaptable to multiple operations
    config = get_operation_config(operation)
    freq_threshold = config['freq_threshold']
    includes_zero = config['includes_zero']

    if model_name.lower() == 'rnn':
        name = 'RNN'
    elif model_name.lower() == 'transformer':
        name = 'Transformer'
    elif model_name.lower() == 'ssm':
        name = 'SSM'

    modulo = dataset.modulo
    dataset_size = modulo if includes_zero else modulo - 1

    # Create Fourier basis with appropriate size
    fourier_basis, fourier_basis_names = create_fourier_analysis_setup(modulo, includes_zero)

    generator = 3

    if includes_zero:
        # For addition/subtraction: exclude padding token
        W_E = W_E[:-1,:]
        W_fc = W_fc[:-1,:]
        sample_interval = modulo
        perm = None  # No reordering needed
    else:
        # For multiplication/division: use first dataset_size weights
        W_E = W_E[:dataset_size, :]
        W_fc = W_fc[:dataset_size, :]
        sample_interval = modulo - 1
        
        # Create permutation using powers of 3
        perm = torch.tensor([generator**i % modulo for i in range(1, modulo)])
        
        # Apply permutation directly (perm-1 for 0-indexing)
        W_E = W_E[perm-1]
        W_fc = W_fc[perm-1]

    '''
    model.eval()
    with torch.no_grad():
        embedded_seq = model.embedding(dataset.dataset)
        output, hidden_states = model.rnn(embedded_seq)
    '''

    # Extract hidden states
    output_h1 = output[::sample_interval, 0, :]
    output_h2 = output[:, 1, :]
    output_h3 = output[:, 2, :]
    
    if not includes_zero:
        # Reorder hidden states for multiplicative operations
        output_h1 = output_h1[perm-1]
        
        # For 2D operations, reorder both a and b dimensions
        # Reshape to (sample_interval, sample_interval, hidden_dim)
        h2_reshaped = output_h2.view(sample_interval, sample_interval, -1)
        h3_reshaped = output_h3.view(sample_interval, sample_interval, -1)
        
        # Reorder the "a" dimension (rows) - first dimension
        h2_a_reordered = h2_reshaped[perm-1]
        h3_a_reordered = h3_reshaped[perm-1]
        
        # Reorder the "b" dimension (columns) - need to transpose, reorder, transpose back
        h2_both_reordered = h2_a_reordered.transpose(0,1)[perm-1].transpose(0,1).contiguous()
        h3_both_reordered = h3_a_reordered.transpose(0,1)[perm-1].transpose(0,1).contiguous()
        
        # Flatten back
        output_h2 = h2_both_reordered.view(-1, h2_both_reordered.shape[-1])
        output_h3 = h3_both_reordered.view(-1, h3_both_reordered.shape[-1])
    else:
        perm = None  # Set perm for consistency in Fourier transform section

    # Compute Fourier transforms
    if not includes_zero:
        # Apply permutation to Fourier transforms as well
        fourier_embed = fourier_basis @ W_E
        fourier_fc = fourier_basis @ W_fc  
        fourier_h1 = fourier_basis @ output_h1
    else:
        fourier_embed = fourier_basis @ W_E
        fourier_fc = fourier_basis @ W_fc
        fourier_h1 = fourier_basis @ output_h1
    
    # Compute norms
    coeffs_embed = fourier_embed.norm(dim=1)
    coeffs_fc = fourier_fc.norm(dim=1)
    coeffs_h1 = fourier_h1.norm(dim=1)
    
    '''
        Made hardcoded frequencies dynamic
        Isn't made in mind for division 
        Likely need to adjust as the research progresses
    '''
    # Apply operation-specific threshold to identify key frequencies
    key_freqs = torch.where(coeffs_embed > freq_threshold, 1, 0).nonzero()
    
    if key_freqs.numel() > 0:
        key_freqs = key_freqs.squeeze().detach().numpy()
        if key_freqs.ndim == 0:  # Handle single frequency case
            key_freqs = np.array([key_freqs.item()])
        key_freqs = key_freqs - 1  # Adjust for 0-indexing
        # Remove negative indices
        key_freqs = key_freqs[key_freqs >= 0]
    else:
        # If no key frequencies found, use a few low frequencies
        key_freqs = np.array([0, 1, 2, 3])
    
    # Compute 2D Fourier transforms for 2nd and 3rd hidden states
    fourier_h2 = fft2d(output_h2, fourier_basis, sample_interval)
    fourier_h3 = fft2d(output_h3, fourier_basis, sample_interval)
    
    # Fix reshaping - should match fourier_basis dimensions
    basis_len = len(fourier_basis)
    coeffs_h2 = fourier_h2.norm(dim=-1).reshape(basis_len, basis_len).detach().numpy()
    coeffs_h3 = fourier_h3.norm(dim=-1).reshape(basis_len, basis_len).detach().numpy()

    # Main Fourier coefficients plot
    plt.rcParams.update({'font.size': 10})
    plt.figure(figsize=(6, 4))
    
    # Plot normalized coefficients - skip DC component (index 0)
    coeffs_len = len(coeffs_embed) - 1
    if model_name == 'rnn':
        plt.stem(range(coeffs_len), 
                (coeffs_embed/coeffs_embed.norm()).detach().numpy()[1:], 
                markerfmt='b^', linefmt='--', label='Embedding')
        plt.stem(range(coeffs_len), 
                (coeffs_fc/coeffs_fc.norm()).detach().numpy()[1:], 
                markerfmt='rs', linefmt='--', label='Unembedding')
    else:
         plt.stem(range(coeffs_len), 
                (coeffs_fc/coeffs_fc.norm()).detach().numpy()[1:], 
                markerfmt='rs', linefmt='--', label='Embedding/Unembedding')
    plt.stem(range(coeffs_len), 
            (coeffs_h1/coeffs_h1.norm()).detach().numpy()[1:], 
            markerfmt='go', linefmt='--', label='1st Hidden State')
            
    plt.legend(loc='best', bbox_to_anchor=(0.5, 0.4))
    
    # EDITED: Improved x-tick labeling logic
    # Find significant frequencies based on embedding coefficients
    normalized_coeffs = (coeffs_embed/coeffs_embed.norm()).detach().numpy()[1:]  # Skip DC
    
    # Find peaks that are above a reasonable threshold
    peak_threshold = 0.1  # Adjust this threshold as needed
    significant_freqs = np.where(normalized_coeffs > peak_threshold)[0]
    
    # Ensure we have some frequencies to show
    if len(significant_freqs) == 0:
        significant_freqs = np.array([0, 1, 2, 3])
    
    # Limit to reasonable number of labels (every 2nd or 3rd frequency)
    step_size = max(1, len(significant_freqs) // 8)  # Show max 8 labels
    display_freqs = significant_freqs[::step_size]
    
    # Make sure indices are valid for fourier_basis_names
    valid_display_freqs = display_freqs[display_freqs < len(fourier_basis_names) - 1]
    
    if len(valid_display_freqs) > 0:
        plt.xticks(valid_display_freqs[::2], [fourier_basis_names[i+1] for i in valid_display_freqs[::2]], rotation=45, ha='right')
    
    plt.xlabel('Frequencies')
    plt.ylabel('Normalized Fourier Coefficient')
    title_suffix = f" (Generator: {generator})" if not includes_zero else ""
    plt.title(f'Fourier Coefficients for Modular {config["display_name"]}{title_suffix}')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'figures', f'fourier_coefficients_{operation}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot h2 and h3 - skip DC component
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    # Plot h2
    axs[0].matshow(coeffs_h2[1:, 1:], cmap='Blues')
    axs[0].set_ylabel('a Component')
    axs[0].xaxis.set_label_position('bottom')
    axs[0].set_xlabel('b Component')
    axs[0].xaxis.tick_bottom()
       # Edited: Better tick positioning for 2D plots
    # Use the same significant frequencies for consistency
    matrix_size = coeffs_h2.shape[0] - 1  # Excluding DC component
    tick_positions = valid_display_freqs[valid_display_freqs < matrix_size]
    
    # Ensure reasonable spacing for readability
    if len(tick_positions) > 4:
        tick_positions = tick_positions[::2]  # Take every other position
    
    if len(tick_positions) > 0:
        axs[0].set_xticks(tick_positions, 
                          [fourier_basis_names[i+1] for i in tick_positions], 
                          rotation=45, fontsize=8, ha='right')
        axs[0].set_yticks(tick_positions, 
                          [fourier_basis_names[i+1] for i in tick_positions], 
                          rotation=0, fontsize=8)
    
    # axs[0].set_title(f'Fourier Coefficients of $h_2$')
    axs[0].set_title(f'2nd Hidden State')
    
    # Plot h3
    axs[1].matshow(coeffs_h3[1:, 1:], cmap='Blues')
    axs[1].yaxis.set_label_position('right')
    axs[1].set_ylabel('a Component')
    axs[1].xaxis.set_label_position('bottom')
    axs[1].set_xlabel('b Component')
    axs[1].xaxis.tick_bottom()
    axs[1].yaxis.tick_right()
    
    if len(tick_positions) > 0:
        axs[1].set_xticks(tick_positions, 
                          [fourier_basis_names[i+1] for i in tick_positions], 
                          rotation=45, fontsize=8, ha='right')
        axs[1].set_yticks(tick_positions, 
                          [fourier_basis_names[i+1] for i in tick_positions], 
                          rotation=0, fontsize=8)
    
    # axs[1].set_title(f'Fourier Coefficients of $h_3$')
    axs[1].set_title(f'3rd Hidden State')

    # fig.suptitle(f'Hidden States in {name} ({config["display_name"]})') 
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'figures', f'fourier_coefficients_h23_{operation}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # SVD analysis of Unembedding Matrix
    U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc)
    colors = ['b','r','g','m','k','y', 'c', 'y']
    markers = ['o', '^', 's', '*', 'x', '.', 'v', 'h']
    plt.figure(figsize=(10,4))
    
    # Ensure we don't exceed available components
    max_components = min(len(key_freqs)//2, len(colors), U_fc.shape[1]//2)
    
    # Max Components for Division only gives 2 components due to the frequency sensitivity
    # So I hardcoded max_components = 7 for Division
    if config['display_name'] == 'Division':
        max_components = 7
    
    for i in range(max_components):
        if 2*(i+1) <= U_fc.shape[1]:  # Ensure we have enough components
            W_fc_trunc = U_fc[:,2*i:2*(i+1)] @ torch.diag(S_fc[2*i:2*(i+1)]) @ Vh_fc[2*i:2*(i+1),:]
            fourier_transformed_fc_trunc = fourier_basis @ W_fc_trunc
            coeffs_trunc_fc = fourier_transformed_fc_trunc.norm(dim=1)

            # plot norms
            plt.stem(range(coeffs_len), 
            (coeffs_trunc_fc/coeffs_trunc_fc.norm()).detach().numpy()[1:], 
            markerfmt=''.join([colors[i],markers[i]]), 
            linefmt=':', basefmt='k-', 
            label=f'Components {2*i+1}, {2*i+2}')

    # FIXED: Use the same tick labeling logic for SVD plot
    if len(valid_display_freqs) > 0:
        plt.xticks(valid_display_freqs[::2], [fourier_basis_names[i+1] for i in valid_display_freqs[::2]], rotation=45, ha='right')
    
    plt.xlabel('Frequencies')
    plt.ylabel('Normalized Fourier Coefficient')
    plt.title(f'Fourier Coefficients of Components of $W_{{fc}}$ ({config["display_name"]}{title_suffix})')
    plt.legend(loc='best', bbox_to_anchor=(0.6, 0.3))
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'figures', f'fourier_components_fc_{operation}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

#currently runs but check if it does properly
def svd_spectrum_analysis_plotting(model, checkpoint, dataset, save_dir, operation, elbows, weights, model_name):
    """
    Perform SVD spectrum analysis plotting for any modular arithmetic operation.
    
    Args:
        model: The trained model
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        save_dir: Directory to save plots
        operation: String indicating the operation
    """
    config = get_operation_config(operation)
    ip_elbow = config['ip_elbow']
    hh_elbow = config['hh_elbow']
    
    modulo = dataset.modulo
    hidden_dim = model.hidden_dim
    
    W_E = weights['Embedding']['weight']
    W_fc = weights['Unembedding']['weight']
    W_ih = weights['Input Hidden']['weight'] if model_name == 'rnn' else None
    W_hh = weights['Hidden Hidden']['weight'] if model_name == 'rnn' else None
    
    with torch.no_grad():
        S_E = torch.linalg.svdvals(W_E)
        S_fc = torch.linalg.svdvals(W_fc)

        S_E /= S_E.sum()
        S_fc /= S_fc.sum()

        if model_name == 'rnn':
            S_ih = torch.linalg.svdvals(W_ih)
            S_hh = torch.linalg.svdvals(W_hh)
            S_ih /= S_ih.sum()
            S_hh /= S_hh.sum()
    
    # Determine expected dimensions based on operation type
    # TODO: what is going on here? I'm pretty expected vocab size is modulo + 1 if we include 0
    includes_zero = config['includes_zero']
    if includes_zero:
        # Addition/Subtraction: dataset size = modulo²
        expected_vocab_size = modulo
    else:
        # Multiplication/Division: dataset size = (modulo-1)²
        expected_vocab_size = modulo - 1
    
    # Create indices based on actual SVD matrix dimensions
    ip_idxs_E = np.arange(1, len(S_E) + 1)  # For embedding weights
    ip_idxs_fc = np.arange(1, len(S_fc) + 1)  # For unembedding weights

    if model_name == 'rnn':
        ip_idxs_ih = np.arange(1, len(S_ih) + 1)  # For input weights
        hh_idxs = np.arange(1, len(S_hh) + 1)  # For hidden weights
    
    # Sanity check: verify dimensions match expected vocabulary size
    print(f"Expected vocab size: {expected_vocab_size}, Actual embedding dim: {len(S_E)}")
    print(f"Operation: {config['display_name']}, Includes zero: {includes_zero}")
    
    # Adjust elbow points if they exceed matrix dimensions
    ip_elbow_E = min(elbows[0], len(S_E))
    ip_elbow_fc = min(elbows[0], len(S_fc))
    
    if model_name == 'rnn':
        ip_elbow_ih = min(elbows[0], len(S_ih))
        hh_elbow_adj = min(elbows[1], len(S_hh))
    
    layers = ['Embedding', 'Unembedding', 'Input', 'Hidden']

    svd_spectra = [S_E, S_fc]
    n_comps = [ip_elbow_E, ip_elbow_fc]

    if model_name == 'rnn':
        svd_spectra.extend([S_ih, S_hh])
        n_comps.extend([ip_elbow_ih, hh_elbow_adj])
    
    print(f'Fraction of Variance in significant components for {config["display_name"]}:')
    for layer, S, n in zip(layers, svd_spectra, n_comps):
        print(f'{layer} Weights, {n} components: {S[:n].sum():.4f}')
    
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(6, 4))
    
    # Plot singular value spectra with correct indices
    plt.loglog(ip_idxs_E, S_E.detach().numpy(), 'b-')
    plt.loglog(ip_idxs_fc, S_fc.detach().numpy(), 'r-')
    
    if model_name == 'rnn':
        plt.loglog(ip_idxs_ih, S_ih.detach().numpy(), 'g-')
        plt.loglog(hh_idxs, S_hh.detach().numpy(), 'k-')
    
    # Highlight specific points (only if elbow points are within bounds)
    if model_name == 'rnn':
        if ip_elbow_E <= len(S_E):
            plt.plot(ip_elbow_E, S_E[ip_elbow_E-1].detach().numpy(), marker='o', markersize=10,
                    color='b', label='Embedding')
        if ip_elbow_fc <= len(S_fc):
            plt.plot(ip_elbow_fc, S_fc[ip_elbow_fc-1].detach().numpy(), marker='s', markersize=10,
                    color='r', label='Unembedding')
        if ip_elbow_ih <= len(S_ih):
            plt.plot(ip_elbow_ih, S_ih[ip_elbow_ih-1].detach().numpy(), marker='^', markersize=10,
                    color='g', label='Input')
        if hh_elbow_adj <= len(S_hh):
            plt.plot(hh_elbow_adj, S_hh[hh_elbow_adj-1].detach().numpy(), marker='d', markersize=10,
                    color='k', label='Hidden')
    else:
        if ip_elbow_fc <= len(S_fc):
            plt.plot(ip_elbow_fc, S_fc[ip_elbow_fc-1].detach().numpy(), marker='s', markersize=10,
                    color='r', label='Embedding/Unembedding')
    
    # Set x-ticks based on actual elbow points
    xtick_labels = []
    xtick_positions = []
    if ip_elbow_E <= len(S_E):
        xtick_positions.append(ip_elbow_E)
        xtick_labels.append(str(ip_elbow_E))
    if model_name == 'rnn' and hh_elbow_adj <= len(S_hh) and hh_elbow_adj != ip_elbow_E:
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
    
def svd_ablation_analysis(model_cfg, weights, checkpoint, dataset, operation, elbows, model_name):
    """
    Perform SVD ablation analysis for any modular arithmetic operation.
    
    Args:
        model_cfg: Model configuration
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
    """
    config = get_operation_config(operation)
    ip_elbow = elbows[0]
    hh_elbow = elbows[1]
    
    model_weights = copy.deepcopy(checkpoint['model'])
    
    W_E = weights['Embedding']['weight']
    W_fc = weights['Unembedding']['weight']

    if model_name == 'rnn':
        W_ih = weights['Input Hidden']['weight']
        W_hh = weights['Hidden Hidden']['weight']

    with torch.no_grad():
        U_E, S_E, Vh_E = torch.linalg.svd(W_E, full_matrices=False)
        U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc, full_matrices=False)

        if model_name == 'rnn':
            U_ih, S_ih, Vh_ih = torch.linalg.svd(W_ih, full_matrices=False)
            U_hh, S_hh, Vh_hh = torch.linalg.svd(W_hh, full_matrices=False)

    # Ablate all insignificant components
    model_weights['embedding.weight'] = U_E[:,:ip_elbow] @ torch.diag(S_E[:ip_elbow]) @ Vh_E[:ip_elbow,:]
    model_weights['fc.weight'] = U_fc[:,:ip_elbow] @ torch.diag(S_fc[:ip_elbow]) @ Vh_fc[:ip_elbow,:]
    
    if model_name == 'rnn':
        model_weights['rnn.weight_ih_l0'] = U_ih[:,:ip_elbow] @ torch.diag(S_ih[:ip_elbow]) @ Vh_ih[:ip_elbow,:]
        model_weights['rnn.weight_hh_l0'] = U_hh[:,:hh_elbow] @ torch.diag(S_hh[:hh_elbow]) @ Vh_hh[:hh_elbow,:]

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)
    accuracy_keep_significant = evaluate_model(model_trunc, dataset.dataset, dataset.labels)
    print(f'Accuracy after ablating insignificant components ({config["display_name"]}): {accuracy_keep_significant:.2f}%')

    # Ablate all significant components
    model_weights['embedding.weight'] = U_E[:,ip_elbow:] @ torch.diag(S_E[ip_elbow:]) @ Vh_E[ip_elbow:,:]
    model_weights['fc.weight'] = U_fc[:,ip_elbow:] @ torch.diag(S_fc[ip_elbow:]) @ Vh_fc[ip_elbow:,:]
    
    if model_name == 'rnn':
        model_weights['rnn.weight_ih_l0'] = U_ih[:,ip_elbow:] @ torch.diag(S_ih[ip_elbow:]) @ Vh_ih[ip_elbow:,:]
        model_weights['rnn.weight_hh_l0'] = U_hh[:,hh_elbow:] @ torch.diag(S_hh[hh_elbow:]) @ Vh_hh[hh_elbow:,:]

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)
    accuracy_ablate_significant = evaluate_model(model_trunc, dataset.dataset, dataset.labels)
    print(f'Accuracy after ablating all significant components ({config["display_name"]}): {accuracy_ablate_significant:.2f}%')


def fourier_ablation_analysis(model_cfg, weights, checkpoint, dataset, operation, elbows, model_name):
    """
    Perform Fourier component ablation analysis for any modular arithmetic operation.
    
    Args:
        model_cfg: Model configuration
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
    """
    config = get_operation_config(operation)
    ip_elbow = elbows[0]
    hh_elbow = elbows[1]
    
    W_E = weights['Embedding']['weight']
    W_fc = weights['Unembedding']['weight']

    if model_name == 'rnn':
        W_ih = weights['Input Hidden']['weight']
        W_hh = weights['Hidden Hidden']['weight']

    with torch.no_grad():
        U_E, S_E, Vh_E = torch.linalg.svd(W_E, full_matrices=False)
        U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc, full_matrices=False)

        if model_name == 'rnn':
            U_ih, S_ih, Vh_ih = torch.linalg.svd(W_ih, full_matrices=False)
            U_hh, S_hh, Vh_hh = torch.linalg.svd(W_hh, full_matrices=False)

    print(f"Fourier ablation analysis for {config['display_name']}:")
    for i in range(ip_elbow // 2): # ip_elbow // 2
        with torch.no_grad():
            model_weights = copy.deepcopy(checkpoint['model'])
            model_weights['embedding.weight'] = U_E[:,:ip_elbow] @ torch.diag(S_E[:ip_elbow]) @ Vh_E[:ip_elbow,:]
            
            if model_name == 'rnn':
                model_weights['rnn.weight_ih_l0'] = U_ih[:,:ip_elbow] @ torch.diag(S_ih[:ip_elbow]) @ Vh_ih[:ip_elbow,:]
                model_weights['rnn.weight_hh_l0'] = U_hh[:,:hh_elbow] @ torch.diag(S_hh[:hh_elbow]) @ Vh_hh[:hh_elbow,:]

            model_single_trunc_weights = copy.deepcopy(model_weights)
            model_all_trunc_weights = copy.deepcopy(model_weights)

            S_fc_single_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_single_trunc[2*i:2*(i+1)] = 0
            # S_fc_single_trunc[i] = 0
            W_fc_single_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_single_trunc) @ Vh_fc[:ip_elbow,:]
            model_single_trunc_weights['fc.weight'] = W_fc_single_trunc

            S_fc_all_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_all_trunc[:2*(i+1)] = 0
            # S_fc_all_trunc[:i+1] = 0
            W_fc_all_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_all_trunc) @ Vh_fc[:ip_elbow,:]
            model_all_trunc_weights['fc.weight'] = W_fc_all_trunc

            model_single_trunc = create_model(model_cfg)
            model_single_trunc.load_state_dict(model_single_trunc_weights)

            model_all_trunc = create_model(model_cfg)
            model_all_trunc.load_state_dict(model_all_trunc_weights)

            single_acc = evaluate_model(model_single_trunc, dataset.dataset, dataset.labels)
            all_acc = evaluate_model(model_all_trunc, dataset.dataset, dataset.labels)

            print(f"  Accuracy after ablating:")
            print(f"    Only Freq {i+1}: {single_acc:.2f}%")
            print(f"    Until Freq {i+1}: {all_acc:.2f}%")

def verify_trigonometric_identity(model, weights, checkpoint, dataset, operation, model_name):
    modulo = dataset.modulo

    config = get_operation_config(operation)
    freq_threshold = config['freq_threshold']
    includes_zero = config['includes_zero']

    m = modulo - 1 if operation in ('multiplication', 'division') else modulo

    fourier_basis, fourier_basis_names = fourier_basis_constructor(m)
    fourier_basis = torch.stack(fourier_basis)

    W_E = weights['Embedding']['weight']
    W_fc = weights['Unembedding']['weight']

    _, _, Vh_fc = torch.linalg.svd(W_fc) # Unembedding is weights[1]

    model.eval()
    with torch.no_grad():
        embedded_seq = model.embedding(dataset.dataset)
        output = weights['Hidden States']['output']
        hidden_states = weights['Hidden States']['hs']
        h3_norm = output[:, 2, :]

        if model_name == 'rnn':
            h3_norm = model.bn(h3_norm)
        elif model_name == 'transformer':
            h3_norm = model.ln_out(h3_norm)

    if operation in ('multiplication', 'division'):
        generator = 3
        W_E = W_E[:m, :]
        perm = torch.tensor([generator**i % modulo for i in range(1, modulo)])
        W_E = W_E[perm-1]
   
        # h3_norm = h3_norm[perm-1]

    if includes_zero:
        coeffs_embed = fourier_basis @ W_E[:-1, :]
    else:
        coeffs_embed = fourier_basis @ W_E

    coeffs_embed = coeffs_embed.norm(dim=1)
    key_freqs = torch.where(coeffs_embed > freq_threshold, 1, 0).nonzero()
    key_freqs = key_freqs.squeeze().detach().numpy() - 1

    const_ind = np.where(key_freqs == -1)[0]
    key_freqs = np.delete(key_freqs, const_ind)

    plotting_coeffs = torch.zeros(m**2, 2)

    for i in range(len(key_freqs)//2):
        mat = h3_norm @ Vh_fc[2*i:2*(i+1)].t()        
        
        if operation in ('multiplication', 'division'):
            mat_reshaped = mat.reshape(m, m, -1)
            
            mat_a_reordered = mat_reshaped[perm-1]
            mat_both_reordered = mat_a_reordered.transpose(0, 1)[perm-1].transpose(0, 1).contiguous()

            mat = mat_both_reordered.reshape(-1, mat_both_reordered.shape[-1])

        plotting_coeffs += fft2d(mat, fourier_basis, m)


    plotting_coeffs = plotting_coeffs.reshape(m, m, 2)
    # plotting_coeffs = plotting_coeffs[1:, 1:, :]

    n_freqs = len(key_freqs)
    res_cos = torch.zeros(n_freqs, 2)
    res_sin = torch.zeros(n_freqs, 2)
    coeff_cos = torch.zeros(n_freqs, 2)
    coeff_sin = torch.zeros(n_freqs, 2)

    print("Verifying trigonometric identities:")
    if operation in ('addition', 'multiplication'):
        for i in range(len(key_freqs)//2):
            f1 = key_freqs[2*i]
            f2 = key_freqs[2*i+1]

            res_cos[i] = 0.5 * (plotting_coeffs[f1+1, f1+1] + plotting_coeffs[f2+1, f2+1])
            res_sin[i] = 0.5 * (plotting_coeffs[f2+1, f1+1] - plotting_coeffs[f1+1, f2+1])
            coeff_cos[i] = 0.5 * (plotting_coeffs[f1+1, f1+1] - plotting_coeffs[f2+1, f2+1])
            coeff_sin[i] = 0.5 * (plotting_coeffs[f2+1, f1+1] + plotting_coeffs[f1+1, f2+1])

            freq_names = (fourier_basis_names[f1+1], fourier_basis_names[f2+1])
            rel_error = ((res_cos[i].pow(2).sum() + res_sin[i].pow(2).sum()) /
                         (coeff_cos[i].pow(2).sum() + coeff_sin[i].pow(2).sum()))
            print(f'\t\t Relative Error for components {freq_names}: {rel_error.sqrt():.2E}')

    elif operation in ('subtraction', 'division'):
        for i in range(len(key_freqs)//2):
            f1 = key_freqs[2*i]
            f2 = key_freqs[2*i+1]

            res_cos[i] = 0.5 * (plotting_coeffs[f1+1, f1+1] + plotting_coeffs[f2+1, f2+1])
            res_sin[i] = 0.5 * (plotting_coeffs[f2+1, f1+1] - plotting_coeffs[f1+1, f2+1])
            coeff_cos[i] = 0.5 * (plotting_coeffs[f1+1, f1+1] - plotting_coeffs[f2+1, f2+1])
            coeff_sin[i] = 0.5 * (plotting_coeffs[f2+1, f1+1] + plotting_coeffs[f1+1, f2+1])

            freq_names = (fourier_basis_names[f1+1], fourier_basis_names[f2+1])
            rel_error = ((coeff_cos[i].pow(2).sum() + coeff_sin[i].pow(2).sum()) /
                         (res_cos[i].pow(2).sum() + res_sin[i].pow(2).sum()))
            print(f'\t\t Relative Error for components {freq_names}: {rel_error.sqrt():.2E}')

            

'''def fourier_ablation_analysis(model_cfg, checkpoint, dataset, operation):
    """
    Perform Fourier component ablation analysis for any modular arithmetic operation.
    
    Args:
        model_cfg: Model configuration
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
    """
    config = get_operation_config(operation)
    # ip_elbow = config['ip_elbow']
    # hh_elbow = config['hh_elbow']
    
    W_E = checkpoint['model']['embedding.weight']
    W_fc = checkpoint['model']['fc.weight']
    W_ih = checkpoint['model']['rnn.weight_ih_l0']
    W_hh = checkpoint['model']['rnn.weight_hh_l0']

    # Find key frequencies
    coeffs_embed = fourier_basis @ W_E[:-1, :]
    coeffs_embed = coeffs_embed.norm(dim=1)
    key_freqs = torch.where(coeffs_embed > freq_threshold, 1, 0).nonzero() # TODO: Hard coded threshold, hacky
    key_freqs = key_freqs.squeeze().detach().numpy() - 1

    const_ind = np.where(key_freqs == -1)[0]
    key_freqs = np.delete(key_freqs, const_ind) # Get rid of const if exists

    # Get elbow thresholds
    ip_elbow = len(key_freqs)
    
    # Iteratively get hh_elbow value by finding number of 

    with torch.no_grad():
        U_E, S_E, Vh_E = torch.linalg.svd(W_E, full_matrices=False)
        U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc, full_matrices=False)
        U_ih, S_ih, Vh_ih = torch.linalg.svd(W_ih, full_matrices=False)
        U_hh, S_hh, Vh_hh = torch.linalg.svd(W_hh, full_matrices=False)

    print(f"Fourier ablation analysis for {config['display_name']}:")
    for i in range(ip_elbow//2):
        with torch.no_grad():
            model_weights = copy.deepcopy(checkpoint['model'])
            model_weights['embedding.weight'] = U_E[:,:ip_elbow] @ torch.diag(S_E[:ip_elbow]) @ Vh_E[:ip_elbow,:]
            model_weights['rnn.weight_ih_l0'] = U_ih[:,:ip_elbow] @ torch.diag(S_ih[:ip_elbow]) @ Vh_ih[:ip_elbow,:]
            model_weights['rnn.weight_hh_l0'] = U_hh[:,:hh_elbow] @ torch.diag(S_hh[:hh_elbow]) @ Vh_hh[:hh_elbow,:]

            model_single_trunc_weights = copy.deepcopy(model_weights)
            model_all_trunc_weights = copy.deepcopy(model_weights)

            S_fc_single_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_single_trunc[2*i:2*(i+1)] = 0
            W_fc_single_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_single_trunc) @ Vh_fc[:ip_elbow,:]
            model_single_trunc_weights['fc.weight'] = W_fc_single_trunc

            S_fc_all_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_all_trunc[:2*(i+1)] = 0
            W_fc_all_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_all_trunc) @ Vh_fc[:ip_elbow,:]
            model_all_trunc_weights['fc.weight'] = W_fc_all_trunc

            model_single_trunc = create_model(model_cfg)
            model_single_trunc.load_state_dict(model_single_trunc_weights)

            model_all_trunc = create_model(model_cfg)
            model_all_trunc.load_state_dict(model_all_trunc_weights)

            single_acc = evaluate_model(model_single_trunc, dataset.dataset, dataset.labels)
            all_acc = evaluate_model(model_all_trunc, dataset.dataset, dataset.labels)

            print(f"  Accuracy after ablating:")
            print(f"    Only Freq {i+1}: {single_acc:.2f}%")
            print(f"    Until Freq {i+1}: {all_acc:.2f}%")
'''

def get_elbow_thresholds(model, weights, checkpoint, dataset, model_cfg, model_name):
    """Get ih and hh elbows"""
    
    W_E = weights['Embedding']['weight']
    W_fc = weights['Unembedding']['weight']

    if model_name == 'rnn':
        W_ih = weights['Input Hidden']['weight']
        W_hh = weights['Hidden Hidden']['weight']

    includes_zero = OPERATION_CONFIGS[dataset.operation]['includes_zero']
    freq_threshold = OPERATION_CONFIGS[dataset.operation]['freq_threshold']

    modulo = dataset.modulo
    dataset_size = modulo if includes_zero else modulo - 1

    # Create Fourier basis with appropriate size
    fourier_basis, fourier_basis_names = create_fourier_analysis_setup(modulo, includes_zero) # Hard code to True so it doesnt return 113x112

    if dataset.operation in ('multiplication', 'division'):
        
        # Create permutation using powers of 3
        generator = 3
        perm = torch.tensor([generator**i % modulo for i in range(1, modulo)])
        
        # Apply permutation directly (perm-1 for 0-indexing)
        W_E = W_E[perm-1]

    # Find key frequencies
    if includes_zero:
        coeffs_embed = fourier_basis @ W_E[:-1, :]
    else:
        coeffs_embed = fourier_basis @ W_E[:, :]
        
    coeffs_embed = coeffs_embed.norm(dim=1)
    # print(coeffs_embed)

    key_freqs = torch.where(coeffs_embed > freq_threshold, 1, 0).nonzero()
    key_freqs = key_freqs.squeeze().detach().numpy() - 1
    print(f'Frequency Indices: {key_freqs}')

    const_ind = np.where(key_freqs == -1)[0]
    key_freqs = np.delete(key_freqs, const_ind)
    ip_elbow = len(key_freqs)

    if model_name != 'rnn':
        return ip_elbow, W_fc.shape[1]

    # Compute SVDs
    with torch.no_grad():
        U_E, S_E, Vh_E = torch.linalg.svd(W_E, full_matrices=False)
        U_fc, S_fc, Vh_fc = torch.linalg.svd(W_fc, full_matrices=False)
        U_ih, S_ih, Vh_ih = torch.linalg.svd(W_ih, full_matrices=False)
        U_hh, S_hh, Vh_hh = torch.linalg.svd(W_hh, full_matrices=False)

    # Find hh_elbow by reducing dimensionality until accuracy drops from 100%
    full_size = W_hh.shape[0] # should be 256
    hh_elbow = full_size

    base_model = create_model(model_cfg)
    base_model.load_state_dict(checkpoint['model'])
    base_acc = evaluate_model(base_model, dataset.dataset, dataset.labels)

    if base_acc < 100.0:
        print(f"Warning: base model accuracy is {base_acc:.2f}%")
        return ip_elbow, full_size

    for r in range(full_size - 1, 0, -1):
        W_hh_trunc = U_hh[:, :r] @ torch.diag(S_hh[:r]) @ Vh_hh[:r, :]
        truncated_weights = copy.deepcopy(checkpoint['model'])
        truncated_weights['rnn.weight_hh_l0'] = W_hh_trunc

        truncated_model = create_model(model_cfg)
        truncated_model.load_state_dict(truncated_weights)

        acc = evaluate_model(truncated_model, dataset.dataset, dataset.labels)

        if acc < 100.0:
            hh_elbow = r + 1
            break

    return ip_elbow, hh_elbow
