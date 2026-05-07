import copy

import torch

from utils import create_model, evaluate_model

from analysis.context import get_svd_factors
from analysis.operation_config import compute_svd_energy_elbow, get_operation_analysis_context, get_operation_config


def _get_mixed_feature_mask(dataset, operation, feature_dim, fourier_reg_mode=None, device=None, dtype=None):
    """Build operation-aware half-feature mask for mixed mode-1 analysis."""
    op_ctx = get_operation_analysis_context(dataset, operation)
    use_split = fourier_reg_mode == 1 and op_ctx['has_mixed_ops']
    if not use_split:
        return None

    if feature_dim < 2:
        raise ValueError(
            f"fourier_reg_mode={fourier_reg_mode} requires feature dim >= 2 for mixed ablation split, got {feature_dim}."
        )

    split_idx = feature_dim // 2
    mask = torch.zeros(feature_dim, device=device, dtype=dtype)
    if op_ctx['is_multiplicative']:
        mask[split_idx:] = 1
    else:
        mask[:split_idx] = 1

    if torch.count_nonzero(mask) == 0:
        raise ValueError('Mixed ablation split produced an empty feature mask.')
    return mask


def _apply_column_mask(weight_matrix, feature_mask):
    """Apply feature mask to matrix input-feature columns."""
    if feature_mask is None:
        return weight_matrix
    return weight_matrix * feature_mask.unsqueeze(0)


def _apply_square_mask(weight_matrix, feature_mask):
    """Apply feature mask to square hidden-to-hidden matrix (rows and columns)."""
    if feature_mask is None:
        return weight_matrix
    return weight_matrix * feature_mask.unsqueeze(0) * feature_mask.unsqueeze(1)

def svd_ablation_analysis_transformer(
    model_cfg, checkpoint, dataset, operation, ip_elbow, fourier_reg_mode=None, analysis_cache=None
):
    """
    Perform SVD ablation analysis for any modular arithmetic operation.
    TRANSFORMER ONLY!

    Args:
        model_cfg: Model configuration
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
        ip_elbow: Number of significant SVD components
    """

    config = get_operation_config(operation)
    
    model_weights = copy.deepcopy(checkpoint['model'])
    
    factors = get_svd_factors(checkpoint, model_type='transformer', cache=analysis_cache)
    U_E, S_E, Vh_E = factors['embedding.weight']
    U_fc, S_fc, Vh_fc = factors['fc.weight']
    feature_mask = _get_mixed_feature_mask(
        dataset,
        operation,
        feature_dim=checkpoint['model']['embedding.weight'].shape[1],
        fourier_reg_mode=fourier_reg_mode,
        device=U_E.device,
        dtype=U_E.dtype,
    )

    # Get operation specific test data for accuracy evaluation
    op_data, op_labels = dataset.get_test_data(operation=operation)

    # Ablate all insignificant components (keep only significant)
    W_E_keep = U_E[:, :ip_elbow] @ torch.diag(S_E[:ip_elbow]) @ Vh_E[:ip_elbow, :]
    W_fc_keep = U_fc[:, :ip_elbow] @ torch.diag(S_fc[:ip_elbow]) @ Vh_fc[:ip_elbow, :]
    model_weights['embedding.weight'] = _apply_column_mask(W_E_keep, feature_mask)
    model_weights['fc.weight'] = _apply_column_mask(W_fc_keep, feature_mask)

    # Identify other operations to test on
    other_ops = [op for op in dataset.operations if op != operation]

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)
    accuracy_keep_significant = evaluate_model(model_trunc, op_data, op_labels)
    print(f'Accuracy after ablating insignificant components ({config["display_name"]}): {accuracy_keep_significant:.2f}%')

    # Evaluate on other operations
    for other_op in other_ops:
        try:
            other_data, other_labels = dataset.get_test_data(operation=other_op)
            acc_other = evaluate_model(model_trunc, other_data, other_labels)
            print(f'  -> Accuracy on {other_op}: {acc_other:.2f}%')
        except Exception as e:
            print(f'  -> Could not evaluate on {other_op}: {e}')

    # Ablate all significant components (keep only insignificant)
    model_weights = copy.deepcopy(checkpoint['model'])
    W_E_drop = U_E[:, ip_elbow:] @ torch.diag(S_E[ip_elbow:]) @ Vh_E[ip_elbow:, :]
    W_fc_drop = U_fc[:, ip_elbow:] @ torch.diag(S_fc[ip_elbow:]) @ Vh_fc[ip_elbow:, :]
    model_weights['embedding.weight'] = _apply_column_mask(W_E_drop, feature_mask)
    model_weights['fc.weight'] = _apply_column_mask(W_fc_drop, feature_mask)

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)
    accuracy_ablate_significant = evaluate_model(model_trunc, op_data, op_labels)
    print(f'Accuracy after ablating all significant components ({config["display_name"]}): {accuracy_ablate_significant:.2f}%')

    # Evaluate on other operations
    for other_op in other_ops:
        try:
            other_data, other_labels = dataset.get_test_data(operation=other_op)
            acc_other = evaluate_model(model_trunc, other_data, other_labels)
            print(f'  -> Accuracy on {other_op}: {acc_other:.2f}%')
        except Exception as e:
            print(f'  -> Could not evaluate on {other_op}: {e}')



def svd_ablation_analysis(
    model_cfg, checkpoint, dataset, operation, ip_elbow, fourier_reg_mode=None, analysis_cache=None
):
    """
    Perform SVD ablation analysis for any modular arithmetic operation.

    Args:
        model_cfg: Model configuration
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
        ip_elbow: Number of significant SVD components
    """
    config = get_operation_config(operation)

    model_weights = copy.deepcopy(checkpoint['model'])

    factors = get_svd_factors(checkpoint, model_type='rnn', cache=analysis_cache)
    U_E, S_E, Vh_E = factors['embedding.weight']
    U_fc, S_fc, Vh_fc = factors['fc.weight']
    U_ih, S_ih, Vh_ih = factors['rnn.weight_ih_l0']
    U_hh, S_hh, Vh_hh = factors['rnn.weight_hh_l0']
    feature_mask = _get_mixed_feature_mask(
        dataset,
        operation,
        feature_dim=checkpoint['model']['embedding.weight'].shape[1],
        fourier_reg_mode=fourier_reg_mode,
        device=U_E.device,
        dtype=U_E.dtype,
    )
    hh_elbow = compute_svd_energy_elbow(S_hh)

    # Get operation specific test data for accuracy evaluation
    op_data, op_labels = dataset.get_test_data(operation=operation)

    # Identify other operations to test on
    other_ops = [op for op in dataset.operations if op != operation]

    # Ablate all insignificant components
    W_E_keep = U_E[:, :ip_elbow] @ torch.diag(S_E[:ip_elbow]) @ Vh_E[:ip_elbow, :]
    W_fc_keep = U_fc[:, :ip_elbow] @ torch.diag(S_fc[:ip_elbow]) @ Vh_fc[:ip_elbow, :]
    W_ih_keep = U_ih[:, :ip_elbow] @ torch.diag(S_ih[:ip_elbow]) @ Vh_ih[:ip_elbow, :]
    W_hh_keep = U_hh[:, :hh_elbow] @ torch.diag(S_hh[:hh_elbow]) @ Vh_hh[:hh_elbow, :]
    model_weights['embedding.weight'] = _apply_column_mask(W_E_keep, feature_mask)
    model_weights['fc.weight'] = _apply_column_mask(W_fc_keep, feature_mask)
    model_weights['rnn.weight_ih_l0'] = _apply_column_mask(W_ih_keep, feature_mask)
    model_weights['rnn.weight_hh_l0'] = _apply_square_mask(W_hh_keep, feature_mask)

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)
    accuracy_keep_significant = evaluate_model(model_trunc, op_data, op_labels)
    print(f'Accuracy after ablating insignificant components ({config["display_name"]}): {accuracy_keep_significant:.2f}%')

    # Evaluate on other operations
    for other_op in other_ops:
        try:
            other_data, other_labels = dataset.get_test_data(operation=other_op)
            acc_other = evaluate_model(model_trunc, other_data, other_labels)
            print(f'  -> Accuracy on {other_op}: {acc_other:.2f}%')
        except Exception as e:
            print(f'  -> Could not evaluate on {other_op}: {e}')

    # Ablate all significant components
    model_weights = copy.deepcopy(checkpoint['model'])
    W_E_drop = U_E[:, ip_elbow:] @ torch.diag(S_E[ip_elbow:]) @ Vh_E[ip_elbow:, :]
    W_fc_drop = U_fc[:, ip_elbow:] @ torch.diag(S_fc[ip_elbow:]) @ Vh_fc[ip_elbow:, :]
    W_ih_drop = U_ih[:, ip_elbow:] @ torch.diag(S_ih[ip_elbow:]) @ Vh_ih[ip_elbow:, :]
    W_hh_drop = U_hh[:, hh_elbow:] @ torch.diag(S_hh[hh_elbow:]) @ Vh_hh[hh_elbow:, :]
    model_weights['embedding.weight'] = _apply_column_mask(W_E_drop, feature_mask)
    model_weights['fc.weight'] = _apply_column_mask(W_fc_drop, feature_mask)
    model_weights['rnn.weight_ih_l0'] = _apply_column_mask(W_ih_drop, feature_mask)
    model_weights['rnn.weight_hh_l0'] = _apply_square_mask(W_hh_drop, feature_mask)

    model_trunc = create_model(model_cfg)
    model_trunc.load_state_dict(model_weights)
    accuracy_ablate_significant = evaluate_model(model_trunc, op_data, op_labels)
    print(f'Accuracy after ablating all significant components ({config["display_name"]}): {accuracy_ablate_significant:.2f}%')

    # Evaluate on other operations
    for other_op in other_ops:
        try:
            other_data, other_labels = dataset.get_test_data(operation=other_op)
            acc_other = evaluate_model(model_trunc, other_data, other_labels)
            print(f'  -> Accuracy on {other_op}: {acc_other:.2f}%')
        except Exception as e:
            print(f'  -> Could not evaluate on {other_op}: {e}')

def fourier_ablation_analysis_transformer(
    model_cfg, checkpoint, dataset, operation, ip_elbow, fourier_reg_mode=None, analysis_cache=None
):
    """
    Perform Fourier component ablation analysis for any modular arithmetic operation.
    TRANSFORMER ONLY!

    Args:
        model_cfg: Model configuration
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
        ip_elbow: Number of significant SVD components
    """
    config = get_operation_config(operation)

    factors = get_svd_factors(checkpoint, model_type='transformer', cache=analysis_cache)
    U_E, S_E, Vh_E = factors['embedding.weight']
    U_fc, S_fc, Vh_fc = factors['fc.weight']
    feature_mask = _get_mixed_feature_mask(
        dataset,
        operation,
        feature_dim=checkpoint['model']['embedding.weight'].shape[1],
        fourier_reg_mode=fourier_reg_mode,
        device=U_E.device,
        dtype=U_E.dtype,
    )

    # Get operation specific test data for accuracy evaluation
    op_data, op_labels = dataset.get_test_data(operation=operation)

    print(f"Fourier ablation analysis for {config['display_name']}:")
    for i in range(ip_elbow//2):
        with torch.no_grad():
            model_weights = copy.deepcopy(checkpoint['model'])
            W_E_keep = U_E[:, :ip_elbow] @ torch.diag(S_E[:ip_elbow]) @ Vh_E[:ip_elbow, :]
            model_weights['embedding.weight'] = _apply_column_mask(W_E_keep, feature_mask)

            model_single_trunc_weights = copy.deepcopy(model_weights)
            model_all_trunc_weights = copy.deepcopy(model_weights)

            S_fc_single_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_single_trunc[2*i:2*(i+1)] = 0
            W_fc_single_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_single_trunc) @ Vh_fc[:ip_elbow,:]
            model_single_trunc_weights['fc.weight'] = _apply_column_mask(W_fc_single_trunc, feature_mask)

            S_fc_all_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_all_trunc[:2*(i+1)] = 0
            W_fc_all_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_all_trunc) @ Vh_fc[:ip_elbow,:]
            model_all_trunc_weights['fc.weight'] = _apply_column_mask(W_fc_all_trunc, feature_mask)

            model_single_trunc = create_model(model_cfg)
            model_single_trunc.load_state_dict(model_single_trunc_weights)

            model_all_trunc = create_model(model_cfg)
            model_all_trunc.load_state_dict(model_all_trunc_weights)

            single_acc = evaluate_model(model_single_trunc, op_data, op_labels)
            all_acc = evaluate_model(model_all_trunc, op_data, op_labels)

            print(f"  Accuracy after ablating:")
            print(f"    Only Freq {i+1}: {single_acc:.2f}%")
            print(f"    Until Freq {i+1}: {all_acc:.2f}%")


def fourier_ablation_analysis(
    model_cfg, checkpoint, dataset, operation, ip_elbow, fourier_reg_mode=None, analysis_cache=None
):
    """
    Perform Fourier component ablation analysis for any modular arithmetic operation.

    Args:
        model_cfg: Model configuration
        checkpoint: Model checkpoint containing weights
        dataset: Dataset object
        operation: String indicating the operation
        ip_elbow: Number of significant SVD components
    """
    config = get_operation_config(operation)

    factors = get_svd_factors(checkpoint, model_type='rnn', cache=analysis_cache)
    U_E, S_E, Vh_E = factors['embedding.weight']
    U_fc, S_fc, Vh_fc = factors['fc.weight']
    U_ih, S_ih, Vh_ih = factors['rnn.weight_ih_l0']
    U_hh, S_hh, Vh_hh = factors['rnn.weight_hh_l0']
    feature_mask = _get_mixed_feature_mask(
        dataset,
        operation,
        feature_dim=checkpoint['model']['embedding.weight'].shape[1],
        fourier_reg_mode=fourier_reg_mode,
        device=U_E.device,
        dtype=U_E.dtype,
    )
    hh_elbow = compute_svd_energy_elbow(S_hh)

    # Get operation specific test data for accuracy evaluation
    op_data, op_labels = dataset.get_test_data(operation=operation)

    print(f"Fourier ablation analysis for {config['display_name']}:")
    for i in range(ip_elbow//2):
        with torch.no_grad():
            model_weights = copy.deepcopy(checkpoint['model'])
            W_E_keep = U_E[:, :ip_elbow] @ torch.diag(S_E[:ip_elbow]) @ Vh_E[:ip_elbow, :]
            W_ih_keep = U_ih[:, :ip_elbow] @ torch.diag(S_ih[:ip_elbow]) @ Vh_ih[:ip_elbow, :]
            W_hh_keep = U_hh[:, :hh_elbow] @ torch.diag(S_hh[:hh_elbow]) @ Vh_hh[:hh_elbow, :]
            model_weights['embedding.weight'] = _apply_column_mask(W_E_keep, feature_mask)
            model_weights['rnn.weight_ih_l0'] = _apply_column_mask(W_ih_keep, feature_mask)
            model_weights['rnn.weight_hh_l0'] = _apply_square_mask(W_hh_keep, feature_mask)

            model_single_trunc_weights = copy.deepcopy(model_weights)
            model_all_trunc_weights = copy.deepcopy(model_weights)

            S_fc_single_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_single_trunc[2*i:2*(i+1)] = 0
            W_fc_single_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_single_trunc) @ Vh_fc[:ip_elbow,:]
            model_single_trunc_weights['fc.weight'] = _apply_column_mask(W_fc_single_trunc, feature_mask)

            S_fc_all_trunc = torch.clone(S_fc[:ip_elbow])
            S_fc_all_trunc[:2*(i+1)] = 0
            W_fc_all_trunc = U_fc[:,:ip_elbow] @ torch.diag(S_fc_all_trunc) @ Vh_fc[:ip_elbow,:]
            model_all_trunc_weights['fc.weight'] = _apply_column_mask(W_fc_all_trunc, feature_mask)

            model_single_trunc = create_model(model_cfg)
            model_single_trunc.load_state_dict(model_single_trunc_weights)

            model_all_trunc = create_model(model_cfg)
            model_all_trunc.load_state_dict(model_all_trunc_weights)

            single_acc = evaluate_model(model_single_trunc, op_data, op_labels)
            all_acc = evaluate_model(model_all_trunc, op_data, op_labels)

            print(f"  Accuracy after ablating:")
            print(f"    Only Freq {i+1}: {single_acc:.2f}%")
            print(f"    Until Freq {i+1}: {all_acc:.2f}%")
