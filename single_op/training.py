import torch
import torch.nn as nn
import numpy as np
import tqdm
import copy

import os
from utils import evaluate_model
from torch.utils.data import DataLoader, random_split
from analysis_utils import create_fourier_analysis_setup


def loss_fn(logits, labels):
    """Compute cross-entropy loss."""
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    
    # Filter out undefined labels (self.modulo + 1) for mod division
    valid_mask = labels < logits.shape[-1]  # or use: labels <= self.modulo
    if valid_mask.sum() == 0:  # No valid labels
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    valid_labels = labels[valid_mask]
    valid_log_probs = log_probs[valid_mask]

    correct_log_probs = valid_log_probs.gather(dim=-1, index=valid_labels[:, None])[:, 0]
    return -correct_log_probs.mean()

def fourier_basis_constructor(p):
    fourier_basis = []
    fourier_basis.append(torch.ones(p)/np.sqrt(p))
    fourier_basis_names = ['Const']
    for i in range(1, p//2 +1):
        fourier_basis.append(torch.cos(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis.append(torch.sin(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')

    return fourier_basis, fourier_basis_names

def fourier_loss_fn(logits, labels, model, mode, coefficient, modulo, includes_zero):
    """
        Compute cross-entropy loss with fourier regularization term.

        mode: assign specific type of regularizer
        coefficient: Numeric scaling factor, ie. Loss = L(w) + coefficient * FW
    """

    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    base_loss = -correct_log_probs.mean()

    p = modulo
    # print(p)
    fourier_basis, fourier_basis_names = create_fourier_analysis_setup(p, includes_zero=includes_zero)
    fourier_basis = fourier_basis.cuda()

    match mode:
        case 1:  
            reg_term = (
                torch.norm(fourier_basis @ (model.embedding.weight[:-1, :]), p=1) + 
                torch.norm(fourier_basis @ (model.fc.weight[:-1, :]), p=1)
            )
        case 2:
            reg_term = (
                torch.norm(fourier_basis @ model.embedding.weight[:-1, :], p=1) + 
                torch.norm(fourier_basis @ model.s4_layer.Ct[:-1, :].float(), p=1)
            )
        case 3:
            reg_term = (
                torch.norm(fourier_basis @ (model.embedding.weight), p=1) + # Shape of FW: [p, hidden_dim=256]
                torch.norm(model.rnn.weight_ih_l0, p=2) + 
                torch.norm(model.rnn.weight_hh_l0, p=2)
            )
        case 4:
            reg_term = (
                torch.norm(fourier_basis @ (model.embedding.weight + model.fc.weight), p=1) + # Shape of FW: [p, hidden_dim=256]
                torch.norm(model.rnn.weight_ih_l0, p=2) + 
                torch.norm(model.rnn.weight_hh_l0, p=2)
            )

    return base_loss + float(coefficient) * reg_term

def setup_optimizer(opt_config, model):
    """Setup optimizer based on config."""
    return torch.optim.Adam(
        model.parameters(),
        lr = opt_config.get('lr', 0.01),
        weight_decay = float(opt_config.get('wd', 5e-5)),
        betas = tuple(opt_config.get('betas', (0.9, 0.99)))
    )

def setup_scheduler(opt_config, optimizer):
    """Setup learning rate scheduler."""
    lr_decay_interval = opt_config.get('lr_decay_interval', 5000)
    epochs = opt_config.get('epochs', 20000)
    milestones = torch.arange(lr_decay_interval, 
                              epochs + lr_decay_interval, 
                              step=lr_decay_interval)
    gamma = opt_config.get('lr_decay', 0.1)
    
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=milestones, 
        gamma=gamma
    )

def train_epoch(model, dataset, optimizer, fourier_reg_mode, reg_param, generalized):
    """Perform one training step."""
    model.train()
    optimizer.zero_grad()

    train_data, train_labels = dataset.get_train_data()
    test_data, test_labels = dataset.get_test_data()
    
    device = next(model.parameters()).device
    train_data = train_data.to(device)
    train_labels = train_labels.to(device)
    
    train_logits = model(train_data)

    p = dataset.modulo

    includes_zero = True if model.vocab_size % 2 == 0 else False

    if not fourier_reg_mode:
        train_loss = loss_fn(train_logits, train_labels)
    else:
        train_loss = fourier_loss_fn(train_logits, train_labels, model, fourier_reg_mode, reg_param, p, includes_zero)
    
    train_loss.backward()
    optimizer.step()
    
    train_accuracy = evaluate_model(model, train_data, train_labels)

    if generalized == -1: # So we don't keep printing out the epoch at which we reach 100%
        return train_loss.item(), train_accuracy, generalized
    
    test_accuracy = evaluate_model(model, test_data, test_labels)

    if test_accuracy >= 100 and generalized == 0:
        return train_loss.item(), train_accuracy, 1

    return train_loss.item(), train_accuracy, 0


def eval_step(model, dataset, fourier_reg_mode, reg_param):
    """Perform one evaluation step."""
    model.eval()
    test_data, test_labels = dataset.get_test_data()

    device = next(model.parameters()).device
    test_data = test_data.to(device)
    test_labels = test_labels.to(device)

    includes_zero = True if model.vocab_size % 2 == 0 else False
    
    with torch.inference_mode():
        test_logits = model(test_data)
        # test_loss = loss_fn(test_logits, test_labels)

        if not fourier_reg_mode:
            test_loss = loss_fn(test_logits, test_labels)
        else:
            test_loss = fourier_loss_fn(test_logits, test_labels, model, fourier_reg_mode, reg_param, dataset.modulo, includes_zero)
    
    test_accuracy = evaluate_model(model, test_data, test_labels)
    return test_loss.item(), test_accuracy


def train(model, dataset, opt_config):
    """Main training loop."""
    num_epochs = opt_config.get('epochs', 20000)
    use_scheduler = opt_config.get('use_scheduler', False)
    save_every = opt_config.get('save_every', 0)
    save_dir = opt_config.get('save_dir', '.')

    fourier_reg_mode = opt_config.get('fourier_regularization', 0)
    reg_param = opt_config.get('regularization_parameter', 0)

    if not os.path.isdir(os.path.join(save_dir, 'checkpoints')):
        os.makedirs(os.path.join(save_dir, 'checkpoints'))

    optimizer = setup_optimizer(opt_config, model)
    if use_scheduler:
        scheduler = setup_scheduler(opt_config, optimizer)

    match fourier_reg_mode:
        case 0: # No Fourier Regularization
            print("Using Cross-Entropy Loss")
        case 1: # Reg. with W_e
            print("Using Cross-Entropy Loss w/ regularization term on W_e")
        case 2: # Reg. with W_fc
            print("Using Cross-Entropy Loss w/ regularization term on W_fc")
        case 3: # Reg. with W_e + W_fc
            print("Using Cross-Entropy Loss w/ regularization term on W_e * W_fc.T")
        case 4: # Activity Regularization
            print("Using activity regularization")
        case _:
            print("Unreadable regularization config, Using Cross-Entropy Loss")
    
    print(f"Starting training for {num_epochs} epochs...")
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    generalized = 0 # Has model reached 100 test accuracy?

    for epoch in tqdm.tqdm(range(num_epochs)):
        # Training step
        train_loss, train_accuracy, generalized = train_epoch(model, dataset, optimizer, fourier_reg_mode, reg_param, generalized)

        if generalized == 1:
            print(f'Model reached 99.9% test accuracy at epoch {epoch + 1}')
            generalized = -1

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluation step
        test_loss, test_accuracy = eval_step(model, dataset, fourier_reg_mode, reg_param)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Scheduler step (if enabled)
        if use_scheduler:        
            scheduler.step()
        
        # Checkpointing
        if save_every > 0:
            if (epoch + 1) % save_every == 0:
                checkpoint = {
                    'model': copy.deepcopy(model.state_dict()),
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy
                }
                torch.save(checkpoint, os.path.join(save_dir, f'checkpoints/{epoch+1}.pt'))
            
    final_checkpoint = {
        'model': copy.deepcopy(model.state_dict()),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_test_loss': test_losses[-1] if test_losses else None,
        'final_train_accuracy': train_accuracies[-1] if train_accuracies else None,
        'final_test_accuracy': test_accuracies[-1] if test_accuracies else None
    }

    torch.save(final_checkpoint, os.path.join(save_dir, f'checkpoints/final.pt'))
        
    print("Training completed!")

    return final_checkpoint
