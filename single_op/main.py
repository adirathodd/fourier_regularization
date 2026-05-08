import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from datasets import * 
from training import *
from utils import *
from fourier_spectrum_analysis import *
import matplotlib.pyplot as plt
import pandas as pd

import yaml
import os

import argparse
parser = argparse.ArgumentParser(description='Script to Train S4 on Modular Arithmetic and perform Analysis')
parser.add_argument('-p', '--path', default='.', type=str, help='Folder containing yaml file.')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('-t', '--train', action='store_true', default=False)
parser.add_argument('-a', '--analysis', action='store_true', default=False)

args = parser.parse_args()

config_path = os.path.join(args.path, 'config.yaml')
if os.path.exists(config_path):
    with open(os.path.join(args.path, 'config.yaml'), 'r') as f:
        config_params = yaml.safe_load(f)
        config_params['training']['save_dir'] = args.path
else:
    config_params = {}
    config_params['data'] = create_default_data_config()
    config_params['model'] = create_default_model_config()
    config_params['training'] = create_default_opt_config()
    save_dir = config_params['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.yaml'),'w') as f:
        yaml.dump(config_params, f, default_flow_style=False, sort_keys=False)

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

# Try the simplest possible CUDA operation
try:
    device = torch.device('cuda')
    print(f"Device object created: {device}")
    
    # Create a tiny tensor and move it
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"Tensor created on CPU: {x.device}")
    
    x_cuda = x.to(device)
    print(f"Successfully moved to CUDA: {x_cuda.device}")
    print("CUDA is working!")
    
except Exception as e:
    print(f"CUDA test failed: {e}")
    import traceback
    traceback.print_exc()

# Create dataset
operation = config_params['data']['operation']
dataset_classes = {
    'addition': ModularAdditionDataset,
    'subtraction': ModularSubtractionDataset,
    'multiplication': ModularMultiplicationDataset,
    'division': ModularDivisionDataset
}

dataset = dataset_classes[operation](config_params['data']).to_device(device)

model = create_model(config_params['model'], device)

print("Created Modular " + operation + " Data\n")
if args.verbose:
    print_dataset_info(dataset)
    print(model)

if args.train:
    
    checkpoint = train(model, dataset, config_params['training'])
    
    if not os.path.exists(os.path.join(args.path, 'figures')):
        os.makedirs(os.path.join(args.path, 'figures'))

    # Plot training curves
    df = pd.DataFrame({
        'Train Loss': checkpoint['train_losses'],
        'Test Loss': checkpoint['test_losses']
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(y=['Train Loss', 'Test Loss'], ax=ax)
    ax.set_title('Training and Test Loss Over Time')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    plt.savefig(os.path.join(args.path, 'figures', 'loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
if args.analysis:
    model, checkpoint = load_model(os.path.join(args.path, 'checkpoints/final.pt'), config_params['model'], device='cpu')
    dataset = dataset.to_device('cpu')

    if not os.path.exists(os.path.join(args.path, 'figures')):
        os.makedirs(os.path.join(args.path, 'figures'))
    
    # Split up analysis based on model type (rnn, transformer, ssm)
    model_name = config_params['model']['model']

    print(f"Accuracy on entire dataset: {evaluate_model(model, dataset.dataset, dataset.labels)}% \n\t Train set accuracy: {checkpoint['final_train_accuracy']}% \n\t Test set accuracy: {checkpoint['final_test_accuracy']}%")

    W_E = checkpoint['model']['embedding.weight']
    W_fc = checkpoint['model']['fc.weight']

    weights = {
        'Embedding': {
            'weight': W_E,
            'r': None # elbows for svd
        },
        'Unembedding': {
            'weight': W_fc,
            'r': None
        }
    }
    
    model.eval()
    with torch.no_grad():
        if model_name == 'rnn':
            
            weights['Input Hidden'] = {
                'weight': checkpoint['model']['rnn.weight_ih_l0']
            }

            weights['Hidden Hidden'] = {
                'weight': checkpoint['model']['rnn.weight_hh_l0']
            }

            embedded_seq = model.embedding(dataset.dataset)
            output, hs = model.rnn(embedded_seq)

            weights['Hidden States'] = {
                'output': output,
                'hs': hs
            }

        elif model_name == 'transformer': 
            
            output = model.get_hidden_states(tokens=dataset.dataset)[1]
            hidden_states = output[-1]

            weights['Hidden States'] = {
                'output': output,
                'hs': hidden_states
            }

        elif model_name == 'ssm':
            output = model.get_internal_states(dataset.dataset)     
            hidden_states = output[-1]  

            weights['Hidden States'] = {
                'output': output,
                'hs': hidden_states
            } 
        else:
            print(f'Unrecognized model type in configuration parameters: {config_params['model']}')

    # Get frequency component thresholds
    ip_elbow, hh_elbow = get_elbow_thresholds(model, weights, checkpoint, dataset, config_params['model'], model_name)

    weights['Embedding']['r'] = ip_elbow
    weights['Unembedding']['r'] = ip_elbow

    print("=== Fourier Spectrum Analysis Report ===\n")
    # Basic model evaluation - in utils.py
    print("1. Model performance:")
    print(f"Accuracy on entire dataset: {evaluate_model(model, dataset.dataset, dataset.labels)}% \n\t Train set accuracy: {checkpoint['final_train_accuracy']}% \n\t Test set accuracy: {checkpoint['final_test_accuracy']}%")
        
    # Fourier coefficient analysis - in fourier_spectrum_analysis.py
    print("2. Generating fourier coefficient analysis plots")
    fourier_spectrum_analysis_plotting(model, weights, dataset, args.path, operation, model_name) #added operation
    print()    

    # SVD analysis - in fourier_spectrum_analysis.py
    print("3. Generating Weight SVD spectrum plots")
    svd_spectrum_analysis_plotting(model, checkpoint, dataset, args.path, operation, (ip_elbow, hh_elbow), weights, model_name)
    print()
    
    # SVD ablation
    print("4. Performing SVD ablation analysis")
    svd_ablation_analysis(config_params['model'], weights, checkpoint, dataset, operation, (ip_elbow, hh_elbow), model_name)
    print()
    
    # Fourier component ablation
    print("5. Performing fourier component ablation analysis")
    fourier_ablation_analysis(config_params['model'], weights, checkpoint, dataset, operation, (ip_elbow, hh_elbow), model_name)
    print()
    # test others first.
    
    # Trigonometric identity verification
    print("6. Trigonometric Identity Verification:")
    verify_trigonometric_identity(model, weights, checkpoint, dataset, operation, model_name)
    print()
        
    print("=== Analysis Complete ===")