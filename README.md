# Sparse Fourier Regularization for Modular Arithmetic Models

Code for modular arithmetic grokking experiments and mechanistic analysis, based on:
`Research Paper.pdf`.

## Overview

This project studies how sequence models learn modular arithmetic tasks such as
`(a op b) mod p` for `op ∈ {+, -, *, /}`.

Core ideas from the paper:
- Generalizing solutions use sparse Fourier structure in embeddings/unembeddings.
- Adding `L1` Fourier regularization can bypass or shorten grokking and speed convergence.
- In multi-task training, same-group operations (e.g., `+` and `-`) tend to share Fourier spectra.
- Cross-group operations (e.g., `+` and `*`) can entangle spectra; group-sparse Fourier penalties help disentangle.

## Repository Focus

- Train RNNs/Transformers on modular arithmetic tasks.
- Analyze learned circuits with Fourier spectrum and SVD ablations.
- Run single-op and multi-op experiments from a unified CLI.

Main entry point: `main.py`

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The repository is divided into two parts - single op and multi op experiments.
The single op directory contains code & configs to run experiments for:
- Single operations (addition, subtraction, multiplication, division) for RNNs, Transformers, and SSMs
- Fourier regularized vs. weight decay models

All other experiments (namely, multiple operations with transformers) belong to the multi op directory.

## Quickstart (Single-Op)

### 1. Train and analyze model

```bash
cd single_op
python main.py --verbose -p "./<folder-containing-config>" -t -a
```

(See /single_op/configs for example config.yaml files to train on)

## Quickstart (Multi-Op)

### 1. Train a model

```bash
cd multi_op
python3 main.py -t -o addition -m transformer
```

### 2. Analyze a trained model

```bash
python3 main.py -a -o addition -m transformer
```

### 3. Train + analyze in one run

```bash
python3 main.py -t -a -o addition subtraction -m transformer
```

### 4. Reuse an existing run directory

```bash
python3 main.py -t -a -p transformer_modular_addition
```

## Config-Driven Workflow

Each run is controlled by `config.yaml` in the run directory.

Recommended loop:
1. Create or edit `config.yaml`.
2. Run training (`-t`).
3. Run analysis (`-a`).
4. Inspect outputs under `<save_dir>/figures/`.

Starter configs:
- `samples/configs/transformer_addition_quickstart.yaml`
- `samples/configs/transformer_multiop_seqcot_sample.yaml`

## Multi-Op Reproducibility

Use the prepared run folders under `paper_rerun_models/`:
- `add_sub`
- `add_mult`
- `add_div`
- `div_sub`
- `div_mult`
- `mult_sub`
- `add_mult_split`

Train one run:

```bash
python3 main.py -t -p paper_rerun_models/add_sub
```

Analyze one run (loads `checkpoints/best.pt`, falls back to `checkpoints/final.pt`):

```bash
python3 main.py -a -p paper_rerun_models/add_sub
```

Run analysis for all prepared multi-op runs:

```bash
for d in paper_rerun_models/*; do
  python3 main.py -a -p "$d"
done
```

Use `notebooks/analysis.ipynb` to check metrics and figures for a selected run:

```bash
jupyter notebook notebooks/analysis.ipynb
```

Inside the notebook:
- set `EXPERIMENT_KEY` to the run you want to inspect
- run cells top-to-bottom to load metrics and render analysis figures

## CLI Flags

- `-t`, `--train`: run training
- `-a`, `--analysis`: run analysis
- `-o`, `--operation`: one or more operations (`addition`, `subtraction`, `multiplication`, `division`)
- `-m`, `--model-type`: `rnn` or `transformer`
- `-p`, `--path`: run directory containing `config.yaml`

## Checkpoints and Outputs

- `best.pt`: best generalization checkpoint (grokked state)
- `final.pt`: final training state
- Figures and analysis artifacts are saved in each run folder.

## Tests

```bash
python3 tests/test_datasets.py
python3 tests/test_fourier_losses.py
```
