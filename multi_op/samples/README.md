# Samples

This folder contains starter assets for a full train + analysis workflow.

## Sample Configs

- `configs/transformer_addition_quickstart.yaml`
- `configs/transformer_multiop_seqcot_sample.yaml`

Use one by copying it into a run directory as `config.yaml`, then run:

```bash
conda run -n research python3 main.py -t -a -p <run_dir>
```

## Notebook

- `train_and_analyze.ipynb`

Notebook flow:
1. Edit the first code cell (all user-facing settings are there).
2. Run the cell that writes `<RUN_NAME>/config.yaml`.
3. Run train + analysis via `main.py`.
4. Review checkpoints and generated figures under `<RUN_NAME>/figures/`.
