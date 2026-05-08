"""Variable-length analysis helpers that avoid hidden-state grid assumptions."""

from __future__ import annotations

import os
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from training.utils import align_running_targets_to_sequence


def _op_token_maps(dataset):
    base = dataset.modulo if dataset.includes_zero else dataset.modulo - 1
    include_equals_token = bool(getattr(dataset, "include_equals_token", True))
    equals_token = base if include_equals_token else None
    first_op_token = base + (1 if include_equals_token else 0)
    op_to_token = {op: first_op_token + i for i, op in enumerate(dataset.operations)}
    token_to_op = {v: k for k, v in op_to_token.items()}
    return equals_token, op_to_token, token_to_op


def _evaluate_final_accuracy(model, data, labels):
    model.eval()
    with torch.inference_mode():
        logits = model(data)
        pred = logits.argmax(dim=-1)
        correct = (pred == labels).sum().item()
        total = labels.numel()
    return (100.0 * correct / total) if total else 0.0


def _evaluate_seq_accuracy(model, data, labels_seq):
    model.eval()
    with torch.inference_mode():
        logits = model(data, return_sequence_logits=True)
        pred = logits.argmax(dim=-1)

        final_labels = labels_seq[:, -1]
        final_mask = final_labels >= 0
        final_total = int(final_mask.sum().item())
        final_correct = (pred[:, -1][final_mask] == final_labels[final_mask]).sum().item()
        final_acc = (100.0 * final_correct / final_total) if final_total else 0.0

        prefix_mask = labels_seq >= 0
        prefix_mask[:, -1] = False
        prefix_total = int(prefix_mask.sum().item())
        prefix_correct = (pred[prefix_mask] == labels_seq[prefix_mask]).sum().item()
        prefix_acc = (100.0 * prefix_correct / prefix_total) if prefix_total else 0.0
    return final_acc, prefix_acc


def per_length_accuracy_breakdown(model, dataset, training_target="last_token"):
    """Compute per-length test accuracy for variable-length datasets."""
    test_splits = dataset.get_test_splits(target="seq" if training_target == "seq_cot" else "final")
    out = {}
    for nterms in sorted(test_splits.keys()):
        data, labels = test_splits[nterms]
        if data.numel() == 0:
            continue
        device = next(model.parameters()).device
        data = data.to(device)
        labels = labels.to(device)
        if training_target == "seq_cot":
            final_acc, prefix_acc = _evaluate_seq_accuracy(model, data, labels)
            out[int(nterms)] = {
                "final_accuracy": final_acc,
                "prefix_accuracy": prefix_acc,
                "num_samples": int(data.shape[0]),
            }
        else:
            out[int(nterms)] = {
                "final_accuracy": _evaluate_final_accuracy(model, data, labels),
                "num_samples": int(data.shape[0]),
            }
    return out


def _sample_varlen_batch(dataset, nterms, num_samples, seed=0):
    """Sample random expressions for a target nterms without constructing full cartesian grids."""
    rng = random.Random(seed)
    modulo = int(dataset.modulo)
    includes_zero = bool(dataset.includes_zero)
    operations = list(dataset.operations)
    mixed_sequence_ops = bool(getattr(dataset, "mixed_sequence_operations", False))

    equals_token, op_to_token, _ = _op_token_maps(dataset)
    include_equals_token = bool(getattr(dataset, "include_equals_token", True))
    valid_divisors = [x for x in range(1, modulo) if np.gcd(x, modulo) == 1]

    seqs = []
    labels = []
    labels_seq = []

    for _ in range(num_samples):
        if len(operations) == 1:
            op_pattern = [operations[0]] * (nterms - 1)
        elif mixed_sequence_ops:
            op_pattern = [rng.choice(operations) for _ in range(nterms - 1)]
        else:
            op = rng.choice(operations)
            op_pattern = [op] * (nterms - 1)

        # Term values in arithmetic space (before token shift).
        term_vals = []
        if includes_zero:
            term_vals.append(rng.randrange(0, modulo))
        else:
            term_vals.append(rng.randrange(1, modulo))

        for op in op_pattern:
            if op == "division":
                term_vals.append(rng.choice(valid_divisors))
            else:
                if includes_zero:
                    term_vals.append(rng.randrange(0, modulo))
                else:
                    term_vals.append(rng.randrange(1, modulo))

        running = [term_vals[0]]
        for i, op in enumerate(op_pattern, start=1):
            rhs = term_vals[i]
            lhs = running[-1]
            if op == "addition":
                nxt = (lhs + rhs) % modulo
            elif op == "subtraction":
                nxt = (lhs - rhs) % modulo
            elif op == "multiplication":
                nxt = (lhs * rhs) % modulo
            elif op == "division":
                nxt = (lhs * pow(rhs, -1, modulo)) % modulo
            else:
                raise ValueError(f"Unsupported operation: {op}")
            running.append(nxt)

        seq_len = (2 * nterms) if include_equals_token else (2 * nterms - 1)
        seq = torch.empty((seq_len,), dtype=torch.long)
        for i in range(nterms):
            seq[2 * i] = term_vals[i] if includes_zero else (term_vals[i] - 1)
            if i < nterms - 1:
                seq[2 * i + 1] = op_to_token[op_pattern[i]]
        if include_equals_token:
            seq[-1] = equals_token

        run_tensor = torch.tensor(running, dtype=torch.long)
        if includes_zero:
            final_label = run_tensor[-1]
            seq_label = align_running_targets_to_sequence(
                run_tensor.unsqueeze(0),
                include_equals_token=include_equals_token,
            ).squeeze(0)
        else:
            final_label = run_tensor[-1] - 1
            seq_label = align_running_targets_to_sequence(
                (run_tensor - 1).unsqueeze(0),
                include_equals_token=include_equals_token,
            ).squeeze(0)

        seqs.append(seq)
        labels.append(final_label)
        labels_seq.append(seq_label)

    return torch.stack(seqs, dim=0), torch.stack(labels, dim=0), torch.stack(labels_seq, dim=0)


def cross_length_generalization_matrix(
    model,
    dataset,
    training_target="last_token",
    num_samples_extrapolation=2048,
    num_extrapolation_lengths=1,
    seed=0,
    save_dir=None,
):
    """Evaluate generalization across lengths (in-distribution test splits + sampled extrapolation lengths)."""
    in_dist = sorted(int(n) for n in getattr(dataset, "nterms_list", []))
    if not in_dist:
        return None

    eval_lengths = list(in_dist)
    max_len = max(in_dist)
    for k in range(1, max(0, int(num_extrapolation_lengths)) + 1):
        eval_lengths.append(max_len + k)

    results = {}
    device = next(model.parameters()).device

    # In-distribution lengths from test splits.
    test_splits = dataset.get_test_splits(target="seq" if training_target == "seq_cot" else "final")
    for nterms in in_dist:
        data, labels = test_splits[nterms]
        data = data.to(device)
        labels = labels.to(device)
        if training_target == "seq_cot":
            final_acc, prefix_acc = _evaluate_seq_accuracy(model, data, labels)
            results[nterms] = {
                "final_accuracy": final_acc,
                "prefix_accuracy": prefix_acc,
                "source": "test_split",
                "num_samples": int(data.shape[0]),
            }
        else:
            results[nterms] = {
                "final_accuracy": _evaluate_final_accuracy(model, data, labels),
                "source": "test_split",
                "num_samples": int(data.shape[0]),
            }

    # Extrapolated lengths via random sampling.
    for nterms in eval_lengths:
        if nterms in results:
            continue
        data, labels, labels_seq = _sample_varlen_batch(
            dataset, nterms=nterms, num_samples=num_samples_extrapolation, seed=seed + nterms
        )
        data = data.to(device)
        if training_target == "seq_cot":
            labels_seq = labels_seq.to(device)
            final_acc, prefix_acc = _evaluate_seq_accuracy(model, data, labels_seq)
            results[nterms] = {
                "final_accuracy": final_acc,
                "prefix_accuracy": prefix_acc,
                "source": "sampled_extrapolation",
                "num_samples": int(data.shape[0]),
            }
        else:
            labels = labels.to(device)
            results[nterms] = {
                "final_accuracy": _evaluate_final_accuracy(model, data, labels),
                "source": "sampled_extrapolation",
                "num_samples": int(data.shape[0]),
            }

    if save_dir is not None:
        os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
        accs = [results[n]["final_accuracy"] for n in eval_lengths]
        x = np.arange(len(eval_lengths))
        colors = [
            "#1f77b4" if results[n]["source"] == "test_split" else "#ff7f0e"
            for n in eval_lengths
        ]

        plt.figure(figsize=(max(6, len(eval_lengths) * 1.0), 3.2))
        bars = plt.bar(x, accs, color=colors, edgecolor="black", linewidth=0.6)
        plt.ylim(0, 100)
        plt.xticks(x, [str(n) for n in eval_lengths])
        plt.xlabel("Evaluation nterms")
        plt.ylabel("Final Accuracy (%)")
        plt.title("Cross-Length Generalization")

        for bar, acc in zip(bars, accs):
            y = min(99.0, acc + 1.2)
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{acc:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.grid(axis="y", linestyle="--", alpha=0.3)
        fig_path = os.path.join(save_dir, "figures", "cross_length_generalization_matrix.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

    return {"eval_lengths": eval_lengths, "results": results}


def token_position_accuracy(model, dataset):
    """Compute per-position token accuracy for seq_cot labels on variable-length test splits."""
    device = next(model.parameters()).device
    test_splits = dataset.get_test_splits(target="seq")
    by_length = {}
    aggregate = defaultdict(lambda: {"correct": 0, "total": 0})
    include_equals_token = bool(getattr(dataset, "include_equals_token", True))

    for nterms in sorted(test_splits.keys()):
        data, labels_seq = test_splits[nterms]
        if data.numel() == 0:
            continue
        data = data.to(device)
        labels_seq = labels_seq.to(device)

        with torch.inference_mode():
            logits = model(data, return_sequence_logits=True)
            pred = logits.argmax(dim=-1)

        seq_len = int(labels_seq.shape[1])
        per_pos = {}
        for pos in range(seq_len):
            mask = labels_seq[:, pos] >= 0
            total = int(mask.sum().item())
            if total == 0:
                continue
            correct = int((pred[:, pos][mask] == labels_seq[:, pos][mask]).sum().item())
            acc = 100.0 * correct / total
            if pos == seq_len - 1:
                role = "equals" if include_equals_token else "final_term"
            else:
                role = "term" if pos % 2 == 0 else "op"
            per_pos[pos] = {"accuracy": acc, "correct": correct, "total": total, "role": role}
            aggregate[pos]["correct"] += correct
            aggregate[pos]["total"] += total

        by_length[int(nterms)] = per_pos

    aggregate_acc = {}
    for pos in sorted(aggregate.keys()):
        total = aggregate[pos]["total"]
        correct = aggregate[pos]["correct"]
        aggregate_acc[pos] = {
            "accuracy": (100.0 * correct / total) if total else 0.0,
            "correct": int(correct),
            "total": int(total),
        }

    return {"by_length": by_length, "aggregate": aggregate_acc}


def operation_pattern_failure_taxonomy(model, dataset, training_target="last_token", top_k=15):
    """Aggregate failure rates by operation pattern (e.g., '+ -', '* / *')."""
    device = next(model.parameters()).device
    _, _, token_to_op = _op_token_maps(dataset)
    test_splits = dataset.get_test_splits(target="seq" if training_target == "seq_cot" else "final")

    pattern_stats = defaultdict(lambda: {"errors": 0, "correct": 0, "total": 0})

    for nterms in sorted(test_splits.keys()):
        data, labels = test_splits[nterms]
        data = data.to(device)
        labels = labels.to(device)

        with torch.inference_mode():
            if training_target == "seq_cot":
                logits = model(data, return_sequence_logits=True)
                pred = logits.argmax(dim=-1)[:, -1]
                y = labels[:, -1]
            else:
                logits = model(data)
                pred = logits.argmax(dim=-1)
                y = labels

        for i in range(data.shape[0]):
            row = data[i]
            ops = []
            for pos in range(1, 2 * nterms - 1, 2):
                op_name = token_to_op.get(int(row[pos].item()), "unknown")
                ops.append(op_name)
            key = tuple(ops)
            ok = bool(pred[i].item() == y[i].item())
            pattern_stats[key]["total"] += 1
            if ok:
                pattern_stats[key]["correct"] += 1
            else:
                pattern_stats[key]["errors"] += 1

    rows = []
    for pattern, stats in pattern_stats.items():
        total = stats["total"]
        errors = stats["errors"]
        rows.append(
            {
                "pattern": pattern,
                "pattern_str": " ".join(pattern),
                "total": int(total),
                "errors": int(errors),
                "error_rate": (100.0 * errors / total) if total else 0.0,
                "accuracy": (100.0 * stats["correct"] / total) if total else 0.0,
            }
        )

    rows.sort(key=lambda r: (-r["error_rate"], -r["total"], r["pattern_str"]))
    return {
        "top_failures": rows[: max(1, int(top_k))],
        "all_patterns": rows,
    }
