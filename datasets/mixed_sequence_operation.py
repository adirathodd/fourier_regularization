import math

import torch

from training.utils import align_running_targets_to_sequence
from .base import CustomDataset


class MixedSequenceOperationDataset(CustomDataset):
    """Dataset for expressions with mixed operations across a single sequence.

    Example for nterms=4:
        a1 op1 a2 op2 a3 op3 a4 =
    where each op_i can independently be any operation listed in `operations`.
    """

    def __init__(self, config, operations):
        if len(operations) < 2:
            raise ValueError("MixedSequenceOperationDataset requires at least two operations.")

        super().__init__(config, operations)
        self.mixed_ops_within_sequence = True

        self.dataset, self.labels, self.labels_seq = self._generate_dataset()
        (
            self.train_data,
            self.train_labels,
            self.train_labels_seq,
            self.test_data,
            self.test_labels,
            self.test_labels_seq,
        ) = self._split_dataset()

    def _generate_dataset(self):
        # Build in a canonical token space, then shift by -1 for multiplicative-only
        # vocabularies to preserve existing no-zero token semantics.
        token_offset = 0 if self.includes_zero else 1
        base_op_tokens = {
            op: torch.tensor(self.op_token_for_index(i) + token_offset, dtype=torch.long)
            for i, op in enumerate(self.operations)
        }
        base_equals_token = (
            torch.tensor(self.equals_token + token_offset, dtype=torch.long)
            if self.include_equals_token
            else None
        )

        if self.includes_zero:
            base_term_domain = torch.arange(self.modulo, dtype=torch.long)
        else:
            base_term_domain = torch.arange(1, self.modulo, dtype=torch.long)

        valid_divisors = torch.tensor(
            [x for x in range(1, self.modulo) if math.gcd(x, self.modulo) == 1],
            dtype=torch.long,
        )
        inverse_cache = {int(x): pow(int(x), -1, self.modulo) for x in valid_divisors.tolist()}

        dataset_parts = []
        labels_parts = []
        labels_seq_parts = []

        op_choices = torch.arange(len(self.operations), dtype=torch.long)
        if self.nterms == 2:
            op_patterns = op_choices.unsqueeze(1)
        else:
            op_patterns = torch.cartesian_prod(*([op_choices] * (self.nterms - 1)))

        for pattern in op_patterns:
            op_names = [self.operations[int(i)] for i in pattern.tolist()]

            domains = [base_term_domain]
            for op in op_names:
                if op == "division":
                    domains.append(valid_divisors)
                else:
                    domains.append(base_term_domain)

            terms = torch.cartesian_prod(*domains)
            nrows = terms.shape[0]

            seq_len = (2 * self.nterms) if self.include_equals_token else (2 * self.nterms - 1)
            seq = torch.empty((nrows, seq_len), dtype=torch.long)
            seq[:, self.term_positions] = terms
            for i, op in enumerate(op_names):
                seq[:, self.op_positions[i]] = base_op_tokens[op]
            if self.include_equals_token:
                seq[:, -1] = base_equals_token

            running = torch.empty_like(terms)
            running[:, 0] = terms[:, 0]
            for i, op in enumerate(op_names, start=1):
                rhs = terms[:, i]
                if op == "addition":
                    running[:, i] = (running[:, i - 1] + rhs) % self.modulo
                elif op == "subtraction":
                    running[:, i] = (running[:, i - 1] - rhs) % self.modulo
                elif op == "multiplication":
                    running[:, i] = (running[:, i - 1] * rhs) % self.modulo
                elif op == "division":
                    inv_rhs = torch.tensor(
                        [inverse_cache[int(x)] for x in rhs.tolist()], dtype=torch.long
                    )
                    running[:, i] = (running[:, i - 1] * inv_rhs) % self.modulo
                else:
                    raise ValueError(f"Unsupported operation in pattern: {op}")

            if self.includes_zero:
                labels = running[:, -1]
                labels_seq = align_running_targets_to_sequence(
                    running,
                    include_equals_token=self.include_equals_token,
                )
                dataset_parts.append(seq)
                labels_parts.append(labels)
                labels_seq_parts.append(labels_seq)
            else:
                shifted_seq = seq - 1
                shifted_running = running - 1
                labels = shifted_running[:, -1]
                labels_seq = align_running_targets_to_sequence(
                    shifted_running,
                    include_equals_token=self.include_equals_token,
                )
                dataset_parts.append(shifted_seq)
                labels_parts.append(labels)
                labels_seq_parts.append(labels_seq)

        dataset = torch.cat(dataset_parts, dim=0)
        labels = torch.cat(labels_parts, dim=0)
        labels_seq = torch.cat(labels_seq_parts, dim=0)
        return dataset, labels, labels_seq
