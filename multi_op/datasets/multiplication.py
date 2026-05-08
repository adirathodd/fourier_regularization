import torch
from training.utils import align_running_targets_to_sequence
from .base import CustomDataset


class MultiplicationModDataset(CustomDataset):
    def __init__(self, config, operations=["multiplication"]):
        if len(operations) != 1 or operations[0] != "multiplication":
            raise ValueError("This dataset is only supposed to be used for the following operation - Multiplication")

        super().__init__(config, operations)
        self.dataset, self.labels, self.labels_seq = self._generate_dataset()
        self.train_data, self.train_labels, self.train_labels_seq, self.test_data, self.test_labels, self.test_labels_seq = self._split_dataset()

    def _generate_dataset(self):
        """Generate all possible modular multiplication expressions with nterms."""
        terms = self._grid_terms(include_zero=False)
        token_offset = 0 if self.includes_zero else 1
        op_token = torch.tensor(self.op_token_for_index(0) + token_offset)
        equals_token = (
            torch.tensor(self.equals_token + token_offset)
            if self.include_equals_token
            else None
        )

        dataset = self._build_sequence(terms, op_token, equals_token)
        running = torch.empty_like(terms)
        running[:, 0] = terms[:, 0]
        for i in range(1, self.nterms):
            running[:, i] = (running[:, i - 1] * terms[:, i]) % self.modulo

        dataset -= 1
        labels = running[:, -1] - 1
        labels_seq = align_running_targets_to_sequence(
            running - 1,
            include_equals_token=self.include_equals_token,
        )

        return dataset, labels, labels_seq
