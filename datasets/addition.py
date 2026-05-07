import torch
from training.utils import align_running_targets_to_sequence
from .base import CustomDataset


class AdditionModDataset(CustomDataset):
    def __init__(self, config, operations=["addition"]):
        if len(operations) != 1:
            raise ValueError("This dataset is only supposed to be used for the following operation - Addition")

        super().__init__(config, operations)
        self.dataset, self.labels, self.labels_seq = self._generate_dataset()
        self.train_data, self.train_labels, self.train_labels_seq, self.test_data, self.test_labels, self.test_labels_seq = self._split_dataset()

    def _generate_dataset(self):
        """Generate all possible modular addition expressions with nterms."""
        terms = self._grid_terms(include_zero=True)
        op_token = torch.tensor(self.op_token_for_index(0))
        equals_token = torch.tensor(self.equals_token) if self.include_equals_token else None

        dataset = self._build_sequence(terms, op_token, equals_token)
        running = torch.cumsum(terms, dim=1) % self.modulo
        labels = running[:, -1]
        labels_seq = align_running_targets_to_sequence(
            running,
            include_equals_token=self.include_equals_token,
        )

        return dataset, labels, labels_seq
