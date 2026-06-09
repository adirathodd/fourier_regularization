import torch
from .base import CustomDataset
from .addition import AdditionModDataset
from .subtraction import SubtractionModDataset
from .multiplication import MultiplicationModDataset
from .division import DivisionModDataset


def _generate_raw(dataset_cls, config, op):
    """Instantiate dataset_cls without running _split_dataset."""
    instance = object.__new__(dataset_cls)
    CustomDataset.__init__(instance, config, [op])
    return instance._generate_dataset()


class MultiOperationDataset(CustomDataset):
    dataset_classes = {
        'addition': AdditionModDataset,
        'subtraction': SubtractionModDataset,
        'multiplication': MultiplicationModDataset,
        'division': DivisionModDataset
    }

    def __init__(self, config, operations):
        super().__init__(config, operations)
        self.datasets_by_op = {}
        self.labels_by_op = {}
        self.labels_seq_by_op = {}

        self.dataset, self.labels, self.labels_seq = self._generate_dataset()
        self.train_data, self.train_labels, self.train_labels_seq, self.test_data, self.test_labels, self.test_labels_seq = self._split_dataset()

    def _generate_dataset(self):
        """Generate dataset for all operations with consistent token mapping."""
        datasets = []
        labels = []
        labels_seq = []
        term_positions = self.term_positions
        op_positions = self.op_positions
        equals_pos = (2 * self.nterms - 1) if self.include_equals_token else None

        for i, op in enumerate(self.operations):
            # Generate raw data without allocating a full split
            dataset_cls = self.dataset_classes[op]
            dataset, label, label_seq = _generate_raw(dataset_cls, self.config, op)

            dataset = dataset.clone()
            label = label.clone()
            label_seq = label_seq.clone()

            # Adjust indices for consistency across operations
            if self.includes_zero and op in ["multiplication", "division"]:
                dataset[:, term_positions] = dataset[:, term_positions] + 1
                label = label + 1
                valid_mask = (label_seq >= 0)
                label_seq[valid_mask] = label_seq[valid_mask] + 1

            # Set operation token from this dataset's shared token map.
            op_token = self.op_token_for_index(i)
            if equals_pos is not None:
                dataset[:, equals_pos] = int(self.equals_token)
            dataset[:, op_positions] = op_token

            datasets.append(dataset)
            labels.append(label)
            labels_seq.append(label_seq)
            self.datasets_by_op[op] = dataset
            self.labels_by_op[op] = label
            self.labels_seq_by_op[op] = label_seq

        # Combine all datasets
        combined_dataset = torch.cat(datasets, dim=0)
        combined_labels = torch.cat(labels, dim=0)
        combined_labels_seq = torch.cat(labels_seq, dim=0)

        return combined_dataset, combined_labels, combined_labels_seq

    def _split_dataset(self):
        """Split dataset per operation to preserve `train_frac` for each operation.

        This performs a stratified split: for each operation's individual
        dataset we compute a train/test split with `self.train_frac`, then
        concatenate all per-op train parts together (and likewise for test).
        Finally we shuffle the combined train and test sets deterministically
        using `self.data_seed` so batching sees mixed operations.
        """
        train_parts = []
        train_labels = []
        train_labels_seq = []
        test_parts = []
        test_labels = []
        test_labels_seq = []

        for i, op in enumerate(self.operations):
            data = self.datasets_by_op[op]
            labels = self.labels_by_op[op]
            labels_seq = self.labels_seq_by_op[op]

            n = data.size(0)
            if n == 0:
                continue

            gen = torch.Generator().manual_seed(int(self.data_seed) + i)
            perm = torch.randperm(n, generator=gen)
            cutoff = int(n * self.train_frac)

            train_idx = perm[:cutoff]
            test_idx = perm[cutoff:]

            train_parts.append(data[train_idx])
            train_labels.append(labels[train_idx])
            train_labels_seq.append(labels_seq[train_idx])
            test_parts.append(data[test_idx])
            test_labels.append(labels[test_idx])
            test_labels_seq.append(labels_seq[test_idx])

        if len(train_parts) == 0:
            return super()._split_dataset()

        train_data = torch.cat(train_parts, dim=0)
        train_lbl = torch.cat(train_labels, dim=0)
        train_lbl_seq = torch.cat(train_labels_seq, dim=0)
        test_data = torch.cat(test_parts, dim=0)
        test_lbl = torch.cat(test_labels, dim=0)
        test_lbl_seq = torch.cat(test_labels_seq, dim=0)

        gen = torch.Generator().manual_seed(int(self.data_seed))
        train_perm = torch.randperm(train_data.size(0), generator=gen)
        test_perm = torch.randperm(test_data.size(0), generator=gen)

        train_data = train_data[train_perm]
        train_lbl = train_lbl[train_perm]
        train_lbl_seq = train_lbl_seq[train_perm]
        test_data = test_data[test_perm]
        test_lbl = test_lbl[test_perm]
        test_lbl_seq = test_lbl_seq[test_perm]

        return train_data, train_lbl, train_lbl_seq, test_data, test_lbl, test_lbl_seq

    def get_sample(self, idx=None, subset='train'):
        """Get a sample from the multi-operation dataset for inspection."""
        if subset == 'train':
            data, labels = self.train_data, self.train_labels
        elif subset == 'test':
            data, labels = self.test_data, self.test_labels
        else:
            data, labels = self.dataset, self.labels

        if idx is None or idx >= len(data):
            idx = torch.randint(0, len(data), (1,)).item()

        sample_data = data[idx]
        sample_label = labels[idx]

        # Decode the operation
        op_token = sample_data[self.op_positions[0]].item()
        op_index = op_token - self.first_op_token

        if 0 <= op_index < len(self.operations):
            operation = self.operations[op_index]
        else:
            operation = "unknown"

        op_symbols = {
            "addition": "+",
            "subtraction": "-",
            "multiplication": "*",
            "division": "÷"
        }
        op_symbol = op_symbols.get(operation, "?")

        # Display values (convert back to human-readable form)
        terms_display = [sample_data[pos].item() for pos in self.term_positions]
        result_display = sample_label.item()

        if not self.includes_zero and operation in ["multiplication", "division"]:
            terms_display = [x + 1 for x in terms_display]
            result_display += 1

        lhs = f" {op_symbol} ".join(str(x) for x in terms_display)

        return {
            'input': sample_data,
            'target': sample_label,
            'operation': operation,
            'equation': (
                f"{lhs} = {result_display} (mod {self.modulo})"
                if self.include_equals_token
                else f"{lhs} -> {result_display} (mod {self.modulo})"
            ),
            'vocab_info': {
                'includes_zero': self.includes_zero,
                'vocab_size': self.get_data_info()['vocab_size'],
                'equals_token': self.equals_token,
                'op_token': op_token
            }
        }

    def _op_token(self, operation):
        return self.op_token_for_index(self.operations.index(operation))

    def _op_filter(self, data, labels, labels_seq, operation):
        mask = (data[:, self.op_positions[0]] == self._op_token(operation))
        return data[mask], labels[mask], labels_seq[mask]

    def get_train_data(self, target='final', operation=None):
        if operation is None:
            return super().get_train_data(target=target)
        if operation not in self.operations:
            raise ValueError(f"No dataset found for operation: {operation}")
        data, lbl, lbl_seq = self._op_filter(
            self.train_data, self.train_labels, self.train_labels_seq, operation
        )
        if target == 'final':
            return data, lbl
        if target == 'seq':
            return data, lbl_seq
        raise ValueError(f"target must be 'final' or 'seq', got {target}")

    def get_test_data(self, target='final', operation=None):
        if operation is None:
            return super().get_test_data(target=target)
        if operation not in self.operations:
            raise ValueError(f"No dataset found for operation: {operation}")
        data, lbl, lbl_seq = self._op_filter(
            self.test_data, self.test_labels, self.test_labels_seq, operation
        )
        if target == 'final':
            return data, lbl
        if target == 'seq':
            return data, lbl_seq
        raise ValueError(f"target must be 'final' or 'seq', got {target}")

    def get_operation_counts(self):
        """Get count of samples per operation."""
        counts = {}
        for i, op in enumerate(self.operations):
            op_token = self.op_token_for_index(i)
            count = (self.dataset[:, self.op_positions[0]] == op_token).sum().item()
            counts[op] = count

        return counts

    def get_full_data(self, operation="all", target='final'):
        if operation == "all":
            if target == 'final':
                return self.dataset, self.labels
            if target == 'seq':
                return self.dataset, self.labels_seq
            raise ValueError(f"target must be 'final' or 'seq', got {target}")

        if operation not in self.operations or operation not in self.datasets_by_op:
            raise ValueError(f"No dataset found for the following operation - {operation}")

        if target == 'final':
            return self.datasets_by_op[operation], self.labels_by_op[operation]
        if target == 'seq':
            return self.datasets_by_op[operation], self.labels_seq_by_op[operation]
        raise ValueError(f"target must be 'final' or 'seq', got {target}")
