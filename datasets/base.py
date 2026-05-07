import torch
from torch.utils.data import Dataset
from abc import abstractmethod


class CustomDataset(Dataset):
    def __init__(self, config: dict, operations: list):
        super().__init__()
        """
        Dataset is shaped as [t1 <op> t2 <op> ... <op> tn (= optional)] -> [c],
        where n = nterms and c is the modular result.
        """
        # config
        self.config = config
        self.operations = operations
        self.modulo = self.config['modulo']
        self.nterms = int(self.config.get('nterms', 2))
        self.train_frac = self.config['train_frac']
        self.data_seed = self.config['data_seed']
        self.include_equals_token = bool(self.config.get('include_equals_token', True))

        if self.nterms < 2:
            raise ValueError(f"nterms must be at least 2, got {self.nterms}")

        # dataset
        self.train_data, self.test_data = None, None
        self.train_labels, self.test_labels = None, None
        self.train_labels_seq, self.test_labels_seq = None, None
        self.dataset, self.labels, self.labels_seq = None, None, None
        self.includes_zero = any(op in ["addition", "subtraction"] for op in self.operations)
        base_term_vocab = self.modulo if self.includes_zero else (self.modulo - 1)
        extra_equals = 1 if self.include_equals_token else 0
        self.vocab_size = base_term_vocab + extra_equals + len(self.operations)

        # validate operations
        for op in operations:
            if op not in {"addition", "subtraction", "multiplication", "division"}:
                raise ValueError(f"Unsupported operation: {op}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    @abstractmethod
    def _generate_dataset(self):
        """Implement accordingly for each operation(s)"""
        pass

    def _grid_terms(self, include_zero=True):
        """Return all nterms-tuples of operands as [num_samples, nterms]."""
        if include_zero:
            rng = torch.arange(self.modulo)
        else:
            rng = torch.arange(1, self.modulo)

        domains = [rng for _ in range(self.nterms)]
        return torch.cartesian_prod(*domains)

    @property
    def special_token_base(self):
        """Token index immediately after term tokens in canonical token space."""
        return self.modulo if self.includes_zero else self.modulo - 1

    @property
    def equals_token(self):
        if not self.include_equals_token:
            return None
        return self.special_token_base

    @property
    def first_op_token(self):
        return self.special_token_base + (1 if self.include_equals_token else 0)

    def op_token_for_index(self, op_index: int) -> int:
        return self.first_op_token + int(op_index)

    def _build_sequence(self, terms, op_token, equals_token=None):
        """Interleave terms and repeated op tokens, optionally append equals.

        terms: [N, nterms]
        returns: [N, 2*nterms] when include_equals_token else [N, 2*nterms-1]
        """
        seq_len = (2 * self.nterms) if self.include_equals_token else (2 * self.nterms - 1)
        dataset = torch.empty((terms.shape[0], seq_len), dtype=torch.long)
        dataset[:, 0:seq_len:2] = terms
        dataset[:, 1:seq_len-1:2] = op_token
        if self.include_equals_token:
            if equals_token is None:
                raise ValueError("equals_token must be provided when include_equals_token=True")
            dataset[:, -1] = equals_token
        return dataset

    @property
    def term_positions(self):
        seq_len = (2 * self.nterms) if self.include_equals_token else (2 * self.nterms - 1)
        return list(range(0, seq_len, 2))

    @property
    def op_positions(self):
        seq_len = (2 * self.nterms) if self.include_equals_token else (2 * self.nterms - 1)
        return list(range(1, seq_len - 1, 2))

    def _split_dataset(self):
        """Split dataset into train and test sets."""
        torch.manual_seed(self.data_seed)

        total_samples = len(self.dataset)
        indices = torch.randperm(total_samples)
        cutoff = int(total_samples * self.train_frac)

        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]

        train_data = self.dataset[train_indices]
        train_labels = self.labels[train_indices]
        train_labels_seq = self.labels_seq[train_indices]
        test_data = self.dataset[test_indices]
        test_labels = self.labels[test_indices]
        test_labels_seq = self.labels_seq[test_indices]

        return train_data, train_labels, train_labels_seq, test_data, test_labels, test_labels_seq

    def to_device(self, device):
        """Set device for training 'cuda' or 'cpu'

        During training, it is expected to move each batch to the GPU
        accordingly rather than allocating memory in GPU for entire dataset.

        """
        self.device = device
        return self

    def get_train_data(self, target='final', operation=None):
        """Get training data and labels."""
        if operation is not None and operation not in self.operations:
            raise ValueError(f"No dataset found for operation: {operation}")
        if target == 'final':
            return self.train_data, self.train_labels
        if target == 'seq':
            return self.train_data, self.train_labels_seq
        raise ValueError(f"target must be 'final' or 'seq', got {target}")

    def get_test_data(self, target='final', operation=None):
        """Get test data and labels."""
        if operation is not None and operation not in self.operations:
            raise ValueError(f"No dataset found for operation: {operation}")
        if target == 'final':
            return self.test_data, self.test_labels
        if target == 'seq':
            return self.test_data, self.test_labels_seq
        raise ValueError(f"target must be 'final' or 'seq', got {target}")

    def get_full_data(self, operation, target='final'):
        """Get complete dataset and labels."""
        if operation not in self.operations:
            raise ValueError(f"No dataset found for the following operation - {operation}")

        if target == 'final':
            return self.dataset, self.labels
        if target == 'seq':
            return self.dataset, self.labels_seq
        raise ValueError(f"target must be 'final' or 'seq', got {target}")

    def get_data_info(self):
        """Get information about the dataset."""
        return {
            'modulo': self.modulo,
            'nterms': self.nterms,
            'operations': self.operations,
            'include_equals_token': self.include_equals_token,
            'total_samples': len(self.dataset),
            'train_samples': len(self.train_data),
            'test_samples': len(self.test_data),
            'train_frac': self.train_frac,
            'vocab_size': self.vocab_size
        }

    def get_sample(self, idx=None, subset='train'):
        """Get a sample from the dataset for inspection."""
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

        op_token = sample_data[self.op_positions[0]].item()
        op_index = op_token - self.first_op_token
        operation = self.operations[op_index] if 0 <= op_index < len(self.operations) else "unkown"

        op_symbols = {
            "addition": "+",
            "subtraction": "-",
            "multiplication": "*",
            "division": "÷"
        }
        op_symbol = op_symbols.get(operation, "?")

        terms_display = [sample_data[pos].item() for pos in self.term_positions]
        result_display = sample_label.item()

        if op_symbol in ("*", "÷"):
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
            )
        }
