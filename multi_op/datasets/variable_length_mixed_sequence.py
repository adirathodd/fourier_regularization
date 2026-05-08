import random

import torch

from .base import CustomDataset
from .addition import AdditionModDataset
from .subtraction import SubtractionModDataset
from .multiplication import MultiplicationModDataset
from .division import DivisionModDataset
from .multi_operation import MultiOperationDataset
from .mixed_sequence_operation import MixedSequenceOperationDataset


class VariableLengthMixedSequenceDataset(CustomDataset):
    """True variable-length dataset (no synthetic upcast terms)."""

    def __init__(self, config, operations):
        nterms_list = config.get("nterms_list")
        if not nterms_list:
            raise ValueError("VariableLengthMixedSequenceDataset requires data.nterms_list")

        cfg = dict(config)
        cfg["nterms"] = max(int(n) for n in nterms_list)
        super().__init__(cfg, operations)
        self.mixed_sequence_operations = bool(config.get("mixed_sequence_operations", False))
        self.mixed_ops_within_sequence = self.mixed_sequence_operations
        self.is_variable_length = True
        self.nterms_list = sorted(set(int(n) for n in nterms_list))

        self.by_nterms = {}
        self.total_samples = 0
        self.train_samples = 0
        self.test_samples = 0

        single_op_classes = {
            "addition": AdditionModDataset,
            "subtraction": SubtractionModDataset,
            "multiplication": MultiplicationModDataset,
            "division": DivisionModDataset,
        }

        for nterms in self.nterms_list:
            cfg_n = dict(config)
            cfg_n["nterms"] = int(nterms)
            if len(self.operations) == 1:
                op = self.operations[0]
                ds_n = single_op_classes[op](cfg_n, [op])
            elif self.mixed_sequence_operations:
                ds_n = MixedSequenceOperationDataset(cfg_n, self.operations)
            else:
                ds_n = MultiOperationDataset(cfg_n, self.operations)
            self.by_nterms[int(nterms)] = {
                "dataset": ds_n.dataset,
                "labels": ds_n.labels,
                "labels_seq": ds_n.labels_seq,
                "train_data": ds_n.train_data,
                "train_labels": ds_n.train_labels,
                "train_labels_seq": ds_n.train_labels_seq,
                "test_data": ds_n.test_data,
                "test_labels": ds_n.test_labels,
                "test_labels_seq": ds_n.test_labels_seq,
            }
            self.total_samples += int(ds_n.dataset.shape[0])
            self.train_samples += int(ds_n.train_data.shape[0])
            self.test_samples += int(ds_n.test_data.shape[0])

        # Expose per-length tensors as lists for compatibility with analysis
        # utilities that can iterate across variable-length shards.
        self.dataset = [self.by_nterms[n]["dataset"] for n in self.nterms_list]
        self.labels = [self.by_nterms[n]["labels"] for n in self.nterms_list]
        self.labels_seq = [self.by_nterms[n]["labels_seq"] for n in self.nterms_list]
        self.train_data = [self.by_nterms[n]["train_data"] for n in self.nterms_list]
        self.train_labels = [self.by_nterms[n]["train_labels"] for n in self.nterms_list]
        self.train_labels_seq = [self.by_nterms[n]["train_labels_seq"] for n in self.nterms_list]
        self.test_data = [self.by_nterms[n]["test_data"] for n in self.nterms_list]
        self.test_labels = [self.by_nterms[n]["test_labels"] for n in self.nterms_list]
        self.test_labels_seq = [self.by_nterms[n]["test_labels_seq"] for n in self.nterms_list]

    def _generate_dataset(self):
        raise NotImplementedError("VariableLengthMixedSequenceDataset uses per-length sub-datasets")

    def _split_dataset(self):
        raise NotImplementedError("VariableLengthMixedSequenceDataset uses per-length sub-datasets")

    def __len__(self):
        return self.total_samples

    def get_data_info(self):
        return {
            "modulo": self.modulo,
            "nterms": self.nterms,
            "nterms_list": list(self.nterms_list),
            "mixed_sequence_operations": self.mixed_sequence_operations,
            "include_equals_token": self.include_equals_token,
            "operations": self.operations,
            "total_samples": self.total_samples,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "train_frac": self.train_frac,
            "vocab_size": self.vocab_size,
        }

    def get_train_splits(self, target="final"):
        out = {}
        for nterms, split in self.by_nterms.items():
            if target == "final":
                out[nterms] = (split["train_data"], split["train_labels"])
            elif target == "seq":
                out[nterms] = (split["train_data"], split["train_labels_seq"])
            else:
                raise ValueError(f"target must be 'final' or 'seq', got {target}")
        return out

    def get_test_splits(self, target="final"):
        out = {}
        for nterms, split in self.by_nterms.items():
            if target == "final":
                out[nterms] = (split["test_data"], split["test_labels"])
            elif target == "seq":
                out[nterms] = (split["test_data"], split["test_labels_seq"])
            else:
                raise ValueError(f"target must be 'final' or 'seq', got {target}")
        return out

    def get_train_data(self, target="final", operation=None):
        raise NotImplementedError("Use get_train_splits() for variable-length datasets")

    def get_test_data(self, target="final", operation=None):
        raise NotImplementedError("Use get_test_splits() for variable-length datasets")

    def get_full_data(self, operation, target="final"):
        raise NotImplementedError("Use per-length splits for variable-length datasets")

    def _decode_equation(self, sample_data):
        op_symbols = {
            "addition": "+",
            "subtraction": "-",
            "multiplication": "*",
            "division": "÷",
        }
        token_to_op = {self.op_token_for_index(i): op for i, op in enumerate(self.operations)}

        terms = []
        ops = []
        for i in range(0, sample_data.shape[0], 2):
            terms.append(int(sample_data[i].item()))
            if i + 1 < sample_data.shape[0]:
                op_name = token_to_op.get(int(sample_data[i + 1].item()), "unknown")
                ops.append(op_symbols.get(op_name, "?"))

        lhs = str(terms[0])
        for i, op in enumerate(ops):
            lhs += f" {op} {terms[i + 1]}"
        return lhs

    def get_sample(self, idx=None, subset="train"):
        if subset not in {"train", "test", "all"}:
            raise ValueError(f"subset must be 'train', 'test', or 'all', got {subset}")

        candidates = []
        for nterms, split in self.by_nterms.items():
            if subset == "train":
                size = int(split["train_data"].shape[0])
            elif subset == "test":
                size = int(split["test_data"].shape[0])
            else:
                size = int(split["dataset"].shape[0])
            if size > 0:
                candidates.append((nterms, size))

        if not candidates:
            raise ValueError("No samples available")

        if idx is None:
            nterms = random.choice(candidates)[0]
            inner_idx = random.randrange(next(s for n, s in candidates if n == nterms))
        else:
            total = sum(s for _, s in candidates)
            if idx < 0 or idx >= total:
                raise IndexError(f"idx out of range: {idx} (total {total})")
            running = idx
            nterms = candidates[0][0]
            inner_idx = 0
            for nt, size in candidates:
                if running < size:
                    nterms = nt
                    inner_idx = running
                    break
                running -= size

        split = self.by_nterms[nterms]
        if subset == "train":
            sample_data = split["train_data"][inner_idx]
            sample_label = split["train_labels"][inner_idx]
        elif subset == "test":
            sample_data = split["test_data"][inner_idx]
            sample_label = split["test_labels"][inner_idx]
        else:
            sample_data = split["dataset"][inner_idx]
            sample_label = split["labels"][inner_idx]

        lhs = self._decode_equation(sample_data)
        if len(self.operations) == 1:
            op_name = self.operations[0]
        elif self.mixed_sequence_operations:
            op_name = "mixed-sequence"
        else:
            op_name = "multi-operation"
        return {
            "input": sample_data,
            "target": sample_label,
            "operation": op_name,
            "equation": (
                f"{lhs} = {int(sample_label.item())} (mod {self.modulo})"
                if self.include_equals_token
                else f"{lhs} -> {int(sample_label.item())} (mod {self.modulo})"
            ),
            "nterms": int(nterms),
        }
