import torch
import math
from training.utils import align_running_targets_to_sequence
from .base import CustomDataset


class DivisionModDataset(CustomDataset):
    def __init__(self, config, operations=["division"]):
        if len(operations) != 1 or operations[0] != "division":
            raise ValueError("This dataset is only supposed to be used for the following operation - Division")

        super().__init__(config, operations)
        self.dataset, self.labels, self.labels_seq = self._generate_dataset()
        self.train_data, self.train_labels, self.train_labels_seq, self.test_data, self.test_labels, self.test_labels_seq = self._split_dataset()

    def _gcd(self, a, b):
        """Calculate Greatest Common Divisor using Euclidean algorithm."""
        while b:
            a, b = b, a % b
        return a

    def _mod_inverse(self, a, m):
        """Calculate modular multiplicative inverse using Extended Euclidean Algorithm."""
        if self._gcd(a, m) != 1:
            raise ValueError(f"Modular inverse of {a} mod {m} does not exist")

        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, y = extended_gcd(a, m)
        return (x % m + m) % m

    def _generate_dataset(self):
        """Generate left-associative modular division expressions with nterms.

        For nterms=3: a / b / c = ((a * b^{-1}) * c^{-1}) mod p
        """
        token_offset = 0 if self.includes_zero else 1
        op_token = torch.tensor(self.op_token_for_index(0) + token_offset)
        equals_token = (
            torch.tensor(self.equals_token + token_offset)
            if self.include_equals_token
            else None
        )

        valid_divisors = torch.tensor([b for b in range(1, self.modulo) if math.gcd(b, self.modulo) == 1], dtype=torch.long)
        inverse_cache = {int(b): pow(int(b), -1, self.modulo) for b in valid_divisors.tolist()}

        numerator_terms = torch.arange(1, self.modulo, dtype=torch.long)
        domains = [numerator_terms] + [valid_divisors for _ in range(self.nterms - 1)]
        terms = torch.cartesian_prod(*domains)

        dataset = self._build_sequence(terms, op_token, equals_token)

        running = torch.empty_like(terms)
        running[:, 0] = terms[:, 0]
        for i in range(1, self.nterms):
            inv_i = torch.tensor([inverse_cache[int(x)] for x in terms[:, i].tolist()], dtype=torch.long)
            running[:, i] = (running[:, i - 1] * inv_i) % self.modulo

        dataset -= 1
        labels = running[:, -1] - 1
        labels_seq = align_running_targets_to_sequence(
            running - 1,
            include_equals_token=self.include_equals_token,
        )

        return dataset, labels, labels_seq
