"""Fourier regularization for modular arithmetic models.

Penalizes embedding and output-projection weights in the Fourier basis to
steer grokking toward sparse, interpretable frequency representations.

Adding a new regularization mode
---------------------------------
1. Add a new ``case N:`` block inside ``fourier_regularization_term``.
2. If it needs a new helper, define it as a module-level ``_`` function below
   the existing helpers.
3. Update ``losses/__init__.py`` if you want to expose new public symbols.
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Literal

from analysis.analysis_utils import create_fourier_analysis_setup, find_smallest_primitive_root


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class _Cache:
    primitive_roots: dict[int, int] = field(default_factory=dict)
    permutations: dict[tuple, torch.Tensor] = field(default_factory=dict)
    fourier_bases: dict[tuple, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.primitive_roots.clear()
        self.permutations.clear()
        self.fourier_bases.clear()


# Per-process singleton — Python's sys.modules guarantees one instance.
_cache = _Cache()


def clear_caches() -> None:
    """Clear all cached Fourier tensors and primitive roots. Useful in tests."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Cache accessors
# ---------------------------------------------------------------------------

def get_cached_primitive_root(modulo: int) -> int:
    if modulo not in _cache.primitive_roots:
        root = find_smallest_primitive_root(modulo)
        if root is None:
            raise ValueError(
                f"Could not find primitive root for modulo {modulo}. "
                "This usually indicates a non-prime modulus, which is unsupported "
                "for multiplicative Fourier regularization."
            )
        _cache.primitive_roots[modulo] = root
    return _cache.primitive_roots[modulo]


def get_cached_permutation(modulo: int, device: str | torch.device) -> torch.Tensor:
    key = (modulo, str(device))
    if key not in _cache.permutations:
        generator = get_cached_primitive_root(modulo)
        perm = torch.tensor(
            [pow(generator, i, modulo) for i in range(1, modulo)], device=device
        )
        expected = torch.arange(1, modulo, device=device)
        if perm.numel() != (modulo - 1) or torch.unique(perm).numel() != (modulo - 1):
            raise ValueError(
                "Primitive-root permutation is not a bijection over non-zero residues: "
                f"modulo={modulo}, generator={generator}."
            )
        if (
            torch.any(perm < 1)
            or torch.any(perm >= modulo)
            or not torch.equal(torch.sort(perm).values, expected)
        ):
            raise ValueError(
                "Primitive-root permutation contains out-of-range values: "
                f"modulo={modulo}, generator={generator}."
            )
        _cache.permutations[key] = perm
    return _cache.permutations[key]


def get_cached_fourier_basis(
    modulo: int, includes_zero: bool, device: str | torch.device
) -> torch.Tensor:
    key = (modulo, includes_zero, str(device))
    if key not in _cache.fourier_bases:
        basis, _ = create_fourier_analysis_setup(modulo, includes_zero=includes_zero)
        _cache.fourier_bases[key] = basis.to(device)
    return _cache.fourier_bases[key]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grouped_l2_norm(tensor: torch.Tensor, num_groups: int) -> torch.Tensor:
    """Group-lasso style norm over the last dimension."""
    groups = torch.chunk(tensor, num_groups, dim=-1)
    return torch.stack([torch.norm(g, p=2) for g in groups]).sum()


def _split_regularization_term(
    w_embedding: torch.Tensor,
    w_fc: torch.Tensor,
    fourier_basis: torch.Tensor,
    modulo: int,
    device: str | torch.device,
    current_op_group: Literal["additive", "multiplicative"],
) -> torch.Tensor:
    """Split-embedding regularization used by mode 1 with mixed additive/multiplicative ops.

    First half of the embedding dim → additive group (canonical residue order).
    Second half → multiplicative group (primitive-root permutation applied so
    multiplication maps to addition in discrete-log space).
    """
    if w_embedding.shape[-1] < 2:
        raise ValueError(
            "Split-embedding regularization requires embedding/unembedding feature dimension >= 2, "
            f"got {w_embedding.shape[-1]}."
        )

    split_idx = w_embedding.shape[-1] // 2
    if split_idx == 0 or split_idx == w_embedding.shape[-1]:
        raise ValueError(
            "Split-embedding regularization produced an empty half. Increase embedding/unembedding feature dimension."
        )

    if current_op_group == "additive":
        return torch.norm(fourier_basis @ w_embedding[:, :split_idx], p=1) + torch.norm(
            fourier_basis @ w_fc[:, :split_idx], p=1
        )

    if current_op_group == "multiplicative":
        perm = get_cached_permutation(modulo, device)
        w_emb_mult = w_embedding[perm - 1, split_idx:]
        w_fc_mult = w_fc[perm - 1, split_idx:]
        return torch.norm(fourier_basis @ w_emb_mult, p=1) + torch.norm(
            fourier_basis @ w_fc_mult, p=1
        )

    raise ValueError(
        f"current_op_group must be 'additive' or 'multiplicative', got {current_op_group!r}."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fourier_regularization_term(
    model: object,
    mode: int,
    modulo: int,
    includes_zero: bool,
    is_multiplicative: bool,
    num_groups: int = 2,
    has_mixed_ops: bool = False,
) -> torch.Tensor:
    """Compute Fourier regularization term only (no base CE loss).

    Args:
        model: Must expose ``model.embedding.weight`` and ``model.fc.weight``.
    mode: Regularization variant 1–7; see module docstring table.
        modulo: Prime modulus *p*.
        includes_zero: Whether 0 is a valid operand (True for add/sub, False for mul/div).
        is_multiplicative: Whether the current operation batch uses the multiplicative group.
        num_groups: Number of chunks for group-lasso modes 5 and 6.
        has_mixed_ops: Mode-1 only. When True, uses split-embedding logic for mixed additive/multiplicative ops.
    """
    device = model.embedding.weight.device
    fourier_basis = get_cached_fourier_basis(
        modulo, includes_zero=(not is_multiplicative), device=device
    )

    if is_multiplicative and includes_zero:
        slc = slice(1, modulo)
    else:
        limit = modulo if includes_zero else modulo - 1
        slc = slice(0, limit)

    w_embedding = model.embedding.weight[slc, :]
    w_fc = model.fc.weight[slc, :]

    # Apply primitive-root permutation for multiplicative ops (except mode 1 with
    # has_mixed_ops, which handles permutation internally via _split_regularization_term).
    if is_multiplicative and not (mode == 1 and has_mixed_ops):
        perm = get_cached_permutation(modulo, device)
        w_embedding = w_embedding[perm - 1]
        w_fc = w_fc[perm - 1]

    match mode:
        case 1:
            if has_mixed_ops:
                current_op_group = "multiplicative" if is_multiplicative else "additive"
                reg_term = _split_regularization_term(
                    w_embedding=w_embedding,
                    w_fc=w_fc,
                    fourier_basis=fourier_basis,
                    modulo=modulo,
                    device=device,
                    current_op_group=current_op_group,
                )
            else:
                reg_term = torch.norm(fourier_basis @ w_embedding, p=1) + torch.norm(
                    fourier_basis @ w_fc, p=1
                )
        case 2:
            reg_term = torch.norm(fourier_basis @ w_embedding, p=1) + torch.norm(
                fourier_basis @ model.s4_layer.Ct[slc, :].float(), p=1
            )
        case 3:
            reg_term = (
                torch.norm(fourier_basis @ w_embedding, p=1)
                + torch.norm(model.rnn.weight_ih_l0, p=2)
                + torch.norm(model.rnn.weight_hh_l0, p=2)
            )
        case 4:
            reg_term = (
                torch.norm(fourier_basis @ (w_embedding + w_fc), p=1)
                + torch.norm(model.rnn.weight_ih_l0, p=2)
                + torch.norm(model.rnn.weight_hh_l0, p=2)
            )
        case 5:
            fw_embedding = fourier_basis @ w_embedding
            fw_fc = fourier_basis @ w_fc
            reg_term = _grouped_l2_norm(fw_embedding, num_groups) + _grouped_l2_norm(
                fw_fc, num_groups
            )
        case 6:
            fw_embedding = fourier_basis @ w_embedding
            reg_term = _grouped_l2_norm(fw_embedding, num_groups) + torch.norm(
                fourier_basis @ w_fc, p=1
            )
        case 7:
            # Mode 7: always use full embedding dimensions (no additive/multiplicative split).
            reg_term = torch.norm(fourier_basis @ w_embedding, p=1) + torch.norm(
                fourier_basis @ w_fc, p=1
            )
        case _:
            raise ValueError(f"Unsupported fourier regularization mode: {mode}")

    return reg_term
