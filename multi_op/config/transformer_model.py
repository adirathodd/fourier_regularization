from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TransformerModelConfig:
    """Configuration for Transformer models."""
    model_type: str = "transformer"
    vocab_size: Optional[int] = None
    embedding_dim: int = 256
    n_heads: int = 1
    n_layers: int = 1
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_len: int = 128
    fixed_positional_encoding: bool = False
    use_rope: bool = False
    rope_base: float = 10000.0
    activation: str = "relu"
    norm_first: bool = False
    mask: bool = False
    tie_weights: bool = False

    def __post_init__(self):
        if self.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.embedding_dim}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if self.embedding_dim % self.n_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by n_heads ({self.n_heads})")
        if self.dim_feedforward <= 0:
            raise ValueError(f"dim_feedforward must be positive, got {self.dim_feedforward}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.activation not in ['relu', 'gelu']:
            raise ValueError(f"activation must be 'relu' or 'gelu', got {self.activation}")
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        if self.rope_base <= 0:
            raise ValueError(f"rope_base must be positive, got {self.rope_base}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TransformerModelConfig':
        return cls(**config_dict)
