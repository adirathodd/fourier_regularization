from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class RNNModelConfig:
    """Configuration for RNN models."""
    model_type: str = "rnn"
    hidden_dim: int = 256
    num_layers: int = 1
    vocab_size: Optional[int] = None

    def __post_init__(self):
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {self.num_layers}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RNNModelConfig':
        return cls(**config_dict)
