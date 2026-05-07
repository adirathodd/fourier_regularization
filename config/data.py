from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict, field


@dataclass
class DataConfig:
    """Configuration for dataset generation."""
    modulo: int = 113
    nterms: int = 2
    nterms_list: Optional[List[int]] = None
    mixed_sequence_operations: bool = False
    include_equals_token: bool = True
    train_frac: float = 0.3
    data_seed: int = 598
    vocab_size: Optional[int] = None
    # includes_zero is always computed from operations in Config.__init__, never persisted
    includes_zero: bool = field(default=True, init=True, repr=True)

    def __post_init__(self):
        if self.modulo <= 0:
            raise ValueError(f"modulo must be positive, got {self.modulo}")
        if not isinstance(self.nterms, int):
            raise ValueError(f"nterms must be an integer, got {type(self.nterms).__name__}")
        if self.nterms < 2:
            raise ValueError(f"nterms must be at least 2, got {self.nterms}")
        if self.nterms_list is not None:
            if not isinstance(self.nterms_list, list) or len(self.nterms_list) == 0:
                raise ValueError("nterms_list must be a non-empty list of integers when provided")
            cleaned = []
            for n in self.nterms_list:
                if not isinstance(n, int):
                    raise ValueError(f"nterms_list entries must be integers, got {type(n).__name__}")
                if n < 2:
                    raise ValueError(f"nterms_list entries must be at least 2, got {n}")
                cleaned.append(int(n))
            self.nterms_list = sorted(set(cleaned))
            self.nterms = max(self.nterms_list)
        if not 0 < self.train_frac < 1:
            raise ValueError(f"train_frac must be between 0 and 1, got {self.train_frac}")
        if self.data_seed < 0:
            raise ValueError(f"data_seed must be non-negative, got {self.data_seed}")
        if not isinstance(self.include_equals_token, bool):
            raise ValueError(
                f"include_equals_token must be a bool, got {type(self.include_equals_token).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('includes_zero', None)  # derived from operations; never persisted
        return d

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        config_dict = {k: v for k, v in config_dict.items() if k != 'includes_zero'}
        return cls(**config_dict)
