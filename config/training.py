from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 10000
    lr: float = 0.001
    wd: float = 5e-5
    beta1: float = 0.9
    beta2: float = 0.99
    use_scheduler: bool = True
    lr_decay_interval: int = 5000
    lr_decay: float = 0.1
    save_every: int = 1000
    save_dir: str = "rnn_modular_addition"
    batch_mode: str = "operation-batch"
    batch_size: Optional[int] = 512
    training_target: str = "last_token"
    cot_prefix_weight: float = 1.0
    cot_final_weight: float = 1.0
    fourier_reg_mode: Optional[int] = None
    fourier_reg_coefficient: float = 0.0
    fourier_reg_groups: int = 2
    dataloader_num_workers: int = 0

    @property
    def betas(self) -> Tuple[float, float]:
        return (self.beta1, self.beta2)

    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.wd < 0:
            raise ValueError(f"wd must be non-negative, got {self.wd}")
        if not 0 <= self.beta1 < 1:
            raise ValueError(f"beta1 must be in [0, 1), got {self.beta1}")
        if not 0 <= self.beta2 < 1:
            raise ValueError(f"beta2 must be in [0, 1), got {self.beta2}")
        if self.lr_decay_interval <= 0:
            raise ValueError(f"lr_decay_interval must be positive, got {self.lr_decay_interval}")
        if not 0 < self.lr_decay <= 1:
            raise ValueError(f"lr_decay must be in (0, 1], got {self.lr_decay}")
        if self.save_every < 0:
            raise ValueError(f"save_every must be non-negative, got {self.save_every}")
        if self.batch_mode not in ['full', 'mini-batch', 'operation-batch']:
            raise ValueError(f"batch_mode must be 'full', 'mini-batch', or 'operation-batch', got {self.batch_mode}")
        if self.batch_mode == 'full':
            self.batch_size = None
        if self.batch_mode in ['mini-batch', 'operation-batch'] and self.batch_size is None:
            raise ValueError("batch_size must be specified for mini-batch and operation-batch modes")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.training_target not in ['last_token', 'seq_cot']:
            raise ValueError(f"training_target must be 'last_token' or 'seq_cot', got {self.training_target}")
        if self.cot_prefix_weight <= 0:
            raise ValueError(f"cot_prefix_weight must be positive, got {self.cot_prefix_weight}")
        if self.cot_final_weight <= 0:
            raise ValueError(f"cot_final_weight must be positive, got {self.cot_final_weight}")
        if self.fourier_reg_mode is not None and self.fourier_reg_mode not in [1, 2, 3, 4, 5, 6, 7]:
            raise ValueError(f"fourier_reg_mode must be one of [1, 2, 3, 4, 5, 6, 7], got {self.fourier_reg_mode}")
        if self.fourier_reg_coefficient < 0:
            raise ValueError(f"fourier_reg_coefficient must be non-negative, got {self.fourier_reg_coefficient}")
        if not isinstance(self.fourier_reg_groups, int):
            raise ValueError(f"fourier_reg_groups must be an int, got {type(self.fourier_reg_groups).__name__}")
        if self.fourier_reg_groups <= 0:
            raise ValueError(f"fourier_reg_groups must be a positive integer, got {self.fourier_reg_groups}")
        if self.dataloader_num_workers < 0:
            raise ValueError(f"dataloader_num_workers must be non-negative, got {self.dataloader_num_workers}")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['betas'] = list(self.betas)
        return data

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        config_dict = config_dict.copy()
        if 'betas' in config_dict:
            betas = config_dict.pop('betas')
            if isinstance(betas, (list, tuple)) and len(betas) == 2:
                config_dict['beta1'] = betas[0]
                config_dict['beta2'] = betas[1]
        return cls(**config_dict)
