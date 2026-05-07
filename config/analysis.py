from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


@dataclass
class AnalysisConfig:
    """Configuration for analysis-time thresholding and heuristics."""

    fourier_norm_threshold: float = 0.1
    freq_threshold_top_fraction: float = 0.1
    freq_threshold_min_components: int = 4
    freq_threshold_fixed: Optional[float] = None

    def __post_init__(self):
        if self.fourier_norm_threshold < 0:
            raise ValueError(
                f"fourier_norm_threshold must be non-negative, got {self.fourier_norm_threshold}"
            )
        if not 0 < self.freq_threshold_top_fraction <= 1:
            raise ValueError(
                f"freq_threshold_top_fraction must be in (0, 1], got {self.freq_threshold_top_fraction}"
            )
        if self.freq_threshold_min_components <= 0:
            raise ValueError(
                f"freq_threshold_min_components must be positive, got {self.freq_threshold_min_components}"
            )
        if self.freq_threshold_fixed is not None and self.freq_threshold_fixed < 0:
            raise ValueError(
                f"freq_threshold_fixed must be non-negative when set, got {self.freq_threshold_fixed}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AnalysisConfig':
        return cls(**config_dict)
