import os
import yaml
from typing import List, Dict, Optional, Union, Any

from .data import DataConfig
from .analysis import AnalysisConfig
from .rnn_model import RNNModelConfig
from .transformer_model import TransformerModelConfig
from .training import TrainingConfig


class Config:
    """Main configuration class for model training."""

    @staticmethod
    def _default_fixed_freq_threshold(model_type: str) -> float:
        return 0.1 if model_type == "transformer" else 1.5

    def __init__(
        self,
        operations: List[str],
        model_type: str = "rnn",
        data_config: Optional[Union[DataConfig, Dict]] = None,
        analysis_config: Optional[Union[AnalysisConfig, Dict]] = None,
        model_config: Optional[Union[RNNModelConfig, TransformerModelConfig, Dict]] = None,
        training_config: Optional[Union[TrainingConfig, Dict]] = None,
    ):
        self.operations = operations
        self.model_type = model_type.lower()

        valid_ops = {'addition', 'subtraction', 'multiplication', 'division'}
        for op in operations:
            if op not in valid_ops:
                raise ValueError(f"Invalid operation: {op}. Must be one of {valid_ops}")

        if self.model_type not in ['rnn', 'transformer']:
            raise ValueError(f"model_type must be 'rnn' or 'transformer', got {self.model_type}")

        includes_zero = any(op in ["addition", "subtraction"] for op in operations)

        if data_config is None:
            self.data = DataConfig(includes_zero=includes_zero)
        elif isinstance(data_config, dict):
            data_config['includes_zero'] = includes_zero
            self.data = DataConfig.from_dict(data_config)
        else:
            self.data = data_config
            self.data.includes_zero = includes_zero

        base_term_vocab = self.data.modulo if includes_zero else (self.data.modulo - 1)
        extra_equals = 1 if self.data.include_equals_token else 0
        self.data.vocab_size = base_term_vocab + extra_equals + len(operations)

        if analysis_config is None:
            self.analysis = AnalysisConfig(
                freq_threshold_fixed=self._default_fixed_freq_threshold(self.model_type)
            )
        elif isinstance(analysis_config, dict):
            self.analysis = AnalysisConfig.from_dict(analysis_config)
        else:
            self.analysis = analysis_config

        if model_config is None:
            if self.model_type == "rnn":
                self.model = RNNModelConfig(vocab_size=self.data.vocab_size)
            else:
                self.model = TransformerModelConfig(vocab_size=self.data.vocab_size)
        elif isinstance(model_config, dict):
            model_config['vocab_size'] = self.data.vocab_size
            if self.model_type == "rnn":
                self.model = RNNModelConfig.from_dict(model_config)
            else:
                self.model = TransformerModelConfig.from_dict(model_config)
        else:
            self.model = model_config
            self.model.vocab_size = self.data.vocab_size

        if training_config is None:
            save_dir = self._generate_save_dir()
            if self.model_type == 'transformer':
                self.training = TrainingConfig(save_dir=save_dir, lr=1.0e-3)
            else:
                self.training = TrainingConfig(save_dir=save_dir)
        elif isinstance(training_config, dict):
            if 'save_dir' not in training_config:
                training_config['save_dir'] = self._generate_save_dir()
            self.training = TrainingConfig.from_dict(training_config)
        else:
            self.training = training_config

        if not getattr(self, '_skip_validation', False):
            self._validate_fourier_group_settings()
            self._validate_mode1_mixed_split_settings()
            self._validate_sequence_training_settings()
            self._validate_dataset_batch_mode_settings()
            self._validate_fourier_regularization_requirements()
            self._validate_rnn_only_modes()

    @staticmethod
    def _is_prime(n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    def _validate_fourier_group_settings(self):
        if self.training.fourier_reg_mode not in [5, 6]:
            return
        feature_dim = self.model.embedding_dim if self.model_type == "transformer" else self.model.hidden_dim
        if self.training.fourier_reg_groups > feature_dim:
            raise ValueError(
                f"fourier_reg_groups ({self.training.fourier_reg_groups}) cannot exceed "
                f"model embedding dimension ({feature_dim}) for mode {self.training.fourier_reg_mode}"
            )

    def _validate_mode1_mixed_split_settings(self):
        if self.training.fourier_reg_mode != 1:
            return
        has_additive = any(op in {'addition', 'subtraction'} for op in self.operations)
        has_multiplicative = any(op in {'multiplication', 'division'} for op in self.operations)
        if not (has_additive and has_multiplicative):
            return
        feature_dim = self.model.embedding_dim if self.model_type == "transformer" else self.model.hidden_dim
        if feature_dim < 2:
            raise ValueError(
                f"fourier_reg_mode=1 with mixed additive and multiplicative operations "
                f"uses a split-half embedding strategy and requires embedding dimension >= 2, "
                f"got {feature_dim}."
            )

    def _validate_sequence_training_settings(self):
        if self.training.training_target != 'seq_cot':
            return
        if self.model_type == 'transformer' and not self.model.mask:
            raise ValueError(
                "For training_target='seq_cot' with transformer models, "
                "set model.mask=True to prevent future-token leakage."
            )

    def _validate_rnn_only_modes(self):
        if self.training.fourier_reg_mode in [3, 4] and self.model_type == "transformer":
            raise ValueError(
                f"fourier_reg_mode={self.training.fourier_reg_mode} is RNN-only "
                "(uses W_ih/W_hh) and cannot be used with model_type='transformer'."
            )

    def _validate_fourier_regularization_requirements(self):
        if self.training.fourier_reg_mode is None or self.training.fourier_reg_coefficient <= 0.0:
            return
        has_additive = any(op in ['addition', 'subtraction'] for op in self.operations)
        has_multiplicative = any(op in ['multiplication', 'division'] for op in self.operations)
        if has_multiplicative and not self._is_prime(int(self.data.modulo)):
            raise ValueError(
                "Fourier regularization for multiplication/division requires a prime modulo "
                f"for primitive-root permutation, got modulo={self.data.modulo}."
            )
        if has_additive and has_multiplicative and self.training.batch_mode != "operation-batch":
            raise ValueError(
                "Fourier regularization with mixed additive and multiplicative operations "
                "requires batch_mode='operation-batch' so operation-specific permutations "
                "are applied correctly."
            )

    def _validate_dataset_batch_mode_settings(self):
        uses_mixed_sequence = bool(getattr(self.data, "mixed_sequence_operations", False))
        uses_variable_length = getattr(self.data, "nterms_list", None) is not None
        if (uses_mixed_sequence or uses_variable_length) and self.training.batch_mode == "operation-batch":
            raise ValueError(
                "batch_mode='operation-batch' is not compatible with mixed ops inside one sequence "
                "or variable-length nterms_list datasets. Use batch_mode='full' or 'mini-batch'."
            )

    def _generate_save_dir(self) -> str:
        ops_str = "_".join(sorted(self.operations))
        return f"{self.model_type}_modular_{ops_str}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operations': self.operations,
            'model_type': self.model_type,
            'data': self.data.to_dict(),
            'analysis': self.analysis.to_dict(),
            'model': self.model.to_dict(),
            'training': self.training.to_dict()
        }

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def resolve(
        cls,
        path: str,
        operations: List[str],
        model_type: str,
        analysis_only: bool = False,
    ) -> tuple:
        """Resolve save dir, load existing config or create a default one.

        Returns (config, save_dir).
        """
        if path == '.':
            ops = '_'.join(sorted(operations))
            path = f'{model_type}_modular_{ops}'

        config_path = os.path.join(path, 'config.yaml')

        if os.path.exists(config_path):
            print(f"Loading configuration from {config_path}")
            config = cls.load(config_path, skip_validation=analysis_only)
            config.training.save_dir = path
        else:
            print("No config file found, creating default configuration.")
            config = cls(
                operations=operations,
                model_type=model_type,
                training_config={'save_dir': path},
            )
            os.makedirs(path, exist_ok=True)
            config.save(config_path)
            print(f"Saved configuration to {config_path}")

        return config, path

    @classmethod
    def load(cls, filepath: str, skip_validation: bool = False) -> 'Config':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        with open(filepath, 'r') as f:
            config_dict = yaml.safe_load(f)
        operations = config_dict.get('operations', ['addition'])
        model_type = config_dict.get('model_type', 'rnn')
        obj = cls.__new__(cls)
        obj._skip_validation = skip_validation
        obj.__init__(
            operations=operations,
            model_type=model_type,
            data_config=config_dict.get('data'),
            analysis_config=config_dict.get('analysis'),
            model_config=config_dict.get('model'),
            training_config=config_dict.get('training')
        )
        return obj

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict):
                    config_obj = getattr(self, key)
                    for sub_key, sub_value in value.items():
                        if hasattr(config_obj, sub_key):
                            setattr(config_obj, sub_key, sub_value)
                else:
                    setattr(self, key, value)

    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  operations={self.operations},\n"
            f"  model_type={self.model_type},\n"
            f"  data={self.data},\n"
            f"  analysis={self.analysis},\n"
            f"  model={self.model},\n"
            f"  training={self.training}\n"
            f")"
        )


# Backward compatibility functions
def create_default_data_config(operations: List[str]) -> Dict[str, Any]:
    includes_zero = any(op in ["addition", "subtraction"] for op in operations)
    config = DataConfig(includes_zero=includes_zero)
    base_term_vocab = config.modulo if includes_zero else (config.modulo - 1)
    extra_equals = 1 if config.include_equals_token else 0
    config.vocab_size = base_term_vocab + extra_equals + len(operations)
    return config.to_dict()


def create_default_model_config(operations: List[str]) -> Dict[str, Any]:
    includes_zero = any(op in ["addition", "subtraction"] for op in operations)
    config = RNNModelConfig()
    base_term_vocab = 113 if includes_zero else 112
    config.vocab_size = base_term_vocab + 1 + len(operations)
    return config.to_dict()


def create_default_opt_config(save_dir: Optional[str] = None) -> Dict[str, Any]:
    config = TrainingConfig(save_dir=save_dir if save_dir else 'rnn_modular_addition')
    return config.to_dict()
