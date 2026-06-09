from __future__ import annotations

import torch
import torch.nn as nn
from layer import S4Layer
import math

from typing import Optional, Type

from aux.encoders import StandardEncoder
from block import S4Block
from dsp.utils import next_pow2


def _parse_pool_kernel(pool_kernel: Optional[int | tuple[int]]) -> int:
    if pool_kernel is None:
        return 1
    elif isinstance(pool_kernel, tuple):
        return pool_kernel[0]
    elif isinstance(pool_kernel, int):
        return pool_kernel
    else:
        raise TypeError(f"Unable to parse `pool_kernel`, got {pool_kernel}")


def _seq_length_schedule(
    n_blocks: int,
    l_max: int,
    pool_kernel: Optional[int | tuple[int]],
) -> list[tuple[int, int]]:
    ppk = _parse_pool_kernel(pool_kernel)

    schedule = list()
    for depth in range(n_blocks + 1):
        l_max_next = max(2, l_max // ppk)
        pool_ok = l_max_next > ppk
        schedule.append((l_max, pool_ok))
        l_max = l_max_next
    return schedule


class S4Model(nn.Module):
    """S4 Model.

    High-level implementation of the S4 model which:

        1. Encodes the input using a linear layer
        2. Applies ``1..n_blocks`` S4 blocks
        3. Decodes the output of step 2 using another linear layer

    Args:
        d_input (int): number of input features
        d_model (int): number of internal features
        d_output (int): number of features to return
        n_blocks (int): number of S4 blocks to construct
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        wavelet_tform (bool): if ``True`` encode signal using a
            continuous wavelet transform (CWT).
        collapse (bool): if ``True`` average over time prior to
            decoding the result of the S4 block(s). (Useful for
            classification tasks.)
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after
            ``S4Layer()``.
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        norm_strategy (str): position of normalization relative to ``S4Layer()``.
            Must be "pre" (before ``S4Layer()``), "post" (after ``S4Layer()``)
            or "both" (before and after ``S4Layer()``).
        pooling (nn.AvgPool1d, nn.MaxPool1d, optional): pooling method to use
            following each ``S4Block()``.

    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_blocks: int,
        n: int,
        l_max: int,
        wavelet_tform: bool = False,
        collapse: bool = False,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
        norm_strategy: str = "post",
        pooling: Optional[nn.AvgPool1d | nn.MaxPool1d] = None,
    ) -> None:
        super().__init__()

        # self.embedding = nn.Embedding(d_input, d_model)
        self.d_input = d_input
        self.d_model = d_model
        self.d_output = d_output
        self.n_blocks = n_blocks
        self.n = n
        self.l_max = l_max
        self.wavelet_tform = wavelet_tform
        self.collapse = collapse
        self.p_dropout = p_dropout
        self.norm_type = norm_type
        self.norm_strategy = norm_strategy
        self.pooling = pooling

        self.vocab_size = self.d_input
        self.hidden_dim = self.d_model

        *self.seq_len_schedule, (self.seq_len_out, _) = _seq_length_schedule(
            n_blocks=n_blocks,
            l_max=next_pow2(l_max) if wavelet_tform else l_max,
            pool_kernel=None if self.pooling is None else self.pooling.kernel_size,
        )

        if wavelet_tform:
            from s4torch.dsp.cwt import Cwt, CwtWithAdapter

            self.encoder = CwtWithAdapter(
                Cwt(next_pow2(self.l_max)),
                d_model=self.d_model,
            )
        else:
            self.embedding = nn.Embedding(self.d_input, self.d_model) # StandardEncoder replaced with embedding

        self.fc = nn.Linear(self.d_model, self.d_output, bias=False)

        # Tie encoder/decoder
        self.embedding.weight = self.fc.weight

        self.blocks = nn.ModuleList(
            [
                S4Block(
                    d_model=d_model,
                    n=n,
                    l_max=seq_len,
                    p_dropout=p_dropout,
                    activation=activation,
                    norm_type=norm_type,
                    norm_strategy=norm_strategy,
                    pooling=pooling if pooling and pool_ok else None,
                )
                for (seq_len, pool_ok) in self.seq_len_schedule
            ]
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, D_OUTPUT]`` if ``collapse``
                is ``True`` and ``[BATCH, SEQ_LEN // (POOL_KERNEL ** n_block), D_INPUT]``
                otherwise, where ``POOL_KERNEL`` is the kernel size of the ``pooling``
                layer. (Note that ``POOL_KERNEL=1`` if ``pooling`` is ``None``.)

        """
        y = self.embedding(u)
        for block in self.blocks:
            y = block(y)
        return self.fc(y.mean(dim=1) if self.collapse else y)

    def get_internal_states(self, u: torch.Tensor) -> torch.Tensor:
        y = self.embedding(u)
        for block in self.blocks:
          y = block(y)

        return y


class RNNModel(nn.Module):
    """RNN model for modular arithmetic tasks."""
    def __init__(self, hidden_dim, n_layers, vocab_size):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.layer_dim = n_layers
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.RNN(hidden_dim, hidden_dim, n_layers, batch_first=True) # batch_first=True (batch_dim, seq_dim, feature_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, tokens):
        xs = self.embedding(tokens)
        hs, hn = self.rnn(xs)
        hs = self.bn(hs[:, -1, :])
        scores = self.fc(hs)
        return scores

    def get_hidden_states(self, tokens):
            """Get all hidden states for analysis."""
            xs = self.embedding(tokens)
            hs, hn = self.rnn(xs)
            return hs
    
    def get_embeddings(self, tokens):
        """Get embeddings for analysis."""
        return self.embedding(tokens)


class FixedPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.).
    Produces a tensor of shape [batch, seq_len, d_model] to be **added** to token embeddings.
    """
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TransformerModel(nn.Module):
    """Vanilla Transformer encoder for modular arithmetic sequences.
      - forward(tokens) -> logits over vocab for the **last** position
      - get_hidden_states(tokens) -> encoder outputs for **all** positions
      - get_embeddings(tokens) -> token embeddings (no positional encoding)
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 256,
                 n_heads: int = 4,
                 n_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.0,
                 max_len: int = 128,
                 fixed_positional_encoding: bool = True,
                 activation: str = "tanh",
                 norm_first=False,
                 tie=False):
        
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.hidden_dim = embedding_dim

        # Positional encoding (sinusoidal by default; otherwise learned)
        if fixed_positional_encoding:
            self.pos_enc = FixedPositionalEncoding(embedding_dim, max_len=max_len)
            self.pos_embedding = None
        else:
            self.pos_enc = None
            self.pos_embedding = nn.Embedding(max_len, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=activation,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_out = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embedding_dim, vocab_size, bias=False)

        # Tie embedding/unembedding
        if tie:
            self.fc.weight = self.embedding.weight

    def forward(self, tokens):
        """tokens: LongTensor [batch, seq_len]
        Returns: logits [batch, vocab] for the **last** position.
        """
        x = self.embedding(tokens)  # [B, T, D]
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        else:
            # learned absolute positions 0..T-1
            positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)

        h = self.encoder(x)                 # [B, T, D]
        h_last = self.ln_out(h[:, -1, :])   # pool by last token
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)            # [B, vocab]
        return logits

    def get_embeddings(self, tokens):
        """Return raw token embeddings (without positional encoding)."""
        return self.embedding(tokens)

    def get_hidden_states(self, tokens=None, embedded_seq=None):
        """Return encoder hidden states for **all** time steps: [batch, seq_len, d_model.

        Usage:
          - Call with `tokens` (LongTensor [B, T]) to perform embedding lookup.
          - Or call with `embedded_seq` (FloatTensor [B, T, D]) if you already
            have embeddings (e.g. after reordering or projecting them).

        Returns:
          tuple of tensors: (embed_plus_pos, layer1_out, ..., final_layer_out)
          Each tensor is shaped [batch, seq_len, d_model]

        """
        if (tokens is None) == (embedded_seq is None):
            raise ValueError("Provide exactly one of `tokens` or `embedded_seq`.")

        if tokens is not None:
            x = self.embedding(tokens)
        else:
            x = embedded_seq

        # add positional encoding
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        else:
            positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = x + self.pos_embedding(positions)

        hidden_states = [x]
        for layer in self.encoder.layers:
            x = layer(x)
            hidden_states.append(x)

        hidden_states = [self.ln_out(h) for h in hidden_states]
        return tuple(hidden_states)