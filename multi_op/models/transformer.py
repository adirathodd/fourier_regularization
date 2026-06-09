import math
import torch
from torch.nn import Module, ModuleList, Embedding, Linear, LayerNorm, Dropout, TransformerEncoder, TransformerEncoderLayer


class FixedPositionalEncoding(Module):
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


class RotaryPositionalEmbeddings(Module):
    """Applies RoPE rotations to per-head query/key tensors.

    Expected tensor shape: [batch, heads, seq_len, head_dim]
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.base = float(base)
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        cache_is_valid = (
            self.cos_cached is not None
            and self.sin_cached is not None
            and self.cos_cached.size(2) >= seq_len
            and self.cos_cached.device == device
            and self.cos_cached.dtype == dtype
        )
        if cache_is_valid:
            return

        theta = 1.0 / (
            self.base ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("t,d->td", positions, theta)
        angles = torch.cat([angles, angles], dim=-1)

        cos = angles.cos().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        sin = angles.sin().to(dtype=dtype).unsqueeze(0).unsqueeze(0)
        self.cos_cached = cos
        self.sin_cached = sin

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        half = self.head_dim // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(2)
        self._build_cache(seq_len, x.device, x.dtype)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        return (x * cos) + (self._rotate_half(x) * sin)


class RoPEMultiheadSelfAttention(Module):
    """Self-attention with RoPE applied to queries and keys."""
    def __init__(self, embedding_dim: int, n_heads: int, dropout: float, rope_base: float = 10000.0):
        super().__init__()
        if embedding_dim % n_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by n_heads ({n_heads}) for RoPE attention"
            )

        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        self.qkv_proj = Linear(embedding_dim, 3 * embedding_dim)
        self.out_proj = Linear(embedding_dim, embedding_dim)
        self.attn_dropout = Dropout(dropout)
        self.out_dropout = Dropout(dropout)
        self.rope = RotaryPositionalEmbeddings(self.head_dim, base=rope_base)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.rope(q)
        k = self.rope(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embedding_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.out_dropout(attn_output)
        return attn_output


class RoPETransformerEncoderLayer(Module):
    """Transformer encoder block using RoPE attention."""
    def __init__(
        self,
        embedding_dim: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if activation not in {"relu", "gelu"}:
            raise ValueError(f"activation must be 'relu' or 'gelu', got {activation}")

        self.norm_first = norm_first
        self.self_attn = RoPEMultiheadSelfAttention(embedding_dim, n_heads, dropout, rope_base=rope_base)
        self.ln1 = LayerNorm(embedding_dim)
        self.ln2 = LayerNorm(embedding_dim)

        self.ffn_in = Linear(embedding_dim, dim_feedforward)
        self.ffn_out = Linear(dim_feedforward, embedding_dim)
        self.ffn_dropout = Dropout(dropout)
        self.activation_fn = torch.relu if activation == "relu" else torch.nn.functional.gelu

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn_in(x)
        x = self.activation_fn(x)
        x = self.ffn_dropout(x)
        x = self.ffn_out(x)
        x = self.ffn_dropout(x)
        return x

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
        if self.norm_first:
            x = x + self.self_attn(self.ln1(x), causal_mask=causal_mask)
            x = x + self._ffn(self.ln2(x))
            return x

        x = self.ln1(x + self.self_attn(x, causal_mask=causal_mask))
        x = self.ln2(x + self._ffn(x))
        return x


class TransformerModel(Module):
    """Vanilla Transformer encoder for modular arithmetic sequences.
      - forward(tokens) -> logits over vocab for the **last** position
      - get_hidden_states(tokens) -> encoder outputs for **all** positions
      - get_embeddings(tokens) -> token embeddings (no positional encoding)
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_len: int = 128,
                 fixed_positional_encoding: bool = True,
                 use_rope: bool = False,
                 rope_base: float = 10000.0,
                 activation: str = "relu",
                 norm_first=False,
                 mask: bool = False,
                 tie_weights: bool = False):

        super().__init__()
        self.mask = mask
        self.use_rope = use_rope
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(vocab_size, embedding_dim)

        # Position handling:
        # - RoPE path rotates q/k directly in attention layers (no additive absolute embedding)
        # - Non-RoPE path keeps existing sinusoidal/learned absolute positions
        if use_rope:
            self.pos_enc = None
            self.pos_embedding = None
        else:
            if fixed_positional_encoding:
                self.pos_enc = FixedPositionalEncoding(embedding_dim, max_len=max_len)
                self.pos_embedding = None
            else:
                self.pos_enc = None
                self.pos_embedding = Embedding(max_len, embedding_dim)

        if use_rope:
            self.encoder = None
            self.encoder_layers = ModuleList([
                RoPETransformerEncoderLayer(
                    embedding_dim=embedding_dim,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    norm_first=norm_first,
                    rope_base=rope_base,
                )
                for _ in range(n_layers)
            ])
        else:
            encoder_layer = TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                activation=activation,
                norm_first=norm_first,
            )
            self.encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.encoder_layers = None

        self.ln_out = LayerNorm(embedding_dim)
        self.dropout = Dropout(dropout)
        self.fc = Linear(embedding_dim, vocab_size)

        if tie_weights:
            self.fc.weight = self.embedding.weight
            self.fc.bias = None  # remove bias to preserve embedding/unembedding symmetry

    def _get_mask(self, sz):
        """Generate a square causal mask."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _get_causal_bool_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tokens, return_sequence_logits=False):
        """tokens: LongTensor [batch, seq_len]
        Returns:
          - logits [batch, vocab] for the **last** position (default)
          - logits [batch, seq_len, vocab] when return_sequence_logits=True
        """
        x = self.embedding(tokens)  # [B, T, D]
        if not self.use_rope:
            if self.pos_enc is not None:
                x = self.pos_enc(x)
            else:
                # learned absolute positions 0..T-1
                positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
                x = x + self.pos_embedding(positions)

        if self.use_rope:
            causal_mask = self._get_causal_bool_mask(x.size(1), x.device) if self.mask else None
            h = x
            for layer in self.encoder_layers:
                h = layer(h, causal_mask=causal_mask)
        else:
            mask = self._get_mask(x.size(1)).to(x.device) if self.mask else None
            h = self.encoder(x, mask=mask)

        if return_sequence_logits:
            h_all = self.ln_out(h)
            h_all = self.dropout(h_all)
            return self.fc(h_all)

        h_last = self.ln_out(h[:, -1, :])   # pool by last token
        h_last = self.dropout(h_last)
        logits = self.fc(h_last)            # [B, vocab]
        return logits

    def get_embeddings(self, tokens):
        """Return raw token embeddings (without positional encoding)."""
        return self.embedding(tokens)

    def get_hidden_states(self, tokens=None, embedded_seq=None):
        """Return encoder hidden states for **all** time steps: [batch, seq_len, d_model].

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

        # Add absolute positional encoding only on non-RoPE runs.
        if not self.use_rope:
            if self.pos_enc is not None:
                x = self.pos_enc(x)
            else:
                positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
                x = x + self.pos_embedding(positions)

        hidden_states = [x]

        if self.use_rope:
            causal_mask = self._get_causal_bool_mask(x.size(1), x.device) if self.mask else None
            for layer in self.encoder_layers:
                x = layer(x, causal_mask=causal_mask)
                hidden_states.append(x)
        else:
            mask = self._get_mask(x.size(1)).to(x.device) if self.mask else None
            for layer in self.encoder.layers:
                x = layer(x, src_mask=mask)
                hidden_states.append(x)

        result = list(hidden_states)
        result[-1] = self.ln_out(result[-1])
        return tuple(result)
