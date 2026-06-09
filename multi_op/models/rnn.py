from torch.nn import Module, Embedding, RNN, Linear, BatchNorm1d


class RNNModel(Module):
    """RNN model for modular arithmetic tasks."""
    def __init__(self, hidden_dim, n_layers, vocab_size):
        super(RNNModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.layer_dim = n_layers
        self.embedding = Embedding(vocab_size, hidden_dim)
        self.rnn = RNN(hidden_dim, hidden_dim, n_layers, batch_first=True) # batch_first=True (batch_len, sequence_len, hidden_size)
        self.fc = Linear(hidden_dim, vocab_size)
        self.bn = BatchNorm1d(hidden_dim)

    def forward(self, tokens, return_sequence_logits=False):
        xs = self.embedding(tokens)
        hs, hn = self.rnn(xs)

        if return_sequence_logits:
            # Apply the same normalization channel-wise across all time steps.
            bsz, seq_len, hidden_dim = hs.shape
            hs_norm = self.bn(hs.reshape(-1, hidden_dim)).reshape(bsz, seq_len, hidden_dim)
            return self.fc(hs_norm)

        hs = self.bn(hs[:, -1, :])
        scores = self.fc(hs)
        return scores

    def get_hidden_states(self, tokens):
        """Get all hidden states for analysis.

        Returns raw RNN outputs (pre-BatchNorm). Note: forward() applies self.bn
        before self.fc, so the representations returned here differ from what the
        classifier sees. To get the normalized final state, apply self.bn(hs[:, -1, :]).
        """
        xs = self.embedding(tokens)
        hs, hn = self.rnn(xs)
        return hs

    def get_embeddings(self, tokens):
        """Get embeddings for analysis."""
        return self.embedding(tokens)
