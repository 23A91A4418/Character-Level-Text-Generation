import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        n_heads=4,
        hidden_dim=256,
        n_layers=2,
        dropout=0.2,
        max_seq_length=100
    ):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional Embedding
        self.positional_embedding = nn.Embedding(
            max_seq_length,
            embedding_dim
        )

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.fc = nn.Linear(
            embedding_dim,
            vocab_size
        )

        self.dropout = nn.Dropout(dropout)

        self.max_seq_length = max_seq_length

    def forward(self, x):
        batch_size, seq_length = x.size()

        positions = torch.arange(
            0,
            seq_length
        ).unsqueeze(0).expand(
            batch_size,
            seq_length
        )

        x = self.embedding(x) + self.positional_embedding(positions)

        x = self.dropout(x)

        x = self.transformer(x)

        x = x.contiguous().view(-1, x.shape[-1])

        output = self.fc(x)

        return output