import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_layers=2):
        super(LSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Convert character IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2
        )

        # Final output layer
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        """
        Forward pass
        x = input sequence
        hidden = hidden state + cell state
        """

        batch_size = x.size(0)

        # Embedding layer
        x = self.embedding(x)

        # LSTM output
        lstm_out, hidden = self.lstm(x, hidden)

        # Reshape for fully connected layer
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # Final predictions
        output = self.fc(lstm_out)

        return output, hidden

    def init_hidden(self, batch_size):
        """
        Initialize hidden state and cell state
        """

        weight = next(self.parameters()).data

        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        )

        return hidden