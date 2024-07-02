import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()

        self.num_tokens = config['num_tokens']
        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.bidirectional = config['bidirectional']
        self.dropout = config['dropout']

        self.embedding = nn.Embedding(self.num_tokens, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout)
        self.fc = nn.Linear(self.hidden_size * (2 if self.bidirectional else 1), self.num_tokens)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
