import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()

        self.num_tokens = config['num_tokens']
        self.nhead = config['nhead']
        self.num_encoder_layers = config['num_encoder_layers']
        self.num_decoder_layers = config['num_decoder_layers']
        self.d_model = config['d_model']
        self.dim_feedforward = config['dim_feedforward']
        self.dropout = config['dropout']

        self.embedding = nn.Embedding(self.num_tokens, self.d_model)
        self.transformer = nn.Transformer(
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_encoder_layers,
            d_model=self.d_model,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(self.d_model, self.num_tokens)

    def forward(self, src, tgt):
        tgt = tgt.unsqueeze(-1)
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        out = self.transformer(src, tgt)
        out = self.fc(out[:, -1])
        return out
