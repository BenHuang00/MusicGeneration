from torch import nn
from mamba_ssm import Mamba as Mamba_ssm


class Mamba(nn.Module):
    def __init__(self, config):
        super().__init__()

        num_tokens = config['num_tokens']

        self.d_model = config['d_model']
        self.d_state = config['d_state']
        self.d_conv = config['d_conv']
        self.expand = config['expand']

        self.dropout = config['dropout']

        self.embedding = nn.Embedding(num_tokens, self.d_model)
        self.dropout1 = nn.Dropout(self.dropout)
        self.mamba = Mamba_ssm(self.d_model, self.d_state, self.d_conv, self.expand)
        self.dropout2 = nn.Dropout(self.dropout)
        self.fc = nn.Linear(self.d_model, num_tokens)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout1(x)
        x = self.mamba(x)
        x = self.dropout2(x)
        out = self.fc(x[:, -1, :])
        return out[:, -1, :]
