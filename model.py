import torch
import torch.nn as nn

class TSPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_nodes):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, num_nodes)

    def forward(self, x):
        h = self.embed(x)
        logits = self.output(h)
        return logits