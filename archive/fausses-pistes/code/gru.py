import torch
import torch.nn as nn

import config


class GRULM(nn.Module):
    def __init__(self, vocab, d_emb=None, d_hidden=None, n_layers=None):
        p = config.GRU
        d_emb = d_emb or p["d_emb"]
        d_hidden = d_hidden or p["d_hidden"]
        n_layers = n_layers or p["n_layers"]
        super().__init__()
        self.emb = nn.Embedding(vocab, d_emb)
        self.rnn = nn.GRU(d_emb, d_hidden, n_layers, batch_first=True)
        self.head = nn.Linear(d_hidden, vocab)
        self.n_layers = n_layers
        self.d_hidden = d_hidden

    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return self.head(out)

    def init_state(self, B):
        return torch.zeros(self.n_layers, B, self.d_hidden, device=self.emb.weight.device)

    def step(self, x_t, h):
        e = self.emb(x_t)
        out, h = self.rnn(e, h)
        return self.head(out[:, -1]), h


def build(vocab):
    return GRULM(vocab)
