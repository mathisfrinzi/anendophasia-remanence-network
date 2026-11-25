import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)
        self.n_head = n_head
        self.d = d

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(y)
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab, d=None, n_layer=None, n_head=None):
        p = config.TRANSFORMER
        d = d or p["d_model"]
        n_layer = n_layer or p["n_layer"]
        n_head = n_head or p["n_head"]
        super().__init__()
        self.max_len = config.COMMON["seq_len"]
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(self.max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # poids lies

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos.weight[:T].unsqueeze(0)
        for b in self.blocks:
            h = b(h)
        return self.head(self.lnf(h))

    def init_state(self, B):
        return torch.zeros(B, 0, dtype=torch.long, device=self.tok.weight.device)

    def step(self, x_t, ctx):
        ctx = torch.cat([ctx, x_t], dim=1)[:, -self.max_len :]
        return self.forward(ctx)[:, -1], ctx


def build(vocab):
    return TransformerLM(vocab)
