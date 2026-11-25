"""Version rapide de LoopLM (amendement 38) : memes parametres, memes calculs, moins d'operations.
 - les images restent en patchs (B, n, C*p*p) pendant toute la boucle : plus de patchify / unpatchify a chaque token (le LayerNorm sur toute l'image est invariant au reordonnancement) ;
 - la couche de transformer est ecrite a la main (projections q,k,v fusionnees, attention SDPA), sans le surcout de nn.TransformerEncoderLayer ;
 - la lecture (tete) est calculee une seule fois apres la boucle, sur tous les tokens.
FastLoop(model) enveloppe un LoopLM existant et partage ses parametres (les points de controle restent valides). Cas pris en charge : loop True ou False, lecture sur Image2, sans no_gate ni t2_noise."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from s4_loop import patchify, unpatchify


def _layer(L, h):
    B, n, d = h.shape; nh = L.self_attn.num_heads
    y = F.layer_norm(h, (d,), L.norm1.weight, L.norm1.bias, L.norm1.eps)
    q, k, v = F.linear(y, L.self_attn.in_proj_weight, L.self_attn.in_proj_bias).view(B, n, 3, nh, d // nh).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, n, d)
    h = h + F.linear(a, L.self_attn.out_proj.weight, L.self_attn.out_proj.bias)
    y = F.layer_norm(h, (d,), L.norm2.weight, L.norm2.bias, L.norm2.eps)
    return h + F.linear(F.gelu(F.linear(y, L.linear1.weight, L.linear1.bias)), L.linear2.weight, L.linear2.bias)


def _t(T, img):                                   # img (B, n, pd) -> residu, oubli (B, n, pd)
    h = T.inp(F.layer_norm(img, img.shape[1:])) + T.pos
    for L in T.enc.layers: h = _layer(L, h)
    return T.out(T.norm(h)).chunk(2, dim=-1)


class FastLoop(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m
        assert not m.no_gate and not m.read_i1 and m.loop in (True, False) and not m.t2_noise, "variante non prise en charge"

    def forward(self, x):
        m = self.m; B, N = x.shape; C, H, W, p = m.C, m.H, m.W, m.t1.p
        dep = patchify(torch.tanh(m.dep(m.emb(x))).view(B * N, C, H, W), p).view(B, N, -1, C * p * p)
        I1 = patchify(m.i1.unsqueeze(0), p).expand(B, -1, -1); I2 = patchify(m.i2.unsqueeze(0), p).expand(B, -1, -1)
        ret1 = m.ret1.view(1, 1, C, 1, 1).expand(1, 1, C, p, p).reshape(1, 1, C * p * p)
        outs = []
        for t in range(N):
            I1 = I1 + dep[:, t]
            r2, g2 = _t(m.t1, I1)
            I2 = torch.sigmoid(g2 + 3.0) * I2 + torch.tanh(r2); outs.append(I2)
            if m.loop:
                r1, g1 = _t(m.t2, I2); I1 = torch.sigmoid(g1 + 3.0) * I1 + torch.tanh(r1)
            else: I1 = torch.sigmoid(ret1) * I1
        frames = unpatchify(torch.stack(outs, 1).view(B * N, -1, C * p * p), C, H, W, p).view(B, N, -1)
        return m.head(frames)
