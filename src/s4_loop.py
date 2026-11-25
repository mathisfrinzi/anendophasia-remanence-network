"""Etape 4 : deux images couplees par deux transformers (idee de Mathis, amendement 12).

Image1 (entree) recoit le token ; T1 lit Image1 en patchs et ecrit un residu dans Image2 (sortie, d'ou on lit la prediction) ;
T2 lit Image2 en patchs et ecrit un residu dans Image1 (retroaction : l'image d'entree se prepare au token suivant).
Aucun historique : chaque transformer ne voit que l'image courante (16 patchs). Toute la memoire est dans les deux images.
loop=False : T2 supprime (Image1 ne recoit que les depots et une retention fixe) : temoin sans retroaction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def patchify(x, p):                     # (B,C,H,W) -> (B, n, C*p*p)
    B, C, H, W = x.shape
    return x.view(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)


def unpatchify(t, C, H, W, p):          # inverse
    B = t.shape[0]
    return t.view(B, H // p, W // p, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


class ImgTransformer(nn.Module):
    """Transformer bidirectionnel sur les patchs d'une image ; sortie : (residu, oubli) de meme forme que l'image."""

    def __init__(self, C, H, W, p, d, n_layer, n_head):
        super().__init__()
        self.C, self.H, self.W, self.p = C, H, W, p
        n, pd = (H // p) * (W // p), C * p * p
        self.inp = nn.Linear(pd, d)
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2 * pd)

    def forward(self, img):
        x = F.layer_norm(img, img.shape[1:])
        h = self.enc(self.inp(patchify(x, self.p)) + self.pos)
        o = self.out(self.norm(h))
        r, g = o.chunk(2, dim=-1)
        return unpatchify(r, self.C, self.H, self.W, self.p), unpatchify(g, self.C, self.H, self.W, self.p)


class LoopLM(nn.Module):
    def __init__(self, vocab, cfg, loop=True):
        super().__init__()
        C, H, W, p = cfg.get("loop_c", 4), cfg.get("loop_h", 8), cfg.get("loop_w", 8), cfg.get("loop_p", 2)
        d, nl, nh = cfg["d_model"], cfg.get("loop_layers", 2), cfg["n_head"]
        self.C, self.H, self.W, self.loop = C, H, W, loop
        self.passes = int(cfg.get("loop_passes", 1))                   # amendement 46 : nombre de passes T1/T2 par token (poids partages)
        self.emb = nn.Embedding(vocab, cfg["d_tok"])
        self.dep = nn.Linear(cfg["d_tok"], C * H * W)                  # motif depose par le token sur Image1
        self.t1 = ImgTransformer(C, H, W, p, d, nl, nh)
        self.t2 = ImgTransformer(C, H, W, p, d, nl, nh) if loop else None   # loop : True | False | "i2"
        self.no_gate, self.read_i1 = cfg.get("no_gate", False), cfg.get("readout", "i2") == "i1"
        self.t2_noise = 0.0                                              # perturbation causale a l'inference
        self.ret1 = nn.Parameter(torch.full((1, C, 1, 1), 1.0))      # retention fixe d'Image1 sans boucle
        self.i1 = nn.Parameter(torch.zeros(C, H, W))
        self.i2 = nn.Parameter(torch.zeros(C, H, W))
        self.head = nn.Sequential(nn.LayerNorm(C * H * W), nn.Linear(C * H * W, 2 * d), nn.GELU(), nn.Linear(2 * d, vocab))
        self.state_out = None

    def forward(self, x, return_frames=False, state=None):
        B, N = x.shape
        dep = torch.tanh(self.dep(self.emb(x))).view(B, N, self.C, self.H, self.W)
        if state is None:
            I1, I2 = self.i1.expand(B, -1, -1, -1), self.i2.expand(B, -1, -1, -1)
        else:
            I1, I2 = state
        outs, frames = [], []
        for t in range(N):
            I1 = I1 + dep[:, t]                                       # le token se depose sur l'image d'entree
            for _ in range(self.passes - 1 if self.loop is True else 0):                       # passes supplementaires (amendement 46) : T1 puis T2, avant la lecture
                r2, g2 = self.t1(I1); I2 = torch.sigmoid(g2 + 3.0) * I2 + torch.tanh(r2)
                r1, g1 = self.t2(I2); I1 = torch.sigmoid(g1 + 3.0) * I1 + torch.tanh(r1)
            r2, g2 = self.t1(I1)
            I2 = (1.0 if self.no_gate else torch.sigmoid(g2 + 3.0)) * I2 + torch.tanh(r2)        # T1 met a jour l'image de sortie
            outs.append(self.head((I1 if self.read_i1 else I2).flatten(1)))                    # prediction lue sur Image2
            if self.loop == "i2":                                     # temoin a capacite egale : T2 ecrit dans Image2, pas de retour vers Image1
                r1, g1 = self.t2(I2)
                I2 = torch.sigmoid(g1 + 3.0) * I2 + torch.tanh(r1)
                I1 = torch.sigmoid(self.ret1) * I1
            elif self.loop:
                r1, g1 = self.t2(I2)                                  # T2 : retroaction vers l'image d'entree
                if self.t2_noise:
                    r1 = r1 + self.t2_noise * torch.randn_like(r1)
                I1 = (1.0 if self.no_gate else torch.sigmoid(g1 + 3.0)) * I1 + torch.tanh(r1)
            else:
                I1 = torch.sigmoid(self.ret1) * I1
            if return_frames:
                frames.append(torch.stack([I1, I2], 1))
        self.state_out = (I1.detach(), I2.detach())
        logits = torch.stack(outs, 1)
        return (logits, torch.stack(frames, 1)) if return_frames else logits
