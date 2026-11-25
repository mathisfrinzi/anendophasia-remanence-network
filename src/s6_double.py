"""Etape 6 (amendement 26) : double boucle. Deux images d'entree (Image1, Image2, moitie de la taille d'Image3) et une image de sortie Image3.

Les tokens se deposent sur Image1 et Image2 (deux read-in distincts). Quatre transformers, a parametres totaux egaux a L_loop :
  A1 : Image1 -> Image3,  A2 : Image2 -> Image3   (aller)
  B1 : Image3 -> Image1,  B2 : Image3 -> Image2   (retour)
La prediction est lue sur Image3. Memoire totale : 2*(2x8x8) + 4x8x8 = 512 valeurs = 2*(4x8x8), comme L_loop.
cycles=1 : un cycle (A puis B) par token (comme L_loop). Le routage (amendement 26b) n'est pas dans ce fichier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from s4_loop import patchify, unpatchify


class EdgeTransformer(nn.Module):
    """Transformer sur les patchs de l'image source (Cs,H,W) ; sortie (residu, oubli) de la forme de l'image cible (Ct,H,W)."""

    def __init__(self, Cs, Ct, H, W, p, d, n_layer, n_head):
        super().__init__()
        self.Ct, self.H, self.W, self.p = Ct, H, W, p
        n = (H // p) * (W // p)
        self.inp = nn.Linear(Cs * p * p, d)
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_head, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 2 * Ct * p * p)

    def forward(self, img):
        x = F.layer_norm(img, img.shape[1:])
        h = self.enc(self.inp(patchify(x, self.p)) + self.pos)
        r, g = self.out(self.norm(h)).chunk(2, dim=-1)
        return unpatchify(r, self.Ct, self.H, self.W, self.p), unpatchify(g, self.Ct, self.H, self.W, self.p)


class DoubleLoopLM(nn.Module):
    def __init__(self, vocab, cfg, d_edge=44, layers=1, decorr=0.0, c1=None, c2=None, d_edge2=None):
        super().__init__()
        C, H, W, p = cfg.get("loop_c", 4), cfg.get("loop_h", 8), cfg.get("loop_w", 8), cfg.get("loop_p", 2)
        Cs = C // 2                                                    # images d'entree : moitie de canaux
        c1, c2, d_edge2 = (Cs if c1 is None else c1), (Cs if c2 is None else c2), (d_edge if d_edge2 is None else d_edge2)   # variante amendement 29 : branche 1 pleine, branche 2 annexe
        nh = cfg["n_head"]
        self.C, self.C1, self.C2, self.H, self.W = C, c1, c2, H, W
        self.emb = nn.Embedding(vocab, cfg["d_tok"])
        self.dep1 = nn.Linear(cfg["d_tok"], c1 * H * W)                # deux read-in distincts
        self.dep2 = nn.Linear(cfg["d_tok"], c2 * H * W)
        self.a1, self.a2 = EdgeTransformer(c1, C, H, W, p, d_edge, layers, nh), EdgeTransformer(c2, C, H, W, p, d_edge2, layers, nh)
        self.b1, self.b2 = EdgeTransformer(C, c1, H, W, p, d_edge, layers, nh), EdgeTransformer(C, c2, H, W, p, d_edge2, layers, nh)
        self.i1 = nn.Parameter(torch.zeros(c1, H, W)); self.i2 = nn.Parameter(torch.zeros(c2, H, W)); self.i3 = nn.Parameter(torch.zeros(C, H, W))
        d = cfg["d_model"]
        self.head = nn.Sequential(nn.LayerNorm(C * H * W), nn.Linear(C * H * W, 2 * d), nn.GELU(), nn.Linear(2 * d, vocab))
        self.decorr, self.aux = decorr, None
        self.dep_scale = (1.0, 1.0)                                    # ablation a l'inference : (echelle depot 1, echelle depot 2)
        self.state_out = None

    def forward(self, x, return_frames=False, state=None):
        B, N = x.shape
        e = self.emb(x)
        d1 = torch.tanh(self.dep1(e)).view(B, N, self.C1, self.H, self.W) * self.dep_scale[0]
        d2 = torch.tanh(self.dep2(e)).view(B, N, self.C2, self.H, self.W) * self.dep_scale[1]
        if self.decorr:                                                # penalite de decorrelation (variante, pre-enregistree)
            a, b = d1.flatten(2), d2.flatten(2)
            self.aux = self.decorr * (F.cosine_similarity(a, b, dim=-1) ** 2).mean()
        if state is None:
            I1, I2, I3 = (q.expand(B, -1, -1, -1) for q in (self.i1, self.i2, self.i3))
        else:
            I1, I2, I3 = state
        outs, frames = [], []
        for t in range(N):
            I1, I2 = I1 + d1[:, t], I2 + d2[:, t]
            ra, ga = self.a1(I1); rb, gb = self.a2(I2)
            I3 = torch.sigmoid(ga + 3.0) * torch.sigmoid(gb + 3.0) * I3 + torch.tanh(ra) + torch.tanh(rb)   # aller : deux ecritures dans Image3
            outs.append(self.head(I3.flatten(1)))                                                         # prediction lue sur Image3
            r1, g1 = self.b1(I3); r2, g2 = self.b2(I3)                                              # retour
            I1 = torch.sigmoid(g1 + 3.0) * I1 + torch.tanh(r1)
            I2 = torch.sigmoid(g2 + 3.0) * I2 + torch.tanh(r2)
            if return_frames:
                frames.append((I1, I2, I3))
        self.state_out = (I1.detach(), I2.detach(), I3.detach())
        logits = torch.stack(outs, 1)
        return (logits, frames) if return_frames else logits
