"""Reseau d'images en couches (amendement 31) : images 2 (entree) - 3 (intermediaire) - 1 (sortie), un transformer par transition orientee entre
couches voisines (aucune arete a l'interieur d'une couche), soit 6 + 3 vers l'avant et 3 + 6 en retour = 18 transformers, calcules par groupes
(poids empiles, un appel par groupe). Aucun historique : toute la memoire est dans les 6 images.
Par token : depot sur les 2 images d'entree ; balayage avant (entree -> intermediaires -> sortie) ; lecture sur l'image de sortie ;
balayage retour (sortie -> intermediaires -> entree). retour=False : pas de balayage retour (temoin sans retroaction)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from s4_loop import patchify, unpatchify


class Stacked(nn.Module):
    """E transformers independants (1 couche pre-norm) sur les patchs d'une image ; entree (B, E, C, H, W) -> (residu, oubli) de meme forme."""
    def __init__(self, E, C, H, W, p, d, nh):
        super().__init__()
        self.E, self.C, self.H, self.W, self.p, self.d, self.nh = E, C, H, W, p, d, nh
        n, pd = (H // p) * (W // p), C * p * p; self.n, self.pd = n, pd
        def w(*s, fan): return nn.Parameter(torch.randn(*s) / math.sqrt(fan))
        self.inp, self.inp_b, self.pos = w(E, pd, d, fan=pd), nn.Parameter(torch.zeros(E, 1, d)), nn.Parameter(torch.randn(E, n, d) * 0.02)
        self.qkv, self.proj = w(E, d, 3 * d, fan=d), w(E, d, d, fan=d)
        self.m1, self.m2 = w(E, d, 4 * d, fan=d), w(E, 4 * d, d, fan=4 * d)
        self.m1_b, self.m2_b = nn.Parameter(torch.zeros(E, 1, 4 * d)), nn.Parameter(torch.zeros(E, 1, d))
        self.out, self.out_b = w(E, d, 2 * pd, fan=d), nn.Parameter(torch.zeros(E, 1, 2 * pd))

    def forward(self, img):
        B, E = img.shape[:2]; C, H, W, p, d, nh = self.C, self.H, self.W, self.p, self.d, self.nh
        x = F.layer_norm(img, img.shape[2:]).reshape(B * E, C, H, W)
        t = patchify(x, p).view(B, E, self.n, self.pd)
        h = torch.einsum("benp,epd->bend", t, self.inp) + self.inp_b.unsqueeze(0) + self.pos.unsqueeze(0)
        q, k, v = torch.einsum("bend,edk->benk", F.layer_norm(h, (d,)), self.qkv).chunk(3, -1)
        sp = lambda z: z.view(B, E, self.n, nh, d // nh).transpose(2, 3)
        a = F.scaled_dot_product_attention(sp(q), sp(k), sp(v)).transpose(2, 3).reshape(B, E, self.n, d)
        h = h + torch.einsum("bend,edk->benk", a, self.proj)
        m = F.gelu(torch.einsum("bend,edk->benk", F.layer_norm(h, (d,)), self.m1) + self.m1_b.unsqueeze(0))
        h = h + torch.einsum("bend,edk->benk", m, self.m2) + self.m2_b.unsqueeze(0)
        o = torch.einsum("bend,edk->benk", F.layer_norm(h, (d,)), self.out) + self.out_b.unsqueeze(0)
        r, g = o.chunk(2, -1)
        un = lambda z: unpatchify(z.reshape(B * E, self.n, self.pd), C, H, W, p).view(B, E, C, H, W)
        return un(r), un(g)


class GraphLM(nn.Module):
    def __init__(self, vocab, cfg, retour=True, widths=(2, 3, 1)):
        super().__init__()
        C, H, W, p, d, nh = cfg.get("g_c", 4), cfg.get("g_h", 4), cfg.get("g_w", 4), cfg.get("g_p", 2), cfg.get("g_d", 32), cfg.get("g_nh", 2)
        self.C, self.H, self.W, self.widths, self.retour = C, H, W, widths, retour
        a, b, c = widths
        self.emb = nn.Embedding(vocab, cfg.get("d_tok", 32)); self.dep = nn.Linear(cfg.get("d_tok", 32), a * C * H * W)
        self.f1, self.f2 = Stacked(b * a, C, H, W, p, d, nh), Stacked(c * b, C, H, W, p, d, nh)
        if retour: self.b1, self.b2 = Stacked(b * c, C, H, W, p, d, nh), Stacked(a * b, C, H, W, p, d, nh)
        self.ret = nn.Parameter(torch.full((1, a, C, 1, 1), 1.0))
        self.init = nn.ParameterList([nn.Parameter(torch.zeros(n, C, H, W)) for n in widths])
        self.head = nn.Sequential(nn.LayerNorm(c * C * H * W), nn.Linear(c * C * H * W, 128), nn.GELU(), nn.Linear(128, vocab))

    @staticmethod
    def step(mod, src, dst):
        """Toutes les transitions src -> dst en un appel ; l'image destination garde (oubli moyen) et recoit la somme des residus."""
        B, ns, nd = src.shape[0], src.shape[1], dst.shape[1]
        r, g = mod(src.unsqueeze(1).expand(B, nd, ns, *src.shape[2:]).reshape(B, nd * ns, *src.shape[2:]))
        r, g = r.view(B, nd, ns, *src.shape[2:]), g.view(B, nd, ns, *src.shape[2:])
        return torch.sigmoid(g.mean(2) + 3.0) * dst + torch.tanh(r).sum(2)

    def forward(self, x):
        B, N = x.shape; a = self.widths[0]
        dep = torch.tanh(self.dep(self.emb(x))).view(B, N, a, self.C, self.H, self.W)
        A, Bm, Cc = [p.expand(B, -1, -1, -1, -1) for p in self.init]; outs = []
        for t in range(N):
            A = A + dep[:, t]
            Bm = self.step(self.f1, A, Bm); Cc = self.step(self.f2, Bm, Cc)
            outs.append(self.head(Cc.flatten(1)))
            if self.retour: Bm = self.step(self.b1, Cc, Bm); A = self.step(self.b2, Bm, A)
            else: A = torch.sigmoid(self.ret) * A
        return torch.stack(outs, 1)


class GraphLM2(nn.Module):
    """Version a nombre de couches quelconque (amendement 33b) : widths = (n0, n1, ..., nL), transitions entre couches voisines seulement.
    widths = (1, 1) est la boucle a 2 images. zero=True : sortie des transformers initialisee a zero (aretes sans effet au depart)."""
    def __init__(self, vocab, widths, c=4, h=4, w=4, p=2, d=32, nh=2, d_tok=32, retour=True, zero=False):
        super().__init__()
        self.C, self.H, self.W, self.widths, self.retour = c, h, w, tuple(widths), retour
        self.emb = nn.Embedding(vocab, d_tok); self.dep = nn.Linear(d_tok, widths[0] * c * h * w)
        self.f = nn.ModuleList([Stacked(widths[i + 1] * widths[i], c, h, w, p, d, nh) for i in range(len(widths) - 1)])
        self.b = nn.ModuleList([Stacked(widths[i] * widths[i + 1], c, h, w, p, d, nh) for i in range(len(widths) - 1)]) if retour else None
        if zero:
            for m in list(self.f) + (list(self.b) if retour else []):
                nn.init.zeros_(m.out); nn.init.zeros_(m.out_b)
        self.ret = nn.Parameter(torch.full((1, widths[0], c, 1, 1), 1.0))
        self.init = nn.ParameterList([nn.Parameter(torch.zeros(n, c, h, w)) for n in widths])
        self.head = nn.Sequential(nn.LayerNorm(widths[-1] * c * h * w), nn.Linear(widths[-1] * c * h * w, 128), nn.GELU(), nn.Linear(128, vocab))

    def forward(self, x):
        B, N = x.shape; L = len(self.widths)
        dep = torch.tanh(self.dep(self.emb(x))).view(B, N, self.widths[0], self.C, self.H, self.W)
        S = [p.expand(B, -1, -1, -1, -1) for p in self.init]; outs = []
        for t in range(N):
            S[0] = S[0] + dep[:, t]
            for i in range(L - 1): S[i + 1] = GraphLM.step(self.f[i], S[i], S[i + 1])
            outs.append(self.head(S[-1].flatten(1)))
            if self.retour:
                for i in range(L - 2, -1, -1): S[i] = GraphLM.step(self.b[i], S[i + 1], S[i])
            else: S[0] = torch.sigmoid(self.ret) * S[0]
        return torch.stack(outs, 1)
