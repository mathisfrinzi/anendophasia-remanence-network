"""Etape 2 : un transformer sur tokens (T) contre un transformer sur la video continue (V).

T : embedding -> transformer causal -> tokens. Deux variantes : position RoPE, ou aucune position (NoPE).
V : embedding -> video continue (une frame 3x8x8 par token, frame_t = oubli * frame_(t-1) + residu, meme
    mecanisme que le bras `cumul` de l'etape 1) -> lecteur spatial local (convolutions) -> le MEME transformer
    causal, SANS aucune position (l'ordre ne peut venir que de la dynamique de la video) -> tokens.
V2 : la frame est TRANSPORTEE d'une fenetre a la suivante (jamais remise a zero dans un flux), et peut avoir deux echelles
    de temps : des canaux "rapides" (retention initiale 0,8) et "lents" (0,99), la porte d'oubli restant apprise.
Seule la representation d'entree change : la sortie reste un token (la sortie en images est une etape ulterieure).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder import TextImageAE


def rope(x, base=10000.0):
    """Rotary embedding ; x : (B, H, T, Dh). Calcule pour n'importe quelle longueur T."""
    T, Dh = x.shape[2], x.shape[3]
    inv = 1.0 / (base ** (torch.arange(0, Dh, 2, device=x.device).float() / Dh))
    f = torch.outer(torch.arange(T, device=x.device).float(), inv)
    cos, sin = f.cos()[None, None], f.sin()[None, None]
    x1, x2 = x[..., : Dh // 2], x[..., Dh // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, d, h, use_rope):
        super().__init__()
        self.h, self.use_rope = h, use_rope
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.use_rope:
            q, k = rope(q), rope(k)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class Stack(nn.Module):
    def __init__(self, d, n_layer, n_head, use_rope):
        super().__init__()
        self.blocks = nn.ModuleList([Block(d, n_head, use_rope) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.ln(x)


class TLM(nn.Module):
    """Transformer causal sur tokens. pos = "rope" ou "none"."""

    def __init__(self, vocab, cfg, pos):
        super().__init__()
        d = cfg["d_model"]
        self.emb = nn.Embedding(vocab, d)
        self.stack = Stack(d, cfg["n_layer"], cfg["n_head"], pos == "rope")
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x, return_frames=False, state=None):
        logits = self.head(self.stack(self.emb(x)))
        return (logits, None) if return_frames else logits


class VLM(nn.Module):
    """Transformer causal sur la video continue : une frame par token, jamais de position explicite.

    scales = 2 : la frame a 2*c canaux (c rapides puis c lents) ; scales = 1 : c canaux, retention initiale 0,95.
    La frame de fin de fenetre est gardee dans self.state_out (sans gradient) pour etre transportee a la fenetre suivante.
    """

    def __init__(self, vocab, cfg, scales=2):
        super().__init__()
        d, dt, k, ch = cfg["d_model"], cfg["d_tok"], cfg["cumul_tok_ch"], cfg["spatial_ch"]
        c, h, w = cfg["latent_c"], cfg["latent_h"], cfg["latent_w"]
        self.g, self.c, self.h, self.w, self.k = scales, c, h, w, k
        C = scales * c
        self.C = C
        self.emb = nn.Embedding(vocab, dt)
        self.tok_patch = nn.Linear(dt, C * h * w)   # motif propose par le token
        self.tok_ch = nn.Linear(dt, k)              # le token, diffuse sur tous les pixels
        self.init_frame = nn.Parameter(torch.zeros(C, h, w))
        self.step_net = nn.Sequential(              # (frame, motif, token) -> (residu, oubli) : ecriture contextuelle
            nn.Conv2d(2 * C + k, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, 2 * C, 3, padding=1),
        )
        init = [3.0] if scales == 1 else [math.log(0.8 / 0.2), math.log(0.99 / 0.01)]   # logit de la retention initiale
        self.register_buffer("gate_bias", torch.cat([torch.full((c,), v) for v in init]).view(1, C, 1, 1))
        self.reader = nn.Sequential(                # etage spatial local devant le transformer
            nn.Conv2d(C, ch, 3, padding=1), nn.GELU(),
            nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.GELU(),
        )
        self.to_d = nn.Linear(ch * ((h + 1) // 2) * ((w + 1) // 2), d)
        self.stack = Stack(d, cfg["n_layer"], cfg["n_head"], use_rope=False)
        self.head = nn.Linear(d, vocab, bias=False)
        self.state_out = None

    def _norm(self, f):
        """Echelle neutralisee, separement par groupe de canaux (le groupe lent est bien plus grand en amplitude)."""
        sh = f.shape
        f = f.reshape(sh[0], self.g, self.c, self.h, self.w)
        f = F.layer_norm(f, f.shape[2:])
        return f.reshape(sh)

    def frames(self, x, state=None):
        """x : (B, N) -> video (B, N+1, C, H, W) ; frame 0 = etat de depart (appris, ou transporte)."""
        B, N = x.shape
        e = self.emb(x)
        p = self.tok_patch(e).view(B, N, self.C, self.h, self.w)
        q = self.tok_ch(e)
        I = self.init_frame.unsqueeze(0).expand(B, self.C, self.h, self.w) if state is None else state
        out = [I]
        for t in range(N):
            inp = torch.cat([self._norm(I), p[:, t], q[:, t].view(B, self.k, 1, 1).expand(B, self.k, self.h, self.w)], dim=1)
            o = self.step_net(inp)
            I = torch.sigmoid(o[:, self.C :] + self.gate_bias) * I + torch.tanh(o[:, : self.C])
            out.append(I)
        return torch.stack(out, dim=1)

    def forward(self, x, return_frames=False, state=None):
        B, N = x.shape
        fr = self.frames(x, state)                                    # (B, N+1, C, h, w)
        self.state_out = fr[:, -1].detach()
        f = self._norm(fr[:, 1:].reshape(B * N, self.C, self.h, self.w))   # frame apres lecture du token t
        z = self.to_d(self.reader(f).reshape(B * N, -1)).view(B, N, -1)
        logits = self.head(self.stack(z))
        return (logits, fr) if return_frames else logits


# Forme de la tache : bloc de 3 lignes x 4 colonnes = 12 amplitudes, posees a une position continue (interpolation bilinéaire).
BLOB_H, BLOB_W = 3, 4


def splat(amp, x, y, H, W):
    """amp : (M, 12) amplitudes ; x, y : (M,) coin haut-gauche continu, dans [0, W-BLOB_W-1] et [0, H-BLOB_H-1].
    Chaque amplitude est repartie entre les 4 pixels voisins (interpolation bilineaire) : differentiable en x et y.
    Retourne (M, H, W)."""
    M = amp.shape[0]
    x0, y0 = x.floor().detach(), y.floor().detach()
    fx, fy = x - x0, y - y0
    ii = torch.arange(BLOB_H, device=amp.device).repeat_interleave(BLOB_W)       # ligne de chaque amplitude
    jj = torch.arange(BLOB_W, device=amp.device).repeat(BLOB_H)                  # colonne
    canvas = amp.new_zeros(M, H * W)
    for dy in (0, 1):
        for dx in (0, 1):
            w = (fy if dy else 1 - fy)[:, None] * (fx if dx else 1 - fx)[:, None]
            r = y0.long()[:, None] + ii[None] + dy
            c = x0.long()[:, None] + jj[None] + dx
            canvas.scatter_add_(1, r * W + c, amp * w)
    return canvas.view(M, H, W)


class DLM(VLM):
    """Image qui evolue : depot du token (sans lire l'image) puis n pas d'evolution autonome (reseau qui ne voit que l'image).

    frame <- retention * frame + depot(token)          (retention apprise par canal, independante du contexte)
    n fois : frame <- frame + tanh(evolution(frame))   (l'image se "prepare" avant le token suivant)
    depot = "free" : motif libre de C*H*W valeurs ; "blob" : par canal une tache de 12 amplitudes a position continue apprise.
    """

    def __init__(self, vocab, cfg, deposit="blob", evolve=1):
        super().__init__(vocab, cfg, scales=2)
        c, h, w, ch, dt = self.c, self.h, self.w, cfg["spatial_ch"], cfg["d_tok"]
        C = self.C
        self.deposit, self.evolve, self.noise = deposit, evolve, cfg.get("rest_noise", 0.0)
        del self.step_net, self.tok_patch, self.tok_ch, self.emb
        if deposit == "free":
            self.emb = nn.Embedding(vocab, dt)
            self.tok_patch = nn.Linear(dt, C * h * w)
        else:
            self.tab = nn.Embedding(vocab, C * (BLOB_H * BLOB_W + 2))           # 12 amplitudes + 2 coordonnees par canal
            nn.init.normal_(self.tab.weight, std=0.5)
        self.evo = nn.Sequential(nn.Conv2d(C, ch, 3, padding=1), nn.GELU(), nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(ch, C, 3, padding=1))
        init = [3.0] if self.g == 1 else [math.log(0.8 / 0.2), math.log(0.99 / 0.01)]
        self.ret_logit = nn.Parameter(torch.cat([torch.full((c,), v) for v in init]).clone())   # retention par canal (apprise)
        with torch.no_grad():
            self.init_frame.fill_(0.01)                                                           # etat de repos
        del self.gate_bias

    def deposits(self, x):
        """x : (B, N) -> (B, N, C, H, W), ne depend que des tokens (calcule en parallele)."""
        B, N = x.shape
        C, h, w = self.C, self.h, self.w
        if self.deposit == "free":
            return torch.tanh(self.tok_patch(self.emb(x))).view(B, N, C, h, w)
        t = self.tab(x).view(B * N, C, BLOB_H * BLOB_W + 2)
        amp = torch.tanh(t[..., : BLOB_H * BLOB_W]).reshape(B * N * C, -1)
        px = torch.sigmoid(t[..., -2]).reshape(-1) * (w - BLOB_W - 1)
        py = torch.sigmoid(t[..., -1]).reshape(-1) * (h - BLOB_H - 1)
        return splat(amp, px, py, h, w).view(B, N, C, h, w)

    def positions(self, ids):
        """Positions (x, y) de chaque canal pour des tokens donnes : (len(ids), C, 2)."""
        t = self.tab(ids).view(len(ids), self.C, BLOB_H * BLOB_W + 2)
        return torch.stack([torch.sigmoid(t[..., -2]) * (self.w - BLOB_W - 1), torch.sigmoid(t[..., -1]) * (self.h - BLOB_H - 1)], -1)

    def frames(self, x, state=None):
        B, N = x.shape
        dep = self.deposits(x)
        I = self.init_frame.unsqueeze(0).expand(B, self.C, self.h, self.w) if state is None else state
        a = torch.sigmoid(self.ret_logit).view(1, self.C, 1, 1)
        out = [I]
        for t in range(N):
            I = a * I + dep[:, t]
            if self.training and self.noise > 0:
                I = I + self.noise * torch.randn_like(I)
            for _ in range(self.evolve):
                I = I + torch.tanh(self.evo(self._norm(I)))
            out.append(I)
        return torch.stack(out, dim=1)


def build(spec, vocab, cfg):
    if spec["kind"] == "T":
        return TLM(vocab, cfg, spec["pos"])
    if spec["kind"] == "D":
        return DLM(vocab, cfg, spec["deposit"], spec["evolve"])
    return VLM(vocab, cfg, spec.get("scales", 2))


def continuity_index(frames, groups=2):
    """Moyenne des indices de continuite de chaque groupe de canaux (rapide, lent) : la penalite ne peut pas
    etre satisfaite seulement par le groupe lent, qui est continu par construction."""
    if groups == 1:
        return TextImageAE.continuity_index(frames)
    B, T, C = frames.shape[:3]
    g = frames.reshape(B, T, groups, C // groups, *frames.shape[3:])
    return torch.stack([TextImageAE.continuity_index(g[:, :, i]) for i in range(groups)]).mean()
