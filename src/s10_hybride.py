"""Etape 10 : transformer a fenetre glissante (Twin) et hybride transformer + remanence (H).

TwinLM : meme modele que T_rope (memes parametres, memes noms) mais l'attention causale est limitee aux `window` derniers tokens
(le token courant compte : t voit t-window+1 .. t). RoPE est relatif et calcule pour toute longueur : pas de limite de longueur.
HybridLM : TwinLM (fenetre w) + une remanence (images I1, I2 de LoopLM, memes calculs que FastLoop) qui parcourt les memes tokens
sequentiellement. A la position t, l'embedding d'entree du transformer recoit en plus Linear(I2 apres le token t-1) (zero pour t=0).
Les images sont pilotees par les embeddings de tokens comme dans LoopLM (depot sur I1, T1 -> I2, T2 -> I1, portes). Les logits
viennent du transformer seul ; la tete de la remanence n'est pas utilisee. Entrainement de bout en bout. Causal : la sortie en t ne
depend que des tokens <= t (I2 lu avant le depot du token t).
Le transformer ne voit que w tokens ; tout ce qui est plus ancien ne peut lui parvenir que par les images (memoire a longue portee).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from s2_models import Block, Stack, TLM, rope
from s4_loop import patchify, LoopLM
from s8_rapide import _t


class WBlock(Block):
    window = None                                     # None : causal complet (comportement de Block)

    def forward(self, x):
        if self.window is None: return super().forward(x)
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.use_rope: q, k = rope(q), rope(k)
        i = torch.arange(T, device=x.device)
        mask = (i[None, :] <= i[:, None]) & (i[None, :] > i[:, None] - self.window)       # (T,T) : j <= t et j > t - w
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class TwinLM(TLM):
    """T_rope a fenetre glissante `window` (memes parametres que T_rope)."""

    def __init__(self, vocab, cfg, window):
        super().__init__(vocab, cfg, "rope"); self.window = window
        for b in self.stack.blocks: b.__class__ = WBlock; b.window = window


class HybridLM(TwinLM):
    """Transformer a fenetre + memoire remanente (voir docstring du module). Images C x H x W (defaut 4x8x8), patch 2."""

    def __init__(self, vocab, cfg, window, C=4, H=8, W=8, p=2):
        super().__init__(vocab, cfg, window)
        self.rem = LoopLM(vocab, dict(cfg, loop_layers=1, loop_c=C, loop_h=H, loop_w=W, loop_p=p), True)
        self.rem.head = nn.Identity()                  # la tete de la remanence est inutilisee
        self.mem = nn.Linear(C * H * W, cfg["d_model"])
        self.C, self.H, self.W, self.p = C, H, W, p

    def etats(self, x):
        """I2 (en patchs, aplati) apres chaque token : (B,N,C*H*W). Meme boucle que FastLoop."""
        m = self.rem; B, N = x.shape; C, H, W, p = self.C, self.H, self.W, self.p
        dep = patchify(torch.tanh(m.dep(m.emb(x))).view(B * N, C, H, W), p).view(B, N, -1, C * p * p)
        I1 = patchify(m.i1.unsqueeze(0), p).expand(B, -1, -1); I2 = patchify(m.i2.unsqueeze(0), p).expand(B, -1, -1); outs = []
        for t in range(N):
            I1 = I1 + dep[:, t]
            r2, g2 = _t(m.t1, I1)
            I2 = torch.sigmoid(g2 + 3.0) * I2 + torch.tanh(r2); outs.append(I2)
            r1, g1 = _t(m.t2, I2); I1 = torch.sigmoid(g1 + 3.0) * I1 + torch.tanh(r1)
        return torch.stack(outs, 1).flatten(2)

    def forward(self, x, return_frames=False, state=None):
        mem = self.mem(F.layer_norm(self.etats(x), (self.C * self.H * self.W,)))          # (B,N,d), indice t = apres le token t
        mem = F.pad(mem[:, :-1], (0, 0, 1, 0))                                           # decale : en t on lit l'etat apres t-1 ; zero en t=0
        logits = self.head(self.stack(self.emb(x) + mem))
        return (logits, None) if return_frames else logits
