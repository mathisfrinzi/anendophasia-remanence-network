"""Rémanence dont T1 et T2 sont les blocs d'un transformer a patchs PRE-ENTRAINE (ViT de timm) ; amendement 39.
On garde les blocs (+ la normalisation finale) du ViT ; on retire son plongement de patchs et sa tete de classification, remplaces par
 - une projection d'entree (pd -> d) et un plongement de position appris ;
 - une projection de sortie (d -> 2 pd) qui donne le residu et l'oubli de l'image.
Memes images (en patchs), memes portes, memes depots que LoopLM ; seule la mecanique interne de T1 et T2 change.
mode : 'ln' (blocs gelés sauf leurs LayerNorm, comme dans les 'frozen pretrained transformers'), 'frozen' (blocs entierement gelés), 'all' (tout entraine, lr des blocs x0,1)."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from s4_loop import patchify, unpatchify


class PreT(nn.Module):
    def __init__(self, vit, pd, n, mode, out_scale=0.1):
        super().__init__()
        d = vit.embed_dim; self.blocks, self.norm = copy.deepcopy(vit.blocks), copy.deepcopy(vit.norm)
        self.inp, self.pos, self.out = nn.Linear(pd, d), nn.Parameter(torch.randn(n, d) * 0.02), nn.Linear(d, 2 * pd)
        with torch.no_grad(): self.out.weight.mul_(out_scale)
        for p in list(self.blocks.parameters()) + list(self.norm.parameters()):
            p.requires_grad = (mode == "all") or (mode == "ln" and p.ndim == 1 and False)
        if mode == "ln":
            for mod in list(self.blocks.modules()) + [self.norm]:
                if isinstance(mod, nn.LayerNorm):
                    for p in mod.parameters(): p.requires_grad = True

    def forward(self, img):                                    # img (B, n, pd) -> (residu, oubli)
        h = self.inp(F.layer_norm(img, img.shape[1:])) + self.pos
        h = self.blocks(h)
        return self.out(self.norm(h)).chunk(2, dim=-1)


class PreLoop(nn.Module):
    """base : un LoopLM (fournit plongement des tokens, depot, tete de lecture, images initiales) dont T1 et T2 sont remplaces."""
    def __init__(self, base, backbone="vit_small_patch16_224", mode="ln", pretrained=True, ckpt=True):
        super().__init__()
        import timm
        vit = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        m = base; p = m.t1.p; self.C, self.H, self.W, self.p, self.ckpt = m.C, m.H, m.W, p, ckpt
        pd, n = m.C * p * p, (m.H // p) * (m.W // p)
        self.emb, self.dep, self.head, self.i1, self.i2 = m.emb, m.dep, m.head, m.i1, m.i2
        self.t1, self.t2 = PreT(vit, pd, n, mode), PreT(vit, pd, n, mode)
        self.state_out = None

    def _step(self, I1, I2, dep_t):
        I1 = I1 + dep_t
        r2, g2 = self.t1(I1); I2 = torch.sigmoid(g2 + 3.0) * I2 + torch.tanh(r2)
        r1, g1 = self.t2(I2); I1 = torch.sigmoid(g1 + 3.0) * I1 + torch.tanh(r1)
        return I1, I2

    def forward(self, x):
        B, N = x.shape; C, H, W, p = self.C, self.H, self.W, self.p
        dep = patchify(torch.tanh(self.dep(self.emb(x))).view(B * N, C, H, W), p).view(B, N, -1, C * p * p)
        I1 = patchify(self.i1.unsqueeze(0), p).expand(B, -1, -1); I2 = patchify(self.i2.unsqueeze(0), p).expand(B, -1, -1); outs = []
        for t in range(N):
            if self.ckpt and torch.is_grad_enabled(): I1, I2 = checkpoint(self._step, I1, I2, dep[:, t], use_reentrant=False)   # recalcul a la retropropagation : memoire divisee par N
            else: I1, I2 = self._step(I1, I2, dep[:, t])
            outs.append(I2)
        return self.head(unpatchify(torch.stack(outs, 1).view(B * N, -1, C * p * p), C, H, W, p).view(B, N, -1))


def param_groups(model, lr, block_scale=0.1):
    """groupes d'optimiseur : blocs pre-entraines a lr*block_scale, le reste a lr (cle 'scale' lue par step_fn)."""
    blocks = [p for t in (model.t1, model.t2) for p in list(t.blocks.parameters()) + list(t.norm.parameters()) if p.requires_grad]
    ids = {id(p) for p in blocks}; rest = [p for p in model.parameters() if p.requires_grad and id(p) not in ids]
    return [dict(params=rest, lr=lr, scale=1.0)] + ([dict(params=blocks, lr=lr * block_scale, scale=block_scale)] if blocks else [])
