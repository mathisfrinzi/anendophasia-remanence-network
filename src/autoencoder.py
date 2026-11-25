import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class SpatialStage(nn.Module):
    """Etage applique au latent (192 valeurs), en residuel.

    kind = "image"  : le latent est lu comme une image C x H x W ; 3 convolutions 3x3
                      (voisinage spatial reel : un pixel ne voit que ses voisins).
    kind = "vector" : le latent est un vecteur plat ; MLP a un nombre de parametres
                      egal a celui de l'etage convolutif (temoin : meme budget, sans geometrie).
    kind = "none"   : identite (temoin : aucun etage).
    """

    def __init__(self, kind, c, h, w, ch):
        super().__init__()
        self.kind, self.c, self.h, self.w = kind, c, h, w
        d = c * h * w
        if kind == "image":
            self.net = nn.Sequential(
                nn.Conv2d(c, ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(ch, c, 3, padding=1),
            )
        elif kind == "vector":
            n_conv = sum(p.numel() for p in SpatialStage("image", c, h, w, ch).net.parameters())
            hid = max(1, round((n_conv - d) / (2 * d + 1)))  # 2*d*hid + hid + d = n_conv
            self.net = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))
        elif kind == "none":
            self.net = None
        else:
            raise ValueError(kind)

    def forward(self, z):  # z : (B, d)
        if self.net is None:
            return z
        if self.kind == "image":
            B = z.shape[0]
            return z + self.net(z.view(B, self.c, self.h, self.w)).view(B, -1)
        return z + self.net(z)


def _transformer(d, n_heads, n_layers):
    layer = nn.TransformerEncoderLayer(
        d_model=d, nhead=n_heads, dim_feedforward=4 * d, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    )
    return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)


class TextImageAE(nn.Module):
    """Autoencodeur texte -> latent 192 valeurs -> texte.

    Un morceau de L caracteres est encode en UN latent de 192 valeurs (une image 3x8x8
    pour les bras "image" et "cumul", un vecteur plat pour les temoins), normalise
    (variance 1 par echantillon) et bruite pendant l'entrainement ; le decodeur relit les
    L caracteres d'un coup (decodage parallele, bidirectionnel) depuis ce seul latent.

    Bras "cumul" (video continue) : l'image est construite caractere par caractere,
    frame_t = oubli_t * frame_{t-1} + residu_t, ou oubli_t et residu_t sortent d'un petit
    reseau convolutif qui voit le caractere ET la frame courante (ecriture contextuelle).
    Aucune position explicite n'est donnee : l'ordre ne peut venir que de la dynamique.
    La suite des frames est la video ; seule la derniere est lue par le decodeur.
    """

    def __init__(self, vocab, L, kind, cfg=None):
        super().__init__()
        cfg = cfg or config.AE
        self.L, self.kind = L, kind
        d = cfg["d_model"]
        c, h, w = cfg["latent_c"], cfg["latent_h"], cfg["latent_w"]
        self.latent_shape = (c, h, w)
        self.dz = c * h * w
        self.noise_std = cfg["noise_std"]
        self.emb = nn.Embedding(vocab, d)
        if kind == "cumul":
            k = cfg["cumul_tok_ch"]
            ch = cfg["spatial_ch"]
            self.k = k
            self.tok_patch = nn.Linear(d, self.dz)   # quoi : motif propose par le caractere
            self.tok_ch = nn.Linear(d, k)            # le caractere, diffuse sur tous les pixels
            self.init_frame = nn.Parameter(torch.zeros(c, h, w))
            self.step_net = nn.Sequential(           # (frame, motif, caractere) -> (residu, oubli)
                nn.Conv2d(2 * c + k, ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(ch, ch, 3, padding=1), nn.GELU(),
                nn.Conv2d(ch, 2 * c, 3, padding=1),
            )
            self.spatial_dec = SpatialStage("image", c, h, w, cfg["spatial_ch"])
        else:
            self.pos_e = nn.Parameter(torch.randn(L, d) * 0.02)
            self.enc = _transformer(d, cfg["n_heads"], cfg["n_layers"])
            self.to_latent = nn.Linear(L * d, self.dz)
            self.spatial_enc = SpatialStage(kind, c, h, w, cfg["spatial_ch"])
            self.spatial_dec = SpatialStage(kind, c, h, w, cfg["spatial_ch"])
        self.from_latent = nn.Linear(self.dz, L * d)
        self.pos_d = nn.Parameter(torch.randn(L, d) * 0.02)
        self.dec = _transformer(d, cfg["n_heads"], cfg["n_layers"])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        self.d = d

    def encode_frames(self, x):
        """Bras cumul : la video (B, L+1, C, H, W), frame 0 = image initiale apprise."""
        B, L = x.shape
        c, h, w = self.latent_shape
        e = self.emb(x)
        p = self.tok_patch(e).view(B, L, c, h, w)
        q = self.tok_ch(e)
        I = self.init_frame.unsqueeze(0).expand(B, c, h, w)
        frames = [I]
        for t in range(L):
            inp = torch.cat([I, p[:, t], q[:, t].view(B, self.k, 1, 1).expand(B, self.k, h, w)], dim=1)
            out = self.step_net(inp)
            res, g = out[:, :c], out[:, c:]
            I = torch.sigmoid(g + 3.0) * I + torch.tanh(res)   # oubli ~0,95 au depart, appris ensuite
            frames.append(I)
        return torch.stack(frames, dim=1)

    def encode(self, x, return_frames=False):
        """x : (B, L) -> latent normalise (B, dz), sans bruit (et, bras cumul, la video si demandee)."""
        B = x.shape[0]
        if self.kind == "cumul":
            frames = self.encode_frames(x)
            z = F.layer_norm(frames[:, -1].reshape(B, -1), (self.dz,))
            return (z, frames) if return_frames else z
        h = self.emb(x) + self.pos_e
        h = self.enc(h)
        z = self.spatial_enc(self.to_latent(h.reshape(B, -1)))
        return F.layer_norm(z, (self.dz,))  # variance 1 : l'echelle ne peut pas servir a stocker de l'info

    def decode(self, z):
        B = z.shape[0]
        z = self.spatial_dec(z)
        h = self.from_latent(z).view(B, self.L, self.d) + self.pos_d
        h = self.dec(h)
        return self.head(self.ln(h))

    def forward(self, x, return_frames=False):
        if return_frames:
            z, frames = self.encode(x, return_frames=True)
        else:
            z = self.encode(x)
        if self.training and self.noise_std > 0:
            z = z + self.noise_std * torch.randn_like(z)
        logits = self.decode(z)
        return (logits, frames) if return_frames else logits

    @staticmethod
    def continuity_index(frames):
        """frames : (B, L+1, ...). Pas moyen entre frames consecutives / distance moyenne entre
        frames de textes differents au meme instant (decalage circulaire du lot). Differentiable."""
        fr = frames.flatten(2)
        step = (fr[:, 1:] - fr[:, :-1]).norm(dim=-1).mean()
        other = (fr[1:, 1:] - fr[:-1, 1:]).norm(dim=-1).mean()
        return step / (other + 1e-8)

    def latent_images(self, x):
        """Latent sans bruit, forme (B, C, H, W), pour visualisation."""
        c, h, w = self.latent_shape
        return self.encode(x).view(-1, c, h, w)
