"""Analyses de structure cachee d'un autoencodeur entraine (voir PROTOCOLE.md, amendement 4).

Trois questions, sans jamais regarder l'image « a l'oeil » :
  1. Ablation par zone : effacer un bloc 2x2 de l'image finale ; quelles positions du texte deviennent fausses ?
     (information localisee dans l'image, ou repartie partout ?)  -- tous les bras.
  2. Cartes d'ecriture (bras cumul) : ou le residu de chaque caractere s'ecrit-il ? La localisation de l'ecriture
     depend-elle de la classe du caractere, du caractere exact, de l'instant t ? (R2 + test par permutation)
  3. Derive spatiale (bras cumul) : le centre de masse de l'ecriture se deplace-t-il avec t ?
"""
import glob
import json
import os

import numpy as np
import torch

import config
import data as D
from autoencoder import TextImageAE
from ae_run import chunk_batch

VOWELS = set("aeiouAEIOU")


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = TextImageAE(len(ck["vocab"]), ck["L"], ck["kind"], ck["cfg"]).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck["vocab"]


@torch.no_grad()
def region_ablation(model, x, block=2):
    """Efface (met a 0, la moyenne du latent normalise) chaque bloc block x block de l'image finale.
    Retourne (precision intacte par position (L,), degats (n_blocs, L) = baisse de precision, forme de la grille)."""
    c, h, w = model.latent_shape
    z = model.encode(x)
    base = (model.decode(z).argmax(-1) == x).float().mean(0)
    zi = z.view(-1, c, h, w)
    drops = []
    for i in range(0, h, block):
        for j in range(0, w, block):
            z2 = zi.clone()
            z2[:, :, i : i + block, j : j + block] = 0
            acc = (model.decode(z2.view(-1, model.dz)).argmax(-1) == x).float().mean(0)
            drops.append((base - acc).clamp(min=0))
    return base.cpu().numpy(), torch.stack(drops).cpu().numpy(), (h // block, w // block)


def localization(Dm):
    """Dm : (n_blocs, L) degats. Part du degat de chaque position concentree sur son pire bloc
    (uniforme = 1/n_blocs) et part du degat de chaque bloc concentree sur sa pire position (uniforme = 1/L)."""
    nb, L = Dm.shape
    per_pos = Dm / (Dm.sum(0, keepdims=True) + 1e-12)
    per_blk = Dm / (Dm.sum(1, keepdims=True) + 1e-12)
    return dict(
        max_share_per_position=float(per_pos.max(0).mean()), uniform_blocks=1.0 / nb,
        max_share_per_block=float(per_blk.max(1).mean()), uniform_positions=1.0 / L,
        mean_total_damage_per_block=float(Dm.sum(1).mean()),
    )


def specificity(Dm, n_perm=200, seed=0):
    """Information mutuelle (nats) entre bloc efface et position abimee, d'apres la matrice de degats (n_blocs, L),
    contre le niveau du hasard (cellules melangees : meme total, meme bruit). p = part des melanges >= observe.
    Proche du hasard = code reparti ou degats negligeables ; au-dessus = chaque zone sert des positions precises."""
    tot = Dm.sum()
    if tot <= 0:
        return dict(mi=0.0, mi_permutation_mean=0.0, p=1.0)
    def mi(M):
        P = M / M.sum()
        e = P.sum(1, keepdims=True) @ P.sum(0, keepdims=True)
        m = P > 0
        return float((P[m] * np.log(P[m] / e[m])).sum())
    obs = mi(Dm)
    rng = np.random.default_rng(seed)
    flat = Dm.flatten()
    null = np.array([mi(rng.permutation(flat).reshape(Dm.shape)) for _ in range(n_perm)])
    return dict(mi=obs, mi_permutation_mean=float(null.mean()), p=float((null >= obs).mean()))


@torch.no_grad()
def write_maps(model, x):
    """Bras cumul : |residu| moyen sur les canaux, (B, L, h, w), et classes de caracteres."""
    fr = model.encode_frames(x)
    return (fr[:, 1:] - fr[:, :-1]).abs().mean(2)


def _r2(X, labels, n_perm=200, seed=0, char_ids=None, cls_of=None):
    """Part de variance de X (N, d) expliquee par les etiquettes (R2 = SSB/SST) + R2 moyen sous permutation.
    Si char_ids et cls_of sont donnes (etiquette = classe d'un caractere), le nul permute l'affectation
    caractere -> classe (memes effectifs) au lieu de melanger les lignes : l'identite du caractere explique deja
    beaucoup, la question est de savoir si les classes lexicales ajoutent quelque chose au-dela d'un regroupement quelconque."""
    N = X.shape[0]
    sst = ((X - X.mean(0)) ** 2).sum()
    def r2(lab):
        ss = 0.0
        for k in np.unique(lab):
            m = lab == k
            ss += m.sum() * ((X[m].mean(0) - X.mean(0)) ** 2).sum()
        return ss / sst
    obs = r2(labels)
    rng = np.random.default_rng(seed)
    if cls_of is not None:
        perm = np.array([r2(rng.permutation(cls_of)[char_ids]) for _ in range(n_perm)])
    else:
        perm = np.array([r2(rng.permutation(labels)) for _ in range(n_perm)])
    return dict(r2=float(obs), r2_permutation_mean=float(perm.mean()), p=float((perm >= obs).mean()))


def _rank(a):
    return np.argsort(np.argsort(a)).astype(float)


def analyze(root, tag, kind, L, val_ids, device, n_chunks=512):
    base = os.path.join(root, "results", "ae_%s_L%d_%s" % (kind, L, tag))
    ck = base + "_model.pt"
    if not os.path.exists(ck):
        print("[analyse %s] pas de modele sauvegarde, saute." % tag)
        return None
    model, vocab = load_checkpoint(ck, device)
    x = chunk_batch(np.random.default_rng(2024), val_ids, n_chunks, L, device)
    out = dict(tag=tag, arm=kind, chunk=L)

    # 1. ablation par zone
    base_acc, Dm, grid = region_ablation(model, x)
    out["ablation"] = dict(intact_char_acc=float(base_acc.mean()), **localization(Dm), specificite=specificity(Dm))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw=dict(width_ratios=[3, 1.2]))
    im = axs[0].imshow(Dm, aspect="auto", cmap="magma")
    axs[0].set_xlabel("position dans le morceau"); axs[0].set_ylabel("bloc 2x2 efface (ligne par ligne)")
    axs[0].set_title("baisse de precision", fontsize=9); fig.colorbar(im, ax=axs[0], fraction=0.03)
    axs[1].imshow(Dm.sum(1).reshape(grid), cmap="magma"); axs[1].set_title("degat total par zone", fontsize=9); axs[1].axis("off")
    fig.savefig(base + "_ablation.png", dpi=120, bbox_inches="tight"); plt.close(fig)

    # 2-3. cartes d'ecriture (bras cumul)
    if kind == "cumul":
        dl = write_maps(model, x)                                   # (B, L, h, w)
        B, _, h, w = dl.shape
        X = dl.reshape(B * L, h * w).cpu().numpy()
        X = X / (X.sum(1, keepdims=True) + 1e-12)                   # ou l'ecriture se fait, pas combien
        ids = x.reshape(-1).cpu().numpy()
        chars = [vocab[i] for i in range(len(vocab))]
        def cls(i):
            ch = chars[i]
            if ch == " ": return 0
            if ch in VOWELS: return 1
            if ch.isalpha(): return 2
            return 3
        cls_of = np.array([cls(i) for i in range(len(vocab))])
        tt = np.tile(np.arange(L), B)
        out["ecriture"] = dict(
            par_classe=_r2(X, cls_of[ids], char_ids=ids, cls_of=cls_of),   # espace / voyelle / consonne / autre ; nul au niveau des caracteres
            par_caractere=_r2(X, ids),
            par_instant=_r2(X, tt),
        )
        mean_t = dl.mean(0).cpu().numpy()                           # (L, h, w)
        ys, xs = np.mgrid[0:h, 0:w]
        com_x = (mean_t * xs).sum((1, 2)) / mean_t.sum((1, 2)); com_y = (mean_t * ys).sum((1, 2)) / mean_t.sum((1, 2))
        t = np.arange(L, dtype=float)
        sp = lambda a: float(np.corrcoef(_rank(t), _rank(a))[0, 1])
        out["derive"] = dict(spearman_com_x_vs_t=sp(com_x), spearman_com_y_vs_t=sp(com_y),
                             com_x_debut_fin=[float(com_x[:3].mean()), float(com_x[-3:].mean())],
                             com_y_debut_fin=[float(com_y[:3].mean()), float(com_y[-3:].mean())],
                             deplacement_px=float(np.hypot(com_x[-3:].mean() - com_x[:3].mean(), com_y[-3:].mean() - com_y[:3].mean())))
        # figures : cartes par classe, par instant
        fig, axs = plt.subplots(1, 4, figsize=(8, 2.3))
        for k, name in enumerate(["espace", "voyelle", "consonne", "autre"]):
            m = X[cls_of[ids] == k]
            axs[k].imshow(m.mean(0).reshape(h, w), cmap="viridis"); axs[k].set_title(name, fontsize=8); axs[k].axis("off")
        fig.suptitle("ou le residu s'ecrit, par classe de caractere", fontsize=9)
        fig.savefig(base + "_write_classes.png", dpi=120, bbox_inches="tight"); plt.close(fig)
        cols = min(L, 16); rows = (L + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(1.0 * cols, 1.1 * rows), squeeze=False)
        for i, a in enumerate(axs.flat):
            a.axis("off")
            if i < L:
                a.imshow(mean_t[i], cmap="viridis"); a.set_title("t=%d" % (i + 1), fontsize=7)
        fig.suptitle("ou le residu s'ecrit, a chaque instant t (moyenne sur les morceaux)", fontsize=8)
        fig.savefig(base + "_write_time.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    with open(base.replace("/ae_", "/an_", 1) + ".json", "w") as f:
        json.dump(out, f, indent=1)
    return out


def summarize(root="."):
    rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, "results", "an_*.json")))]
    lines = ["# Analyses de structure cachee", "",
             "- Ablation : information mutuelle (nats) entre le bloc 2x2 efface et la position du texte abimee, contre le hasard (cellules melangees) ; p = part des melanges au moins aussi structures. Degat moyen par bloc = somme sur les positions de la baisse de precision quand un bloc est efface (faible = code tres redondant).",
             "- Ecriture (bras cumul) : part de variance (R2) de la **localisation** de l'ecriture expliquee par le caractere exact / sa classe (espace, voyelle, consonne, autre) / l'instant t, contre le hasard. Le R2 par caractere est attendu par construction (l'ecriture part d'un motif propre au caractere) ; la classe est comparee a des regroupements arbitraires de caracteres.",
             "- Derive : correlation de rang entre t et la position (x, y) du centre de masse de l'ecriture ; deplacement en pixels entre le debut et la fin du morceau.", "",
             "| modele | precision intacte | degat moyen par bloc | ablation : MI (hasard), p | R2 caractere (hasard) | R2 classe (hasard), p | R2 instant (hasard) | derive x / y, deplacement |",
             "|:--|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        a = r["ablation"]; e = r.get("ecriture"); d = r.get("derive"); sp = a["specificite"]
        g = lambda k: "%.3f (%.3f)" % (e[k]["r2"], e[k]["r2_permutation_mean"]) if e else "-"
        gc = "%.3f (%.3f), p=%.2f" % (e["par_classe"]["r2"], e["par_classe"]["r2_permutation_mean"], e["par_classe"]["p"]) if e else "-"
        dd = "%.2f / %.2f, %.1f px" % (d["spearman_com_x_vs_t"], d["spearman_com_y_vs_t"], d["deplacement_px"]) if d else "-"
        lines.append("| %s %s L=%d | %.1f %% | %.3f | %.3f (%.3f), p=%.2f | %s | %s | %s | %s |" % (
            r["arm"], r["tag"], r["chunk"], 100 * a["intact_char_acc"], a["mean_total_damage_per_block"],
            sp["mi"], sp["mi_permutation_mean"], sp["p"], g("par_caractere"), gc, g("par_instant"), dd))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(root, "results", "an_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt
