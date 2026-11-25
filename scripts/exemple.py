"""Un exemple à lancer soi-même : on prend un modèle entraîné et on regarde ses images évoluer, token par token.

    python scripts/exemple.py boucle                       # L_loop (partie 4) suit une suite de permutations S3
    python scripts/exemple.py boucle --n 60 --graine 3     # plus long, autre suite
    python scripts/exemple.py phrase                       # V λ=0 (partie 2) lit une phrase
    python scripts/exemple.py phrase "Tom saw a big dog. The dog ran to the park."

Écrit un GIF animé et une planche fixe dans figures/exemples/ (ou --sortie). CPU, quelques secondes.
Modèles utilisés (déjà dans le dépôt) : results/cpu_L_loop_s0_model.pt (boucle, 0,25 M paramètres, entraînée sur
12 éléments) et results/s2_V_lam0_model.pt (vidéo, 5,7 M paramètres, TinyStories). « phrase » demande `pip install tokenizers`.
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BOULES = ["#2a78d6", "#eb6834", "#1baf7a"]         # les trois objets permutés
OK, FAUX = "#008300", "#e34948"
plt.rcParams.update({"font.size": 10, "figure.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK})


def rgb(img):
    """(3, H, W) quelconque -> image RGB lisible : chaque canal centré-réduit puis écrasé dans [0, 1]."""
    x = img.reshape(3, -1)
    x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
    return (1 / (1 + np.exp(-1.2 * x))).reshape(img.shape).transpose(1, 2, 0)


def canaux(ax, img, vmax):
    """Les 4 canaux d'une image 4×8×8 côte à côte en 2×2, échelle divergente commune."""
    C, H, W = img.shape
    grand = np.full((2 * H + 1, 2 * W + 1), np.nan)
    for c in range(C):
        i, j = divmod(c, 2)
        grand[i * (H + 1): i * (H + 1) + H, j * (W + 1): j * (W + 1) + W] = img[c]
    cm = plt.get_cmap("RdBu_r").with_extremes(bad=SURF)
    return ax.imshow(grand, cmap=cm, vmin=-vmax, vmax=vmax, interpolation="nearest")


def boules(ax, perm, y, x0=0.0, r=0.32, dx=0.85):
    for k, o in enumerate(perm):
        ax.add_patch(Circle((x0 + k * dx, y), r, fc=BOULES[o], ec=SURF, lw=1.5))


# ---------------------------------------------------------------- la boucle sur S3
def boucle(n, graine, sortie):
    import s3_tasks as s
    from cpu_boucle import SPECS, ST

    G, mult = s.group(3)
    perms = list(itertools.permutations(range(3)))
    m = s.build(SPECS["L_loop"], G, s.model_cfg(ST))
    m.load_state_dict(torch.load(ROOT / "results" / "cpu_L_loop_s0_model.pt", map_location="cpu"))
    m.eval()
    x, y = s.make_batch(np.random.default_rng(graine), G, mult, 1, n)
    with torch.no_grad():
        logits, fr = m(x, return_frames=True)               # fr : (1, n, 2, 4, 8, 8) = [image 1, image 2] après chaque token
    pred = logits[0].argmax(-1).numpy()
    vrai = y[0].numpy()
    tok = x[0].numpy()
    fr = fr[0].numpy()
    acc = (pred == vrai).mean()
    print(f"L_loop, {n} éléments de S3 (entraîné sur 12) : {100 * acc:.0f} % de bonnes réponses")
    v1, v2 = np.percentile(np.abs(fr[:, 0]), 98), np.percentile(np.abs(fr[:, 1]), 98)

    fig = plt.figure(figsize=(9, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.2], width_ratios=[1, 1, 1.1])
    a1, a2, at = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ab = fig.add_subplot(gs[1, :])

    def dessine(t):
        for ax in (a1, a2, at, ab):
            ax.clear(); ax.axis("off")
        canaux(a1, fr[t, 0], v1); a1.set_title("image 1 (entrée)\n4 canaux 8×8", fontsize=9.5)
        canaux(a2, fr[t, 1], v2); a2.set_title("image 2 (sortie)\n4 canaux 8×8", fontsize=9.5)
        at.set_xlim(-0.6, 2.6); at.set_ylim(-0.2, 3.6)
        at.text(-0.5, 3.35, f"token {t + 1} / {n}", fontsize=10, fontweight="bold")
        at.text(-0.5, 2.85, "permutation lue :", fontsize=8.5, color=INK2); boules(at, perms[tok[t]], 2.45, r=0.22, dx=0.6)
        at.text(-0.5, 1.85, "état vrai :", fontsize=8.5, color=INK2); boules(at, perms[vrai[t]], 1.45, r=0.22, dx=0.6)
        bon = pred[t] == vrai[t]
        at.text(-0.5, 0.85, "état prédit :", fontsize=8.5, color=INK2); boules(at, perms[pred[t]], 0.45, r=0.22, dx=0.6)
        at.text(1.95, 0.45, "✓" if bon else "✗", fontsize=16, color=OK if bon else FAUX, va="center")
        ab.set_xlim(-0.5, n - 0.5); ab.set_ylim(-0.6, 1.4)
        ab.axvspan(-0.5, 11.5, color="#efeeea", lw=0)
        ab.text(5.5, 1.15, "longueur d'entraînement", ha="center", fontsize=8, color=INK2)
        ab.text(min(n - 1, 11.5 + (n - 12) / 2), 1.15, "au-delà : jamais vu", ha="center", fontsize=8, color=INK2)
        cols = [OK if p == v else FAUX for p, v in zip(pred[: t + 1], vrai[: t + 1])]
        ab.scatter(range(t + 1), [0.35] * (t + 1), c=cols, s=28, marker="s")
        ab.scatter(range(t + 1, n), [0.35] * (n - t - 1), c="#d9d8d3", s=28, marker="s")
        fig.suptitle("La boucle suit une composition de permutations ; toute sa mémoire est dans les deux images",
                     x=0.02, ha="left", fontsize=11, fontweight="bold")

    os.makedirs(sortie, exist_ok=True)
    anim = FuncAnimation(fig, dessine, frames=n, interval=350)
    anim.save(Path(sortie) / "boucle_s3.gif", writer=PillowWriter(fps=3), dpi=70)
    plt.close(fig)

    # planche fixe : les deux images à quelques instants
    ts = [0, 1, 2, 5, 11, 23, n - 1] if n > 24 else list(range(min(n, 7)))
    fig, axs = plt.subplots(2, len(ts), figsize=(1.6 * len(ts), 3.9))
    for j, t in enumerate(ts):
        for i, v in enumerate((v1, v2)):
            canaux(axs[i, j], fr[t, i], v); axs[i, j].set_xticks([]); axs[i, j].set_yticks([])
            for sp in axs[i, j].spines.values():
                sp.set_visible(False)
        bon = pred[t] == vrai[t]
        axs[0, j].set_title(f"token {t + 1} {'✓' if bon else '✗'}", fontsize=9, color=OK if bon else FAUX)
    axs[0, 0].set_ylabel("image 1", fontsize=9); axs[1, 0].set_ylabel("image 2", fontsize=9)
    fig.suptitle(f"L_loop sur {n} éléments de S3 (entraînée sur 12) : {100 * acc:.0f} % de bonnes réponses",
                 x=0.02, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(Path(sortie) / "boucle_s3.png", dpi=110); plt.close(fig)
    print("écrit", Path(sortie) / "boucle_s3.gif", "et .png")


# ---------------------------------------------------------------- la vidéo sur une phrase
def phrase(texte, sortie):
    from tokenizers import Tokenizer
    import s2_models as M

    tk = Tokenizer.from_file(str(ROOT / "tokenizer" / "bpe_4096.json"))
    ids = tk.encode(texte).ids
    toks = [tk.decode([i]) for i in ids]
    ck = torch.load(ROOT / "results" / "s2_V_lam0_model.pt", map_location="cpu", weights_only=False)
    m = M.build(ck["spec"], ck["V"], ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        logits, fr = m(torch.tensor([ids]), return_frames=True)
    p = torch.softmax(logits[0], -1)
    fr = fr[0].numpy()                                      # (N+1, 6, 8, 8) : 3 canaux rapides puis 3 lents
    pas = np.linalg.norm((fr[1:] - fr[:-1]).reshape(len(ids), -1), axis=1)
    N = len(ids)

    fig = plt.figure(figsize=(9.5, 4.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.3], width_ratios=[1, 1, 1.3])
    a1, a2, at = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ab = fig.add_subplot(gs[1, :])

    def dessine(t):
        for ax in (a1, a2, at):
            ax.clear(); ax.axis("off")
        ab.clear()
        a1.imshow(rgb(fr[t + 1, :3]), interpolation="nearest"); a1.set_title("canaux rapides\n(oublient en ≈ 5 tokens)", fontsize=9)
        a2.imshow(rgb(fr[t + 1, 3:]), interpolation="nearest"); a2.set_title("canaux lents\n(oublient en ≈ 100 tokens)", fontsize=9)
        lu = "".join(toks[: t + 1])
        lu = ("…" + lu[-44:]) if len(lu) > 45 else lu
        at.set_xlim(0, 1); at.set_ylim(0, 1)
        at.text(0, 0.97, f"token {t + 1} / {N} : « {toks[t].strip() or toks[t]} »", fontsize=10, fontweight="bold", va="top")
        at.text(0, 0.84, lu, fontsize=8, color=INK2, va="top", wrap=True)
        suite = f"« {toks[t + 1].strip()} »" if t + 1 < N else "(fin)"
        at.text(0, 0.6, f"le modèle prédit ensuite (vrai : {suite}) :", fontsize=8.5, color=INK2)
        top = p[t].topk(4)
        for k, (v, j) in enumerate(zip(top.values, top.indices)):
            vrai = t + 1 < N and j.item() == ids[t + 1]
            at.barh(0.48 - k * 0.12, v.item(), height=0.08, color=OK if vrai else "#c9c8c2")
            at.text(v.item() + 0.02, 0.48 - k * 0.12, f"« {tk.decode([j.item()]).strip()} » {100 * v.item():.0f} %", fontsize=8.5, va="center")
        ab.bar(range(t + 1), pas[: t + 1], color="#1baf7a", width=0.7)
        ab.bar(range(t + 1, N), pas[t + 1:], color="#e3e2de", width=0.7)
        ab.set_xlim(-0.6, N - 0.4); ab.set_ylim(0, pas.max() * 1.1); ab.set_ylabel("taille du pas", fontsize=8.5, color=INK2)
        ab.set_xticks(range(N)); ab.set_xticklabels([s.strip() or "␣" for s in toks], rotation=70, fontsize=7)
        for s_ in ("top", "right"):
            ab.spines[s_].set_visible(False)
        fig.suptitle("V λ=0 lit une phrase : une image par token, aucune position", x=0.02, ha="left", fontsize=11, fontweight="bold")

    os.makedirs(sortie, exist_ok=True)
    anim = FuncAnimation(fig, dessine, frames=N, interval=500)
    anim.save(Path(sortie) / "video_phrase.gif", writer=PillowWriter(fps=2), dpi=70)
    plt.close(fig)
    print("écrit", Path(sortie) / "video_phrase.gif")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("quoi", choices=["boucle", "phrase"])
    ap.add_argument("texte", nargs="?", default="Lily was hungry. She went to the kitchen and ate a red apple. Then she ate a big rock.")
    ap.add_argument("--n", type=int, default=40, help="boucle : nombre d'éléments de S3")
    ap.add_argument("--graine", type=int, default=0, help="boucle : graine de la suite tirée")
    ap.add_argument("--sortie", default=str(ROOT / "figures" / "exemples"))
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.quoi == "boucle":
        boucle(a.n, a.graine, a.sortie)
    else:
        phrase(a.texte, a.sortie)
