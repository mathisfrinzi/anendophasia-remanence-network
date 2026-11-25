"""Schémas pédagogiques du rapport (un ou deux par partie). Aucun calcul : des dessins.

    python scripts/schemas.py        # écrit figures/schemas/*.svg
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "anendophasia"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "schemas"

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a85"
BLUE, ORANGE, AQUA, GREEN, VIOLET, YELLOW, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#008300", "#6250d6", "#eda100", "#e34948", "#8a8a85"
PALE = {"blue": "#e4eefa", "aqua": "#dcf3ea", "violet": "#e9e6fa", "green": "#dcefdc", "gray": "#efeeea", "yellow": "#fbf0d6", "red": "#fbe3e3"}
DASH = (0, (4, 3))
plt.rcParams.update({"font.size": 10, "text.color": INK, "figure.facecolor": SURF, "savefig.facecolor": SURF})


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_facecolor(SURF)
    return fig, ax


def box(ax, x, y, w, h, text="", fc="#ffffff", ec="#9c9b95", fs=9.5, bold=False, ls="-", color=INK, lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=lw, ls=ls))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p, q, text=None, color=INK2, ls="-", rad=0.0, fs=8.5, off=(0, 0.14), lw=1.5, ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw, color=color, ls=ls,
                                 connectionstyle="arc3,rad=%s" % rad, shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha=ha, fontsize=fs, color=color, linespacing=1.25)


def pixels(ax, x, y, s, seed, n=6, cmap="viridis", hot=None, lo=0.25, hi=0.75):
    rng = np.random.default_rng(seed); c = s / n
    cm = plt.get_cmap(cmap)
    for i in range(n):
        for j in range(n):
            v = rng.uniform(lo, hi)
            if hot and (i, j) in hot:
                v = 0.97
            ax.add_patch(Rectangle((x + j * c, y + (n - 1 - i) * c), c, c, fc=cm(v), ec=SURF, lw=0.6))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    if os.environ.get("APERCU"):
        fig.savefig(Path(os.environ["APERCU"]) / f"schema_{name}.png", bbox_inches="tight", dpi=70)
    plt.close(fig)
    print("écrit figures/schemas/%s.svg" % name)


# ---------------------------------------------------------------- 0. l'idée
def idee():
    fig, ax = canvas(13, 4.4)
    ax.text(0.1, 4.15, "Un modèle de langage qui pense en images", fontsize=13, fontweight="bold")
    ax.text(0.1, 3.8, "Au lieu d'un monologue de tokens, l'état interne est une image (puis une vidéo) que le modèle invente et met à jour token après token.",
            fontsize=9.5, color=INK2)
    names = ["texte", "encodeur\nappris", "image / vidéo\ninventée par\nle modèle", "transformer(s)", "image de\nsortie", "décodeur", "texte"]
    xs = [0.1, 1.85, 3.65, 5.95, 8.0, 9.85, 11.6]; ws = [1.2, 1.4, 1.95, 1.6, 1.4, 1.3, 1.2]
    for i, (n, x, w) in enumerate(zip(names, xs, ws)):
        img = n.startswith("image")
        box(ax, x, 1.55, w, 1.3, n, fc=PALE["aqua"] if img else "#ffffff", ec=AQUA if img else "#9c9b95")
        if i < 6:
            arrow(ax, (x + w, 2.2), (xs[i + 1], 2.2))
    for y, x0, x1, c, t, ls in [(1.15, 0.1, 5.6, AQUA, "Partie 1 : texte → image → même texte (autoencodeur). Fait.", "-"),
                                (0.65, 3.65, 7.55, VIOLET, "Parties 2 à 6 : image(s) → transformer(s) → token suivant. Fait.", "-"),
                                (0.15, 8.0, 12.8, GRAY, "Sortie en image + décodeur : pas encore fait.", DASH)]:
        ax.plot([x0, x1], [y + 0.2, y + 0.2], color=c, lw=3, ls=ls, solid_capstyle="round")
        ax.text(x0, y - 0.13, t, color=c if c != GRAY else INK2, fontsize=9)
    save(fig, "idee")


def frame():
    fig, ax = canvas(13, 4.3)
    toks = ["« je »", "« mange »", "« une »"]
    for k in range(4):
        x = 0.4 + k * 3.15
        pixels(ax, x, 1.75, 1.6, 10 + k, hot=[(1 + k % 3, 1 + k), (2 + k % 3, 1 + k)] if k else None)
        ax.text(x + 0.8, 1.47, ["image de départ (apprise)", "après le token 1", "après le token 2", "après le token 3"][k], ha="center", fontsize=8.5, color=INK2)
        if k:
            arrow(ax, (x - 1.2, 2.55), (x - 0.12, 2.55))
            ax.text(x - 0.66, 3.0, toks[k - 1], fontsize=10, ha="center", fontweight="bold")
            ax.text(x - 0.66, 2.3, "écrit un\nrésidu", fontsize=8, ha="center", color=INK2, va="top")
    ax.text(6.4, 0.85, "image(t)  =  oubli(t) ⊙ image(t − 1)  +  résidu(image(t − 1), token t)", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.4, 0.38, "L'oubli est une porte apprise par pixel ; deux groupes de canaux : « rapides » (≈ 5 tokens) et « lents » (≈ 100 tokens).\n"
            "Aucune position n'est fournie : l'ordre n'existe que dans l'image qui s'accumule.", ha="center", fontsize=9, color=INK2, va="center", linespacing=1.4)
    ax.text(0.1, 4.05, "Le mécanisme : une frame par token", fontsize=12, fontweight="bold")
    save(fig, "frame")


# ---------------------------------------------------------------- 1. autoencodeur
def autoencodeur():
    fig, ax = canvas(13, 4.1)
    ax.text(0.1, 3.85, "Partie 1 : un morceau de texte doit traverser une image de 192 valeurs", fontsize=12, fontweight="bold")
    box(ax, 0.1, 1.6, 1.7, 1.1, "« sick. Shorts a »\n(L caractères)", fs=9)
    box(ax, 2.3, 1.6, 1.7, 1.1, "encodeur\n(transformer)")
    box(ax, 6.4, 1.6, 1.8, 1.1, "décodeur\n(transformer)")
    box(ax, 8.7, 1.6, 1.7, 1.1, "« sick. Shorts a »\n(relu)", fs=9)
    arrow(ax, (1.8, 2.15), (2.3, 2.15)); arrow(ax, (8.2, 2.15), (8.7, 2.15))
    arrows = [("cumul", "vidéo continue : une frame\npar caractère, avec oubli", AQUA, 3.0), ("image", "image d'un coup", BLUE, 2.35),
              ("vecteur", "vecteur plat de 192", YELLOW, 1.7), ("aucun", "pas d'étage spatial", GRAY, 1.05)]
    for name, t, c, y in arrows:
        box(ax, 4.5, y - 0.25, 1.4, 0.5, name, fc="#ffffff", ec=c, fs=9, bold=True)
        ax.text(6.0, y, "", fontsize=8)
        arrow(ax, (4.0, 2.15), (4.5, y), color=c, lw=1.1); arrow(ax, (5.9, y), (6.4, 2.15), color=c, lw=1.1)
        ax.text(10.8, y, t, fontsize=8.5, color=INK2, va="center")
        ax.plot([10.55, 10.7], [y, y], color=c, lw=4, solid_capstyle="round")
    pixels(ax, 4.62, 3.35, 0.36, 2, n=4); pixels(ax, 5.05, 3.35, 0.36, 3, n=4); pixels(ax, 5.48, 3.35, 0.36, 5, n=4)
    ax.text(0.1, 0.35, "Mesures : caractères justes et morceaux exacts, à L = 8, 16, 32, 64 ; puis une pénalité λ qui force deux frames successives à se ressembler (vidéo continue).",
            fontsize=9, color=INK2)
    save(fig, "autoencodeur")


# ---------------------------------------------------------------- 2. modèles de l'étape 2
def modeles():
    fig, ax = canvas(13, 6.0)
    def row(y, title, boxes, col):
        ax.text(0.1, y + 1.05, title, fontsize=10.5, fontweight="bold", color=col)
        x = 0.1
        for t, w, fc in boxes:
            box(ax, x, y - 0.08, w, 0.96, t, fc=fc)
            if x > 0.1:
                arrow(ax, (x - 0.25, y + 0.4), (x, y + 0.4))
            x += w + 0.25
    row(4.3, "Transformer ordinaire : T_rope (positions rotatives) ou T_nope (aucune position)",
        [("tokens", 1.2, "#fff"), ("plongement\n(+ RoPE ou rien)", 2.0, "#fff"), ("transformer causal\n(voit tous les tokens passés)", 3.1, PALE["blue"]), ("token\nsuivant", 1.3, "#fff")], BLUE)
    row(2.4, "Vidéo V : l'écriture lit l'image courante",
        [("tokens", 1.2, "#fff"), ("image(t) = oubli ⊙ image(t−1)\n+ résidu(image(t−1), token)", 3.4, PALE["aqua"]), ("lecteur\nconvolutif", 1.4, "#fff"),
         ("transformer causal\nSANS position", 2.3, PALE["blue"]), ("token\nsuivant", 1.2, "#fff")], AQUA)
    row(0.5, "Image qui évolue D : le dépôt ne lit pas l'image",
        [("tokens", 1.2, "#fff"), ("dépôt par le token seul\n(tache de 12 pixels\nà position apprise)", 2.6, PALE["green"]), ("évolution libre\n0, 1 ou 2 pas", 1.6, PALE["green"]),
         ("lecteur", 1.0, "#fff"), ("transformer causal\nSANS position", 2.3, PALE["blue"]), ("token\nsuivant", 1.2, "#fff")], GREEN)
    ax.text(0.1, 0.05, "V et D : une frame 6×8×8 par token ; le transformer lit la suite des frames. ≈ 6 M paramètres partout, TinyStories, BPE 4 096, fenêtres de 256 tokens.",
            fontsize=8.5, color=INK2)
    save(fig, "modeles")


# ---------------------------------------------------------------- 3. tâches d'état
def taches():
    fig, ax = canvas(13, 4.6)
    ax.text(0.1, 4.35, "Partie 3 : suivre un état, la composition de permutations", fontsize=12, fontweight="bold")
    ax.text(0.1, 4.0, "Chaque token est une permutation de 3 objets ; à chaque pas, le modèle doit dire l'ordre courant (produit cumulé). 6 états possibles.",
            fontsize=9.5, color=INK2)
    cols = [BLUE, ORANGE, AQUA]
    state = [0, 1, 2]
    perms = [(1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 0, 2), (0, 2, 1)]
    names = ["échange\n1↔2", "échange\n2↔3", "échange\n1↔3", "échange\n1↔2", "échange\n2↔3"]
    x0 = 0.5
    def draw_state(x, st):
        for k, o in enumerate(st):
            ax.add_patch(Circle((x + k * 0.38, 2.6), 0.15, fc=cols[o], ec=SURF, lw=1.5))
    draw_state(x0, state); ax.text(x0 + 0.38, 2.15, "départ", ha="center", fontsize=8.5, color=INK2)
    for i, (p, n) in enumerate(zip(perms, names)):
        state = [state[p[k]] for k in range(3)]
        x = x0 + (i + 1) * 2.2
        arrow(ax, (x - 1.25, 2.6), (x - 0.3, 2.6))
        box(ax, x - 1.1, 3.0, 0.7, 0.62, n, fs=7.5, fc=PALE["gray"], ec=GRAY)
        draw_state(x, state)
        ax.text(x + 0.38, 2.15, f"cible t = {i + 1}", ha="center", fontsize=8.5, color=INK2)
    ax.text(12.55, 2.6, "…", fontsize=16, va="center")
    # longueurs
    ax.add_patch(Rectangle((0.5, 1.0), 24 / 256 * 11.5, 0.35, fc=VIOLET, ec="none"))
    ax.add_patch(Rectangle((0.5 + 24 / 256 * 11.5, 1.0), (232 / 256) * 11.5, 0.35, fc=PALE["violet"], ec="none"))
    ax.text(0.5, 0.75, "entraînement : 24 éléments", fontsize=8.5, color=VIOLET, va="top")
    ax.text(12.0, 0.75, "évaluation jusqu'à 256 éléments (extrapolation en longueur)", fontsize=8.5, color=INK2, ha="right", va="top")
    ax.text(0.1, 0.2, "S2 = parité (2 états), S3 = 3 objets (6 états, le plus petit groupe non commutatif), S5 = 5 objets (120 états, non résoluble).",
            fontsize=8.5, color=INK2)
    save(fig, "taches")


# ---------------------------------------------------------------- 4. la boucle
def boucle():
    fig, ax = canvas(13, 5.6)
    ax.text(0.1, 5.35, "Partie 4 : deux images, deux transformers, une boucle", fontsize=12, fontweight="bold")
    box(ax, 0.3, 2.9, 2.4, 1.9, "", fc=PALE["blue"], ec=BLUE); box(ax, 9.0, 2.9, 2.4, 1.9, "", fc=PALE["violet"], ec=VIOLET)
    ax.text(1.5, 4.45, "Image 1 (entrée)", ha="center", fontsize=9.5, fontweight="bold"); ax.text(10.2, 4.45, "Image 2 (sortie)", ha="center", fontsize=9.5, fontweight="bold")
    pixels(ax, 1.05, 3.05, 0.9, 3); pixels(ax, 9.75, 3.05, 0.9, 5, cmap="magma", lo=0.3, hi=0.8)
    box(ax, 4.3, 3.45, 2.8, 0.85, "T1 : lit les 16 patchs\nde l'image 1")
    box(ax, 4.3, 1.05, 2.8, 0.85, "T2 : lit les 16 patchs\nde l'image 2")
    arrow(ax, (2.7, 3.87), (4.3, 3.87)); arrow(ax, (7.1, 3.87), (9.0, 3.87), "écrit un résidu\n(porte d'oubli)", off=(0, 0.12))
    arrow(ax, (10.2, 2.9), (7.1, 1.48), "lit", off=(0.3, 0.05))
    arrow(ax, (4.3, 1.48), (1.5, 2.9), "retour : écrit un résidu\ndans l'image 1", off=(-0.55, -0.6), color=VIOLET, lw=2.2)
    ax.text(0.3, 4.95, "le token t se dépose sur l'image 1", fontsize=9.5, color=INK2)
    arrow(ax, (11.4, 3.85), (12.0, 3.85)); ax.text(11.55, 3.3, "tête de lecture\n→ token t + 1", fontsize=8.5, color=INK2)
    ax.text(5.7, 0.35, "Pas d'historique : chaque transformer ne voit que l'image courante, toute la mémoire est dans les deux images (4×8×8 chacune).\n"
            "Témoin L_noloop : pas de T2, l'image 1 ne reçoit que les dépôts des tokens (rétention fixe).", ha="center", fontsize=9, color=INK2, va="center", linespacing=1.4)
    save(fig, "boucle")


def controles():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    specs = [("L_loop", "T2 lit l'image 2\net écrit dans l'image 1", VIOLET, "i1"), ("L_i2", "T2 lit l'image 2\net écrit dans l'image 2", YELLOW, "i2"),
             ("L_wide", "pas de T2,\nT1 à 4 couches", RED, "wide"), ("L_noloop", "pas de T2,\nT1 à 2 couches", GRAY, None)]
    for ax, (name, t, c, mode) in zip(axs, specs):
        ax.set_xlim(0, 3.2); ax.set_ylim(0, 3.4); ax.axis("off")
        box(ax, 0.1, 1.8, 0.9, 0.8, "image 1", fc=PALE["blue"], ec=BLUE, fs=8.5)
        box(ax, 2.2, 1.8, 0.9, 0.8, "image 2", fc=PALE["violet"], ec=VIOLET, fs=8.5)
        box(ax, 1.1, 2.0, 0.95, 0.42, "T1 ×4" if mode == "wide" else "T1", fs=8.5, lw=2.4 if mode == "wide" else 1.4, ec=RED if mode == "wide" else "#9c9b95")
        arrow(ax, (1.0, 2.2), (1.1, 2.2)); arrow(ax, (2.05, 2.2), (2.2, 2.2))
        if mode in ("i1", "i2"):
            box(ax, 1.1, 0.85, 0.95, 0.42, "T2", fs=8.5)
            arrow(ax, (2.65, 1.8), (2.05, 1.06))
            if mode == "i1":
                arrow(ax, (1.1, 1.06), (0.55, 1.8), color=VIOLET, lw=2.2)
            else:
                arrow(ax, (2.05, 1.0), (2.75, 1.8), color=YELLOW, lw=2.2, rad=0.4)
        ax.text(0.1, 3.1, name, fontsize=11, fontweight="bold", color=c if c != GRAY else INK)
        ax.text(0.1, 0.35, t, fontsize=8.5, color=INK2, va="center")
    fig.suptitle("Quatre variantes à comparer : seul L_loop renvoie de l'information vers l'image d'entrée", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "controles")


# ---------------------------------------------------------------- 5. mémoire longue
def memoire():
    fig, ax = canvas(13, 4.0)
    ax.text(0.1, 3.8, "Partie 5 : suivre S3 et, en même temps, retenir une valeur", fontsize=12, fontweight="bold")
    x = 0.3
    box(ax, x, 1.7, 0.9, 0.75, "v = 5", fc=PALE["yellow"], ec=YELLOW, bold=True)
    ax.text(x + 0.45, 1.4, "valeur\n(0 à 7)", ha="center", fontsize=8, color=INK2, va="top")
    x += 1.2
    for i in range(9):
        if i == 6:
            ax.text(x + 0.3, 2.07, "…", fontsize=16, va="center"); x += 0.8; continue
        box(ax, x, 1.7, 0.62, 0.75, "g%s" % {7: "ₙ₋₁", 8: "ₙ"}.get(i, "₁₂₃₄₅₆"[min(i, 5)]), fc=PALE["gray"], ec=GRAY, fs=9)
        ax.text(x + 0.31, 1.4, "état S3", ha="center", fontsize=7.5, color=INK2, va="top")
        x += 0.8
    box(ax, x + 0.1, 1.7, 0.9, 0.75, "?", fc=PALE["violet"], ec=VIOLET, bold=True)
    ax.text(x + 0.55, 1.4, "requête →\nprédire v", ha="center", fontsize=8, color=INK2, va="top")
    ax.annotate("", xy=(x + 0.55, 2.55), xytext=(0.75, 2.55), arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=1.8, connectionstyle="arc3,rad=-0.12"))
    ax.text((x + 1.3) / 2, 3.25, "n éléments à traverser sans oublier v", ha="center", fontsize=9, color=INK2)
    ax.text(10.4, 2.3, "Entraînement : n de 4 à 24\nÉvaluation : n = 12 à 96\nHasard : S3 17 %, rappel 12,5 %", fontsize=9, color=INK2, va="center", linespacing=1.5)
    ax.text(0.1, 0.3, "Variante : la même tâche avec des images 4×8×8 (256 valeurs), 4×4×4 (64) et 2×4×4 (32), pour voir quand la mémoire « gratuite » du témoin s'épuise.",
            fontsize=8.5, color=INK2)
    save(fig, "memoire")


# ---------------------------------------------------------------- 6. langage
def langage():
    fig, ax = canvas(13, 4.0)
    ax.text(0.1, 3.75, "Partie 6 : du vrai texte, sur CPU", fontsize=12, fontweight="bold")
    box(ax, 0.1, 0.7, 6.2, 2.7, "", fc=PALE["gray"], ec=GRAY)
    ax.text(0.3, 3.15, "Histoires synthétiques (29 mots)", fontsize=10, fontweight="bold")
    story = ("tom has the key . lea has the cup . zoe has the hat .\ntom is in the park . lea is in the shop . …\n"
             "tom swaps with lea . zoe goes to the school .\n…  (E événements)\n\nwho has the key ? lea .\nwhere is zoe now ? school .")
    ax.text(0.3, 2.85, story, fontsize=8.5, family="monospace", va="top", linespacing=1.45)
    box(ax, 6.7, 0.7, 6.2, 2.7, "", fc=PALE["blue"], ec=BLUE)
    ax.text(6.9, 3.15, "Wikitext-2 (2,9 M tokens BPE)", fontsize=10, fontweight="bold")
    ax.text(6.9, 2.85, "Texte encyclopédique réel. Fenêtres d'entraînement\nde 64 tokens, évaluation sur 256 :\n\n"
            "• bits par caractère dans la longueur (0–64)\n• et au-delà (64–256), sans frame transportée\n\n"
            "Références : unigramme (2,75 bpc) et T_rope.", fontsize=9, va="top", linespacing=1.45)
    ax.text(0.1, 0.25, "« Qui » demande de lier un nom à un objet puis de suivre les échanges ; « où » ne demande que le dernier déplacement.",
            fontsize=8.5, color=INK2)
    save(fig, "langage")


# ---------------------------------------------------------------- feuille de route
def feuille_de_route():
    fig, ax = canvas(13.5, 3.7)
    ax.text(0.1, 3.45, "Le chemin parcouru, et ce que chaque étape a répondu", fontsize=12, fontweight="bold")
    steps = [("1", "Le texte tient-il\ndans une image ?", "oui ≤ 32 car.", AQUA), ("2", "Une vidéo sans position\nvaut-elle un transformer ?", "oui, et extrapole", AQUA),
             ("3", "La vidéo suit-elle\nun état ?", "parité oui,\nS3 non", YELLOW), ("4", "Une boucle à deux\nimages aide-t-elle ?", "oui : S3 appris\net généralisé", VIOLET),
             ("5", "La boucle garde-t-elle\nune mémoire longue ?", "oui jusqu'à 96", VIOLET), ("6", "Et sur du\nvrai texte ?", "à 2,5 % de\nT_rope", YELLOW),
             ("…", "Dyck, clé-valeur, S5\nsortie en image", "en attente", GRAY)]
    w, gap = 1.7, 0.18
    for i, (n, q, a, c) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, 1.35, w, 1.5, "", fc="#ffffff", ec=c, lw=1.8, ls=DASH if c == GRAY else "-")
        ax.text(x + 0.15, 2.6, n, fontsize=13, fontweight="bold", color=c if c != GRAY else INK2, va="center")
        ax.text(x + w / 2, 2.0, q, ha="center", va="center", fontsize=8.3, linespacing=1.3)
        box(ax, x + 0.12, 0.45, w - 0.24, 0.7, a, fc=PALE[{AQUA: "aqua", YELLOW: "yellow", VIOLET: "violet", GRAY: "gray"}[c]], ec=c, fs=8.3)
        if i < len(steps) - 1:
            arrow(ax, (x + w, 2.1), (x + w + gap, 2.1), lw=1.2)
    ax.text(0.1, 0.1, "Couleur : vert d'eau = la vidéo suffit ; violet = il faut la boucle ; jaune = mitigé ; gris = pas encore mesuré.", fontsize=8.5, color=INK2)
    save(fig, "feuille_de_route")


SCHEMAS = [idee, frame, autoencodeur, modeles, taches, boucle, controles, memoire, langage, feuille_de_route]

if __name__ == "__main__":
    for f in SCHEMAS:
        f()
