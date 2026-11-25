"""Graphes du rapport, relus depuis results/ (aucun entraînement, quelques secondes).

    python scripts/figures.py            # écrit figures/*.svg
    python scripts/figures.py p4_        # seulement les figures dont le nom commence par p4_

Une figure par fonction ; le nom de la fonction est le nom du fichier. Les couleurs suivent le modèle, jamais son rang :
T_rope bleu, T_nope orange, V (vidéo) vert d'eau, V λ=1 magenta, D (image qui évolue) vert, L_loop violet,
témoins sans boucle en gris.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["svg.hashsalt"] = "anendophasia"     # SVG reproductibles (diffs git propres)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "figures" / "historique"

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a85", "#e3e2de"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY = "#8a8a85"
COL = {
    "T_rope": BLUE, "T_nope": ORANGE, "V_lam0": AQUA, "V1_lam0": YELLOW, "V_lam1": MAGENTA, "V_lam1_ext": MAGENTA,
    "D_free_e1": GREEN, "D_blob_e0": GREEN, "D_blob_e1": GREEN, "L_loop": VIOLET, "L_loop1": VIOLET,
    "L_noloop": GRAY, "L_noloop1": GRAY, "L_wide": RED, "L_i2": YELLOW, "unigramme": MUTED,
}
LAB = {
    "T_rope": "T_rope", "T_nope": "T_nope", "V_lam0": "V λ=0", "V1_lam0": "V λ=0, 1 échelle", "V_lam1": "V λ=1 (8 000 pas)",
    "V_lam1_ext": "V λ=1 (24 000 pas)", "D_free_e1": "D motif libre, 1 pas", "D_blob_e0": "D tache, 0 pas",
    "D_blob_e1": "D tache, 1 pas", "L_loop": "L_loop", "L_loop1": "L_loop1", "L_noloop": "L_noloop",
    "L_noloop1": "L_noloop1", "L_wide": "L_wide", "L_i2": "L_i2",
}
DASH = (0, (4, 3))

plt.rcParams.update({
    "font.size": 10, "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": "#bdbcb6", "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2.0, "lines.markersize": 6,
})


# ---------------------------------------------------------------- outils
def J(name):
    with open(R / name, encoding="utf-8") as f:
        return json.load(f)


def has(name):
    return (R / name).exists()


def st(task, model, seeds=range(10)):
    """Résultats finaux des graines disponibles d'une tâche d'état (st_TASK_MODEL_sK.json)."""
    return [J(f"st_{task}_{model}_s{s}.json")["final"] for s in seeds if has(f"st_{task}_{model}_s{s}.json")]


def clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def chance(ax, y, text="hasard", x=None, ha="right"):
    ax.axhline(y, color=INK2, lw=1, ls=DASH, zorder=1)
    xl = ax.get_xlim()
    ax.text(xl[1] if x is None else x, y + 1.2, text, fontsize=8.5, color=INK2, ha=ha, va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    if os.environ.get("APERCU"):                       # aperçu PNG pour relire les figures (hors dépôt)
        fig.savefig(Path(os.environ["APERCU"]) / f"{name}.png", bbox_inches="tight", dpi=80)
    plt.close(fig)
    print("écrit figures/%s.svg" % name)


def endlabel(ax, x, y, text, color=INK, dy=0, fs=9, ha="left", weight="normal"):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", ha=ha, fontsize=fs, color=color,
                fontweight=weight)


def seeds_strip(ax, xs, vals, color, jitter=0.08, size=34, zorder=3):
    rng = np.random.default_rng(0)
    for x, v in zip(xs, vals):
        v = np.atleast_1d(v)
        ax.scatter(x + rng.uniform(-jitter, jitter, len(v)), v, s=size, color=color, edgecolor=SURF, lw=1.2, zorder=zorder)


# ================================================================ partie 1 : autoencodeur
ARM = {"cumul": (AQUA, "vidéo continue (cumul)"), "image": (BLUE, "image d'un coup"),
       "vector": (YELLOW, "vecteur plat"), "none": (GRAY, "sans étage spatial")}


def p1_capacite():
    L = [8, 16, 32, 64]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True)
    for ax, key, ttl in zip(axs, ["char_acc", "exact_acc"], ["Caractères relus correctement", "Morceaux relus sans aucune faute"]):
        for arm, (c, lab) in ARM.items():
            y = [100 * J(f"ae_{arm}_L{l}.json")["final"][key] for l in L]
            ax.plot(L, y, "-o", color=c, lw=2.4 if arm == "cumul" else 1.8, mec=SURF, mew=1.2, label=lab, zorder=3 if arm == "cumul" else 2)
        for lab, l, f in [("30 000 pas", 32, "ae_cumul_L32_long30k.json"), ("30 000 pas", 64, "ae_cumul_L64_long30k.json"),
                          ("image 16×16", 64, "ae_cumul_L64_img16.json")]:
            v = 100 * J(f)["final"][key]
            ax.plot([l], [v], "D", color=AQUA, ms=7, mec=INK, mew=0.8, zorder=4)
            endlabel(ax, l, v, lab, color=INK2, fs=8, dy={(64, "30 000 pas"): -7, (32, "30 000 pas"): -9}.get((l, lab), 0))
        ax.set_xscale("log", base=2); ax.set_xticks(L); ax.set_xticklabels(L); ax.set_xlim(6.5, 130)
        ax.set_xlabel("longueur du morceau (caractères)"); ax.set_title(ttl); clean(ax)
    axs[0].set_ylim(48, 101.5); axs[0].set_ylabel("%")
    axs[0].axhline(99, color=INK2, lw=1, ls=DASH); axs[0].text(7, 97.4, "critère pré-enregistré : 99 %", fontsize=8.5, color=INK2)
    axs[1].set_ylim(-3, 103)
    axs[1].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Combien de texte tient dans une image 3×8×8 (192 valeurs) ?", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "p1_capacite")


def p1_continuite():
    """Compromis continuité / fidélité (λ), en petits multiples : pas de double axe."""
    tags, lams = ["", "_pas03", "_pas1", "_pas3"], ["0", "0,3", "1", "3"]
    d = [J(f"ae_cumul_L16{t}.json") for t in tags]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8))
    series = [(axs[0], [100 * x["final"]["exact_acc"] for x in d], "Morceaux exacts (%)", (88, 100), "%.1f"),
              (axs[1], [100 * x["final"]["char_acc"] for x in d], "Caractères justes (%)", (98, 100.2), "%.1f"),
              (axs[2], [x["continuity"]["index"] for x in d], "Indice de continuité (bas = continu)", (0, 0.85), "%.2f")]
    for k, (ax, y, ttl, yl, fmt) in enumerate(series):
        if k < 2:          # échelle tronquée : des points, pas des barres
            ax.plot(range(4), y, "-o", color=AQUA, ms=8, mec=SURF, mew=1.5)
        else:
            ax.bar(range(4), y, color=AQUA, width=0.55)
        for i, v in enumerate(y):
            ax.text(i, v + (yl[1] - yl[0]) * 0.03, fmt % v, ha="center", fontsize=9)
        ax.set_xticks(range(4)); ax.set_xticklabels(lams); ax.set_xlim(-0.5, 3.5); ax.set_ylim(*yl); ax.set_title(ttl, fontsize=10.5)
        ax.set_xlabel("pénalité de continuité λ"); clean(ax)
    axs[2].axhline(0.5, color=INK2, lw=1, ls=DASH); axs[2].text(3.4, 0.52, "seuil « continu » 0,5", fontsize=8, color=INK2, ha="right")
    fig.suptitle("Rendre la vidéo continue coûte ~7 points de morceaux exacts, presque rien en caractères (L = 16)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "p1_continuite")


def p1_apprentissage():
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    shades = {8: "#9fe0c6", 16: "#4fc79c", 32: AQUA, 64: "#0d7a54"}
    for L, c in shades.items():
        cu = J(f"ae_cumul_L{L}.json")["curve"]
        ax.plot([p["step"] for p in cu], [100 * p["char_acc"] for p in cu], color=c, lw=2)
        if L == 64:
            endlabel(ax, cu[-1]["step"], 100 * cu[-1]["char_acc"], f"L = {L}", color=INK2)
    endlabel(ax, 8000, 99.5, "L = 8, 16, 32", color=INK2, dy=-10)
    for L in (32, 64):
        cu = J(f"ae_cumul_L{L}_long30k.json")["curve"]
        ax.plot([p["step"] for p in cu], [100 * p["char_acc"] for p in cu], color=shades[L], lw=1.6, ls=DASH)
        endlabel(ax, cu[-1]["step"], 100 * cu[-1]["char_acc"], f"L = {L}, 30 000 pas", color=INK2)
    ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("caractères relus correctement (%)"); ax.set_ylim(0, 102); ax.set_xlim(0, 36000)
    ax.set_title("Vidéo continue : les morceaux longs demandent seulement plus d'entraînement"); clean(ax)
    fig.tight_layout(); save(fig, "p1_apprentissage")


def p1_position():
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for f, c, lab in [("ae_cumul_L32.json", AQUA, "L = 32, 8 000 pas"), ("ae_cumul_L64.json", "#0d7a54", "L = 64, 8 000 pas"),
                      ("ae_cumul_L64_long30k.json", VIOLET, "L = 64, 30 000 pas"), ("ae_cumul_L64_img16.json", BLUE, "L = 64, image 16×16")]:
        a = J(f)["final"]["acc_by_pos"]
        x = (np.arange(len(a)) + 0.5) / len(a)
        ax.plot(x, 100 * np.array(a), "-", color=c, lw=2, label=lab)
    ax.set_xlabel("position dans le morceau (0 = premier caractère écrit, 1 = dernier)")
    ax.set_ylabel("caractères justes (%)"); ax.set_ylim(0, 102); ax.set_xlim(0, 1)
    ax.set_title("À 64 caractères, les erreurs se concentrent au milieu du morceau ; le début et la fin restent lisibles")
    ax.legend(fontsize=8.5, loc="lower right"); clean(ax)
    fig.tight_layout(); save(fig, "p1_position")


# ================================================================ partie 2 : transformer nourri par la vidéo
S2 = ["T_rope", "T_nope", "V_lam0", "V1_lam0", "V_lam1_ext", "D_free_e1", "D_blob_e0", "D_blob_e1"]


def p2_extrapolation():
    keys = ["bpc_0_256", "bpc_256_512", "bpc_512_1024"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axs[0]
    main = ["T_rope", "T_nope", "V_lam0", "V_lam1_ext"]
    for n in S2:
        y = [J(f"s2_{n}.json")["extrapolation"][k] for k in keys]
        if n in main:
            ax.plot(range(3), y, "-o", color=COL[n], lw=2.4, mec=SURF, mew=1.2, zorder=3)
            endlabel(ax, 2, y[-1], LAB[n], color=INK, dy={"V_lam1_ext": -6, "V_lam0": 6}.get(n, 0))
        else:
            ax.plot(range(3), y, "-", color="#c9c8c2", lw=1.2, zorder=1)
    ax.plot([], [], color="#c9c8c2", lw=1.2, label="autres variantes (V 1 échelle, D)")
    ax.set_xticks(range(3)); ax.set_xticklabels(["0–256\n(longueur d'entraînement)", "256–512", "512–1 024"])
    ax.set_xlim(-0.2, 2.7); ax.set_ylim(0.55, 1.8); ax.set_ylabel("bits par caractère (bas = mieux)")
    ax.set_xlabel("position des tokens prédits"); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("TinyStories : au-delà de 256 tokens, la vidéo se dégrade peu"); clean(ax)
    ax = axs[1]
    y = [J(f"s2_{n}.json")["extrapolation"]["bpc_512_1024"] - J(f"s2_{n}.json")["extrapolation"]["bpc_0_256"] for n in S2]
    order = np.argsort(y)
    ax.barh(range(len(S2)), [y[i] for i in order], color=[COL[S2[i]] for i in order], height=0.6)
    for k, i in enumerate(order):
        ax.text(y[i] + 0.015, k, "+%.2f" % y[i], va="center", fontsize=8.5)
    ax.set_yticks(range(len(S2))); ax.set_yticklabels([LAB[S2[i]] for i in order]); ax.invert_yaxis()
    ax.set_xlabel("dégradation : bpc(512–1 024) − bpc(0–256)"); ax.set_xlim(0, 1.25)
    ax.set_title("Prix de l'extrapolation, du meilleur au pire"); clean(ax, grid="x")
    fig.tight_layout(); save(fig, "p2_extrapolation")


def p2_apprentissage():
    fig, ax = plt.subplots(figsize=(9, 4.3))
    for n in ["T_rope", "T_nope", "V_lam0", "V_lam1", "V_lam1_ext", "D_blob_e1"]:
        cu = J(f"s2_{n}.json")["curve"]
        x, y = [p["step"] for p in cu], [p["bpc"] for p in cu]
        ax.plot(x, y, marker="o", color=COL[n], ms=4, lw=1.4 if n == "V_lam1" else 2, ls=DASH if n == "V_lam1" else "-", mec=SURF)
        ax.lines[-1].set_label(LAB[n])
    ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("bits par caractère, validation 0–256"); ax.set_ylim(0.6, 2.0); ax.set_xlim(0, 28500)
    ax.set_title("La pénalité de continuité (λ=1) ralentit l'apprentissage ; avec 3× plus de pas elle s'en rapproche"); clean(ax)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)
    fig.tight_layout(); save(fig, "p2_apprentissage")


def p2_cout():
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for n in S2:
        d = J(f"s2_{n}.json")
        x, y = d["seconds"] / 60, d["extrapolation"]["bpc_512_1024"]
        ax.scatter(x, y, s=70, color=COL[n], edgecolor=SURF, lw=1.5, zorder=3)
        endlabel(ax, x, y, LAB[n], color=INK2, fs=8.5, dy={"D_free_e1": 7, "D_blob_e1": -6, "T_nope": 0}.get(n, 0))
    ax.set_xlabel("temps d'entraînement sur T4 (minutes)"); ax.set_ylabel("bpc aux positions 512–1 024 (bas = mieux)")
    ax.set_xlim(0, 150); ax.set_ylim(0.75, 1.85)
    ax.set_title("Mieux extrapoler coûte 4 à 9 fois le temps d'un transformer"); clean(ax, grid="both")
    fig.tight_layout(); save(fig, "p2_cout")


def p2_flux():
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for n in ["T_rope", "T_nope", "V_lam0", "V_lam1_ext"]:
        y = J(f"s2_{n}.json")["stream_bpc"]
        ax.plot(range(1, len(y) + 1), y, "-o", color=COL[n], mec=SURF, mew=1.2)
        endlabel(ax, len(y), y[-1], LAB[n], fs=8.5, dy={"V_lam0": -5, "T_nope": 3}.get(n, 0))
    ax.set_xlabel("rang de la fenêtre de 256 tokens dans le flux (la frame est transportée d'une fenêtre à l'autre)")
    ax.set_ylabel("bpc"); ax.set_xlim(0.7, 9.6); ax.set_xticks(range(1, 9))
    ax.set_title("Flux de 8 fenêtres : transporter la frame n'aide pas mesurablement sur des histoires courtes"); clean(ax)
    fig.tight_layout(); save(fig, "p2_flux")


def p2_rappel_listes():
    d = J("s2_recall.json")
    first = {"T_rope": 12.53, "T_nope": 12.55, "V_lam0": 12.37, "V1_lam0": 12.34}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, dist in zip(axs, (0, 64)):
        for m in ["T_rope", "T_nope", "V_lam0", "V1_lam0"]:
            y = [d[m][f"n{n}_d{dist}"]["second_nll"] for n in (8, 16, 32, 64)]
            ax.plot(range(4), y, "-o", color=COL[m], mec=SURF, mew=1.2, label=LAB[m])
        ax.axhline(7.44, color=INK2, lw=1, ls=DASH); ax.text(3, 7.6, "tirage uniforme (7,44)", fontsize=8.5, color=INK2, ha="right")
        ax.axhspan(12.3, 12.6, color="#efeeea", zorder=0); ax.text(0, 12.75, "première copie (aucun rappel possible)", fontsize=8.5, color=INK2)
        ax.set_xticks(range(4)); ax.set_xticklabels(["8", "16", "32", "64"]); ax.set_xlabel("longueur de la liste à recopier (tokens)")
        ax.set_title(f"Distraction de {dist} tokens"); ax.set_ylim(7, 13.3); clean(ax)
    axs[0].set_ylabel("perte sur la seconde copie (nats par token)"); axs[1].legend(fontsize=8.5, loc="center right")
    fig.suptitle("Rappel de listes aléatoires : aucun modèle de l'étape 2 ne recopie (précision 0 %)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "p2_rappel_listes")


# ================================================================ partie 3 : suivi d'état
BUCK = ["acc_0_24", "acc_24_48", "acc_48_96", "acc_96_256"]
BUCK_LAB = ["0–24\n(entraînement)", "24–48", "48–96", "96–256"]
ETAT = ["T_rope", "T_nope", "V_lam0", "D_blob_e0", "D_blob_e1", "L_loop", "L_noloop"]


def p3_etat():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, task, ttl, ch in zip(axs, ["S2", "S3"], ["Parité, S2 (2 états)", "Permutations de 3 objets, S3 (6 états)"], [50, 100 / 6]):
        for m in ETAT:
            r = st(task, m)
            y = [100 * np.median([x[k] for x in r]) for k in BUCK]
            hi = m in ("L_loop", "V_lam0", "T_rope")
            ax.plot(range(4), y, marker="o", color=COL[m], lw=2.6 if hi else 1.3, ms=6 if hi else 4, mec=SURF, alpha=1 if hi else 0.7,
                    ls="-" if m != "D_blob_e0" else DASH, label=LAB[m], zorder=3 if hi else 2)
        ax.set_xticks(range(4)); ax.set_xticklabels(BUCK_LAB); ax.set_ylim(0, 105); ax.set_xlim(-0.2, 3.2)
        ax.set_xlabel("position dans la suite"); ax.set_title(ttl); clean(ax); chance(ax, ch)
    axs[0].set_ylabel("précision, médiane de 3 graines (%)")
    axs[1].legend(fontsize=8.5, ncol=2, loc="center right")
    fig.tight_layout(); save(fig, "p3_etat")


def p3_graines():
    """Chaque graine est un point : la médiane cache des succès isolés."""
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    models = ["T_rope", "T_nope", "V_lam0", "D_blob_e0", "D_blob_e1", "L_noloop", "L_loop"]
    for ax, task, ch in zip(axs, ["S2", "S3", "S5"], [50, 100 / 6, 100 / 120]):
        for i, m in enumerate(models):
            r = st(task, m)
            if not r:
                ax.text(i, 8, "non\nmesuré", ha="center", fontsize=7.5, color=MUTED)
                continue
            seeds_strip(ax, [i], [[100 * x["acc_96_256"] for x in r]], COL[m])
        ax.set_xticks(range(len(models))); ax.set_xticklabels([LAB[m].replace(", ", "\n") for m in models], fontsize=8, rotation=0)
        ax.set_ylim(-3, 106); ax.set_xlim(-0.6, len(models) - 0.4); ax.set_title(f"{task}, positions 96–256", fontsize=10.5)
        clean(ax); chance(ax, ch)
    axs[0].set_ylabel("précision de chaque graine (%)")
    fig.suptitle("Graine par graine, à 4 à 10 fois la longueur d'entraînement : seules la vidéo (parité) et la boucle (parité, S3) tiennent", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "p3_graines")


def p3_long():
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    for m in ["T_rope", "V_lam0", "D_blob_e1"]:
        for tag, ls, lw in [("S3", DASH, 1.4), ("S3L", "-", 2.4)]:
            r = st(tag, m)
            y = [100 * np.median([x[k] for x in r]) for k in BUCK]
            ax.plot(range(4), y, marker="o", color=COL[m], ls=ls, lw=lw, ms=5, mec=SURF)
        endlabel(ax, 0, 100 * np.median([x["acc_0_24"] for x in st("S3L", m)]), LAB[m], ha="right", fs=8.5, dy={"V_lam0": -5, "T_rope": 5}.get(m, 0))
    ax.add_artist(ax.legend(handles=[Line2D([], [], color=INK2, ls=DASH, lw=1.4, label="4 000 pas (S3)"),
                                     Line2D([], [], color=INK2, lw=2.4, label="12 000 pas (S3L)")], fontsize=8.5, loc="upper right"))
    ax.set_xticks(range(4)); ax.set_xticklabels(BUCK_LAB); ax.set_ylim(0, 105); ax.set_xlim(-0.9, 3.2)
    ax.set_ylabel("précision médiane (%)"); ax.set_title("S3 entraîné 3× plus longtemps : tout le monde apprend, personne n'extrapole"); clean(ax); chance(ax, 100 / 6)
    fig.tight_layout(); save(fig, "p3_long")


def p3_apprentissage():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.9), sharey=True)
    for ax, task in zip(axs, ["S2", "S3"]):
        for m in ["T_rope", "V_lam0", "L_loop", "L_noloop"]:
            for s in range(3):
                cu = J(f"st_{task}_{m}_s{s}.json")["curve"]
                ax.plot([p["step"] for p in cu], [100 * p["acc_train_len"] for p in cu], color=COL[m], lw=1.6, alpha=0.85,
                        label=LAB[m] if s == 0 else None)
        ax.set_xlabel("pas (la longueur supervisée grandit de 4 à 24 pendant les 40 premiers %)"); ax.set_title(f"{task} : précision sur la longueur supervisée, 3 graines")
        clean(ax)
    axs[0].set_ylabel("%"); axs[0].set_ylim(0, 103)
    fig.legend(*axs[0].get_legend_handles_labels(), fontsize=9, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(); save(fig, "p3_apprentissage")


# ================================================================ partie 4 : la boucle
CB = ["acc_0_12", "acc_12_24", "acc_24_48", "acc_48_96"]
CB_LAB = ["0–12\n(entraînement)", "12–24", "24–48", "48–96"]
CTRL = ["L_loop", "L_i2", "L_wide", "L_noloop"]


def p4_controles():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    for j, m in enumerate(CTRL):
        r = st("S3c", m)
        med = [100 * np.median([x[k] for x in r]) for k in CB]
        off = (j - 1.5) * 0.08
        ax.plot(np.arange(4) + off, med, "-", color=COL[m], lw=2.2, zorder=2)
        seeds_strip(ax, np.arange(4) + off, [[100 * x[k] for x in r] for k in CB], COL[m], jitter=0.025, size=22)
        endlabel(ax, 3 + off, med[-1], LAB[m], dy={"L_wide": 8, "L_noloop": -8}.get(m, 0))
    ax.set_xticks(range(4)); ax.set_xticklabels(CB_LAB); ax.set_xlim(-0.3, 3.6); ax.set_ylim(0, 105)
    ax.set_ylabel("précision (%), 5 graines (points) et médiane (ligne)"); ax.set_title("S3, mêmes conditions pour tous : seule la boucle généralise")
    clean(ax); chance(ax, 100 / 6, x=-0.25, ha="left")
    ax = axs[1]
    for i, m in enumerate(CTRL):
        r = st("S3c", m)
        p = J(f"st_S3c_{m}_s0.json")["n_params"] / 1000
        ax.barh(i, p, color=COL[m], height=0.55)
        ax.text(p + 4, i, "%d k paramètres" % p, va="center", fontsize=8.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(["L_loop : T1 + T2, T2 écrit\ndans l'image 1 (retour)", "L_i2 : T1 + T2, T2 écrit\ndans l'image 2 (pas de retour)",
                                                 "L_wide : T1 seul, 4 couches", "L_noloop : T1 seul, 2 couches"], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlim(0, 340); ax.set_title("Capacité : L_wide et L_i2 égalent la boucle"); clean(ax, grid="x")
    fig.tight_layout(); save(fig, "p4_controles")


def p4_graines():
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, task, ch in zip(axs, ["S2", "S3"], [50, 100 / 6]):
        for j, m in enumerate(["L_loop", "L_noloop"]):
            for s, x in enumerate(st(task, m)):
                y = [100 * x[k] for k in BUCK]
                ax.plot(np.arange(4) + (j - 0.5) * 0.06, y, "-o", color=COL[m], lw=1.6, ms=5, mec=SURF, alpha=0.9, label=LAB[m] if s == 0 else None)
        ax.set_xticks(range(4)); ax.set_xticklabels(BUCK_LAB); ax.set_ylim(0, 105); ax.set_title(f"{task}, taille normale (GPU) : une ligne par graine")
        clean(ax); chance(ax, ch)
    axs[0].set_ylabel("précision (%)"); axs[1].legend(fontsize=9, loc="center right")
    fig.tight_layout(); save(fig, "p4_graines")


def p4_ablations():
    rows = [("L_loop (référence)", "st_S3c_L_loop_s0.json", VIOLET), ("1 couche par transformer", "st_S3c_A_1layer_s0.json", VIOLET),
            ("patchs 4×4 (4 patchs)", "st_S3c_A_p4_s0.json", "#a79cec"), ("sans porte d'oubli", "st_S3c_A_nogate_s0.json", GRAY),
            ("lecture sur l'image 1", "st_S3c_A_readI1_s0.json", GRAY)]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    for i, (lab, f, c) in enumerate(rows):
        d = J(f)
        y = [100 * d["final"][k] for k in CB]
        ax.plot(range(4), y, marker="o", color=c, lw=2.2, mec=SURF, mew=1.2, ls="-" if i < 3 else DASH)
        endlabel(ax, 3, y[-1], "%s (%d k)" % (lab, d["n_params"] / 1000), fs=8.5, dy={0: 9, 1: -1, 3: 6, 4: -6}.get(i, 0))
    ax.set_xticks(range(4)); ax.set_xticklabels(CB_LAB); ax.set_ylim(0, 105); ax.set_xlim(-0.2, 4.6); ax.set_ylabel("précision sur S3 (%)")
    ax.set_title("Ablations (graine 0) : la porte d'oubli et la lecture sur l'image 2 sont nécessaires, la profondeur non"); clean(ax); chance(ax, 100 / 6, x=-0.15, ha="left")
    fig.tight_layout(); save(fig, "p4_ablations")


def p4_causal():
    d = J("cpu_boucle_causal_s0.json")
    sig = [0.0, 0.1, 0.3, 1.0]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.9), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    shades = ["#c9c3f1", "#a79cec", "#7f71df", VIOLET]
    for k, c, lab in zip(CB, shades, ["0–12", "12–24", "24–48", "48–96"]):
        y = [100 * d[f"bruit_T2_{s}"][k] for s in sig]
        ax.plot(range(4), y, "-o", color=c, mec=SURF, mew=1.2, label=f"positions {lab}")
    ax.set_xticks(range(4)); ax.set_xticklabels(["0", "0,1", "0,3", "1"]); ax.set_ylim(0, 105)
    ax.set_xlabel("bruit ajouté au résidu de T2 à l'inférence (σ)"); ax.set_ylabel("précision sur S3 (%)")
    ax.set_title("Perturber le retour casse le modèle : T2 est réellement utilisé"); ax.legend(fontsize=8.5, loc="lower left"); clean(ax); chance(ax, 100 / 6)
    ax = axs[1]
    steps = [5, 11, 23, 47]
    for key, c, mk in [("Image1", BLUE, "o"), ("Image2", VIOLET, "s")]:
        ax.plot(range(4), [100 * v for v in d["sonde_lineaire"][key]], "-" + mk, color=c, mec=SURF, mew=1.2, ms=8 if mk == "o" else 6,
                label=f"sonde linéaire sur {key.replace('Image', 'image ')}")
    ax.plot(range(4), [100 * v for v in d["acc_modele_aux_memes_pas"]], ":", color=INK2, lw=1.5, label="précision du modèle")
    ax.set_xticks(range(4)); ax.set_xticklabels([f"pas {s}" for s in steps]); ax.set_ylim(0, 105)
    ax.set_title("L'état de S3 se lit linéairement dans les deux images"); ax.legend(fontsize=8.5, loc="lower left"); clean(ax); chance(ax, 100 / 6)
    fig.tight_layout(); save(fig, "p4_causal")


def p4_cout():
    models = ["T_rope", "T_nope", "V_lam0", "D_blob_e0", "D_blob_e1", "L_noloop", "L_loop"]
    t = [np.mean([J(f"st_S3_{m}_s{s}.json")["seconds"] for s in range(3)]) / 60 for m in models]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.barh(range(len(models)), t, color=[COL[m] for m in models], height=0.6)
    for i, v in enumerate(t):
        ax.text(v + 0.2, i, "%.1f min  (×%.0f)" % (v, v / t[0]), va="center", fontsize=8.5)
    ax.set_yticks(range(len(models))); ax.set_yticklabels([LAB[m] for m in models]); ax.invert_yaxis(); ax.set_xlim(0, 23)
    ax.set_xlabel("minutes pour 4 000 pas sur S3 (T4, moyenne de 3 graines)")
    ax.set_title("Le prix de la boucle : ~19 fois le temps d'un transformer (une boucle Python par token)"); clean(ax, grid="x")
    fig.tight_layout(); save(fig, "p4_cout")


# ================================================================ partie 5 : mémoire longue
def _mem(model, n, key):
    return [x[str(n)][key] for x in st("S3M", model) if str(n) in x]


def p5_memoire():
    ns = [12, 24, 48, 96]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for ax, key, ttl, ch in zip(axs, ["s3", "recall"], ["Suivre S3 pendant le délai", "Rappeler la valeur à la fin"], [100 / 6, 12.5]):
        for m in ["L_loop1", "L_noloop1", "T_rope"]:
            v = [[100 * a for a in _mem(m, n, key)] for n in ns]
            med = [np.median(a) for a in v]
            ax.fill_between(range(4), [min(a) for a in v], [max(a) for a in v], color=COL[m], alpha=0.15, lw=0)
            ax.plot(range(4), med, "-o", color=COL[m], mec=SURF, mew=1.2)
            endlabel(ax, 3, med[-1], LAB[m], dy={"L_loop1": 6}.get(m, 0))
        ax.set_xticks(range(4)); ax.set_xticklabels(["n = %d" % n for n in ns]); ax.set_xlim(-0.2, 3.7); ax.set_ylim(0, 105)
        ax.set_xlabel("nombre d'éléments de S3 entre la valeur et la question (entraînement : 4 à 24)"); ax.set_title(ttl)
        clean(ax); chance(ax, ch, x=-0.15, ha="left")
    axs[0].set_ylabel("précision (%) : médiane et étendue de 3 graines")
    fig.tight_layout(); save(fig, "p5_memoire")


def p5_taille_image():
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    sizes = [("", "4×8×8 = 256 valeurs"), ("_4x4x4", "4×4×4 = 64 valeurs"), ("_2x4x4", "2×4×4 = 32 valeurs")]
    for ax, (tag, ttl) in zip(axs, sizes):
        for m in ["L_loop1", "L_noloop1"]:
            ns = [n for n in (12, 24, 32, 48, 64, 96) if _mem(m + tag, n, "recall")]
            v = [[100 * a for a in _mem(m + tag, n, "recall")] for n in ns]
            seeds_strip(ax, ns, v, COL[m], jitter=1.2, size=22)
            ax.plot(ns, [np.median(a) for a in v], "-", color=COL[m], lw=2, label=LAB[m])
        ax.axhline(90, color=INK2, lw=1, ls=DASH); ax.text(10, 91.5, "90 %", fontsize=8, color=INK2)
        ax.set_xticks([12, 24, 48, 64, 96]); ax.set_xlabel("distance n"); ax.set_title("Image " + ttl, fontsize=10); ax.set_ylim(-3, 105); clean(ax)
    axs[0].set_ylabel("rappel de la valeur (%)"); axs[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Plus l'image est petite, plus tôt le témoin oublie ; la boucle tient jusqu'à 96 même avec 32 valeurs",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(); save(fig, "p5_taille_image")


def p5_sondes():
    """Lisibilité linéaire de la valeur par canal, au fil du délai : où vit la mémoire ?"""
    ts = ["t1", "t5", "t15", "t30", "t48"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f1fc", "#c9c3f1", "#8f82e4", VIOLET, "#2c1f8a"])
    for ax, s in zip(axs, range(3)):
        so = J(f"st_S3M_L_loop1_s{s}.json")["sondes"]
        M = np.array([[so[t]["Image1_canaux"][c] for t in ts] for c in range(4)] + [[so[t]["Image2_canaux"][c] for t in ts] for c in range(4)])
        im = ax.imshow(100 * M, cmap=cmap, vmin=50, vmax=100, aspect="auto")
        for i in range(8):
            for j in range(5):
                v = 100 * M[i, j]
                ax.text(j, i, "%.0f" % v, ha="center", va="center", fontsize=7.5, color="white" if v > 85 else INK)
        ax.axhline(3.5, color=SURF, lw=3)
        ax.set_xticks(range(5)); ax.set_xticklabels([t[1:] for t in ts]); ax.set_xlabel("pas après la valeur")
        ax.set_title(f"graine {s}", fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[0].set_yticks(range(8)); axs[0].set_yticklabels([f"image 1, canal {c}" for c in range(4)] + [f"image 2, canal {c}" for c in range(4)], fontsize=8.5)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.01); cb.set_label("sonde linéaire (%)", color=INK2); cb.outline.set_visible(False)
    fig.suptitle("La valeur se lit dans presque tous les canaux des deux images : elle circule, elle n'est pas rangée",
                 x=0.01, y=1.04, ha="left", fontsize=12, fontweight="bold")
    save(fig, "p5_sondes")


# ================================================================ partie 6 : langage
def p6_synthetique():
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
    for ax, key, ttl, ch in zip(axs, ["who", "where"], ["« Qui a l'objet ? » (liaison nom ↔ objet)", "« Où est X ? » (dernier déplacement)"], [100 / 3, 25]):
        for m in ["T_rope", "V_lam0", "L_loop1"]:
            f = J(f"lm_synth_v2_{m}_s0.json")["final"]
            y = [100 * f[e][key] for e in ("8", "16", "32")]
            ax.plot(range(3), y, "-o", color=COL[m], mec=SURF, mew=1.2)
            endlabel(ax, 2, y[-1], LAB[m], dy={"L_loop1": 6, "T_rope": -5}.get(m, 0) if key == "who" else 0)
        ax.set_xticks(range(3)); ax.set_xticklabels(["E = 8\n(entraînement)", "E = 16", "E = 32"]); ax.set_xlim(-0.2, 2.6); ax.set_ylim(0, 105)
        ax.set_title(ttl); ax.set_xlabel("événements dans l'histoire"); clean(ax); chance(ax, ch, x=-0.15, ha="left")
    axs[0].set_ylabel("réponses justes (%), 1 graine")
    fig.tight_layout(); save(fig, "p6_synthetique")


def p6_wikitext():
    ms = ["unigramme", "T_rope", "V_lam0", "L_loop1"]
    keys, xl = ["0-64", "64-128", "128-256"], ["0–64\n(entraînement)", "64–128", "128–256"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axs[0]
    for m in ms[1:]:
        f = J(f"lm_wiki_{m}_s0.json")["final"]["256"]
        y = [f[k] for k in keys]
        ax.plot(range(3), y, "-o", color=COL[m], mec=SURF, mew=1.2)
        endlabel(ax, 2, y[-1], LAB[m], dy={"T_rope": -6, "L_loop1": 6}.get(m, 0))
    ax.axhline(2.198, color=INK2, lw=1, ls=DASH); ax.text(1.0, 2.195, "critère « apprend » : 20 % sous l'unigramme", fontsize=8, color=INK2, va="top", ha="center")
    ax.set_xticks(range(3)); ax.set_xticklabels(xl); ax.set_xlim(-0.2, 2.6); ax.set_ylim(2.05, 2.24)
    ax.set_ylabel("bits par caractère (bas = mieux)"); ax.set_title("Wikitext-2, 2 000 pas : la boucle à 2,5 % de T_rope"); clean(ax)
    ax = axs[1]
    for i, m in enumerate(ms[1:]):
        d = J(f"lm_wiki_{m}_s0.json")
        ax.scatter(d["seconds"] / 60, d["final"]["256"]["0-64"], s=40 + d["n_params"] / 6000, color=COL[m], edgecolor=SURF, lw=1.5, zorder=3)
        endlabel(ax, d["seconds"] / 60, d["final"]["256"]["0-64"], "%s, %.2f M param." % (LAB[m], d["n_params"] / 1e6), fs=8.5, dy=12)
    ax.set_xlim(0, 32); ax.set_ylim(2.05, 2.24); ax.set_xlabel("minutes sur CPU"); ax.set_title("Coût : la boucle prend 8× le temps"); clean(ax, grid="both")
    fig.tight_layout(); save(fig, "p6_wikitext")


# ================================================================ partie 7 : analyses
def p7_bruit_pca():
    d = J("analyses_cpu.json")["modeles"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.9))
    ax = axs[0]
    for m, lab in [("V_lam0", "V λ=0"), ("V_lam1_ext", "V λ=1 (24 000 pas)")]:
        y = [100 * (d[m][f"bruit_{s}"] / d[m]["base"] - 1) for s in ("0.1", "0.3", "1.0")]
        ax.plot(range(3), y, "-o", color=COL[m], mec=SURF, mew=1.2)
        endlabel(ax, 2, y[-1], lab)
    ax.set_xticks(range(3)); ax.set_xticklabels(["0,1", "0,3", "1"]); ax.set_xlim(-0.2, 2.9)
    ax.set_xlabel("bruit sur l'image lue (fraction de son écart-type)"); ax.set_ylabel("hausse de la perte (%)")
    ax.set_title("La vidéo plus continue n'est pas plus robuste au bruit"); clean(ax)
    ax = axs[1]
    k = [1, 2, 4, 8, 16, 32]
    for m, lab in [("V_lam0", "V λ=0"), ("V_lam1_ext", "V λ=1 (24 000 pas)")]:
        y = [d[m][f"pca_{i}"] for i in k]
        ax.plot(k, y, "-o", color=COL[m], mec=SURF, mew=1.2)
        ax.axhline(d[m]["base"], color=COL[m], lw=1, ls=":")
        endlabel(ax, 32, y[-1], lab, dy=-6 if m == "V_lam0" else 6)
    ax.text(1, 0.86, "pointillés : sans projection (192 directions)", fontsize=8, color=INK2)
    ax.set_xscale("log", base=2); ax.set_xticks(k); ax.set_xticklabels(k); ax.set_xlim(0.8, 120)
    ax.set_xlabel("directions principales gardées (sur 192)"); ax.set_ylabel("bpc"); ax.set_title("λ=1 cache l'information dans des directions de faible variance"); clean(ax)
    fig.tight_layout(); save(fig, "p7_bruit_pca")


def p7_positions_blob():
    c = J("analyses_cpu.json")["classes"]
    names = sorted(c, key=lambda k: c[k][1] - c[k][2])
    fig, ax = plt.subplots(figsize=(9, 3.9))
    for i, n in enumerate(names):
        nb, intra, inter, p = c[n]
        ax.plot([intra, inter], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter([inter], [i], s=60, color=GRAY, edgecolor=SURF, lw=1.2, zorder=2)
        ax.scatter([intra], [i], s=60, color=GREEN, edgecolor=SURF, lw=1.2, zorder=3)
        if p < 0.05 or p > 0.95:
            ax.text(2.16, i, ("p < 0,003" if p < 0.003 else "p = %.3f" % p).replace(".", ",") if p < 0.5 else "", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(["%s (%d)" % (n, c[n][0]) for n in names], fontsize=8.5)
    ax.set_xlabel("distance moyenne entre positions de dépôt du modèle D"); ax.set_xlim(1.55, 2.25)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GREEN, label="au sein de la classe"),
                       Line2D([], [], marker="o", ls="", color=GRAY, label="entre classes (témoin)")], fontsize=8.5, loc="upper left")
    ax.set_title("Où D dépose ses tokens : seuls les nombres se regroupent nettement"); clean(ax, grid="x")
    fig.tight_layout(); save(fig, "p7_positions_blob")


def p7_surprise():
    """Surprise du token suivant, mot par mot, sur une phrase (results/s2_decodage.txt, scripts/demo_phrase.py)."""
    rows = [l.rstrip("\n").split(" | ") for l in open(R / "s2_decodage.txt", encoding="utf-8") if l.count(" | ") == 5]
    nxt = [r[1].strip("'") for r in rows]
    a, b = [float(r[3]) for r in rows], [float(r[5]) for r in rows]
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, a, 0.38, color=COL["V_lam0"], label="V λ=0")
    ax.bar(x + 0.2, b, 0.38, color=COL["V_lam1_ext"], label="V λ=1 (24 000 pas)")
    ax.set_xticks(x); ax.set_xticklabels(nxt, rotation=60, fontsize=8.5, ha="right")
    ax.set_ylabel("surprise du token (bits)"); ax.set_xlim(-0.7, len(rows) - 0.3)
    i = nxt.index(" rock")
    ax.text(i, b[i] + 0.4, "mot absurde", ha="center", fontsize=8.5, color=INK2); ax.set_ylim(0, 15.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("« Lily was hungry. She went to the kitchen and ate a red apple. Then she ate a big rock. » : token prédit, surprise de chaque modèle")
    clean(ax); fig.tight_layout(); save(fig, "p7_surprise")


def fp_bras_initiaux():
    """Premier plan (7 bras, niveau caractère), abandonné après deux bras."""
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for f, c, lab in [("bras1_transformer_metrics.json", BLUE, "bras 1 : transformer (4,0 M)"), ("bras2_gru_metrics.json", ORANGE, "bras 2 : GRU (3,7 M)")]:
        cu = J(f)["curve"]
        x, y = [p["step"] for p in cu], [p["val_loss"] / np.log(2) for p in cu]
        ax.plot(x, y, marker="o", ms=4, color=c, mec=SURF)
        endlabel(ax, x[-1], y[-1], lab, fs=8.5)
    ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("bits par caractère (validation)"); ax.set_xlim(0, 11500)
    ax.set_title("Premier plan : un GRU bat le transformer au niveau caractère (TinyStories, 8 000 pas)"); clean(ax)
    fig.tight_layout(); save(fig, "fp_bras_initiaux")


# ================================================================ bilan
def bilan_carte():
    """Une case par (question, architecture) : précision ou écart normalisés à l'échelle hasard → parfait."""
    def ext(task, m, k="acc_96_256", ch=0.5):
        r = st(task, m)
        return None if not r else (np.median([x[k] for x in r]) - ch) / (1 - ch)
    rows = [
        ("Parité, positions 96–256", lambda m: ext("S2", m)),
        ("S3, positions 96–256", lambda m: ext("S3", m, ch=1 / 6)),
        ("S3 réduit, positions 48–96 (5 graines)", lambda m: ext("S3c", {"V_lam0": "-", "T_rope": "-"}.get(m, m), "acc_48_96", 1 / 6)),
        ("S5, dans la longueur", lambda m: ext("S5", m, "acc_0_24", 1 / 120)),
        ("Mémoire : rappel à n = 96", lambda m: (lambda v: None if not v else (np.median(v) - 0.125) / 0.875)(_mem({"L_loop": "L_loop1", "L_noloop": "L_noloop1"}.get(m, m), 96, "recall"))),
    ]
    cols = ["T_rope", "V_lam0", "D_blob_e1", "L_noloop", "L_wide", "L_i2", "L_loop"]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, (_, f) in enumerate(rows):
        for j, m in enumerate(cols):
            v = f(m)
            if v is not None:
                M[i, j] = max(0.0, v)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#f3f1fc", "#c9c3f1", "#8f82e4", VIOLET, "#2c1f8a"])
    fig, ax = plt.subplots(figsize=(11, 3.7))
    ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color=MUTED, fontsize=9)
            else:
                ax.text(j, i, "%.0f" % (100 * M[i, j]), ha="center", va="center", fontsize=9.5, color="white" if M[i, j] > 0.6 else INK)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([LAB[c] for c in cols]); ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color=SURF, lw=3); ax.tick_params(which="minor", length=0); ax.tick_params(length=0)
    ax.text(-0.5, len(rows) - 0.25, "Médiane des graines, ramenée sur 0 = hasard → 100 = parfait. « — » : non mesuré.", fontsize=8.5, color=INK2, va="top")
    ax.set_title("Carte des résultats : là où il faut garder un état longtemps, seule la boucle tient", pad=28)
    fig.tight_layout(); save(fig, "bilan_carte")


def p6_corpus():
    """Six corpus réels, CPU, 2 000 pas : bpc relatif à T_rope dans la longueur d'entraînement et au-delà."""
    cs = [("Wikitext-2", "wiki"), ("Code Python", "code_python"), ("PTB", "ptb"), ("Contes Gutenberg", "gutenberg_contes"),
          ("Gutenberg complet", "gutenberg_tout"), ("Shakespeare", "shakespeare")]
    f = lambda c, m: J("lm_wiki_%s_s0.json" % m if c == "wiki" else "lm_%s_%s_s0.json" % (c, m))["final"]["256"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax, k, tt in zip(axs, ("0-64", "128-256"), ("Positions 0–64 (longueur d'entraînement)", "Positions 128–256 (extrapolation)")):
        for i, (lab, c) in enumerate(cs):
            t = f(c, "T_rope")[k]
            for j, m in enumerate(("V_lam0", "L_loop1")):
                v = 100 * (f(c, m)[k] / t - 1); y = i + (j - 0.5) * 0.3
                ax.barh(y, v, height=0.27, color=COL[m]); ax.text(v + (0.4 if v >= 0 else -0.4), y, "%+.1f %%" % v, va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color=INK, lw=1); ax.set_xlim(-16, 12); ax.set_title(tt, fontsize=10); ax.set_xlabel("bpc relatif à T_rope (négatif = mieux que T_rope)"); clean(ax, grid="x")
    axs[0].set_yticks(range(len(cs))); axs[0].set_yticklabels([c[0] for c in cs]); axs[0].invert_yaxis()
    axs[0].legend(handles=[Line2D([], [], color=COL[m], lw=6, label=LAB[m]) for m in ("V_lam0", "L_loop1")], fontsize=8.5, loc="center left")
    fig.suptitle("Six corpus réels : la boucle reste à quelques % de T_rope, et gagne partout au-delà de 64 tokens sauf sur Wikitext-2", fontsize=10.5)
    fig.tight_layout(); save(fig, "p6_corpus")


def bilan_langage():
    """Les trois mesures de langage côte à côte, en écart relatif à T_rope (référence)."""
    rows = []
    e = {n: J(f"s2_{n}.json")["extrapolation"] for n in ("T_rope", "V_lam0", "D_blob_e1")}
    rows.append(("TinyStories 0–256", {m: e[m]["bpc_0_256"] / e["T_rope"]["bpc_0_256"] - 1 for m in e}))
    rows.append(("TinyStories 512–1 024", {m: e[m]["bpc_512_1024"] / e["T_rope"]["bpc_512_1024"] - 1 for m in e}))
    w = {m: J(f"lm_wiki_{m}_s0.json")["final"]["256"] for m in ("T_rope", "V_lam0", "L_loop1")}
    rows.append(("Wikitext-2 0–64", {m: w[m]["0-64"] / w["T_rope"]["0-64"] - 1 for m in w}))
    rows.append(("Wikitext-2 128–256", {m: w[m]["128-256"] / w["T_rope"]["128-256"] - 1 for m in w}))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    models = ["V_lam0", "D_blob_e1", "L_loop1"]
    for i, (lab, d) in enumerate(rows):
        for j, m in enumerate(models):
            if m in d:
                v = 100 * d[m]
                y = i + (j - 1) * 0.22
                ax.barh(y, v, height=0.2, color=COL[m])
                ax.text(v + (0.8 if v >= 0 else -0.8), y, "%+.1f %%" % v, va="center", ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows]); ax.invert_yaxis(); ax.set_xlim(-35, 22)
    ax.set_xlabel("bpc relatif à T_rope (négatif = mieux que T_rope)")
    ax.legend(handles=[Line2D([], [], color=COL[m], lw=6, label=LAB[m]) for m in models], fontsize=8.5, loc="lower left")
    ax.set_title("Sur du texte : jamais loin de T_rope, nettement mieux seulement en extrapolation sur TinyStories"); clean(ax, grid="x")
    fig.tight_layout(); save(fig, "bilan_langage")


def p9_pointeurs():
    """Poursuite de pointeurs sur liste circulaire (amendement 23) : précision par nombre de sauts h et effet des tours de réflexion k."""
    import glob
    D = {}
    for f in glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "reseau_*_s*.json")):
        r = json.load(open(f)); D.setdefault(r["model"], {})[r["seed"]] = r["grille"]
    spec = [("L_loop1_k", "boucle, entraînée avec pauses", "#6250d6", "-"), ("L_loop1_k0", "boucle, jamais de pause", "#6250d6", "--"),
            ("L_noloop1_k", "sans retour T2", "#8a8a85", "-"), ("T_rope_k", "T_rope (2 couches)", "#2a78d6", "-")]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for m, lab, c, ls in spec:
        for ax, k in ((axs[0], 0), ):
            v = np.array([[100 * g[str(h)][str(k)] for h in range(1, 6)] for g in D[m].values()])
            ax.plot(range(1, 6), np.median(v, 0), ls, color=c, lw=2, label=lab)
            for row in v: ax.plot(range(1, 6), row, "o", color=c, ms=3.5, alpha=.45)
    axs[0].axvspan(3.5, 5.5, color="#eeeeea", zorder=0); axs[0].text(4.5, 103, "hors entraînement", ha="center", fontsize=8.5, color=INK2)
    axs[0].set_xlabel("nombre de sauts h (entraînement : 1 à 3)"); axs[0].set_ylabel("précision (%), k = 0 pauses"); axs[0].set_xticks(range(1, 6)); axs[0].set_ylim(0, 110)
    axs[0].set_title("Sauts enchaînés sans temps de réflexion", fontsize=10); axs[0].legend(fontsize=8.5, loc="center right"); clean(axs[0])
    KS = [0, 2, 4, 8, 12]
    for m, lab, c, ls in spec[:2]:
        v = np.array([[100 * g["3"][str(k)] for k in KS] for g in D[m].values()])
        axs[1].plot(range(len(KS)), np.median(v, 0), ls, color=c, lw=2)
        for row in v: axs[1].plot(range(len(KS)), row, "o", color=c, ms=3.5, alpha=.45)
    axs[1].set_xticks(range(len(KS))); axs[1].set_xticklabels(KS); axs[1].set_ylim(0, 110); axs[1].set_xlabel("jetons de pause k donnés au test")
    axs[1].set_title("h = 3 : pauses ajoutées au test (tirets : jamais vues à l'entraînement)", fontsize=10); clean(axs[1])
    fig.suptitle("Liste circulaire de 6 symboles : la boucle chaîne 3 sauts sans réfléchir, pas un de plus", fontsize=10.5)
    fig.tight_layout(); save(fig, "p9_pointeurs")


def p10_s4():
    """S4 (amendements 27, 27b, 29) : précision selon la longueur n, et courbes d'apprentissage à n = 24, 15 000 pas."""
    import glob
    spec = [("T_rope", "T_rope", "#2a78d6"), ("L_loop1", "boucle L_loop1", "#6250d6"), ("L_dloop1", "double loop (demi-taille)", "#e0762a"),
            ("L_dloop1_full", "double loop, branche principale pleine", "#3a9d5d")]
    D = {}
    for m, _, _ in spec:
        for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "s4_%s_s*_p15000.json" % m))):
            r = json.load(open(f))
            if r.get("done"): D.setdefault(m, []).append(r)
    ns = ["12", "24", "48", "96", "192"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for m, lab, c in spec:
        if m not in D: continue
        v = np.array([[100 * r["final"][n] for n in ns] for r in D[m]])
        axs[0].plot(range(len(ns)), np.median(v, 0), "-", color=c, lw=2, label=lab + (" (%d graines)" % len(v) if len(v) > 1 else ""))
        for row in v: axs[0].plot(range(len(ns)), row, "o", color=c, ms=3.5, alpha=.45)
        for r in D[m]:
            axs[1].plot([h["step"] for h in r["hist"]], [100 * h["24"] for h in r["hist"]], "-", color=c, lw=1.6, alpha=.9)
    axs[0].axhline(100 / 24, color=MUTED, lw=1, ls=":"); axs[0].text(-0.2, 100 / 24 + 2.5, "hasard", fontsize=8.5, color=INK2, ha="left")
    axs[0].axvspan(1.5, 4.5, color="#eeeeea", zorder=0); axs[0].text(3.0, 103, "au-delà de l'entraînement (n \u2264 24)", ha="center", fontsize=8.5, color=INK2)
    axs[0].set_xticks(range(len(ns))); axs[0].set_xticklabels(ns); axs[0].set_ylim(0, 112); axs[0].set_xlabel("longueur de la séquence n"); axs[0].set_ylabel("précision (%)")
    axs[0].set_title("Précision finale selon la longueur", fontsize=10); axs[0].legend(fontsize=8.5, loc="center right"); clean(axs[0])
    axs[1].set_xlabel("pas d'entraînement"); axs[1].set_ylabel("précision à n = 24 (%)"); axs[1].set_ylim(0, 105)
    axs[1].set_title("Apprentissage à n = 24 : décrochage brutal, entre 5 000 et 14 000 pas selon la graine", fontsize=10); clean(axs[1])
    fig.suptitle("S4 (24 états), 15 000 pas : la boucle et le double loop à branche pleine apprennent (2 graines sur 3), T_rope et le demi-taille non", fontsize=10.5)
    fig.tight_layout(); save(fig, "p10_s4")


def p11_silverbox():
    """Silverbox (amendement 28) : RMSE de simulation (mV) sur la flèche de test, globale et par quart d'amplitude croissante."""
    R = lambda m: J("silverbox_%s_s0.json" % m)
    lin = R("lineaire")["lag96"]["test"]
    spec = [("lineaire", "régression linéaire (96 retards)", "#8a8a85", lin), ("GRU", "GRU (2 000 pas)", "#d98324", R("GRU")["test"]),
            ("LSTM", "LSTM (2 000 pas)", "#eda100", R("LSTM")["test"]), ("GRU_20k", "GRU (20 000 pas)", "#b36a1c", R("GRU_20k")["test"]),
            ("LSTM_20k", "LSTM (20 000 pas)", "#c98d00", R("LSTM_20k")["test"]), ("T_rope", "T_rope", "#2a78d6", R("T_rope")["test"]),
            ("L_noloop1", "boucle sans retour T2", "#9a9a9a", R("L_noloop1")["test"]), ("L_loop1", "boucle (L_loop1)", "#6250d6", R("L_loop1")["test"])]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for i, (m, lab, c, r) in enumerate(spec):
        axs[0].barh(i, r["rmse_mV"], color=c, height=.7); axs[0].text(r["rmse_mV"] + .6, i, "%.1f" % r["rmse_mV"], va="center", fontsize=9)
    axs[0].set_yticks(range(len(spec))); axs[0].set_yticklabels([s[1] for s in spec], fontsize=9); axs[0].invert_yaxis(); axs[0].set_xlim(0, 50)
    axs[0].axvline(lin["rmse_mV"] / 2, color=INK2, ls=":", lw=1); axs[0].text(lin["rmse_mV"] / 2 + .4, 7.55, "critère (a) : 7 mV", fontsize=8, color=INK2)
    axs[0].set_xlabel("RMSE de simulation sur la flèche de test (mV, bas = mieux)"); axs[0].set_title("Toute la flèche de test", fontsize=10); clean(axs[0], grid="x")
    for m, lab, c, r in spec:
        if m in ("GRU_20k", "LSTM_20k", "L_noloop1"): continue
        axs[1].plot(range(1, 5), r["quarts_mV"], "o-", color=c, lw=2, label=lab)
    axs[1].set_xticks(range(1, 5)); axs[1].set_xticklabels(["1er quart\n(|u| < 0,04)", "2e", "3e", "4e quart\n(|u| jusqu'à 0,149 :\nhors plage d'entraînement)"], fontsize=8.5)
    axs[1].set_ylabel("RMSE (mV)"); axs[1].set_title("Par quart d'amplitude croissante", fontsize=10); axs[1].legend(fontsize=8.5, loc="upper left"); clean(axs[1])
    fig.suptitle("Silverbox : la boucle fait 10,5 mV contre 9,6 pour un GRU ; dans la plage d'entraînement, aucun réseau ne bat la régression linéaire", fontsize=10.5)
    fig.tight_layout(); save(fig, "p11_silverbox")


# ================================================================ partie 14 : courbes d'apprentissage
def _logs():
    return J("courbes_logs.json")


def _curv(keys, field="loss"):
    """moyenne, min et max sur les graines des courbes (memes pas) ; keys : cles de courbes_logs.json."""
    L = _logs(); cs = [L[k] for k in keys if k in L]
    if not cs: return None
    n = min(len(c) for c in cs); x = [c[i]["step"] for i in range(n) for c in cs[:1]]
    v = np.array([[c[i][field] for i in range(n)] for c in cs])
    return np.array(x), v.mean(0), v.min(0), v.max(0)


def _jcurve(files, key="curve", field="loss"):
    rs = [J(f) for f in files if has(f)]
    if not rs: return None
    cs = [r[key] for r in rs]; n = min(len(c) for c in cs)
    v = np.array([[c[i][field] for i in range(n)] for c in cs])
    return np.array([cs[0][i]["step"] for i in range(n)]), v.mean(0), v.min(0), v.max(0)


def _line(ax, c, color, lab=None, ls="-", lw=2, band=True, endl=False):
    if c is None: return
    x, m, lo, hi = c
    ax.plot(x, m, ls, color=color, lw=lw, label=lab)
    if band and len(lo) and np.any(hi - lo > 1e-9): ax.fill_between(x, lo, hi, color=color, alpha=.15, lw=0)


def p14_etat():
    """Courbes de perte de l'entrainement : S3, S3 + memoire longue, S4, S5 (bande = min/max sur les graines)."""
    L = _logs(); fig, axs = plt.subplots(2, 2, figsize=(12.5, 8))
    GRUc, LSTMc = AQUA, YELLOW
    ax = axs[0, 0]
    for m, lab, c in (("L_loop", "boucle", VIOLET), ("L_noloop", "sans retour T2", GRAY), ("L_wide", "4 couches, sans retour T2 (L_wide)", RED), ("L_i2", "T2 écrit dans l'image 2 (L_i2)", MAGENTA)):
        _line(ax, _jcurve(["st_S3c_%s_s%d.json" % (m, k) for k in range(5)]), c, lab)
    _line(ax, _curv(["rnn_S3_GRU_%d|S3 GRU|%d" % (k, k) for k in range(3)]), GRUc, "GRU"); _line(ax, _curv(["rnn_S3_LSTM_%d|S3 LSTM|%d" % (k, k) for k in range(3)]), LSTMc, "LSTM")
    ax.set_title("S3 (permutations de 6 éléments), 2 000 pas", fontsize=10); ax.set_ylabel("perte d'entraînement"); ax.legend(fontsize=8.5); clean(ax)
    ax = axs[0, 1]
    _line(ax, _curv(["m0|S3M L_loop1|0", "m1|S3M L_loop1|1", "m0|S3M L_loop1|2"]), VIOLET, "boucle (1 couche)")
    _line(ax, _curv(["m1|S3M L_noloop1|0", "m0|S3M L_noloop1|1", "m1|S3M L_noloop1|2"]), GRAY, "sans retour T2")
    _line(ax, _curv(["m0|S3M T_rope|0", "m1|S3M T_rope|1", "m0|S3M T_rope|2"]), BLUE, "T_rope")
    _line(ax, _curv(["rnn_S3M_GRU_%d|S3M GRU|%d" % (k, k) for k in range(3)]), GRUc, "GRU"); _line(ax, _curv(["rnn_S3M_LSTM_%d|S3M LSTM|%d" % (k, k) for k in range(3)]), LSTMc, "LSTM")
    ax.set_title("S3 + mémoire longue (S3M), 3 000 pas", fontsize=10); ax.legend(fontsize=8.5); clean(ax)
    ax = axs[1, 0]
    for m, lab, c in (("T_rope", "T_rope", BLUE), ("L_loop1", "boucle", VIOLET), ("L_dloop1_full", "double loop", GREEN), ("G231", "réseau 2-3-1", RED)):
        _line(ax, _jcurve(["s4_%s_s%d_p15000.json" % (m, k) for k in range(3)], key="hist"), c, lab)
    ax.axhline(np.log(24), color=MUTED, lw=1, ls=":"); ax.text(15000, np.log(24) + .05, "hasard (ln 24)", ha="right", fontsize=8.5, color=INK2)
    ax.set_title("S4 (24 états), 15 000 pas : plateau puis décrochage", fontsize=10); ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("perte d'entraînement"); ax.legend(fontsize=8.5); clean(ax)
    ax = axs[1, 1]
    for m, lab, c in (("T_rope", "T_rope", BLUE), ("L_loop1", "boucle", VIOLET), ("G231", "réseau 2-3-1", RED), ("G231_ff", "réseau 2-3-1 sans retour", GRAY)):
        _line(ax, _jcurve(["s5_%s_s0.json" % m]), c, lab)
    _line(ax, _curv(["s5_GRU|S5 GRU|0"]), GRUc, "GRU")
    ax.axhline(np.log(120), color=MUTED, lw=1, ls=":"); ax.text(8000, np.log(120) + .03, "hasard (ln 120)", ha="right", fontsize=8.5, color=INK2)
    ax.set_title("S5 (120 états), 8 000 pas : la perte monte avec le curriculum, personne ne décroche", fontsize=9.5); ax.set_xlabel("pas d'entraînement"); clean(ax)
    axs[0, 0].set_xlabel("pas d'entraînement"); axs[0, 1].set_xlabel("pas d'entraînement")
    fig.suptitle("Pertes d'entraînement : plateau initial, puis chute brutale quand le mécanisme est trouvé (jamais pour S5)", fontsize=10.5)
    fig.tight_layout(); save(fig, "p14_etat")


def p14_pointeurs():
    """Liste circulaire : pertes (h <= 3 avec pauses, h <= 3 petite taille / taille de la boucle, h <= 5)."""
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8)); GRUc, LSTMc = AQUA, YELLOW; ln6 = np.log(6)
    ax = axs[0, 0]
    _line(ax, _curv(["wr0|L_loop1_k|%d" % k for k in range(3)]), VIOLET, "boucle (pauses à l'entraînement)")
    _line(ax, _curv(["wr1|L_loop1_k0|%d" % k for k in range(3)]), VIOLET, "boucle (jamais de pause)", ls="--")
    _line(ax, _curv(["wr0|L_noloop1_k|%d" % k for k in range(3)]), GRAY, "sans retour T2")
    _line(ax, _curv(["wr1|T_rope_k|%d" % k for k in range(3)]), BLUE, "T_rope")
    _line(ax, _curv(["rnn_PTR_GRU_%d|PTR GRU|%d" % (k, k) for k in range(3)]), GRUc, "GRU"); _line(ax, _curv(["rnn_PTR_LSTM_%d|PTR LSTM|%d" % (k, k) for k in range(3)]), LSTMc, "LSTM")
    ax.axhline(ln6, color=MUTED, lw=1, ls=":"); ax.set_title("h \u2264 3 : le retour T2 ou une récurrence font chuter la perte", fontsize=10); ax.set_ylabel("perte d'entraînement"); ax.legend(fontsize=8.5); clean(ax)
    ax = axs[0, 1]
    _line(ax, _curv(["h5_L_loop1|H5 L_loop1|0", "h5_L_loop1_1|H5 L_loop1|1", "h5_L_loop1_2|H5 L_loop1|2"]), VIOLET, "boucle (3 graines)")
    for k, (kk, ls) in enumerate((("h5_L_loop1|H5 L_loop1|0", ":"), ("h5_L_loop1_1|H5 L_loop1|1", "--"), ("h5_L_loop1_2|H5 L_loop1|2", "-."))):
        c = _curv([kk]); ax.plot(c[0], c[1], ls, color=VIOLET, lw=1.3, alpha=.8)
    _line(ax, _curv(["h5_GRU|H5 GRU|0"]), GRUc, "GRU"); _line(ax, _curv(["h5_G231|H5 G231|0"]), RED, "réseau 2-3-1")
    ax.axhline(ln6, color=MUTED, lw=1, ls=":"); ax.set_title("h \u2264 5 : le GRU descend, la boucle plafonne (graine 0 : tardivement)", fontsize=10); ax.legend(fontsize=8.5); clean(ax)
    for ax, pre, ttl in ((axs[1, 0], "diag", "Petite taille (images 4×4×4, d = 32)"), (axs[1, 1], "full", "Taille de la boucle (images 4×8×8, d = 64)")):
        for nm, c in (("1-1", VIOLET), ("2-1", GREEN), ("1-1-1", RED), ("2-2-1", ORANGE), ("2-3-1", GRAY), ("2-3-1z", MAGENTA)):
            k = "%s_%s|DIAG %s|0" % (pre, nm, nm)
            if k in _logs(): _line(ax, _curv([k]), c, nm)
        ax.axhline(ln6, color=MUTED, lw=1, ls=":"); ax.set_title("Échelle de croissance, h \u2264 3 : " + ttl, fontsize=10); ax.set_xlabel("pas d'entraînement"); ax.legend(fontsize=8.5, ncol=2); clean(ax)
    axs[1, 0].set_ylabel("perte d'entraînement")
    for ax in axs[0]: ax.set_xlabel("pas d'entraînement")
    fig.suptitle("Liste circulaire (n = 6) : le hasard est à ln 6 = 1,79 ; sans mécanisme, la perte reste collée à ce plateau", fontsize=10.5)
    fig.tight_layout(); save(fig, "p14_pointeurs")


def p14_reel():
    """Silverbox (RMSE de validation pendant l'entrainement), histoires synthetiques et Wikitext (bpc 0-64)."""
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3)); L = _logs(); ax = axs[0]
    sb = {"GRU": (AQUA, ["sb_rest|GRU|0", "sb_GRU_1|GRU|1", "sb_GRU_2|GRU|2"]), "LSTM": (YELLOW, ["sb_rest|LSTM|0", "sb_LSTM_1|LSTM|1", "sb_LSTM_2|LSTM|2"]),
          "T_rope": (BLUE, ["sb_rest|T_rope|0", "sb_T_rope_1|T_rope|1", "sb_T_rope_2|T_rope|2"]), "boucle": (VIOLET, ["sb_loop|L_loop1|0", "sb_L1_1|L_loop1|1", "sb_L1_2|L_loop1|2"]), "boucle sans T2": (GRAY, ["sb_rest|L_noloop1|0"])}
    for lab, (c, ks) in sb.items(): _line(ax, _curv(ks, "val_mV"), c, lab)
    ax.axhline(5.46, color=INK2, lw=1, ls=DASH); ax.text(250, 5.7, "régression linéaire (validation) : 5,5 mV", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yscale("log"); ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("RMSE de validation (mV)"); ax.set_title("Silverbox : tous plafonnent vers 7 mV", fontsize=10); ax.legend(fontsize=8.5); clean(ax)
    ax = axs[1]
    for lab, c, ks in (("boucle", VIOLET, ["lm_loop|LM L_loop1|0"]), ("T_rope", BLUE, ["lm_t|LM T_rope|0"]), ("V λ=0", AQUA, ["lm_t|LM V_lam0|0"])):
        _line(ax, _curv(ks, "reponse"), c, lab, band=False)
    ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("précision sur la réponse"); ax.set_title("Histoires synthétiques (qui / où)", fontsize=10); ax.legend(fontsize=8.5); clean(ax)
    ax = axs[2]
    for lab, c, ks in (("boucle", VIOLET, ["w_loop|WIKI L_loop1|0"]), ("T_rope", BLUE, ["w_t|WIKI T_rope|0"]), ("V λ=0", AQUA, ["w_t|WIKI V_lam0|0"])):
        _line(ax, _curv(ks, "bpc_0_64"), c, lab, band=False)
    ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("bits par caractère (positions 0-64)"); ax.set_title("Wikitext-2 : la boucle apprend plus lentement, puis rattrape", fontsize=10); ax.legend(fontsize=8.5); clean(ax)
    fig.tight_layout(); save(fig, "p14_reel")


def p14_rapide():
    """Version rapide contre version d'origine de la remanence (amendement 38) : liste circulaire h <= 3, memes graine et donnees, 3 graines."""
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True)
    for k in range(3):
        o, r = J("verif_rapide_origine_s%d.json" % k)["origine"], J("verif_rapide_rapide_s%d.json" % k)["rapide"]
        for row, (fld, fn) in enumerate((("acc", lambda c: 100 * c["acc"][2]), ("loss", lambda c: c["loss"]))):
            ax = axs[row, k]
            ax.plot([c["step"] for c in o["curve"]], [fn(c) for c in o["curve"]], "-o", color=VIOLET, lw=2.2, ms=4, label="origine")
            ax.plot([c["step"] for c in r["curve"]], [fn(c) for c in r["curve"]], "--s", color=ORANGE, lw=2, ms=4, label="rapide")
            clean(ax)
            if row == 0: ax.set_title("graine %d" % k, fontsize=10); ax.set_ylim(0, 105)
            else: ax.axhline(np.log(6), color=MUTED, lw=1, ls=":"); ax.set_xlabel("pas d'entraînement")
    axs[0, 0].set_ylabel("précision à h = 3 (%)"); axs[1, 0].set_ylabel("perte d'entraînement"); axs[0, 0].legend(fontsize=9)
    fig.suptitle("Version rapide contre origine, liste circulaire h \u2264 3 : les deux apprennent ou non selon la graine, pas selon la version", fontsize=10.5)
    fig.tight_layout(); save(fig, "p14_rapide")


FIGS = [p1_capacite, p1_continuite, p1_apprentissage, p1_position,
        p2_extrapolation, p2_apprentissage, p2_cout, p2_flux, p2_rappel_listes,
        p3_etat, p3_graines, p3_long, p3_apprentissage,
        p4_controles, p4_graines, p4_ablations, p4_causal, p4_cout,
        p5_memoire, p5_taille_image, p5_sondes,
        p6_synthetique, p6_wikitext, p6_corpus, p9_pointeurs, p11_silverbox, p10_s4, p14_etat, p14_pointeurs, p14_reel, p14_rapide,
        p7_bruit_pca, p7_positions_blob, p7_surprise, fp_bras_initiaux,
        bilan_carte, bilan_langage]

if __name__ == "__main__":
    pref = sys.argv[1] if len(sys.argv) > 1 else ""
    for f in FIGS:
        if f.__name__.startswith(pref):
            f()
