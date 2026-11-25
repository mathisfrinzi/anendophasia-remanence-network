"""Figures du bilan (README et docs/RESULTATS.md). Usage : python scripts/figures_bilan.py
Lit results/*.json, ecrit figures/bilan_*.png. Couleurs fixes par famille de modele :
Remanence Network = bleu, T_rope (transformer) = orange, GRU = vert d'eau, LSTM = jaune."""
import json, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
COL = dict(R="#2a78d6", T="#eb6834", GRU="#1baf7a", LSTM="#eda100", grey="#8a8985")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 6, "savefig.dpi": 160, "savefig.bbox": "tight", "figure.facecolor": "white"})
EDGES = ["0-16", "16-32", "32-64", "64-128", "128-256", "256-512"]


def fam(m):
    return "R" if m.startswith("R_") or "loop" in m else "T" if m.startswith("T_") else "GRU" if "GRU" in m else "LSTM" if "LSTM" in m else "grey"


def wavg(r, ks):
    w = [int(k.split("-")[1]) - int(k.split("-")[0]) for k in ks]
    return sum(r[k] * x for k, x in zip(ks, w)) / sum(w)


def load(pat):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, pat))):
        d = json.load(open(f)); out[d["model"]] = d
    return out


LABEL = {"T_rope": "T_rope 0,62 M", "T_96x2": "T_rope 1,01 M", "R_4x8x8": "Remanence 4×8×8 (0,95 M)", "R_4x16x16": "Remanence 4×16×16 (1,11 M)",
         "GRU_256": "GRU 256 (1,56 M)", "GRU_1024": "GRU 1024 (7,8 M)"}


def fig_params():
    """bpc (0-128) en fonction du nombre de parametres, Wikitext-2 et TinyStories (amendement 40)."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (pat, title) in zip(axs, [("wikitext2_*_s0_taille.json", "Wikitext-2"), ("ts_*_s0_taille.json", "TinyStories")]):
        runs = load(pat)
        for f in ("T", "R", "GRU"):
            pts = sorted((d["n_params"], wavg(d["bpc"], EDGES[:4]), m) for m, d in runs.items() if fam(m) == f)
            if not pts: continue
            name = {"T": "Transformer (T_rope)", "R": "Remanence Network", "GRU": "GRU"}[f]
            ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "o-", color=COL[f], label=name)
            for n, b, m in pts:
                ax.annotate(("%.3f" % b).replace(".", ","), (n / 1e6, b), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK2)
        ax.set_xscale("log"); ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0,5", "1", "2", "4", "8"]); ax.minorticks_off()
        ax.set_xlabel("paramètres (millions, échelle log)"); ax.set_ylabel("bits par caractère (0-128)")
        ax.set_title(title + " — plus bas = mieux")
        if title == "TinyStories":
            ax.set_ylim(1.05, 1.42)
            ax.annotate("4×16×16 : 2,525\n(n'a pas démarré, hors cadre)", xy=(1.107, 1.42), xytext=(1.5, 1.37), fontsize=7.5, color=COL["R"],
                        arrowprops=dict(arrowstyle="->", color=COL["R"], lw=1), annotation_clip=False)
    axs[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Langage, 2 000 pas, fenêtres de 128, une graine", fontsize=9, color=INK2, y=1.02)
    fig.savefig(os.path.join(FIG, "bilan_langage_parametres.png")); plt.close(fig)


def fig_position():
    """bpc par tranche de position : au-dela de 128 tokens (jamais vus a l'entrainement)."""
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=False)
    sel = {"Wikitext-2": ("wikitext2_*_s0_taille.json", ["T_96x2", "R_4x16x16", "GRU_256"]),
           "TinyStories": ("ts_*_s0_taille.json", ["T_rope", "R_4x8x8", "GRU_256"])}
    for ax, (title, (pat, models)) in zip(axs, sel.items()):
        runs = load(pat)
        for m in models:
            d = runs[m]; ax.plot(range(len(EDGES)), [d["bpc"][k] for k in EDGES], "o-", color=COL[fam(m)], label=LABEL[m])
        ax.axvspan(3.5, 5.3, color="#f3f2ee", zorder=0); ax.text(4.4, ax.get_ylim()[1], "au-delà de la\nlongueur d'entraînement", ha="center", va="top", fontsize=7.5, color=INK2)
        ax.set_xticks(range(len(EDGES))); ax.set_xticklabels(EDGES, fontsize=7.5); ax.set_xlabel("position du token dans le texte")
        ax.set_ylabel("bits par caractère"); ax.set_title(title); ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.savefig(os.path.join(FIG, "bilan_langage_position.png")); plt.close(fig)


def fig_courbes():
    """Perte d'entrainement : plateau de depart de la remanence."""
    runs = load("wikitext2_*_s0_taille.json")
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for m, ls in (("T_96x2", "-"), ("GRU_256", "-"), ("R_4x8x8", "--"), ("R_4x16x16", "-")):
        c = runs[m]["curve"]; ax.plot([x[0] for x in c], [x[1] for x in c], ls, color=COL[fam(m)], lw=1.6, label=LABEL[m])
    ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("perte (entropie croisée)"); ax.set_ylim(4, 8.6)
    ax.set_title("Wikitext-2 : la Remanence démarre plus tard"); ax.legend(fontsize=7.5)
    fig.savefig(os.path.join(FIG, "bilan_courbes_perte.png")); plt.close(fig)


def fig_cout():
    """Temps d'entrainement (2 000 pas, Wikitext-2, XPU)."""
    runs = load("wikitext2_*_s0_taille.json")
    order = ["T_96x2", "GRU_256", "GRU_1024", "R_4x8x8", "R_4x16x16"]
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for i, m in enumerate(order):
        s = runs[m]["seconds"]; ax.barh(i, s / 60, color=COL[fam(m)], height=0.6)
        ax.text(s / 60 * 1.15, i, ("%.0f s" % s) if s < 120 else ("%.0f min" % (s / 60) if s < 3600 else "%d h %02d" % (s // 3600, s % 3600 // 60)), va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([LABEL[m] for m in order], fontsize=8); ax.set_xscale("log"); ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1 000"]); ax.minorticks_off()
    ax.set_xlabel("minutes (échelle log)"); ax.set_title("Coût : 2 000 pas sur Wikitext-2 (GPU Intel XPU)"); ax.grid(axis="y", visible=False)
    ax.set_xlim(0.5, 2000)
    fig.savefig(os.path.join(FIG, "bilan_cout.png")); plt.close(fig)


def _seeds(pat):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, pat)))]


def _band(ax, xs, runs, getter, color, label, ls="-", lw=2):
    import numpy as np
    Y = np.array([[getter(d, x) for x in xs] for d in runs]) * 100
    med = np.median(Y, 0); ax.plot(range(len(xs)), med, ls, color=color, lw=lw, marker="o", ms=4, label=label + (" (%d graines)" % len(runs) if len(runs) > 1 else " (1 graine)"))
    if len(runs) > 1: ax.fill_between(range(len(xs)), Y.min(0), Y.max(0), color=color, alpha=0.12, lw=0)


def fig_etat():
    """Taches a etat : S3 (controles a capacite egale), S3M (memoire), S4 ; mediane des graines, bande min-max."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    ax = axs[0]; B = ["0_12", "12_24", "24_48", "48_96"]; g = lambda d, b: d["final"]["acc_" + b]
    _band(ax, B, _seeds("st_S3c_L_loop_s?.json"), g, COL["R"], "Remanence")
    _band(ax, B, _seeds("rnn_S3_GRU_s?.json"), g, COL["GRU"], "GRU", ls="--")
    _band(ax, B, _seeds("rnn_S3_LSTM_s?.json"), g, COL["LSTM"], "LSTM", ls=":")
    _band(ax, B, _seeds("st_S3c_L_i2_s?.json"), g, "#8a8985", "témoin : T2 sans retour")
    _band(ax, B, _seeds("st_S3c_L_wide_s?.json"), g, "#b5b4ae", "témoin : T1 élargi", ls="--")
    ax.axhline(100 / 6, color=INK2, lw=0.8, ls=":"); ax.text(3, 100 / 6 + 2, "hasard", fontsize=7, color=INK2, ha="right")
    ax.axvspan(0.5, 3.4, color="#f3f2ee", zorder=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(["0-12", "12-24", "24-48", "48-96"]); ax.set_xlabel("longueur de la suite (entraînement ≤ 12)")
    ax.set_ylabel("précision (%)"); ax.set_title("S3 : suivre un état (6 états)"); ax.set_ylim(0, 105); ax.legend(fontsize=6.8, loc="center right", bbox_to_anchor=(1, 0.66))
    ax.text(0.1, 93, "Remanence, GRU et LSTM : 100 %", fontsize=7, color=INK2)
    ax = axs[1]; N = ["12", "24", "48", "96"]; g = lambda d, n: d["final"][n]["recall"]
    _band(ax, N, _seeds("st_S3M_L_loop1_s?.json"), g, COL["R"], "Remanence")
    _band(ax, N, _seeds("rnn_S3M_GRU_s?.json"), g, COL["GRU"], "GRU", ls="--")
    _band(ax, N, _seeds("st_S3M_T_rope_s?.json"), g, COL["T"], "T_rope")
    _band(ax, N, _seeds("st_S3M_L_noloop1_s?.json"), g, "#8a8985", "témoin sans T2")
    ax.axhline(12.5, color=INK2, lw=0.8, ls=":"); ax.axvspan(1.5, 3.4, color="#f3f2ee", zorder=0)
    ax.set_xticks(range(4)); ax.set_xticklabels(N); ax.set_xlabel("longueur n (entraînement ≤ 24)")
    ax.set_ylabel("rappel de la valeur (%)"); ax.set_title("S3M : retenir une valeur en suivant S3"); ax.set_ylim(0, 105); ax.legend(fontsize=6.4, loc="lower left", bbox_to_anchor=(-0.01, 0.1))
    ax = axs[2]; N = ["12", "24", "48", "96", "192"]; g = lambda d, n: d["final"][n]
    _band(ax, N, _seeds("s4_L_loop1_s?_p15000.json"), g, COL["R"], "Remanence")
    _band(ax, N, _seeds("s4_T_rope_s0_p15000.json"), g, COL["T"], "T_rope")
    ax.axhline(100 / 24, color=INK2, lw=0.8, ls=":"); ax.axvspan(1.5, 4.4, color="#f3f2ee", zorder=0)
    ax.set_xticks(range(5)); ax.set_xticklabels(N); ax.set_xlabel("longueur n (entraînement ≤ 24)")
    ax.set_ylabel("précision (%)"); ax.set_title("S4 : 24 états, 15 000 pas"); ax.set_ylim(0, 105); ax.legend(fontsize=6.8, loc="center right", bbox_to_anchor=(1, 0.33))
    ax.text(2.0, 62, "1 graine sur 3 reste\nà 52-70 %", fontsize=7, color=COL["R"])
    fig.suptitle("Zone grise = longueurs jamais vues à l'entraînement. Trait = médiane, bande = min-max des graines.", fontsize=8.5, color=INK2, y=1.02)
    fig.savefig(os.path.join(FIG, "bilan_taches_etat.png")); plt.close(fig)


def fig_silverbox():
    """Silverbox : erreur de simulation (test, mV), 3 graines."""
    rows = [("GRU", "GRU"), ("LSTM", "LSTM"), ("L_loop1", "R"), ("T_rope", "T")]
    names = {"GRU": "GRU", "LSTM": "LSTM", "L_loop1": "Remanence", "T_rope": "T_rope"}
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for i, (m, f) in enumerate(rows):
        v = [d["test"]["rmse_mV"] for d in _seeds("silverbox_%s_s?.json" % m)]
        ax.barh(i, sum(v) / len(v), color=COL[f], height=0.55, alpha=0.9)
        ax.scatter(v, [i] * len(v), color=INK, s=10, zorder=3)
        ax.text(max(v) + 0.3, i, ("%.2f mV" % (sum(v) / len(v))).replace(".", ","), va="center", fontsize=8)
    ax.axvline(14.03, color=INK2, ls=":", lw=1); ax.text(14.2, -0.45, "régression linéaire", fontsize=7, color=INK2, va="center")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([names[m] for m, _ in rows]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("erreur de simulation, test (mV) — plus bas = mieux"); ax.set_title("Silverbox (identification de système), 3 graines"); ax.set_xlim(0, 18.5)
    fig.savefig(os.path.join(FIG, "bilan_silverbox.png")); plt.close(fig)


if __name__ == "__main__":
    for f in (fig_params, fig_position, fig_courbes, fig_cout, fig_etat, fig_silverbox):
        f(); print("ok", f.__name__)
