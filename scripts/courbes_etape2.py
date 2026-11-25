# Courbes bpc(pas) de l etape 2. A lancer depuis la racine du depot : python scripts/courbes_etape2.py
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = "results/"
def load(n):
    d = json.load(open(R + "s2_%s_progress.json" % n)); d = d if isinstance(d, list) else d["curve"]
    return [c["step"] for c in d], [c["bpc"] for c in d]
S = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"
series = [("T_rope", "#2a78d6", "T_rope (transformer, RoPE)"), ("T_nope", "#eb6834", "T_nope (transformer sans position)"),
          ("V_lam0", "#1baf7a", "V_lam0 (video, sans penalite)"), ("V1_lam0", "#eda100", "V1_lam0 (video, une echelle)"),
          ("V_lam1", "#e87ba4", "V_lam1 (video, penalite lambda = 1)")]
data = {}
for n, _, _ in series:
    x, y = load(n)
    if n == "V_lam1":
        xe, ye = load("V_lam1_ext"); x, y = x + xe, y + ye
    data[n] = (x, y)
fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor=S)
for ax, (lo, hi, title) in zip(axs, ((0.65, 2.35, "Vue d'ensemble"), (0.66, 1.05, "Zoom : 0,66 a 1,05 bpc"))):
    ax.set_facecolor(S)
    for n, col, lab in series:
        x, y = data[n]
        ax.plot(x, y, color=col, lw=2, marker="o", ms=4.5, mec=S, mew=1.2, label=lab)
    ax.axvline(8000, color=INK2, lw=1, ls=(0, (4, 4)), alpha=.7)
    ax.set_ylim(lo, hi); ax.set_xlim(500, 24500)
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel("pas d'entrainement (1 pas = 32 sequences de 256 tokens)", color=INK2, fontsize=10)
    ax.set_ylabel("bpc de validation (plus bas = mieux)", color=INK2, fontsize=10)
    ax.grid(axis="y", color="#e3e2de", lw=.8); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb6")
    ax.tick_params(colors=INK2)
axs[1].text(8250, 1.035, "reprise de V_lam1\n(nouveau planning)", color=INK2, fontsize=9, va="top")
axs[1].annotate("0,773", (24000, data["V_lam1"][1][-1]), xytext=(23200, 0.835), color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=.8))
axs[1].annotate("0,732", (8000, data["V_lam0"][1][-1]), xytext=(9000, 0.712), color=INK, fontsize=9, arrowprops=dict(arrowstyle="-", color=INK2, lw=.8))
axs[1].annotate("0,706", (8000, data["T_rope"][1][-1]), xytext=(9000, 0.682), color=INK, fontsize=9, annotation_clip=False)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2)
fig.suptitle("Etape 2 : bpc de validation au fil de l'entrainement", color=INK, fontsize=13, x=0.06, ha="left")
fig.tight_layout(rect=(0, 0.1, 1, 0.95))
fig.savefig("results/s2_courbes_bpc.png", dpi=150, facecolor=S)
