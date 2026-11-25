"""Amendement 44 : bpc (0-64 et 64-256) en fonction du nombre de parametres, T_rope agrandi vs remanence agrandie."""
import json, glob, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(__file__), "..")
rows = []
for f in sorted(glob.glob(os.path.join(R, "results", "wikitext2_*_s0_a44.json"))):
    d = json.load(open(f)); r = d["bpc"]; W = lambda ks: sum(r[k] * (int(k.split("-")[1]) - int(k.split("-")[0])) for k in ks) / sum(int(k.split("-")[1]) - int(k.split("-")[0]) for k in ks)   # moyenne ponderee par la longueur des tranches
    rows.append((d["model"], d["n_params"], {"0-64": W(["0-16", "16-32", "32-64"]), "64-256": W(["64-128", "128-256"]), **r}))
rows.sort(key=lambda r: r[1])
for m, n, b in rows: print("%-11s %9d  " % (m, n) + "  ".join("%s: %.3f" % kv for kv in b.items()))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
for i, key in enumerate(["0-64", "64-256"]):
    for pre, col, lab in (("T_", "#d95f02", "T_rope agrandi (largeur 64→192)"), ("R_", "#1b6ca8", "rémanence (taille d'image)")):
        pts = [(n, b[key], m) for m, n, b in rows if m.startswith(pre) and key in b]
        if not pts: continue
        ax[i].plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "o-", color=col, label=lab)
        for n, v, m in pts: ax[i].annotate(m.replace("T_", "").replace("R_", ""), (n / 1e6, v), textcoords="offset points", xytext=(3, 5), fontsize=7)
    ax[i].set_title("positions %s" % key); ax[i].set_xlabel("paramètres (millions)"); ax[i].set_ylabel("bits par caractère"); ax[i].grid(alpha=.3)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(os.path.join(R, "figures", "historique", "p44_taille_vs_trope.png"), dpi=130); print("ok")
