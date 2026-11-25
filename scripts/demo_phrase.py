# Image token par token + predictions du mot suivant. A lancer depuis la racine : python scripts/demo_phrase.py
import sys, torch, numpy as np, json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "src")
from tokenizers import Tokenizer
import s2_models as M
tok = Tokenizer.from_file("tokenizer/bpe_4096.json")
text = "Lily was hungry. She went to the kitchen and ate a red apple. Then she ate a big rock."
ids = tok.encode(text).ids
toks = [tok.decode([i]) for i in ids]
print(len(ids), toks)
R = "results/"
def load(name):
    ck = torch.load(R + "s2_%s_model.pt" % name, map_location="cpu", weights_only=False)
    m = M.build(ck["spec"], ck["V"], ck["cfg"]); m.load_state_dict(ck["state"]); return m.eval()
x = torch.tensor([ids])
out = {}
for name in ("V_lam0", "V_lam1_ext"):
    m = load(name)
    with torch.no_grad():
        logits, fr = m(x, return_frames=True)
    lp = torch.log_softmax(logits[0], -1)
    fr = fr[0]                                     # (N+1, 6, 8, 8)
    step = (fr[1:] - fr[:-1]).flatten(1).norm(dim=1).numpy()
    out[name] = dict(fr=fr, lp=lp, step=step)
    surpr = [-lp[i, ids[i + 1]].item() / np.log(2) for i in range(len(ids) - 1)]
    out[name]["surpr"] = surpr
    print(name, "pas moyen", step.mean().round(3), "indice (une phrase)", float(M.continuity_index(fr[None], 2)))
# ---- tableau de decodage
lines = []
for i in range(len(ids) - 1):
    row = [repr(toks[i]), repr(toks[i + 1])]
    for n in out:
        lp = out[n]["lp"][i]; top = lp.topk(3)
        row.append(" / ".join("%s %.0f%%" % (repr(tok.decode([j.item()])), 100 * np.exp(v.item())) for v, j in zip(top.values, top.indices)))
        row.append("%.1f" % out[n]["surpr"][i])
    lines.append(" | ".join(row))
open("results/s2_decodage.txt", "w").write("\n".join(lines))
print("\n".join(lines))
# ---- figure
N = len(ids)
fig = plt.figure(figsize=(max(14, N * 0.52), 11.5), facecolor="#fcfcfb")
gs = fig.add_gridspec(7, N + 1, height_ratios=[1, 1, 1, 0.12, 1, 1, 1.5], hspace=0.12, wspace=0.05)
def show(ax, f, m):
    img = np.clip(0.5 + 0.5 * f.permute(1, 2, 0).numpy() / m, 0, 1)
    ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
rows = {"V_lam0": (0, "V_lam0\n(sans penalite)"), "V_lam1_ext": (4, "V_lam1_ext\n(lambda = 1)")}
for name, (r0, lab) in rows.items():
    fr = out[name]["fr"]
    mf = fr[:, :3].abs().max().item(); ms = fr[:, 3:].abs().max().item()
    for t in range(N):
        show(fig.add_subplot(gs[r0, t]), fr[t + 1, :3], mf)
        show(fig.add_subplot(gs[r0 + 1, t]), fr[t + 1, 3:], ms)
        if r0 == 0 and t == 0:
            pass
    a = fig.add_subplot(gs[r0, N]); a.axis("off"); a.text(0, 0.5, "rapide", fontsize=9, color="#52514e", va="center")
    a = fig.add_subplot(gs[r0 + 1, N]); a.axis("off"); a.text(0, 0.5, "lent", fontsize=9, color="#52514e", va="center")
    ax = fig.add_subplot(gs[r0 + 2, :N] if r0 == 0 else gs[r0 + 2, :N])
    ax.plot(range(N), out[name]["step"], color="#2a78d6" if name == "V_lam0" else "#e87ba4", lw=2, marker="o", ms=4)
    ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(0, None); ax.set_xticks([]); ax.set_facecolor("#fcfcfb")
    ax.set_ylabel("taille du pas\n" + lab, fontsize=8.5, color="#52514e")
    ax.grid(axis="y", color="#e3e2de", lw=.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
# etiquettes de tokens en haut
for t in range(N):
    fig.axes[t * 2].set_title(toks[t].replace(" ", "·") or "?", rotation=90, fontsize=9, color="#0b0b0b", pad=3)
fig.suptitle("Image qui evolue, token par token : phrase « %s »" % text, fontsize=11, color="#0b0b0b", x=0.02, ha="left")
fig.savefig("results/s2_film_frames.png", dpi=140, facecolor="#fcfcfb", bbox_inches="tight")
