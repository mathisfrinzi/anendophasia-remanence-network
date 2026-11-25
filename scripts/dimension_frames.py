# Dimension effective et concentration des frames (V_lam0 vs V_lam1_ext). A lancer depuis la racine : python scripts/dimension_frames.py
import sys, torch, numpy as np
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("idx2", "scripts/indice_continuite_textes.py")
src = open("scripts/indice_continuite_textes.py").read().split("for name in")[0]
ns = {}; exec(src, ns)
x = torch.tensor(ns["ids"]); M = ns["M"]
for name in ("V_lam0", "V_lam1_ext"):
    ck = torch.load("results/s2_%s_model.pt" % name, map_location="cpu", weights_only=False)
    m = M.build(ck["spec"], ck["V"], ck["cfg"]); m.load_state_dict(ck["state"]); m.eval()
    with torch.no_grad():
        _, fr = m(x, return_frames=True)
    f = fr[:, 1:].reshape(-1, 2, 3 * 64).numpy()           # (exemples, groupe, 192)
    for g, gn in enumerate(("rapide", "lent")):
        a = f[:, g]; a = a - a.mean(0)
        ev = np.clip(np.linalg.eigvalsh(np.cov(a.T)), 0, None)
        pr = ev.sum() ** 2 / (ev ** 2).sum()
        var = a.var(0); top = np.sort(var)[::-1]
        k = int(0.1 * len(var))
        print("%-11s %-6s dimension effective %.1f / 192 ; part de la variance dans les 10 %% de valeurs les plus variables : %.0f %%" % (name, gn, pr, 100 * top[:k].sum() / top.sum()))
