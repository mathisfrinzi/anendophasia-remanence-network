"""Lit results/reseau_*_s*.json et applique les criteres de l'amendement 23."""
import json, glob, os, sys
R = os.path.join(os.path.dirname(__file__), "..", "results")
M = ["L_loop1_k", "L_loop1_k0", "L_noloop1_k", "T_rope_k"]
D = {m: {} for m in M}
for f in sorted(glob.glob(os.path.join(R, "reseau_*_s*.json"))):
    r = json.load(open(f)); D[r["model"]][r["seed"]] = r["grille"]
acc = lambda m, s, h, k: 100 * D[m][s][str(h)][str(k)]
print("Precision (%) par graine : h=1..5 a k=0 | k=8 | k=12")
for m in M:
    for s in sorted(D[m]):
        print("%-12s s%d  k0 %s | k8 %s | k12 %s" % (m, s, *[[round(acc(m, s, h, k)) for h in range(1, 6)] for k in (0, 8, 12)]))
def n_ok(m, cond):
    seeds = sorted(D[m]); return sum(1 for s in seeds if cond(s)), len(seeds)
print()
for m in M:
    print("(0) %s : h=1,k=8 < 60 %% sur %d/%d graines" % ((m,) + n_ok(m, lambda s: acc(m, s, 1, 8) < 60)))
L, L0, N = "L_loop1_k", "L_loop1_k0", "L_noloop1_k"
print("(a) boucle >= 90 %% a h=1,2,3 (k=8) : %d/%d" % n_ok(L, lambda s: all(acc(L, s, h, 8) >= 90 for h in (1, 2, 3))))
print("(b) k8-k0 >= 20 pts a h=2 ET h=3 : %d/%d" % n_ok(L, lambda s: all(acc(L, s, h, 8) - acc(L, s, h, 0) >= 20 for h in (2, 3))))
print("(c) L_k(k=8) - L_k0(k=0) >= 20 pts a h=3 : %d/%d" % sum(1 for s in D[L] if s in D[L0] and acc(L, s, 3, 8) - acc(L0, s, 3, 0) >= 20) if False else
      "(c) L_k(k=8) - L_k0(k=0) >= 20 pts a h=3 : %d/%d" % (sum(1 for s in D[L] if s in D[L0] and acc(L, s, 3, 8) - acc(L0, s, 3, 0) >= 20), len(D[L])))
print("(d) L_k - L_noloop_k >= 20 pts a h=3, k=8 : %d/%d" % (sum(1 for s in D[L] if s in D[N] and acc(L, s, 3, 8) - acc(N, s, 3, 8) >= 20), len(D[L])))
print("(e) L_k >= 60 %% a h=4 ou 5 avec k=12 : %d/%d" % n_ok(L, lambda s: max(acc(L, s, 4, 12), acc(L, s, 5, 12)) >= 60))
