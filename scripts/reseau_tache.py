"""Tache « poursuite de pointeurs » sans raccourci, pour le reseau de memoires (amendement 23).
Contexte : n paires (a, f(a)) en ordre aleatoire ; puis [Q] x [H_h] puis k pauses [P] ; reponse = f^h(x) lue a la derniere position.
f est soit une permutation cyclique unique de n symboles (« liste circulaire » : f^h(x) != f^h'(x) pour h != h' < n), soit une fonction aleatoire."""
import numpy as np


class Tache:
    def __init__(self, n=12, hmax=8, kind="cycle"):
        self.n, self.hmax, self.kind = n, hmax, kind
        self.Q, self.H0, self.P = n, n + 1, n + 1 + hmax; self.V = n + 2 + hmax

    def fonction(self, rng, B):
        n = self.n
        if self.kind == "cycle":
            c = np.argsort(rng.random((B, n)), 1); f = np.empty((B, n), dtype=np.int64)
            rows = np.arange(B)[:, None]; f[rows, c] = np.roll(c, -1, 1)
        else:
            f = rng.integers(0, n, (B, n))
        return f

    def batch(self, rng, B, h, k):
        """h : entier ou tableau (B,) ; k : entier. Retourne x (B, 2n+3+k), y (B,) et le depart (B,)."""
        n = self.n; f = self.fonction(rng, B); h = np.broadcast_to(np.asarray(h), (B,))
        ordre = np.argsort(rng.random((B, n)), 1); rows = np.arange(B)[:, None]
        pairs = np.stack([ordre, f[rows, ordre]], 2).reshape(B, 2 * n)
        x0 = rng.integers(0, n, B); y = x0.copy()
        for t in range(self.hmax):
            m = h > t; y[m] = f[np.arange(B)[m], y[m]]
        seq = np.concatenate([pairs, np.full((B, 1), self.Q), x0[:, None], (self.H0 + h - 1)[:, None], np.full((B, k), self.P)], 1)
        return seq.astype(np.int64), y.astype(np.int64), x0


def raccourcis(n, kind, hs=range(1, 7), N=200_000, seed=0):
    """Precision maximale d'un predicteur qui ne calcule pas la chaine complete : constante, x, f^h' (h' != h)."""
    t = Tache(n, 8, kind); rng = np.random.default_rng(seed); f = t.fonction(rng, N); x0 = rng.integers(0, n, N); rows = np.arange(N)
    chaine = [x0]
    for _ in range(max(hs)): chaine.append(f[rows, chaine[-1]])
    out = {}
    for h in hs:
        y = chaine[h]; cst = np.bincount(y, minlength=n).max() / N
        sh = max(((chaine[j] == y).mean() for j in range(0, max(hs) + 1) if j != h))
        out[h] = dict(constante=cst, autre_saut=sh, hasard=1 / n)
    return out


if __name__ == "__main__":
    for kind in ("cycle", "fonction"):
        for n in (8, 12, 16, 32):
            r = raccourcis(n, kind); print("%-8s n=%2d  " % (kind, n) + "  ".join("h%d: raccourci %.2f (cst %.2f)" % (h, v["autre_saut"], v["constante"]) for h, v in r.items()))
