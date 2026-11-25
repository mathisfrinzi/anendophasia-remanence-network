"""ListOps réduit (Nangia et Bowman, 2018) : expressions imbriquées sur les chiffres 0-9, à évaluer.

    [MAX 2 9 [MIN 4 7 ] 0 ]  ->  9

Opérateurs : MAX, MIN, MED (médiane, arrondie à l'inférieur pour un nombre pair d'arguments), SM (somme modulo 10).
Chaque opérateur a 2 à 4 arguments ; la profondeur d'une expression est le nombre maximal d'opérateurs imbriqués.
On tire une expression de profondeur exactement d, de longueur ≤ maxlen tokens. Un token par symbole ; l'opérateur
et son crochet ouvrant sont un seul token (« [MAX »), comme dans ListOps.

    python scripts/listops_data.py      # vérifications et références triviales (majorité, heuristiques)
"""
import numpy as np

OPS = ["[MAX", "[MIN", "[MED", "[SM"]
ITOS = ["<PAD>", "<PAUSE>"] + [str(i) for i in range(10)] + OPS + ["]"]
STOI = {w: i for i, w in enumerate(ITOS)}
PAD, PAUSE, CLOSE = 0, 1, STOI["]"]
V = len(ITOS)
DIGIT0 = STOI["0"]                   # chiffre c -> token DIGIT0 + c


def apply(op, args):
    if op == "[MAX":
        return max(args)
    if op == "[MIN":
        return min(args)
    if op == "[MED":
        return int(sorted(args)[(len(args) - 1) // 2])
    return sum(args) % 10


def tree(rng, d):
    """Arbre de profondeur exactement d (d = 0 : un chiffre)."""
    if d == 0:
        return int(rng.integers(0, 10))
    n = int(rng.integers(2, 5))
    deep = int(rng.integers(0, n))                       # au moins un argument porte la profondeur d - 1
    args = [tree(rng, d - 1) if i == deep else tree(rng, int(rng.integers(0, d))) for i in range(n)]
    return (OPS[int(rng.integers(0, 4))], args)


def value(t):
    return t if isinstance(t, int) else apply(t[0], [value(a) for a in t[1]])


def tokens(t):
    if isinstance(t, int):
        return [str(t)]
    out = [t[0]]
    for a in t[1]:
        out += tokens(a)
    return out + ["]"]


def sample(rng, d, maxlen=64):
    while True:
        t = tree(rng, d)
        tk = tokens(t)
        if len(tk) <= maxlen:
            return [STOI[w] for w in tk], value(t)


def batch(rng, depths, n, maxlen=64, k=0):
    """n expressions, profondeur tirée dans `depths`, remplissage à gauche, k pauses à la fin. -> x (n, L+k), y (n,), d (n,)"""
    ds = rng.choice(depths, n)
    seqs, ys = zip(*[sample(rng, int(d), maxlen) for d in ds])
    L = max(len(s) for s in seqs)
    x = np.full((n, L + k), PAUSE, dtype=np.int64)
    x[:, :L] = PAD
    for i, s in enumerate(seqs):
        x[i, L - len(s):L] = s
    return x, np.array(ys), ds


def _check():
    t = ("[MAX", [2, 9, ("[MIN", [4, 7]), 0])
    assert " ".join(tokens(t)) == "[MAX 2 9 [MIN 4 7 ] 0 ]" and value(t) == 9
    assert apply("[MED", [3, 1, 2]) == 2 and apply("[MED", [4, 1, 3, 2]) == 2 and apply("[SM", [7, 8, 9]) == 4
    rng = np.random.default_rng(0)
    for d in range(1, 6):
        s, y = sample(rng, d)
        def depth(seq):
            m = c = 0
            for i in seq:
                c += ITOS[i] in OPS
                c -= i == CLOSE
                m = max(m, c)
            return m
        assert depth(s) == d, (d, depth(s))
    print("vérifications OK")


def references(n=20000):
    """Précision de règles triviales par profondeur : majorité, dernier chiffre, max de tous les chiffres, opérateur
    extérieur appliqué à ses arguments chiffres seuls."""
    rng = np.random.default_rng(1)
    print("prof. | long. médiane | majorité | dernier chiffre | max des chiffres | op. extérieur sur ses chiffres")
    for d in range(1, 6):
        S = [sample(rng, d) for _ in range(n)]
        ys = np.array([y for _, y in S])
        maj = np.bincount(ys, minlength=10).max() / n
        last = np.mean([[i - DIGIT0 for i in s if DIGIT0 <= i < DIGIT0 + 10][-1] == y for s, y in S])
        mx = np.mean([max(i - DIGIT0 for i in s if DIGIT0 <= i < DIGIT0 + 10) == y for s, y in S])
        def outer(s):
            op, lvl, args = ITOS[s[0]], 0, []
            for i in s[1:]:
                w = ITOS[i]
                if w in OPS:
                    lvl += 1
                elif w == "]":
                    lvl -= 1
                elif lvl == 0:
                    args.append(int(w))
            return apply(op, args) if args else 0
        out = np.mean([outer(s) == y for s, y in S])
        print("%d | %d | %.1f %% | %.1f %% | %.1f %% | %.1f %%" % (d, np.median([len(s) for s, _ in S]), 100 * maj, 100 * last, 100 * mx, 100 * out))


if __name__ == "__main__":
    _check()
    references()
