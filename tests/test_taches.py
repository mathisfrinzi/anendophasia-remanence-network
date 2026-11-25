"""Les générateurs de tâches calculent bien ce qu'ils prétendent."""
import itertools

import numpy as np
import pytest

import s3_tasks
import s5_tasks


@pytest.mark.parametrize("k,ordre", [(2, 2), (3, 6), (5, 120)])
def test_groupe(k, ordre):
    G, mult = s3_tasks.group(k)
    assert G == ordre
    assert set(np.unique(mult)) == set(range(G))                     # fermé
    e = list(itertools.permutations(range(k))).index(tuple(range(k)))
    assert (mult[e] == np.arange(G)).all() and (mult[:, e] == np.arange(G)).all()
    a, b, c = np.random.default_rng(0).integers(0, G, (3, 200))
    assert (mult[mult[a, b], c] == mult[a, mult[b, c]]).all()        # associatif
    commutatif = (mult == mult.T).all()
    assert commutatif == (k == 2)                                    # S3 et S5 ne commutent pas


def test_produit_cumule():
    G, mult = s3_tasks.group(3)
    g, p = s3_tasks.make_batch(np.random.default_rng(0), G, mult, 8, 30)
    g, p = g.numpy(), p.numpy()
    for b in range(8):
        acc = g[b, 0]
        assert p[b, 0] == acc
        for t in range(1, 30):
            acc = mult[acc, g[b, t]]
            assert p[b, t] == acc


def test_dyck():
    x, d = s5_tasks.dyck_batch(np.random.default_rng(0), 64, 200)
    x, d = x.numpy(), d.numpy()
    pas = np.where(x == 0, 1, -1)
    assert (np.cumsum(pas, 1) == d).all()                            # la cible est la profondeur courante
    assert d.min() >= 0 and d.max() <= s5_tasks.DEPTH


def test_cle_valeur():
    seq, y = s5_tasks.kv_batch(np.random.default_rng(0), 64, 6)
    seq, y = seq.numpy(), y.numpy()
    for b in range(64):
        paires = dict(zip(seq[b, 0:12:2], seq[b, 1:12:2]))
        assert len(paires) == 6                                      # clés distinctes
        assert seq[b, 12] == s5_tasks.KV_Q
        assert paires[seq[b, 13]] == y[b]
