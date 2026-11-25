"""Propriétés des modèles qui doivent tenir quels que soient les poids.

- causalité : la prédiction au pas t ne dépend pas des tokens d'après ;
- état transporté : couper une séquence en deux et transporter l'état donne le même résultat ;
- la boucle sans historique : son état tient entièrement dans les deux images.
"""
import pytest
import torch

import s3_tasks
from autoencoder import TextImageAE
from s4_loop import LoopLM, patchify, unpatchify

VOCAB = 11
SPECS = [
    dict(name="T_rope", kind="T", pos="rope"),
    dict(name="T_nope", kind="T", pos="none"),
    dict(name="V_lam0", kind="V", scales=2),
    dict(name="V1_lam0", kind="V", scales=1),
    dict(name="D_blob_e1", kind="D", deposit="blob", evolve=1),
    dict(name="D_free_e0", kind="D", deposit="free", evolve=0),
    dict(name="L_loop", kind="L", loop=True),
    dict(name="L_noloop", kind="L", loop=False),
    dict(name="L_i2", kind="L", loop="i2"),
    dict(name="L_loop1", kind="L", loop=True, layers=1),
]


def build(spec, cfg):
    torch.manual_seed(0)
    return s3_tasks.build(spec, VOCAB, cfg).eval()


@pytest.mark.parametrize("spec", SPECS, ids=[s["name"] for s in SPECS])
def test_forme_et_causalite(spec, petit_cfg):
    m = build(spec, petit_cfg)
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, VOCAB, (3, 9), generator=g)
    with torch.no_grad():
        y = m(x)
        assert y.shape == (3, 9, VOCAB)
        assert torch.isfinite(y).all()
        x2 = x.clone()
        x2[:, 5:] = (x2[:, 5:] + 1) % VOCAB                  # on change le futur
        y2 = m(x2)
    assert torch.allclose(y[:, :5], y2[:, :5], atol=1e-5), "une prédiction dépend de tokens futurs"
    assert not torch.allclose(y[:, 5:], y2[:, 5:]), "les tokens changés n'ont aucun effet"


@pytest.mark.parametrize("spec", [s for s in SPECS if s["kind"] in ("V", "D", "L")], ids=lambda s: s["name"])
def test_etat_transporte(spec, petit_cfg):
    """Pour les modèles récurrents, traiter [a, b] d'un coup ou a puis b avec l'état transporté donne pareil...
    à condition que le lecteur ne voie pas les frames passées : vrai pour la boucle, pas pour V/D (transformer causal
    sur toute la fenêtre). Pour V/D on vérifie seulement que la frame finale est la même."""
    m = build(spec, petit_cfg)
    x = torch.randint(0, VOCAB, (2, 10), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        y_tout = m(x)
        s_tout = m.state_out
        y_a = m(x[:, :4])
        y_b = m(x[:, 4:], state=m.state_out)
        s_b = m.state_out
    flat = lambda s: torch.cat([t.flatten() for t in s]) if isinstance(s, tuple) else s.flatten()
    assert torch.allclose(flat(s_tout), flat(s_b), atol=1e-5)
    if spec["kind"] == "L":
        assert torch.allclose(y_tout, torch.cat([y_a, y_b], 1), atol=1e-5)


def test_boucle_sans_t2_quand_noloop(petit_cfg):
    assert build(dict(kind="L", loop=False), petit_cfg).t2 is None
    assert build(dict(kind="L", loop=True), petit_cfg).t2 is not None


def test_bruit_sur_t2_change_la_prediction(petit_cfg):
    """La perturbation causale de la partie 4 agit bien sur le retour T2 → image 1."""
    m = build(dict(kind="L", loop=True), petit_cfg)
    x = torch.randint(0, VOCAB, (2, 6), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        y0 = m(x)
        m.t2_noise = 1.0
        torch.manual_seed(0)
        y1 = m(x)
    assert torch.allclose(y0[:, 0], y1[:, 0], atol=1e-5)      # le premier token est lu avant tout retour
    assert not torch.allclose(y0[:, 1:], y1[:, 1:])


def test_patchs_aller_retour():
    x = torch.randn(2, 4, 8, 8)
    for p in (2, 4):
        t = patchify(x, p)
        assert t.shape == (2, (8 // p) ** 2, 4 * p * p)
        assert torch.equal(unpatchify(t, 4, 8, 8, p), x)


@pytest.mark.parametrize("kind", ["cumul", "image", "vector", "none"])
def test_autoencodeur(kind):
    import config
    cfg = dict(config.AE, d_model=32, n_layers=1, n_heads=2, spatial_ch=8)
    torch.manual_seed(0)
    m = TextImageAE(20, 8, kind, cfg).eval()
    x = torch.randint(0, 20, (3, 8))
    with torch.no_grad():
        y = m(x)
        z = m.encode(x)
    assert y.shape == (3, 8, 20)
    assert z.shape == (3, 192)
    assert torch.allclose(z.var(-1, unbiased=False), torch.ones(3), atol=1e-3)   # l'échelle ne stocke rien
    if kind == "cumul":
        _, fr = m(x, return_frames=True)
        assert fr.shape == (3, 9, 3, 8, 8)                                      # une frame par caractère + la frame de départ


def test_indice_de_continuite():
    """Une vidéo immobile a un indice nul ; des frames indépendantes, un indice d'ordre 1."""
    torch.manual_seed(0)
    fixe = torch.randn(16, 1, 3, 8, 8).expand(16, 10, 3, 8, 8)
    assert TextImageAE.continuity_index(fixe) < 1e-6
    bruit = torch.randn(16, 10, 3, 8, 8)
    assert 0.8 < TextImageAE.continuity_index(bruit) < 1.2
