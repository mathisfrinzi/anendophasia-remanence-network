"""Toutes les figures et tous les schémas se refont depuis results/, sans erreur."""
import figures
import schemas


def test_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(figures, "OUT", tmp_path)
    for f in figures.FIGS:
        f()
    assert len(list(tmp_path.glob("*.svg"))) == len(figures.FIGS)


def test_schemas(tmp_path, monkeypatch):
    monkeypatch.setattr(schemas, "OUT", tmp_path)
    for f in schemas.SCHEMAS:
        f()
    assert len(list(tmp_path.glob("*.svg"))) == len(schemas.SCHEMAS)


def test_exemple_boucle(tmp_path):
    """L'exemple charge le modèle entraîné de la partie 4 et le fait tourner (suite courte)."""
    import exemple
    exemple.boucle(6, 0, tmp_path)
    assert (tmp_path / "boucle_s3.gif").exists() and (tmp_path / "boucle_s3.png").exists()
