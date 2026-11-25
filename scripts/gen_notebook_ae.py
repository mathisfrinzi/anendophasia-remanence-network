#!/usr/bin/env python3
"""Genere notebooks/ae_visuel.ipynb : UNE cellule Colab qui lance l'autoencodeur texte <-> image.

Depot prive, aucun token : la cellule deploie le depot depuis le zip televerse
(GitHub -> Code -> Download ZIP ; choisir la branche autoencodeur-visuel). Les resultats
reviennent par telechargement (zip results_ae.zip, toutes les heures et a la fin).
Reprise : televerser aussi results_ae.zip ; les mesures deja faites sont sautees."""
import json
import os

CELL = r'''# =============================================================================
# Autoencodeur texte <-> image | CELLULE UNIQUE : Execution -> Tout executer (GPU T4)
# Duree estimee : ~1 h 30 a 2 h 30 (4 tailles de morceau x 4 bras, 8 000 pas chacun ; le bras cumul est sequentiel, donc plus lent).
# Au premier lancement, un BOUTON de televersement apparait : choisir le zip du depot
# (GitHub -> Code -> Download ZIP, branche autoencodeur-visuel). En cas de reprise,
# choisir AUSSI results_ae.zip (selection multiple) : les mesures deja faites sont sautees.
# Toutes les heures et a la fin, results_ae.zip est TELECHARGE dans ton navigateur :
# c'est la sauvegarde ; la joindre a la discussion pour lire les resultats.
# =============================================================================
import os
import shutil
import sys
import time
import zipfile

import torch

print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "AUCUN (Execution -> Modifier le type d'execution -> GPU T4)")


def deploy():
    """Deploie le depot depuis le zip televerse ; fusionne results_ae.zip. Retourne la racine du depot."""
    cwd0 = os.getcwd()
    if os.path.exists(os.path.join("src", "ae_run.py")):
        return cwd0
    zips = sorted(f for f in os.listdir(cwd0) if f.endswith(".zip"))
    if not zips:
        try:
            from google.colab import files as colab_files
            print("Clique sur le bouton ci-dessous et televerse le zip du depot (plus results_ae.zip pour reprendre).")
            colab_files.upload()
            zips = sorted(f for f in os.listdir(cwd0) if f.endswith(".zip"))
        except ImportError:
            pass
    assert zips, "Aucun zip present : televerse le zip du depot (GitHub -> Code -> Download ZIP)."
    repo, res = None, []
    os.makedirs("_unzip", exist_ok=True)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            if any(n.endswith("src/ae_run.py") for n in names):
                dest = os.path.join("_unzip", z[:-4])
                zf.extractall(dest)
                for root, dirs, fs in os.walk(dest):
                    if os.path.exists(os.path.join(root, "src", "ae_run.py")):
                        repo = os.path.abspath(root)
            elif any(n.startswith("results/ae_") for n in names):
                res.append(z)
    assert repo, "Le zip du depot (contenant src/ae_run.py) est introuvable : verifie que c'est la branche autoencodeur-visuel."
    for z in res:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(repo)
        print("Resultats precedents fusionnes :", z)
    return repo


ROOT = deploy()
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
print("Depot :", ROOT)

_last = [0.0]


def save_and_download(force=False):
    """Reconstruit results_ae.zip et le telecharge (toutes les heures, ou a la fin)."""
    if not force and time.time() - _last[0] < 3600:
        return
    _last[0] = time.time()
    base = os.path.join("/content" if os.path.isdir("/content") else ROOT, "results_ae")
    shutil.make_archive(base, "zip", ROOT, "results")
    try:
        from google.colab import files as colab_files
        colab_files.download(base + ".zip")
    except ImportError:
        print("Sauvegarde :", base + ".zip")


import ae_run

_last[0] = time.time()
__RUN__
save_and_download(force=True)

# Images : latents (bras image et cumul) et film des frames (bras cumul)
import glob
from IPython.display import Image, display
for p in sorted(glob.glob("results/ae_cumul_L*_frames.png") + glob.glob("results/ae_*_L*_latents.png")):
    print(p)
    display(Image(p))
'''


def build(cell, name):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": [
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": cell.splitlines(keepends=True)}
        ],
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks", "historique", name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("ecrit :", os.path.normpath(out))


def main():
    build(CELL.replace("__RUN__", "ae_run.run_all(ROOT, on_done=save_and_download)"), "ae_visuel.ipynb")
    comp = (CELL.replace("__RUN__", "ae_run.run_complement(ROOT, on_done=save_and_download)")
            .replace("Autoencodeur texte <-> image | CELLULE UNIQUE", "Autoencodeur : MESURES SUPPLEMENTAIRES | CELLULE UNIQUE")
            .replace("# Duree estimee : ~1 h 30 a 2 h 30 (4 tailles de morceau x 4 bras, 8 000 pas chacun ; le bras cumul est sequentiel, donc plus lent).",
                     "# Duree estimee : ~15 min (3 modeles a 16 caracteres, sauvegardes puis analyses : structure cachee de l'image)."))
    build(comp, "ae_visuel_complement.ipynb")
    s2 = (CELL.replace("__RUN__", "s2_run.run_all(ROOT, on_done=save_and_download)")
          .replace("import ae_run", "import s2_run")
          .replace("src/ae_run.py", "src/s2_run.py").replace("results/ae_", "results/s2_")
          .replace("results_ae", "results_s2")
          .replace("Autoencodeur texte <-> image | CELLULE UNIQUE", "Etape 2 : transformer sur tokens, texte vs video | CELLULE UNIQUE")
          .replace("# Duree estimee : ~1 h 30 a 2 h 30 (4 tailles de morceau x 4 bras, 8 000 pas chacun ; le bras cumul est sequentiel, donc plus lent).",
                   "# Duree estimee : ~2 h 30 sur T4 (5 modeles, 8 000 pas ; les modeles video sont plus lents)."))
    tail = s2[s2.index("# Images : latents"):]
    s2 = s2.replace(tail, "# Resume\nprint(open(os.path.join(ROOT, 'results', 's2_summary.md'), encoding='utf-8').read())\n")
    build(s2, "etape2.ipynb")
    ext = (s2.replace("s2_run.run_all(ROOT, on_done=save_and_download)",
                      "s2_run.run_all(ROOT, cfg=s2_run.config.S2_EXT, models=s2_run.config.S2_EXT['run'], on_done=save_and_download)")
           .replace("results_s2", "results_s2x").replace("'s2_summary.md'", "'s2x_summary.md'")
           .replace("Etape 2 : transformer sur tokens, texte vs video", "Etape 2 : prolongation de V_lam1 (8 000 -> 24 000 pas)")
           .replace("# Duree estimee : ~2 h 30 sur T4 (5 modeles, 8 000 pas ; les modeles video sont plus lents).",
                    "# Duree estimee : ~2 h sur T4 (16 000 pas supplementaires). Le zip du depot (branche main) contient deja le modele de depart."))
    build(ext, "etape2_prolongation.ipynb")
    dyn = (s2.replace("s2_run.run_all(ROOT, on_done=save_and_download)",
                      "s2_run.run_all(ROOT, cfg=s2_run.config.S2_DYN, models=s2_run.config.S2_DYN['run'], on_done=save_and_download)")
           .replace("results_s2", "results_s2d").replace("'s2_summary.md'", "'s2d_summary.md'")
           .replace("Etape 2 : transformer sur tokens, texte vs video", "Etape 2 : image qui evolue (depot par le token + evolution)")
           .replace("# Duree estimee : ~2 h 30 sur T4 (5 modeles, 8 000 pas ; les modeles video sont plus lents).",
                    "# Duree estimee : ~3 h 45 sur T4 (4 modeles dans l'ordre D_free_e1, D_blob_e1, D_blob_e0, D_blob_e2 ; le zip est telecharge a chaque heure, tu peux interrompre apres le 3e)."))
    build(dyn, "etape2_dynamique.ipynb")
    st = (CELL.replace("__RUN__", "s3_tasks.run_all(ROOT, on_done=save_and_download)")
          .replace("import ae_run", "import s3_tasks")
          .replace("src/ae_run.py", "src/s3_tasks.py").replace("results/ae_", "results/st_")
          .replace("results_ae", "results_st")
          .replace("Autoencodeur texte <-> image | CELLULE UNIQUE", "Etape 3 : suivi d'etat (permutations) | CELLULE UNIQUE")
          .replace("# Duree estimee : ~1 h 30 a 2 h 30 (4 tailles de morceau x 4 bras, 8 000 pas chacun ; le bras cumul est sequentiel, donc plus lent).",
                   "# Duree estimee : ~2 h sur T4 (3 groupes x 3 graines x 5 modeles ; les resultats deja faits sont sautes a la reprise)."))
    tail = st[st.index("# Images : latents"):]
    st = st.replace(tail, "# Resume\nprint(open(os.path.join(ROOT, 'results', 'st_summary.md'), encoding='utf-8').read())\n")
    build(st, "etape3_etat.ipynb")
    stl = (st.replace("s3_tasks.run_all(ROOT, on_done=save_and_download)", "s3_tasks.run_all(ROOT, st=s3_tasks.ST_LONG, tasks=s3_tasks.TASKS_LONG, on_done=save_and_download)")
           .replace("results_st", "results_stl").replace("'st_summary.md'", "'stl_summary.md'")
           .replace("Etape 3 : suivi d'etat (permutations)", "Etape 3 bis : S3, entrainement long, 6 graines")
           .replace("# Duree estimee : ~2 h sur T4 (3 groupes x 3 graines x 5 modeles ; les resultats deja faits sont sautes a la reprise).",
                    "# Duree estimee : ~2 h sur T4 (6 graines x 3 modeles x 12 000 pas ; les resultats deja faits sont sautes a la reprise)."))
    build(stl, "etape3_long.ipynb")
    s4 = (st.replace("s3_tasks.run_all(ROOT, on_done=save_and_download)", "s3_tasks.run_all(ROOT, st=s3_tasks.ST_LOOP, tasks=s3_tasks.TASKS_LOOP, on_done=save_and_download)")
          .replace("results_st", "results_st4").replace("'st_summary.md'", "'st4_summary.md'")
          .replace("Etape 3 : suivi d'etat (permutations)", "Etape 4 : deux images couplees (boucle T1/T2)")
          .replace("# Duree estimee : ~2 h sur T4 (3 groupes x 3 graines x 5 modeles ; les resultats deja faits sont sautes a la reprise).",
                   "# Duree estimee : ~1 h 30 a 2 h sur T4 (S2 et S3, 3 graines, 2 modeles a boucle ; T_rope et V_lam0 sont deja dans le zip du depot et sautes)."))
    build(s4, "etape4_boucle.ipynb")
    s5 = (st.replace("s3_tasks.run_all(ROOT, on_done=save_and_download)",
                     "s5_tasks.run_all(ROOT, on_done=save_and_download)\n"
                     "s3_tasks.run_all(ROOT, st=dict(s3_tasks.ST_LOOP, summary_name='st5s5_summary.md'), tasks={'S5': s3_tasks.TASKS['S5']}, on_done=save_and_download)")
          .replace("import s3_tasks", "import s3_tasks, s5_tasks")
          .replace("results_st", "results_st5").replace("'st_summary.md'", "'st5_summary.md'")
          .replace("Etape 3 : suivi d'etat (permutations)", "Etape 5 : Dyck, cle-valeur, S5 (boucle / sans boucle)")
          .replace("# Duree estimee : ~2 h sur T4 (3 groupes x 3 graines x 5 modeles ; les resultats deja faits sont sautes a la reprise).",
                   "# Duree estimee (mesures de l'etape 4) : Dyck ~1 h + cle-valeur ~1 h 30, puis S5 ~3 h (tu peux t'arreter apres cle-valeur : le zip est telecharge a chaque heure)."))
    build(s5, "etape5_taches.ipynb")


if __name__ == "__main__":
    main()
