[← Résumé](../README.md) · [Résultats](RESULTATS.md) · [English version](REPRODUCE.en.md)

# Reproduire les résultats

Toutes les commandes se lancent depuis la racine du dépôt. Chaque run écrit un JSON dans `results/` ; les scripts CPU et `suite.py` sautent un run dont le fichier existe déjà. La graine se donne en argument (0, puis 1 et 2 pour 3 graines). Durées mesurées sur un CPU à 2 cœurs, sauf mention.

```bash
pip install -r requirements.txt
pytest -q                          # 35 tests, ~10 s : modèles, tâches, figures relues depuis results/
python scripts/figures_bilan.py    # figures du README et de RESULTATS.md
```

## Tâches à état (CPU)

| résultat | commande | durée | sortie |
|---|---|---|---|
| S3 et contrôles à capacité égale (A15) | `python scripts/cpu_boucle.py L_loop 0` (aussi `L_noloop`, `L_wide`, `L_i2`, ablations `A_*`) | ~11 min | `results/st_S3c_*` |
| S3M, retenir une valeur (A17) | `python scripts/cpu_memoire.py L_loop1 0` | ~11 min | `results/st_S3M_*` |
| S4, 24 états (A27b) | `python scripts/cpu_s4.py L_loop1 0 15000` (aussi `T_rope`, `L_dloop1`) | ~50 min | `results/s4_*_p15000.json` |
| liste circulaire (A23) | `python scripts/reseau_exp.py L_loop1_k 0` puis `python scripts/reseau_resume.py` | ~40 min | `results/reseau_*` |
| GRU / LSTM sur S3, S3M, listes (A30) | `python scripts/rnn_taches.py S3 GRU 0` (tâches `S3`, `S3M`, `PTR` ; modèles `GRU`, `LSTM`) | 0,5-3 min | `results/rnn_*` |
| Silverbox (A28) | `python scripts/silverbox.py L_loop1 0` (aussi `GRU`, `LSTM`, `T_rope`, `lineaire`) | ~31 min (GRU 2 min) | `results/silverbox_*` |
| passes par token (A46) | `python scripts/cpu_s4.py L_loop1_p2 0 4000` (`p1`, `p2`, `p4`) | 14-50 min | `results/s4_L_loop1_p*` |
| entraînement alterné (A45) | `python scripts/cpu_alt.py alt 0 15000` (`alt` ou `joint`) | ≤ 1 h (arrêt dès 99 %) | `results/alt_*` |

## Langage

Le script principal est [`scripts/ts_remanence.py`](../scripts/ts_remanence.py) ; il télécharge le corpus au premier lancement (Hugging Face, avec une copie GitHub de secours pour Wikitext-2).

| résultat | commande | matériel | durée |
|---|---|---|---|
| taille d'image contre GRU et T_rope (A40) | `python scripts/suite.py taille --device cuda` (ou `xpu`) | GPU | 1 à 10 min (T_rope, GRU) à 4 h 40 (rémanence 4×16×16) par modèle |
| T_rope à paramètres égaux (complément A40) | `python scripts/ts_remanence.py --dataset wikitext2 --window 128 --batch 32 --steps 2000 --eval-len 512 --tag _taille --models T_96x2 --device cuda` | GPU | ~1 min |
| courbe bpc / paramètres sur CPU (A44) | `python scripts/ts_remanence.py --dataset wikitext2 --window 64 --batch 16 --steps 1500 --eval-len 256 --tag _a44 --no-gen --device cpu --models T_64x2 T_96x2 T_128x2 T_192x2 R_4x8x8 R_4x16x16 R_8x16x16` | CPU | 1 à 3 min (T_rope) à 30 min (rémanence) |
| entraînement long (A44) | même commande avec `--steps 6000 --tag _a44long --models T_96x2 R_4x8x8` | CPU | ~36 min (rémanence) |
| balayage du learning rate (A44) | même commande avec `--lr 2e-3 --tag _lr2e-3` | CPU | idem |
| six corpus, fenêtres de 64 (A20, A22) | `python scripts/prepare_corpus.py` et `python scripts/prepare_corpus2.py`, puis `CORPUS=wikitext2 python scripts/cpu_wikitext.py L_loop1 0` (`CORPUS` : `wikitext2`, `shakespeare`, `ptb`, `gutenberg_contes`, `gutenberg_tout`, `code_python`) | CPU | ~25 min |

Sur Colab : ouvrir [`notebooks/suite_colab.ipynb`](../notebooks/suite_colab.ipynb), choisir le plan (`taille`, `memoire`, `hybride`, `sondes`) et lancer les cellules.

Noms de modèles acceptés par `ts_remanence.py` : `T_rope`, `T_<largeur>x<couches>` (par exemple `T_96x2`), `R_<canaux>x<hauteur>x<largeur>` (rémanence, par exemple `R_4x16x16`), `GRU_<taille>`, `Twin_<fenêtre>` (transformer à fenêtre glissante), `H_<fenêtre>` (hybride).

## Étapes préliminaires (GPU, notebooks)

Les étapes 1 à 4 (autoencodeur, vidéo, état, première boucle sur GPU) se relancent depuis les notebooks de [`notebooks/historique/`](../notebooks/historique/) ; leur détail est dans l'[ancienne page de reproduction](../archive/rapport/reproduire.md). Le protocole complet de chaque expérience est dans [PROTOCOLE.md](preenregistrements/PROTOCOLE.md).
