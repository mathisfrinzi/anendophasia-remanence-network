# Scripts

Tous se lancent depuis la racine : `python scripts/<nom>.py …`. Ils restent à plat dans ce dossier parce qu'ils s'importent entre eux et ajoutent `src/` au chemin. Commandes de chaque résultat : [docs/REPRODUIRE.md](../docs/REPRODUIRE.md). Le numéro « A » est celui de l'amendement de [PROTOCOLE.md](../docs/preenregistrements/PROTOCOLE.md).

## Expériences actuelles

| script | rôle |
|---|---|
| `ts_remanence.py` | langage sur CPU ou GPU : TinyStories, Wikitext-2, PG-19, Wikipédia ; modèles `R_*`, `T_*`, `GRU_*`, `Twin_*`, `H_*` (A36-A44) |
| `suite.py` | enchaîne les runs de `ts_remanence.py` par plan : `taille`, `memoire`, `hybride`, `sondes` (A40-A43) |
| `sondes_histoires.py` | sondes linéaires sur des histoires synthétiques (A43, non lancé) |
| `cpu_s4.py` | S4 (24 états) ; aussi les passes par token `L_loop1_pK` (A27, A46) |
| `cpu_alt.py` | entraînement alterné images / transformers sur S4 (A45) |
| `rnn_taches.py` | GRU et LSTM sur S3, S3M, liste circulaire (A30) ; fournit la classe `RNN` |
| `silverbox.py` | identification du système Silverbox (A28) |

## Expériences des étapes précédentes (reproductibles)

| script | rôle |
|---|---|
| `cpu_boucle.py`, `cpu_boucle_causal.py` | S3, contrôles à capacité égale, ablations ; bruit et sondes (A15, A16) |
| `cpu_memoire.py`, `cpu_double.py`, `cpu_s3m2.py` | S3M, double loop, deux valeurs (A17, A18, A26, A33) |
| `cpu_wikitext.py`, `prepare_corpus.py`, `prepare_corpus2.py`, `resume_lm.py` | six corpus sur CPU (A20, A22) |
| `cpu_langage.py` | histoires synthétiques (A19) |
| `reseau_exp.py`, `reseau_h5.py`, `reseau_resume.py`, `reseau_tache.py`, `reseau_verif.py` | liste circulaire et pointeurs (A23, A32) ; les deux derniers sont des bibliothèques |
| `cpu_reflexion.py` | permutation avec pauses (A21, invalidée) |
| `cpu_babi.py`, `cpu_listops.py`, `listops_data.py` | bAbI, ListOps (A24, A25) |
| `cpu_s5.py` | S5 et réseau d'images 2-3-1 (A31) |
| `cpu_ud.py` | étiquetage UD English-EWT (A35) |
| `analyses_cpu.py`, `dimension_frames.py`, `indice_continuite_textes.py`, `demo_phrase.py` | analyses de la partie 7 (à lancer depuis la racine) |
| `gen_notebook_ae.py` | génère les notebooks de `notebooks/historique/` |

## Figures

| script | sortie |
|---|---|
| `figures_bilan.py` | `figures/bilan_*.png` (README, RESULTATS.md) |
| `figures.py` | `figures/historique/*.svg` (rapports d'étape) |
| `schemas.py` | `figures/schemas/*.svg` |
| `exemple.py` | `figures/exemples/` (GIF des images pendant S3) |
| `courbe_a44.py`, `courbes_logs.py` | figure de l'amendement 44 ; courbes rassemblées depuis les journaux |

## Diagnostics ponctuels (gardés pour traçabilité)

`diag_graphe.py` (échec du réseau 2-3-1), `xpu_diag.py` (GPU Intel), `verif_rapide.py` et `bench_rapide.py` (équivalence et vitesse de la version rapide, A38).
