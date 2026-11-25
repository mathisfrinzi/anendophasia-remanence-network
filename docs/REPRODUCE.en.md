[Version française](REPRODUIRE.md)

[← Summary](../README.en.md) · [Results](RESULTS.en.md)

# Reproducing the results

All commands are run from the repository root. Each run writes a JSON file to `results/`; the CPU scripts and `suite.py` skip a run whose file already exists. The seed is given as an argument (0, then 1 and 2 for 3 seeds). Durations were measured on a 2-core CPU unless stated otherwise.

```bash
pip install -r requirements.txt
pytest -q                          # 35 tests, ~10 s: models, tasks, figures re-read from results/
python scripts/figures_bilan.py    # figures of the README and of RESULTS.en.md
```

## State tasks (CPU)

| result | command | duration | output |
|---|---|---|---|
| S3 and equal-capacity controls (A15) | `python scripts/cpu_boucle.py L_loop 0` (also `L_noloop`, `L_wide`, `L_i2`, ablations `A_*`) | ~11 min | `results/st_S3c_*` |
| S3M, retaining a value (A17) | `python scripts/cpu_memoire.py L_loop1 0` | ~11 min | `results/st_S3M_*` |
| S4, 24 states (A27b) | `python scripts/cpu_s4.py L_loop1 0 15000` (also `T_rope`, `L_dloop1`) | ~50 min | `results/s4_*_p15000.json` |
| circular list (A23) | `python scripts/reseau_exp.py L_loop1_k 0` then `python scripts/reseau_resume.py` | ~40 min | `results/reseau_*` |
| GRU / LSTM on S3, S3M, lists (A30) | `python scripts/rnn_taches.py S3 GRU 0` (tasks `S3`, `S3M`, `PTR`; models `GRU`, `LSTM`) | 0.5-3 min | `results/rnn_*` |
| Silverbox (A28) | `python scripts/silverbox.py L_loop1 0` (also `GRU`, `LSTM`, `T_rope`, `lineaire`) | ~31 min (GRU 2 min) | `results/silverbox_*` |
| passes per token (A46) | `python scripts/cpu_s4.py L_loop1_p2 0 4000` (`p1`, `p2`, `p4`) | 14-50 min | `results/s4_L_loop1_p*` |
| alternating training (A45) | `python scripts/cpu_alt.py alt 0 15000` (`alt` or `joint`) | ≤ 1 h (stops at 99%) | `results/alt_*` |

## Language

The main script is [`scripts/ts_remanence.py`](../scripts/ts_remanence.py); it downloads the corpus on first launch (Hugging Face, with a fallback GitHub copy for Wikitext-2).

| result | command | hardware | duration |
|---|---|---|---|
| image size versus GRU and T_rope (A40) | `python scripts/suite.py taille --device cuda` (or `xpu`) | GPU | 1 to 10 min (T_rope, GRU) to 4 h 40 (remanence 4×16×16) per model |
| T_rope at equal parameters (A40 supplement) | `python scripts/ts_remanence.py --dataset wikitext2 --window 128 --batch 32 --steps 2000 --eval-len 512 --tag _taille --models T_96x2 --device cuda` | GPU | ~1 min |
| bpc / parameters curve on CPU (A44) | `python scripts/ts_remanence.py --dataset wikitext2 --window 64 --batch 16 --steps 1500 --eval-len 256 --tag _a44 --no-gen --device cpu --models T_64x2 T_96x2 T_128x2 T_192x2 R_4x8x8 R_4x16x16 R_8x16x16` | CPU | 1 to 3 min (T_rope) to 30 min (remanence) |
| long training (A44) | same command with `--steps 6000 --tag _a44long --models T_96x2 R_4x8x8` | CPU | ~36 min (remanence) |
| learning-rate sweep (A44) | same command with `--lr 2e-3 --tag _lr2e-3` | CPU | same |
| six corpora, windows of 64 (A20, A22) | `python scripts/prepare_corpus.py` and `python scripts/prepare_corpus2.py`, then `CORPUS=wikitext2 python scripts/cpu_wikitext.py L_loop1 0` (`CORPUS`: `wikitext2`, `shakespeare`, `ptb`, `gutenberg_contes`, `gutenberg_tout`, `code_python`) | CPU | ~25 min |

On Colab: open [`notebooks/suite_colab.ipynb`](../notebooks/suite_colab.ipynb), choose the plan (`taille`, `memoire`, `hybride`, `sondes`) and run the cells.

Model names accepted by `ts_remanence.py`: `T_rope`, `T_<width>x<layers>` (for example `T_96x2`), `R_<channels>x<height>x<width>` (remanence, for example `R_4x16x16`), `GRU_<size>`, `Twin_<window>` (sliding-window transformer), `H_<window>` (hybrid).

## Preliminary steps (GPU, notebooks)

Steps 1 to 4 (autoencoder, video, state, first loop on GPU) can be rerun from the notebooks in [`notebooks/historique/`](../notebooks/historique/); their details are in the [old reproduction page](../archive/rapport/reproduire.md) (in French). The full protocol of each experiment is in [PROTOCOLE.md](preenregistrements/PROTOCOLE.md) (in French).
