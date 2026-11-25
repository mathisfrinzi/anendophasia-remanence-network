# Etape 2 : transformer sur tokens contre transformer sur la video continue

Bits par caractere (bpc) de validation, plus bas = mieux. Fenetres d'entrainement de 256 tokens ; au-dela, extrapolation (fenetres de 1024 tokens, par tranche de position).

| modele | params | bpc a 256 tokens | tokens 0-256 | tokens 256-512 | tokens 512-1024 | continuite | duree |
|:--|--:|--:|--:|--:|--:|--:|--:|
| T_rope | 6046464 | 0.706 | 0.625 | 0.775 | 1.164 | - | 14 min |
| V_lam0 | 5733972 | 0.732 | 0.645 | 0.686 | 0.846 | 0.43 | 57 min |
| D_free_e1 | 5727180 | 0.722 | 0.647 | 0.735 | 1.164 | 0.45 | 51 min |
| D_blob_e1 | 5497420 | 0.725 | 0.656 | 0.751 | 1.327 | 0.50 | 51 min |

Flux de 8 fenetres contigues (2048 tokens) : bpc par numero de fenetre. Le modele video transporte sa frame ; le transformer sur tokens n'a aucune memoire d'une fenetre a l'autre.

| modele | fenetre 1 | fenetre 2 | fenetre 4 | fenetre 8 |
|:--|--:|--:|--:|--:|
| T_rope | 0.752 | 0.745 | 0.723 | 0.705 |
| V_lam0 | 0.780 | 0.761 | 0.747 | 0.733 |
| D_free_e1 | 0.772 | 0.762 | 0.742 | 0.731 |
| D_blob_e1 | 0.776 | 0.761 | 0.742 | 0.734 |

Reference (niveau caractere, bras 1-2, 300 000 histoires) : transformer 1,24 bpc ; GRU 1,14 bpc.
