# Etape 2 : transformer sur tokens contre transformer sur la video continue

Bits par caractere (bpc) de validation, plus bas = mieux. Fenetres d'entrainement de 256 tokens ; au-dela, extrapolation (fenetres de 1024 tokens, par tranche de position).

| modele | params | bpc a 256 tokens | tokens 0-256 | tokens 256-512 | tokens 512-1024 | continuite | duree |
|:--|--:|--:|--:|--:|--:|--:|--:|
| T_rope | 6046464 | 0.706 | 0.625 | 0.775 | 1.164 | - | 14 min |
| V_lam0 | 5733972 | 0.732 | 0.645 | 0.686 | 0.846 | 0.43 | 57 min |
| V_lam1 | 5733972 | 1.800 | 1.728 | 1.716 | 1.677 | 0.03 | 58 min |
| T_nope | 6046464 | 0.736 | 0.662 | 0.892 | 1.746 | - | 13 min |
| V1_lam0 | 5704686 | 0.761 | 0.680 | 0.735 | 0.914 | 0.73 | 56 min |

Flux de 8 fenetres contigues (2048 tokens) : bpc par numero de fenetre. Le modele video transporte sa frame ; le transformer sur tokens n'a aucune memoire d'une fenetre a l'autre.

| modele | fenetre 1 | fenetre 2 | fenetre 4 | fenetre 8 |
|:--|--:|--:|--:|--:|
| T_rope | 0.752 | 0.745 | 0.723 | 0.705 |
| V_lam0 | 0.780 | 0.761 | 0.747 | 0.733 |
| V_lam1 | 1.838 | 1.797 | 1.755 | 1.807 |
| T_nope | 0.790 | 0.779 | 0.763 | 0.754 |
| V1_lam0 | 0.805 | 0.804 | 0.773 | 0.781 |

Reference (niveau caractere, bras 1-2, 300 000 histoires) : transformer 1,24 bpc ; GRU 1,14 bpc.
