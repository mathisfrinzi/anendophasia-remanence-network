# Etape 2 : transformer sur tokens contre transformer sur la video continue

Bits par caractere (bpc) de validation, plus bas = mieux. Fenetres d'entrainement de 256 tokens ; au-dela, extrapolation (fenetres de 1024 tokens, par tranche de position).

| modele | params | bpc a 256 tokens | tokens 0-256 | tokens 256-512 | tokens 512-1024 | continuite | duree |
|:--|--:|--:|--:|--:|--:|--:|--:|
| T_rope | 6046464 | 0.706 | 0.625 | 0.775 | 1.164 | - | 14 min |
| V_lam0 | 5733972 | 0.732 | 0.645 | 0.686 | 0.846 | 0.43 | 57 min |
| V_lam1 | 5733972 | 1.800 | 1.728 | 1.716 | 1.677 | 0.03 | 58 min |
| V_lam1_ext | 5733972 | 0.773 | 0.695 | 0.718 | 0.819 | 0.01 | 123 min |

Flux de 8 fenetres contigues (2048 tokens) : bpc par numero de fenetre. Le modele video transporte sa frame ; le transformer sur tokens n'a aucune memoire d'une fenetre a l'autre.

| modele | fenetre 1 | fenetre 2 | fenetre 4 | fenetre 8 |
|:--|--:|--:|--:|--:|
| T_rope | 0.752 | 0.745 | 0.723 | 0.705 |
| V_lam0 | 0.780 | 0.761 | 0.747 | 0.733 |
| V_lam1 | 1.838 | 1.797 | 1.755 | 1.807 |
| V_lam1_ext | 0.828 | 0.802 | 0.774 | 0.772 |

Les modeles *_ext sont la prolongation d'un modele de 8 000 pas jusqu'a 24000 pas (budget inegal avec les autres lignes ; la duree indiquee est celle de la prolongation seule).

Reference (niveau caractere, bras 1-2, 300 000 histoires) : transformer 1,24 bpc ; GRU 1,14 bpc.
