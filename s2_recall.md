# Rappel de listes (tokens aleatoires, seconde copie)

Perte en nats par token (plus bas = meilleur rappel) ; entre parentheses la precision top-1. Un tirage uniforme sur les 1700 tokens vaudrait 7.44 nats, mais ces modeles, entraines sur des histoires, jugent ces tokens aleatoires bien plus improbables que cela : un resultat proche de la premiere copie signifie « aucun rappel ».

Distraction d = 0 tokens

| modele | n = 8 | n = 16 | n = 32 | n = 64 |
|:--|--:|--:|--:|--:|
| T_rope | 11.85 (0 %) | 12.25 (0 %) | 12.41 (0 %) | 12.48 (0 %) |
| T_nope | 11.91 (0 %) | 12.34 (0 %) | 12.44 (0 %) | 12.48 (0 %) |
| V_lam0 | 11.70 (0 %) | 12.09 (0 %) | 12.05 (0 %) | 12.12 (0 %) |
| V1_lam0 | 11.85 (0 %) | 12.25 (0 %) | 12.19 (0 %) | 12.16 (0 %) |

Distraction d = 64 tokens

| modele | n = 8 | n = 16 | n = 32 | n = 64 |
|:--|--:|--:|--:|--:|
| T_rope | 12.48 (0 %) | 12.58 (0 %) | 12.56 (0 %) | 12.58 (0 %) |
| T_nope | 12.52 (0 %) | 12.59 (0 %) | 12.56 (0 %) | 12.55 (0 %) |
| V_lam0 | 12.09 (0 %) | 12.17 (0 %) | 12.17 (0 %) | 12.18 (0 %) |
| V1_lam0 | 12.10 (0 %) | 12.18 (0 %) | 12.21 (0 %) | 12.18 (0 %) |

Premiere copie (temoin : aucun rappel possible) : T_rope 12.53, T_nope 12.55, V_lam0 12.37, V1_lam0 12.34 nats.
