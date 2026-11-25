# Etape 3 : suivi d'etat (precision par position, mediane [min-max] sur les graines)

Entrainement sur des suites de 24 elements ; tranches de position : 0-24, 24-48, 48-96, 96-256. Les tranches au-dela de 24 mesurent l'extrapolation en longueur.

## S2 (hasard : 50.0 %)

| modele | params | pos. 0-24 | pos. 24-48 | pos. 48-96 | pos. 96-256 | graines >= 95 % dans la longueur d'entrainement |
|:--|--:|--:|--:|--:|--:|--:|
| T_rope | 793856 | 99 [99-99] | 52 [49-56] | 50 [50-50] | 50 [50-50] | 3 / 3 |
| V_lam0 | 862644 | 100 [100-100] | 100 [100-100] | 100 [100-100] | 96 [71-100] | 3 / 3 |
| L_loop | 894406 | 100 [100-100] | 100 [75-100] | 100 [50-100] | 100 [50-100] | 3 / 3 |
| L_noloop | 489254 | 81 [81-83] | 50 [49-50] | 50 [50-50] | 50 [50-50] | 0 / 3 |

Par graine (precision en % dans la longueur d'entrainement | au-dela de 48) :

- T_rope : s0 : 99 | 50 ; s1 : 99 | 50 ; s2 : 99 | 50
- V_lam0 : s0 : 100 | 96 ; s1 : 100 | 71 ; s2 : 100 | 100
- L_loop : s0 : 100 | 100 ; s1 : 100 | 50 ; s2 : 100 | 100
- L_noloop : s0 : 83 | 50 ; s1 : 81 | 50 ; s2 : 81 | 50

## S3 (hasard : 16.7 %)

| modele | params | pos. 0-24 | pos. 24-48 | pos. 48-96 | pos. 96-256 | graines >= 95 % dans la longueur d'entrainement |
|:--|--:|--:|--:|--:|--:|--:|
| T_rope | 794880 | 76 [65-80] | 20 [18-20] | 17 [16-17] | 17 [17-17] | 0 / 3 |
| V_lam0 | 863412 | 63 [61-100] | 23 [22-100] | 17 [16-100] | 17 [17-99] | 1 / 3 |
| L_loop | 895690 | 100 [30-100] | 100 [16-100] | 100 [17-100] | 100 [17-100] | 2 / 3 |
| L_noloop | 490538 | 53 [47-55] | 17 [16-17] | 17 [16-17] | 17 [17-17] | 0 / 3 |

Par graine (precision en % dans la longueur d'entrainement | au-dela de 48) :

- T_rope : s0 : 76 | 17 ; s1 : 65 | 17 ; s2 : 80 | 17
- V_lam0 : s0 : 61 | 17 ; s1 : 63 | 17 ; s2 : 100 | 99
- L_loop : s0 : 30 | 17 ; s1 : 100 | 100 ; s2 : 100 | 100
- L_noloop : s0 : 53 | 17 ; s1 : 47 | 17 ; s2 : 55 | 17

