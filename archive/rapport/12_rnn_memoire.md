[Sommaire](README.md) · [← Partie 11](11_silverbox.md)

# Partie 12 — Les RNN réussissent aussi les tâches de mémoire de la rémanence

> **Ce qu'il faut retenir**
>
> - Un **GRU** et un **LSTM** (1 couche, 256 unités, ≈ 250 et 330 k paramètres) résolvent **S3**, **S3 + mémoire longue** et la **liste circulaire jusqu'à 3 sauts** avec 100 % de précision, sur 3 graines sur 3, en 30 à 200 s (la rémanence : 675 s sur S3, 2 400 s sur la liste).
> - Ils échouent comme la rémanence à **4 sauts** (≈ 5 %, hasard ≈ 17 %) : la limite à h = 3 n'est donc pas propre à la rémanence, c'est celle de l'entraînement jusqu'à h = 3.
> - **Lecture honnête** : la rémanence n'apporte pas une capacité qu'un RNN n'aurait pas. Le retour T2 donne à un transformer un état récurrent, que le GRU et le LSTM ont d'office. L'intérêt de la rémanence est de garder l'architecture transformer (attention sur les mots, lecture parallèle du contexte) tout en ayant cet état, pas de battre les RNN sur ces tâches.

## 12.1 Montage (amendement 30)

`scripts/rnn_taches.py` : mêmes données, pas, optimiseur, curriculum et évaluations que la boucle. S3 : entraînement sur longueur 12, évaluation jusqu'à 96 (2 000 pas). S3M : 3 000 pas, rappel à n = 12…96. Liste circulaire : n = 6, h ≤ 3, k pauses aléatoires ≤ 8, 8 000 pas, évaluation h = 1…5. Graines 0, 1, 2.

## 12.2 Résultats

| Tâche | GRU | LSTM | Rémanence (réf.) |
|---|---|---|---|
| S3, précision 48–96 | 0,9998 / 0,996 / 1,0 | 1,0 / 1,0 / 1,0 | 1,0 |
| S3M, rappel et S3 à n = 96 | 1,0 sur 3 graines | 1,0 sur 3 graines | 0,98 (rappel), S3 1,0 |
| Liste circulaire h = 1, 2, 3 (k = 8) | 1,0 sur 3 graines | 1,0 sur 3 graines | 1,0 |
| Liste circulaire h = 4 / 5 | 0,06–0,12 / 0,01–0,03 | 0,02–0,09 / 0,01–0,03 | 0,12 / 0,02 |

Critères pré-enregistrés : S3 atteint (pronostic 80 %) ; S3M **atteint** (pronostic 55 %, sous-estimé) ; liste circulaire h = 3 ≥ 0,9 **atteint** (pronostic 30 %, nettement sous-estimé) ; h = 4 > 0,5 non atteint (pronostic 3 %, correct). Les pauses n'aident pas non plus ici (précision identique pour tous les k).

## 12.3 Ce que ça change dans le récit

Les conclusions sur la rémanence (partie 6, 8, 9) restent valides **par rapport aux témoins transformers** : sans retour T2, un transformer d'une couche ne suit pas l'état ; avec, il le suit. Mais ce n'est plus un argument contre les RNN. Le constat est : la rémanence rend à un transformer la mémoire d'état d'un RNN, avec un coût de calcul bien plus élevé (14 à 20 fois le GRU sur Silverbox, 5 à 20 fois sur S3 et la liste). Le test qui reste ouvert est celui où l'attention aide en plus de l'état, par exemple le langage sur de longs contextes, où la rémanence a été à égalité avec T_rope sans RNN pour comparer.

Code : `scripts/rnn_taches.py`. Résultats : `results/rnn_*`. Protocole : amendement 30.

## 12.4 S5 avec un réseau d'images en couches 2-3-1 : résultat nul (amendement 31)

Le réseau (6 images en couches de 2, 3 et 1, un transformer par transition entre couches voisines, 18 transformers, ≈ 288 k paramètres, `src/s7_graphe.py`) tourne aussi vite que la rémanence grâce au regroupement des arêtes par paire de couches. Sur **S5** (120 états, hasard 0,8 %, 8 000 pas, 1 graine), **aucun modèle n'apprend** : réseau en couches, sans retour, rémanence, GRU et T_rope restent à ≈ 9–13 % pour n ≤ 12 (seule la première position, triviale, est apprise) et au hasard au-delà ; la perte reste à 4,3–4,4 (ln 120 = 4,79). Le GRU lui-même n'y arrive pas. Ce budget est donc trop faible pour juger : S4 n'a décroché qu'après 5 000–6 000 pas sur 15 000. Pour que ce test départage quelque chose, il faudrait un budget plus long (≥ 30 000 pas, probablement sur GPU) ou un curriculum plus progressif.

## 12.5 Le réseau 2-3-1 sur S4 et la liste circulaire jusqu'à h = 5 (amendement 32)

- **S4** (15 000 pas, 1 graine) : le réseau 2-3-1 plafonne à 39 % (n = 12) et 32 % (n = 24), comme T_rope (40 %) et le double loop (41 %) ; la rémanence à 2 images fait 100 %.
- **Liste circulaire, entraînée jusqu'à h = 5** : le **GRU** fait 100 % de h = 1 à 5. La **rémanence**, graine 0, monte à 0,99 / 0,65 / 1,00 / 0,65 / 0,96 ; mais **les graines 1 et 2 ne suivent pas** (hasard, et 0,47 / 0,25 / 0,25 / 0,26 / 0,46) : l'entraînement jusqu'à h = 5 n'est pas fiable pour la rémanence, contrairement à h ≤ 3 (3 graines sur 3 en partie 9). Le résultat de la graine 0 ne suffit donc pas à dire que la limite à 3 sauts « venait de l'entraînement » (correction faite après l'amendement 33b). Le **réseau 2-3-1** reste au hasard dès h = 1 : il n'a pas appris la tâche.
- Avec n = 6, h = 6 rend le point de départ et h = 7, 8 reviennent à h = 1, 2 : aucun modèle n'exploite cette périodicité (≤ 0,27).
- **Lecture** : à ce budget et à une graine, l'architecture massive n'apporte rien et est plus difficile à optimiser que la boucle. Je ne peux pas dire si c'est un défaut de conception (initialisation, gating, nombre d'arêtes) ou de budget.

## 12.6 Pourquoi le réseau 2-3-1 échoue : échelle de croissance (amendement 33b)

Liste circulaire h ≤ 3, 6 000 pas, images 4×4×4 et d = 32 pour tous, graine 0, précision à h = 1 / 2 / 3 : **1-1** (la rémanence à 2 images, petite) 0,51 / 0,26 / 0,21 ; 2-1 : 0,66 / 0,28 / 0,27 ; 1-1-1, 2-2-1, 2-3-1 et 2-3-1 avec sortie des transformers initialisée à zéro : hasard (0,16–0,18). **Dès le premier échelon, la rémanence à 2 images en petite taille n'apprend pas** : l'échec de `G231` ne peut donc pas être attribué aux arêtes en plus, la réduction de taille suffit à casser la tâche, et l'initialisation à zéro n'y change rien. Pour comparer des arêtes, il faut garder la taille de la rémanence (images 4×8×8, d = 64), ce qui coûte 5 à 8 fois plus de calcul par arête : à faire sur GPU.

## 12.7 L'échelle de croissance à la taille de la rémanence (amendement 34)

Images 4×8×8, d = 64, liste circulaire h ≤ 3, 6 000 pas, 1 graine, précision à h = 1 / 2 / 3 : **1-1** (la rémanence seule) 0,98 / 0,50 / 0,36 ; **2-1** (deux entrées) 0,98 / 0,54 / 0,62 ; **1-1-1** (une couche intermédiaire) au hasard (0,16–0,18). Le seuil pré-enregistré (≥ 0,9 jusqu'à h = 3) n'est atteint par aucun, **y compris la rémanence seule** : 6 000 pas sans pauses ne suffisent pas, donc l'échelle n'est pas interprétable de façon stricte. À titre descriptif : ajouter une seconde entrée ne gêne pas l'apprentissage, ajouter une couche intermédiaire l'empêche de démarrer. Si ça se confirme sur d'autres graines et avec un budget plus long (GPU), l'obstacle serait la profondeur par token, pas le nombre d'images ni d'arêtes.
