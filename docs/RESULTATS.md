[English version](RESULTS.en.md)

# Résultats détaillés

Tous les chiffres viennent de `results/*.json`. « A12 » renvoie à l'amendement 12 de [PROTOCOLE.md](preenregistrements/PROTOCOLE.md), où se trouvent le montage complet, les critères et pronostics écrits avant la mesure, et les résultats bruts. Précision en %, bpc = bits par caractère (plus bas = mieux), « graine » = initialisation aléatoire. Le hasard est indiqué entre parenthèses.

Le modèle. Deux images de 4×8×8 (256 nombres chacune) forment toute la mémoire ; il n'y a ni attention sur le passé ni position. Pour chaque token : (1) le token se dépose sur l'image 1 ; (2) T1, un petit transformer sur les 16 patchs de l'image 1, écrit dans l'image 2 à travers une porte d'oubli apprise ; (3) une tête lit l'image 2 et prédit ; (4) T2 lit l'image 2 et réécrit l'image 1 (même porte). Code : [`src/s4_loop.py`](../src/s4_loop.py) (`LoopLM`), version rapide équivalente [`src/s8_rapide.py`](../src/s8_rapide.py).

## Sommaire

1. [État : la rémanence suit-elle un état ?](#1-état--la-rémanence-suit-elle-un-état-)
2. [Mémoire : retient-elle une valeur ?](#2-mémoire--retient-elle-une-valeur-)
3. [Contre les réseaux récurrents](#3-contre-les-réseaux-récurrents)
4. [Langage](#4-langage)
5. [Taille d'image et paramètres égaux](#5-taille-dimage-et-paramètres-égaux)
6. [Variantes de l'architecture](#6-variantes-de-larchitecture)
7. [Vitesse d'apprentissage, graines, coût](#7-vitesse-dapprentissage-graines-coût)
8. [Étapes préliminaires : texte → image, vidéo](#8-étapes-préliminaires--texte--image-vidéo)
9. [Corrections faites en cours de route](#9-corrections-faites-en-cours-de-route)
10. [Ce qui reste ouvert](#10-ce-qui-reste-ouvert)

---

## 1. État : la rémanence suit-elle un état ?

Verdict : oui, solide sur S3 (5 graines sur 5, contrôles à capacité égale).

S3 : on lit une suite de permutations de 3 éléments et on doit donner leur composition à chaque pas (6 états, hasard 16,7 %). Entraînement sur des suites de ≤ 12, test jusqu'à 96. CPU, 2 000 pas, d = 64 (A15).

| modèle | paramètres | 0-12 | 12-24 | 24-48 | 48-96 |
|---|---|---|---|---|---|
| **rémanence** | 252 k | 100 ×5 | 100 ×5 | 100 ×5 | 100 ×5 |
| témoin : T2 écrit dans l'image 2 (pas de retour) | 252 k | 64-79 | 32-35 | 29-33 | 27-32 |
| témoin : T1 élargi, sans T2 | 248 k | 59-66 | 16-20 | 16-17 | 16-17 |
| sans T2 | 148 k | 58-66 | 17-18 | 16-17 | 16-17 |

- C'est le retour T2 → image 1 qui fait l'état : à paramètres égaux, les deux témoins restent au hasard hors entraînement.
- Ablations (A16, 1 graine) : sans porte d'oubli → 17-22 % ; lecture sur l'image 1 au lieu de l'image 2 → 17 % ; une seule couche par transformer → toujours 100 %.
- Sur GPU (entraînement sur 24, test jusqu'à 256, A12) : 2 graines sur 3 à 100 % jusqu'à 256 ; T_rope 17 % au-delà de 48.
- S4 (24 états, hasard 4,2 %, 15 000 pas, A27b) : précision à n = 24 / 192 : rémanence 100 / 100, 70 / 52, 100 / 100 (3 graines) ; T_rope 40 / 8,5 (1 graine). À 4 000 pas, aucun modèle ne décolle.
- S5 (120 états, A10, A31) : personne n'apprend en 8 000 pas, GRU compris (≈ 9-13 % dans la longueur d'entraînement, hasard au-delà).

## 2. Mémoire : retient-elle une valeur ?

Verdict : oui, solide (3 graines sur 3).

S3M : une valeur parmi 8 est donnée au début, puis n éléments de S3, puis on demande la valeur (hasard 12,5 %). Entraînement n ≤ 24, 3 000 pas (A17).

| rappel de la valeur à n = | 48 | 96 |
|---|---|---|
| **rémanence** | 100 ×3 | 98,0 / 100 / 97,7 |
| sans T2 | 80,5 / 91,0 / 96,5 | 6,2 / 55,5 / 42,6 |
| T_rope | 14,8 / 9,0 / 74,6 | 1,2 / 0,0 / 22,3 |
| GRU, LSTM | 100 ×3 | 100 ×3 |

- Avec des images 8 fois plus petites (2×4×4 = 32 nombres), la rémanence garde 84 à 100 % à n = 96 (A18).
- Sondes linéaires : la valeur est lisible dans les deux images à 94-100 % pendant tout le délai, répartie dans tous les canaux.
- Deux valeurs à retenir (A33) : 2 graines sur 3 réussissent.

## 3. Contre les réseaux récurrents

Verdict : la rémanence ne fait pas mieux qu'un GRU ou un LSTM (A30, A28, 3 graines).

| tâche | GRU | LSTM | rémanence |
|---|---|---|---|
| S3, précision 48-96 | 99,6-100 (≈ 30 s) | 100 (≈ 31 s) | 100 (≈ 675 s) |
| S3M, rappel à n = 96 | 100 (≈ 55 s) | 100 | 97,7-100 (≈ 670 s) |
| liste circulaire, 3 sauts | 100 (≈ 190 s) | 100 | 100 (≈ 2 400 s) |
| liste circulaire, 4 et 5 sauts (jamais vus) | 1-12 | 1-9 | 0-13 |
| Silverbox, erreur de test (mV) | **9,56** ± 0,02 | 9,60 ± 0,03 | 10,82 ± 0,56 |

- Silverbox (A28) : identification d'un circuit électronique non linéaire. T_rope 15,29 mV, régression linéaire 14,03, rémanence sans T2 41,96. Sans T2, l'erreur est presque 4 fois plus grande : le retour compte aussi ici. Aucun réseau ne fait mieux que le linéaire dans la plage d'entraînement.
- Liste circulaire (A23) : suivre h pointeurs dans une liste de 6. La rémanence réussit 3 sauts sur 3 graines ; T_rope plafonne à 21-26 % à 3 sauts. Les « pauses de réflexion » avant la réponse n'aident pas. Entraînée jusqu'à 5 sauts (A32), la rémanence n'y arrive que pour 1 graine sur 3 ; le GRU y arrive (1 graine testée).

## 4. Langage

Verdict : la rémanence est au niveau d'un transformer de même taille dans la longueur d'entraînement, plus stable au-delà, derrière un GRU. Une graine par configuration.

Six corpus sur CPU (A20, A22 ; fenêtres de 64, 2 000 pas, ≈ 1 M de paramètres ; bpc 0-64 / 128-256) :

| corpus | T_rope | rémanence |
|---|---|---|
| Wikitext-2 | **2,086** / 2,096 | 2,139 / 2,102 |
| code Python | **1,739** / 1,844 | 1,795 / **1,791** |
| Penn Treebank | 1,622 / 1,635 | **1,604 / 1,578** |
| contes Gutenberg | **2,051** / 2,199 | 2,071 / **2,016** |
| Gutenberg complet | **2,190** / 2,494 | 2,229 / **2,278** |
| Shakespeare | **2,338** / 2,603 | 2,425 / **2,414** |

Dans la fenêtre d'entraînement, la rémanence est à −1 / +4 % de T_rope ; au-delà, T_rope se dégrade de 6 à 14 % sur 4 corpus et la rémanence le dépasse. Le contrôle qui entraînerait T_rope sur des fenêtres plus longues (A37) n'a pas pu être interprété (budgets trop courts).

TinyStories et Wikitext-2 sur GPU (A40 ; 2 000 pas, fenêtres de 128 ; bpc 0-128 / 128-512) :

| modèle | paramètres | TinyStories | Wikitext-2 | durée (Wikitext-2) |
|---|---|---|---|---|
| T_rope | 0,62 M | 1,296 / 1,390 | 2,100 / 2,185 | 82 s |
| T_rope (largeur 96) | 1,01 M | – | **1,997** / 2,060 | 75 s |
| rémanence 4×8×8 | 0,95 M | 1,302 / 1,250 | 2,139 / 2,156 | 2 h 52 |
| rémanence 4×16×16 | 1,11 M | 2,525 / 2,506 (n'a pas démarré) | 2,024 / **2,040** | 4 h 40 |
| GRU 256 | 1,56 M | 1,281 / 1,227 | 2,085 / 2,091 | 3,6 min |
| GRU 1024 | 7,8 M | **1,095 / 1,041** | **1,930 / 1,932** | 4,8 min |

- À paramètres comparables (≈ 1 M), T_rope fait 1,3 % mieux que la rémanence dans la fenêtre, 1 % moins bien au-delà : égalité à une graine près.
- Le GRU 256 égale ou bat la rémanence 4×8×8 sur les deux corpus.
- Autres tâches de langage, non discriminantes : étiquetage morpho-syntaxique UD (86,4 % pour tous, A35), bAbI et ListOps (A24, A25 : le transformer n'atteint pas le niveau requis, arrêt).

## 5. Taille d'image et paramètres égaux

Verdict : grossir l'image n'aide pas de façon reproductible ; le transformer profite plus de sa taille.

- A40 (GPU) : passer de 4×8×8 à 4×16×16 améliore la rémanence de 5,4 % sur Wikitext-2, mais sur TinyStories le 4×16×16 reste au plateau (perte 6,8). Le GRU gagne 7 à 15 % en passant de 256 à 1 024.
- A44 (CPU, Wikitext-2, fenêtres de 64, 1 500 pas) :

| modèle | paramètres | bpc 0-64 |
|---|---|---|
| T_rope 64×2 | 0,62 M | 2,102 |
| T_rope 96×2 | 1,01 M | 1,989 |
| T_rope 128×2 | 1,45 M | 1,921 |
| T_rope 192×2 | 2,46 M | 1,827 |
| rémanence 4×8×8 | 0,95 M | 2,129 |
| rémanence 4×16×16 | 1,11 M | 2,146 |
| rémanence 8×16×16 | 1,32 M | 2,995 (n'a pas démarré) |

- Plus long (6 000 pas) : l'écart rémanence 4×8×8 / T_rope 96×2 passe de 7,0 à 4,0 % (1,815 contre 1,746) ; au-delà de 64 tokens, égalité (1,809 contre 1,808).
- Learning rate (5e-4, 1e-3, 2e-3) : les deux modèles préfèrent 2e-3, l'écart reste de 7 à 8,5 %. Le lr unique de 1e-3 n'explique pas le retard.

## 6. Variantes de l'architecture

Aucune ne bat la rémanence de base.

| variante | idée | résultat | source |
|---|---|---|---|
| double loop | 2 images d'entrée, 4 transformers | égal sur S3M, S4, listes ; 2× plus lent | A26, A29, A33 |
| réseau d'images 2-3-1 | 6 images en couches, 18 transformers | ne démarre pas (S4, S5, listes) ; ajouter une couche intermédiaire bloque l'apprentissage | A31-A35b |
| ViT pré-entraîné | T1, T2 = blocs de `vit_small` (43 M) | non concluant (600 pas, pas de contrôle) | A39 |
| entraînement alterné | images puis transformers | plus lent (S4 : 0 graine sur 3 contre 2 sur 3) | A45 |
| plusieurs passes par token | T1→T2 répété 2 ou 4 fois (à la Ouro / Huginn) | aucun effet à 4 000 pas sur S4, coût ×2 et ×4 | A46 |
| version rapide | mêmes calculs, moins d'opérations | sorties identiques à 1e-6, ×1,4 à ×1,8 sur CPU | A38 |

## 7. Vitesse d'apprentissage, graines, coût

- Plateau puis chute : toutes les rémanences restent d'abord au niveau du hasard, puis décrochent brutalement. Le GRU décroche 2 à 4 fois plus tôt (S3M : ≤ 500 pas contre 750 à 1 500 ; listes : 1 500 pas contre 2 500 à 3 500). Sur S4, la rémanence décroche à 6 000 et 13 000 pas, et une graine jamais. Plusieurs résultats nuls (S5, réseau 2-3-1, passes par token) sont peut-être des plateaux non franchis faute de budget.
- Graines : la rémanence échoue sur 1 graine sur 3 dans plusieurs tâches (S3 GPU, S4, listes à 5 sauts) ; les tailles d'image 4×16×16 et 8×16×16 ne démarrent pas sur certains jeux. Les résultats à une graine sont à lire avec cette variance en tête.
- Coût : 8× T_rope sur CPU (Wikitext-2), 26× sur GPU Intel (TinyStories), 49 à 224× sur GPU Intel (A40, selon le corpus et la taille) ; 12 à 50× un GRU. La boucle est limitée par le lancement de nombreuses petites opérations, pas par le calcul : un GPU ne l'accélère presque pas.

## 8. Étapes préliminaires : texte → image, vidéo

- Texte dans une image (A1-A3) : une image de 192 nombres, écrite caractère par caractère comme une vidéo continue, se relit à 99,9 % pour 16 caractères, 97,9 % pour 32 ; 53 % à 64 (87,9 % en 30 000 pas, alors que 32 caractères passent à 99,6 %).
- Vidéo d'images lue par un transformer sans position (A5-A7, TinyStories, 1 graine) : au niveau de T_rope dans la longueur d'entraînement (0,645 contre 0,625 bpc), bien meilleure loin au-delà (0,846 contre 1,164 à 512-1 024 tokens).
- La vidéo seule ne suffit pas à suivre un état (A10, A11) : S3 appris par 1 graine sur 3. D'où l'ajout du retour T2 : la rémanence.
- Images lisibles ? (A14) Non : pas de structure visible, la continuité imposée n'apporte ni robustesse au bruit ni compacité.

## 9. Corrections faites en cours de route

- La permutation de 6 (A21) laissait un raccourci (« répondre x ») : tâche refaite en liste circulaire (A23).
- Un bug du générateur d'histoires synthétiques (A19) a invalidé la première série.
- « La limite à 3 sauts vient de l'entraînement » (A32, 1 graine) : retiré après 2 autres graines (A33b).
- « La grande image place la rémanence devant le transformer » (A40) : vrai seulement contre un transformer plus petit ; à paramètres égaux, égalité.
- L'hypothèse implicite des débuts, « les RNN échoueront sur les tâches de mémoire », a été réfutée (A30) : le projet est passé de « seule architecture qui apprend S3 » à « seul transformer sans accès au passé qui apprend S3 ».

## 10. Ce qui reste ouvert

1. Échelle : tout est ≤ 8 M de paramètres, ≤ 15 000 pas. Les LoopLM récents (Ouro, Huginn) ne marchent qu'à des milliards de tokens, et bouclent en profondeur, pas en mémoire.
2. Démarrage : pourquoi la rémanence reste longtemps au plateau, et pourquoi les grandes images ne démarrent pas. Une stabilisation de type Parcae (contraintes sur l'injection de l'état) n'a pas été essayée.
3. Mémoire très longue (PG-19, A41) et hybride transformer + rémanence (A42) : préparés, pas lancés.
4. Contrôle du transformer entraîné sur des fenêtres longues : non fait proprement (A37 non interprétable).
5. Ce que contiennent les images sur du langage (sondes, A43) : préparé, pas lancé.
