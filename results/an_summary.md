# Analyses de structure cachee

- Ablation : information mutuelle (nats) entre le bloc 2x2 efface et la position du texte abimee, contre le hasard (cellules melangees) ; p = part des melanges au moins aussi structures. Degat moyen par bloc = somme sur les positions de la baisse de precision quand un bloc est efface (faible = code tres redondant).
- Ecriture (bras cumul) : part de variance (R2) de la **localisation** de l'ecriture expliquee par le caractere exact / sa classe (espace, voyelle, consonne, autre) / l'instant t, contre le hasard. Le R2 par caractere est attendu par construction (l'ecriture part d'un motif propre au caractere) ; la classe est comparee a des regroupements arbitraires de caracteres.
- Derive : correlation de rang entre t et la position (x, y) du centre de masse de l'ecriture ; deplacement en pixels entre le debut et la fin du morceau.

| modele | precision intacte | degat moyen par bloc | ablation : MI (hasard), p | R2 caractere (hasard) | R2 classe (hasard), p | R2 instant (hasard) | derive x / y, deplacement |
|:--|--:|--:|--:|--:|--:|--:|--:|
| cumul an_base L=16 | 99.8 % | 0.110 | 0.301 (0.371), p=1.00 | 0.095 (0.007) | 0.028 (0.011), p=0.00 | 0.175 (0.002) | -0.56 / 0.56, 0.3 px |
| cumul an_pas1 L=16 | 99.0 % | 0.287 | 0.145 (0.223), p=1.00 | 0.097 (0.007) | 0.029 (0.012), p=0.01 | 0.062 (0.002) | 0.85 / 0.61, 0.3 px |
| image an_image L=16 | 100.0 % | 0.009 | 0.979 (1.225), p=1.00 | - | - | - | - |
