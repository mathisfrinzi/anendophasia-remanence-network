# Protocole pré-enregistré : étape 1 : autoencodeur texte ↔ image

> Écrit le 2026-11-17 **avant toute mesure**. Rien ici ne change après le premier pas d'entraînement ; toute déviation est consignée dans les résultats.

## Question

Un morceau de L caractères peut-il être encodé en **une image** (3×8×8 = 192 valeurs) puis relu à l'identique ? Jusqu'à quelle taille L ? Et surtout : peut-il l'être **de façon continue**, comme une vidéo où chaque caractère ajoute un résidu à l'image déjà là (un « fil » de pensée qui s'oublie peu à peu), plutôt qu'en rangeant d'un coup un vecteur dans un vecteur ? Le second cas est le seul qui ne soit pas trivial : l'encodage d'un seul coup n'est qu'une compression, la vidéo continue est l'idée de départ.

C'est l'étape 1 de la direction « le LLM travaille sur une représentation visuelle apprise du texte » : encodeur texte→image, puis (étape 2, plus tard) un transformer qui opère sur ces images, et le même décodeur image→texte. L'étape 2 ne se lance que si l'étape 1 fonctionne.

## Montage

- **Données** : TinyStories (streaming, dans l'ordre) ; 50 000 premières histoires pour l'entraînement, 2 000 suivantes pour la validation. Niveau caractère, vocabulaire = caractères triés construits sur train + val. Morceaux = fenêtres aléatoires de L caractères.
- **Encodeur** : embedding + position → transformer (2 couches, d = 128) → projection linéaire vers 192 valeurs → étage « spatial » → normalisation (variance 1 par échantillon) → bruit gaussien σ = 0,1 à l'entraînement seulement. Le bruit et la normalisation rendent la capacité du latent mesurable (l'échelle ne peut pas stocker d'information).
- **Décodeur** : étage spatial → projection linéaire → transformer bidirectionnel (2 couches, d = 128) → logits ; les L caractères sont relus **d'un coup** depuis ce seul latent.
- **Quatre bras** ; le décodeur est le même partout (étage spatial image → projection → transformer bidirectionnel) :
  - `cumul` (**bras principal, vidéo continue**) : l'image est construite caractère par caractère, `frame_t = oubli_t × frame_(t−1) + résidu_t`, où l'oubli (∈ ]0,1[, ≈ 0,95 au départ puis appris) et le résidu (tanh) sortent d'un petit réseau convolutif (3 convolutions 3×3, 32 canaux) qui reçoit la frame courante **et** le caractère (écriture contextuelle). **Aucune position explicite** : l'ordre ne peut venir que de la dynamique. La suite des frames est la vidéo ; seule la dernière est lue par le décodeur ;
  - `image` : encodage d'un seul coup (transformer sur tout le morceau), puis étage de 3 convolutions 3×3 sur l'image 3×8×8 ;
  - `vector` : même chose avec un latent plat et un MLP au même nombre de paramètres que l'étage convolutif (témoin) ;
  - `none` : pas d'étage (témoin).
- **Tailles de morceau** : L ∈ {8, 16, 32, 64}. Les nombres de paramètres varient avec L (projections L·d ↔ 192) ; ils sont égaux entre `image`, `vector` et `none` à L fixé (à quelques dizaines près). **`cumul` en a environ moitié moins** (son encodeur est un petit réseau convolutif, pas un transformer) : c'est un handicap assumé, un succès de `cumul` n'en est que plus net.
- **Entraînement** : Adam lr 1e-3, warmup 200 pas, clip 1,0, lot 128, 8 000 pas, graine 42, mêmes lots pour tous les bras. Aucun réglage par bras.
- **Métriques** (validation, lots fixes, graine 123) : proportion de caractères relus correctement (`char_acc`), proportion de morceaux entièrement exacts, perte en nats/caractère.

## Critères

- **Reconstruction réussie** pour (bras, L) : `char_acc` finale ≥ **99 %**.
- **Résultat principal** : pour chaque bras, le **plus grand L réussi** ; le chiffre qui compte est celui de `cumul` (la vidéo continue), les autres bras sont des références.
- **L'étape 2 avec vidéo continue est justifiée** si `cumul` réussit pour un L ≥ 16 (une image doit représenter assez de texte pour raccourcir sensiblement la séquence). Si `cumul` échoue mais qu'un bras `image` réussit, l'étape 2 ne peut se faire qu'avec des images indépendantes ; consigné tel quel.
- **La forme image apporte quelque chose** à ce stade si `image` réussit pour un L strictement plus grand que `vector` **et** que `none` ; sinon, consigné « pas d'effet mesurable de la forme image sur la reconstruction ».
- **Continuité (bras `cumul`)** : indice = pas moyen entre deux frames consécutives ÷ distance moyenne entre frames de textes différents au même instant (val.). Lecture pré-enregistrée : **≤ 0,5 clairement continue, 0,5–0,8 partiellement, > 0,8 non continue** (images qui sautent). Ce n'est pas un critère de réussite, mais un « succès continu » exige une reconstruction réussie **et** un indice ≤ 0,8.
- **Profil d'oubli** : précision par position dans le morceau (1re à dernière), pour `cumul` comme pour les autres : montre si le fil oublie le début.
- NO-GO consigné avec la même rigueur qu'un succès.

## Limites assumées

- Un seul tirage (une graine) par cas : une différence de quelques points entre bras n'est pas concluante. L'écart doit être net pour être retenu.
- Le décodage parallèle est plus difficile que le décodage autorégressif ; un échec à grand L peut venir de là autant que de la capacité de l'image. Dans ce cas, on le consigne, on ne conclut pas sur la capacité.
- Le test n'évalue **pas** le raisonnement ni la perplexité : seulement la fidélité du canal texte → image → texte.

## Amendement 1 (2026-11-17, avant toute mesure)

Ajout du bras `cumul`, de l'indice de continuité, du profil d'oubli et du film des frames, à la demande explicite de privilégier la vidéo continue (« une pensée est un fil qu'on oublie au fur et à mesure »). Les sections précédentes sont inchangées, sauf le passage de trois à quatre bras et la hiérarchie des résultats. Transparence : les seuils de continuité (0,5 / 0,8) ont été fixés après un test de plomberie sur du **texte synthétique** (mots tirés au hasard) où l'indice valait ~0,65 ; aucune mesure sur TinyStories n'existait. Ce test montre seulement que le bras apprend et que l'ordre peut être retrouvé sans position explicite.

## Amendement 2 (2026-11-17, après la mesure principale, avant les mesures ci-dessous)

Résultat principal connu : `cumul` réussit à L = 8 et 16, échoue à 32 (97,9 %) et 64 (53,0 %), les courbes montent encore à 8 000 pas ; `image`, `vector`, `none` réussissent partout. Verdicts officiels inchangés (8 000 pas, config d'origine). Cinq **mesures supplémentaires** sont ajoutées, étiquetées (`tag`) et rapportées à part (`results/ae_complement.md`) ; elles ne remplacent pas le tableau principal :

- `long30k` : `cumul` à L = 32 et 64 avec 30 000 pas (sous-entraînement ou limite dure ?) ;
- `bruit03` : `cumul` à L = 16, bruit σ = 0,3 au lieu de 0,1 sur l'image avant le décodeur (le latent est déjà bruité à σ = 0,1 dans le protocole principal) ; teste si un bruit plus fort, qui oblige à des codes plus robustes, permet encore la reconstruction ;
- `img16` : `cumul` à L = 16 et 64, image 3×16×16 (768 valeurs) au lieu de 3×8×8, 8 000 pas ; teste si la mémoire est limitée par la taille de l'image. Le décodeur a plus de paramètres (projection 768 → L·d) : la comparaison n'est pas à budget égal.

Lecture pré-enregistrée : un run supplémentaire « réussit » au même critère (≥ 99 % des caractères) ; un seul tirage, donc seuls les écarts nets comptent.

## Amendement 3 (2026-11-17, avant ces mesures)

**Résultats des mesures supplémentaires de l'amendement 2** (8 000 pas sauf mention ; critère : ≥ 99 % des caractères) : bruit σ = 0,3, L = 16 : 99,9 % (réussi) ; image 3×16×16, L = 16 : 99,9 % (réussi), L = 64 : 92,6 % (échec) ; 30 000 pas, L = 32 : 99,6 % (réussi), L = 64 : 87,9 % (échec). Indice de continuité 0,69–0,73 quand ça réussit ; 0,27–0,32 à L = 64 sur 3×8×8 (mémoire saturée, pas un progrès). Les frames ressemblent à une texture de bruit qui s'épaissit, pas à des formes.

**Pénalité sur la taille des pas** (bras `cumul` seulement) : `perte = entropie croisée + λ × indice de continuité`, où l'indice est celui défini plus haut, calculé de façon différentiable sur le lot d'entraînement (pas moyen entre frames consécutives ÷ distance moyenne entre frames de textes différents au même instant). Invariante d'échelle (la normalisation finale rendrait une pénalité en valeur absolue contournable). Mesures : L = 16, 8 000 pas, 3×8×8, bruit 0,1, λ ∈ {0,3 ; 1 ; 3}, tags `pas03`, `pas1`, `pas3`.

Lecture pré-enregistrée : un λ « fonctionne » si la reconstruction reste ≥ 99 % **et** l'indice de continuité mesuré en validation passe sous 0,5. **Limite assumée** : la pénalité vise l'indice lui-même, donc le voir baisser est attendu et ne prouve pas à lui seul que les images ont une structure de formes ; il faut aussi regarder les frames (jugement qualitatif, consigné comme tel). Test de plomberie sur texte synthétique (pas une mesure) : à L = 8, λ = 1 a ramené l'indice de 0,65 à 0,20 avec 99,8 % de caractères relus.

## Amendement 4 (2026-11-17)

**Résultats de la pénalité (amendement 3)**, L = 16, 8 000 pas : λ = 0,3 / 1 / 3 → caractères relus 99,4 / 99,3 / 99,2 % (tous ≥ 99 % : réussis), morceaux exacts 92,7 / 91,2 / 90,4 % (contre 98,1 % sans pénalité), indice de continuité **0,10 / 0,09 / 0,07** (clairement continu, ≤ 0,5). Critère de l'amendement 3 rempli pour les trois λ. Précision de lecture : l'indice a baissé par les deux bouts, pas moyen entre frames 4,3 → 2,3 / 1,4 / 0,9, mais aussi distance moyenne entre frames de textes différents 6,0 → 22,4 / 16,2 / 12,2 ; il faut donc lire « petits pas relativement à l'écart entre textes », pas « images qui bougent peu en absolu ». Les frames évoluent visiblement par glissement progressif (un motif persiste d'un caractère au suivant) ; elles restent une mosaïque de pixels, sans forme reconnaissable. Jugement qualitatif.

**Analyses de structure cachée** (`src/ae_analysis.py`), sur trois modèles entraînés à L = 16 avec les mêmes réglages que les mesures précédentes et sauvegardés : `cumul` sans pénalité, `cumul` λ = 1, `image` (témoin d'un seul coup). 512 morceaux de validation. Hypothèse à tester (Mathis) : l'image n'a aucune raison d'avoir une forme lisible pour l'œil, mais peut porter une structure qu'on ne voit pas.

1. **Ablation par zone** (tous les bras) : un bloc 2×2 de l'image finale est mis à 0 (moyenne du latent normalisé), on mesure la baisse de précision par position du texte. Statistique : information mutuelle entre bloc effacé et position abîmée, contre le hasard (cellules de la matrice mélangées, 200 tirages). *Lecture pré-enregistrée* : information **localisée** si p < 0,01 **et** dégât moyen par bloc ≥ 0,05 ; sinon, « code réparti / redondant » (c'est le cas attendu si les dégâts sont négligeables).
2. **Cartes d'écriture** (bras `cumul`) : où le résidu s'écrit à chaque pas (|résidu| moyen sur les canaux, normalisé en distribution sur les pixels). Part de variance (R²) de cette localisation expliquée par : le caractère exact (**attendu par construction**, l'écriture part d'un motif propre au caractère : non interprété), la classe du caractère (espace, voyelle, consonne, autre) contre des regroupements arbitraires des mêmes caractères en classes de mêmes effectifs (200 tirages), et l'instant t contre le hasard. *Lecture pré-enregistrée* : **structure lexicale** si le R² de classe a p < 0,01 contre ce nul.
3. **Dérive spatiale** (bras `cumul`) : corrélation de rang entre t et la position du centre de masse de l'écriture. *Lecture pré-enregistrée* : dérive si |ρ| ≥ 0,7 sur x ou y **et** déplacement du centre de masse entre début et fin ≥ 0,5 pixel (grille 8×8).

Test de plomberie sur texte synthétique (modèles sous-entraînés, pas une mesure) : aucune structure détectée, le nul se comporte normalement.

### Résultats de l'amendement 4 (17 nov. 2026, L = 16, 8 000 pas, 512 morceaux de validation)

Réplications : `cumul` sans pénalité 99,8 % / 97,5 % de morceaux exacts / indice 0,72 (mesure précédente : 99,9 / 98,1 / 0,71) ; `cumul` λ = 1 : 99,2 / 89,4 / 0,08 (précédente : 99,3 / 91,2 / 0,09). Les mesures se répliquent.

| lecture pré-enregistrée | `cumul` sans pénalité | `cumul` λ = 1 | `image` (un seul coup) |
|:--|:--|:--|:--|
| Ablation : information localisée (p < 0,01 et dégât ≥ 0,05) | **non** : MI 0,301 sous le hasard (0,371), p = 1,00, dégât moyen par bloc 0,110 | **non** : MI 0,145 (0,223), p = 1,00, dégât 0,287 | **non** : dégât 0,009 (code très redondant) |
| Structure lexicale (R² de classe, p < 0,01 contre regroupements arbitraires) | **oui, effet faible** : R² 0,028 contre 0,011, p < 0,005 | **oui, effet faible** : 0,029 contre 0,012, p = 0,005 | - |
| Dérive spatiale (|ρ| ≥ 0,7 et déplacement ≥ 0,5 px) | **non** : ρ = −0,56 / 0,56, 0,28 px | **non** : ρ = 0,85 / 0,61, mais 0,25 px | - |

Non pré-enregistré mais mesuré : R² de la localisation d'écriture expliquée par l'instant t = **0,175** (hasard 0,002) sans pénalité, 0,062 avec λ = 1 ; par le caractère exact 0,095 / 0,097 (hasard 0,007). L'écriture dépend donc davantage de l'instant que du caractère, sans aucune position fournie au modèle. Les cartes moyennes par t (`..._write_time.png`) montrent un schéma qui revient (centre au début et à la fin, bande en haut vers t = 3–5, pic sur la colonne gauche vers t = 7–11), pas un balayage monotone : c'est pourquoi le centre de masse ne bouge que de ~0,3 px. L'ablation montre aussi des zones d'image très utilisées (centre-haut) et d'autres presque inutilisées (rangée du bas). Les 16 positions du texte ne sont en revanche pas associées à des zones précises. L'impression d'un front de couleur se déplaçant de gauche à droite (frames de λ = 3, un seul morceau) n'est pas confirmée par la mesure sur les modèles analysés (λ = 0 et 1).

## Amendement 5 (2026-11-17) : étape 2, transformer sur tokens, texte vs vidéo

Écrit **avant** toute mesure de l'étape 2. Décision de Mathis : passer directement à l'étape 2 avec λ = 0 et λ = 1 (sans grille de λ supplémentaire à L = 32/64).

**Données et métrique.** TinyStories, tokens BPE (vocabulaire 4 096, `tokenizers`), fenêtres de 256 tokens. Métrique : bits par caractère (bpc) = perte en bits par token ÷ caractères par token (mesuré sur le texte décodé). Sortie : toujours des tokens (la sortie en image est une étape ultérieure).

**Modèles** (même données, mêmes 8 000 pas, même planning d'apprentissage, graine 42) :
- `T_rope` : transformer causal, 5 couches, d = 256, position RoPE.
- `T_nope` : même transformer sans aucune position (témoin : que fait le transformer sans la vidéo ?).
- `V_lam0` / `V_lam1` : chaque token écrit un résidu dans une frame continue 3×8×8 (mécanisme `cumul` de l'étape 1, une frame par token), lue par un petit module convolutif puis donnée au **même** transformer causal **sans position** (l'ordre n'est présent que dans la vidéo). Pénalité de continuité λ = 0 ou 1, introduite linéairement sur les 1 000 premiers pas (ajouté après un test court où λ = 1 démarrait lentement).

**Équité.** Paramètres égaux à ~5 % près (T 6,05 M, V 5,70 M) ; **calcul non égal** (V est plus lent : boucle séquentielle de frames). Écart assumé et déclaré.

**Évaluation.** bpc de validation à 256 tokens, et extrapolation à 1 024 tokens par tranche de position : [0,256), [256,512), [512,1024).

**Critères pré-enregistrés.**
1. V est **compétitif** si son bpc sur [0,256) est à moins de 5 % de celui de `T_rope`.
2. V **extrapole mieux** si son bpc sur [512,1024) est inférieur d'au moins 3 % à celui de `T_rope` **et** de `T_nope`.
3. La pénalité est **utile** si `V_lam1` bat `V_lam0` de plus de 2 % sur l'un des deux critères ; **neutre** entre ±2 % ; **nuisible** sinon.
4. Indice de continuité de `V_lam1` < 0,5 pour que la comparaison λ = 0/1 ait un sens.

**Simplifications assumées.** Une seule graine (les écarts < 2 % ne sont pas interprétés). Fenêtres indépendantes (la frame repart de la frame initiale apprise à chaque fenêtre ; pas de flux continu sur tout le texte). Entrée seulement : les tokens passent par la vidéo, la sortie reste des tokens. Test de plomberie sur texte synthétique : λ = 1 démarre lentement à très court entraînement (pas une mesure ; la rampe répond à cela, sans garantie).

Artefacts : `results/s2_*.json`, `results/s2_summary.md`, modèles `results/s2_*_model.pt`.

## Amendement 6 (2026-11-17) : étape 2, version V2 (écrit avant toute mesure de l'étape 2)

Il remplace les modèles vidéo de l'amendement 5 (aucune mesure n'avait été faite). Décisions de Mathis : la frame n'est **jamais remise à zéro** dans un flux ; deux échelles de temps ; convolution spatio-temporelle reportée à après un premier résultat.

**Changements.** (1) *Frame transportée* : à l'entraînement, chaque ligne du lot suit un flux contigu de 8 fenêtres de 256 tokens, la frame passe d'une fenêtre à la suivante, gradient coupé entre fenêtres ; le transformer ne voit que la fenêtre courante. (2) *Deux échelles* : la frame a 6 canaux, 3 rapides (rétention initiale 0,8, mémoire de ~5 tokens) et 3 lents (0,99, ~100 tokens) ; la porte reste apprise (c'est un a priori, pas une contrainte). Le lecteur et le réseau d'écriture normalisent chaque groupe séparément (le groupe lent est bien plus ample). (3) La pénalité de continuité est la **moyenne des indices des deux groupes** (le groupe lent est continu par construction ; la moyenne empêche d'y échapper). Les transformers T voient les mêmes fenêtres (même ordre de données), sans mémoire entre fenêtres.

**Modèles.** `T_rope`, `T_nope`, `V_lam0`, `V_lam1` (2 échelles), et `V1_lam0` (1 échelle, rétention 0,95, **ablation** : les deux échelles aident-elles ?). Paramètres : T 6,05 M ; V 5,73 M ; V1 5,70 M.

**Nouvelle mesure : le flux.** 16 flux de 8 fenêtres contigues de validation (2 048 tokens) ; bpc par numéro de fenêtre ; les modèles vidéo transportent leur frame, T non.

**Critères pré-enregistrés (ajoutent aux critères 1 à 4 de l'amendement 5, qui restent valables).**
5. *Mémoire utile* : la frame transportée aide si le bpc de `V_lam0` ou `V_lam1` à la fenêtre 8 est inférieur d'au moins 2 % à celui de la fenêtre 1 **et** d'au moins 2 % sous celui de `T_rope` à la fenêtre 8. (Les textes sont des histoires courtes indépendantes : l'attente est modeste, la baisse vient de ce que la frame sait déjà qu'on est au milieu d'un texte, pas d'un long contexte réel.)
6. *Deux échelles utiles* : `V_lam0` bat `V1_lam0` de plus de 2 % (bpc à 256 tokens ou à la fenêtre 8) ; neutre entre ±2 % ; nuisible sinon.

**Limites assumées.** Une seule graine. La pénalité sur les modèles V2 n'a pas été validée à grande échelle (test court : λ = 1 démarre lentement, la rampe de 1 000 pas y répond sans garantie). Les histoires TinyStories étant courtes, le gain de mémoire sur le flux est probablement faible ; une vraie preuve de contexte long demanderait un autre jeu de données.

### Résultats de l'étape 2 (18 nov. 2026, 8 000 pas, une graine ; `results/s2_summary.md`)

| | bpc à 256 | tokens 512–1024 | fenêtre 1 → 8 du flux | continuité |
|:--|--:|--:|--:|--:|
| `T_rope` | 0,706 | 1,164 | 0,752 → 0,705 | - |
| `T_nope` | 0,736 | 1,746 | 0,790 → 0,754 | - |
| `V_lam0` (2 échelles) | 0,732 | **0,846** | 0,780 → 0,733 | 0,43 |
| `V_lam1` (2 échelles, λ = 1) | 1,800 | 1,677 | 1,838 → 1,807 | 0,03 |
| `V1_lam0` (1 échelle) | 0,761 | 0,914 | 0,805 → 0,781 | 0,73 |

- **Critère 1 (compétitif)** : réussi. `V_lam0` est à 3,7 % de `T_rope` à 256 tokens, sans position (au niveau de `T_nope`).
- **Critère 2 (extrapolation)** : réussi. Sur 512–1 024 tokens, `V_lam0` est 27 % sous `T_rope` et 52 % sous `T_nope`. Réserve : `T_rope` est un RoPE simple, sans technique d'extension de longueur.
- **Critère 3 (pénalité)** : `V_lam1` n'avait pas fini d'apprendre (reste à ~2,23 bpc pendant 6 000 pas, descend à 1,80 à la fin, courbe encore en baisse). Lecture mécanique à 8 000 pas : « nuisible » ; lecture retenue : **indéterminé** (budget d'entraînement insuffisant), à tester avec un entraînement plus long ou une pénalité introduite tard. Aucune explication du blocage n'est établie (frames non examinées).
- **Critère 4** : l'indice de `V_lam1` (0,03) est < 0,5, mais le modèle n'a pas appris le langage ; la comparaison λ = 0/1 n'a pas de sens à ce stade.
- **Critère 5 (frame transportée utile)** : **échec** de la mesure. La baisse fenêtre 1 → 8 est la même pour `V_lam0` (−6,0 %) et `T_rope` (−6,2 %), qui n'a pas de mémoire ; l'écart V − T reste ~+3,7 %. Les histoires courtes ne mesurent pas un contexte long ; cela ne prouve pas que la frame transportée est inutile.
- **Critère 6 (deux échelles utiles)** : réussi. `V_lam0` bat `V1_lam0` de 3,8 % à 256 tokens et de 6,1 % à la fenêtre 8 (> 2 %).
- Continuité de `V_lam0` (0,43) : moyenne des deux groupes, probablement flattée par le groupe lent ; `V1_lam0` (une échelle) : 0,73.

Modèles sauvegardés : `results/s2_*_model.pt`.

## Amendement 7 (2026-11-18) : prolongation de `V_lam1` (écrit avant la mesure)

Contexte : `V_lam1` (étape 2) est resté ~2,23 bpc pendant 6 000 pas puis a commencé à descendre (1,80 à 8 000 pas, courbe encore en baisse) ; le critère 3 est indéterminé (voir résultats de l'étape 2). Hypothèse de Mathis : la structure imposée par λ = 1 se forme lentement, il faut plus de pas.

**Protocole.** On repart des poids de `s2_V_lam1_model.pt` (8 000 pas), avec un optimiseur neuf, et on entraîne jusqu'à **24 000 pas** au total (16 000 de plus), λ = 1 constant, données nouvelles (les flux continuent selon le numéro de pas). Taux d'apprentissage : échauffement de 200 pas jusqu'à 1e-3, puis cosinus jusqu'à 1e-4. Autres réglages inchangés. Même évaluation que l'étape 2 (bpc à 256 tokens, extrapolation, flux, continuité). Résultat dans `results/s2_V_lam1_ext.json` et `s2x_summary.md` ; `s2_V_lam1` reste tel quel.

**Critères pré-enregistrés.** (a) *sorti du blocage* : bpc à 256 tokens < 1,0. (b) *compétitif* : bpc à 256 tokens ≤ 0,769 (à moins de 5 % de `V_lam0` à 8 000 pas, 0,732) **et** continuité < 0,5. (c) Si (b) est atteint, `V_lam1` aura obtenu en 24 000 pas ce que `V_lam0` a en 8 000 ; la conclusion « la pénalité est utile / neutre / nuisible » exigerait alors un `V_lam0` aussi entraîné 24 000 pas (budget égal), non fait ici. Si (a) n'est pas atteint, on ne conclut rien de plus que « 24 000 pas ne suffisent pas ».

**Limites assumées.** Budget inégal avec les autres modèles (8 000 pas) ; une seule graine ; le nouvel échauffement et le nouveau cosinus ne sont pas une continuation exacte du premier run.

### Résultats de l'amendement 7 (18 nov. 2026, `results/s2x_summary.md`)

`V_lam1_ext` (24 000 pas au total) : bpc à 256 tokens **0,773** (contre 1,800 à 8 000 pas ; `V_lam0` 0,732 à 8 000 pas ; `T_rope` 0,706), tranche 512–1 024 : **0,819** (`V_lam0` 0,846 ; `T_rope` 1,164), flux : fenêtre 1 → 8 : 0,828 → 0,772, continuité 0,01. Courbe : 1,44 (9 000 pas), sous 1,0 vers 13 000, 0,87 (16 000), 0,78 (22 000), 0,773 (24 000) ; la baisse ralentit en fin de planning (taux d'apprentissage vers 1e-4). Figure : `results/s2_courbes_bpc.png`.
- (a) sorti du blocage (< 1,0) : **réussi**.
- (b) compétitif (≤ 0,769 et continuité < 0,5) : **raté de peu** (0,773, soit +5,6 % par rapport à `V_lam0`).
- (c) Aucune conclusion sur l'utilité de la pénalité : budget inégal (24 000 pas contre 8 000). Il faudrait `V_lam0` entraîné aussi 24 000 pas ; non fait. Hypothèse de Mathis (non testée) : `V_lam1` rattraperait `V_lam0` avec plus de pas.

**Mesures complémentaires sur du texte réel** (tokenizer `tokenizer/bpe_4096.json`, 5 textes de 100 tokens écrits à la main, `scripts/indice_continuite_textes.py`) : indice de continuité rapide / lent / moyenne : `V_lam0` 0,84 / 0,25 / **0,55** ; `V_lam1_ext` 0,04 / 0,04 / **0,04**. La continuité de `V_lam0` indiquée précédemment (0,43) est donc flattée par le groupe lent : **le groupe rapide de `V_lam0` n'est pas continu** (0,84). Sur de très courts textes (14 tokens) les indices sont plus élevés (0,86 pour `V_lam1_ext`) car les images de textes différents ne se sont pas encore éloignées : l'indice n'est interprétable qu'avec des textes assez longs.

**Exemple illustré** (`scripts/demo_phrase.py` : `results/s2_film_frames.png`, `results/s2_decodage.txt`) sur une seule phrase, anecdotique : l'image de `V_lam0` fait de grands pas aux points (fins de phrase), celle de `V_lam1_ext` varie peu et concentre ses écritures dans un coin ; le mot absurde « rock » (surprise 8,8 bits pour `V_lam0`, 13,7 pour `V_lam1_ext`) ne provoque **pas** de pas plus grand que les autres : rien n'indique pour l'instant que l'image « retravaille » plus après un mot inattendu.

**Mesure exploratoire de l'espace utilisé** (`scripts/dimension_frames.py`, 5 textes de 100 tokens, 500 images, une seule mesure) : dimension effective de la variance des frames (rapport de participation, sur 192 valeurs) : `V_lam0` rapide 9,9 et lent 1,0 ; `V_lam1_ext` rapide **2,4** et lent 1,8. Part de la variance portée par les 10 % de valeurs les plus variables : `V_lam0` rapide 20 %, lent 33 % ; `V_lam1_ext` rapide **99 %**, lent 83 %. La pénalité de continuité s'accompagne donc d'une variance concentrée sur très peu de valeurs. Limite : ce n'est pas la capacité d'information (des directions de faible variance peuvent encore être lues). Hypothèse non testée : l'indice étant un rapport (pas / distance entre textes), la concentration sur quelques valeurs de forte amplitude est une façon bon marché de le satisfaire.

## Amendement 8 (2026-11-18) : rappel de listes, test exploratoire (écrit avant la mesure)

Origine : les études sur l'anendophasie (Nedergaard et Lupyan) rapportent une moins bonne mémoire verbale à court terme (listes de mots) sans différence de raisonnement général. Question : une vidéo continue qui oublie progressivement rappelle-t-elle moins bien une liste exacte qu'un transformer sur tokens ?

**Test** (`src/s2_recall.py`, sans entraînement, CPU) sur `T_rope`, `T_nope`, `V_lam0`, `V1_lam0` (modèles de l'étape 2 ; `V_lam1` exclu : pas fini d'apprendre). Une liste de n tokens tirés au hasard (identifiants 300–1 999 du vocabulaire BPE : des tokens fréquents, pas de vrais mots, faute de tokenizer ici), éventuellement suivie de d tokens aléatoires de distraction, puis la même liste répétée. n ∈ {8, 16, 32, 64}, d ∈ {0, 64}, 256 séquences par réglage. Mesure : perte (nats) et précision top-1 sur la **seconde** copie ; première copie en témoin (hasard : ln 1 700 ≈ 7,44 nats). Position : la copie reste dans les 256 tokens d'entraînement.

**Attente écrite d'avance** (sans critère de réussite) : V rappelle moins bien que `T_rope` ; si V rappelle aussi bien ou mieux, c'est que l'attention sur les frames passées compense l'oubli de la frame. Une différence n'est lue que si elle dépasse 0,5 nat en moyenne sur la seconde copie ; une seule graine, tokens aléatoires et non du texte : exploratoire, aucune conclusion sur la cognition humaine.

### Résultat de l'amendement 8 (18 nov. 2026)

**Test non informatif (effet plancher).** Aucun des quatre modèles ne rappelle quoi que ce soit : précision top-1 de 0 % partout, et la perte de la seconde copie (11,7 à 12,5 nats) est équivalente à celle de la première (12,3 à 12,5 nats). Les modèles ne copient donc pas du tout des tokens aléatoires vus quelques tokens plus tôt, ni `T_rope` ni `V`. Ce n'est pas un résultat sur la mémoire verbale à court terme : le test ne distingue pas les modèles. Explication possible, non vérifiée : de petits modèles entraînés 8 000 pas sur des histoires n'ont pas développé de mécanisme de copie de séquences arbitraires. Les listes de tokens aléatoires ne sont d'ailleurs pas de vrais mots (faute de tokenizer dans cet environnement). Un test utile demanderait de vrais mots (tokenizer et données TinyStories, par exemple sur Colab) et probablement des textes à contenu répété naturellement (noms propres qui reviennent dans une histoire). Aucune conclusion sur l'attente écrite à l'amendement 8.

## Amendement 9 (2026-11-18) : l'image qui évolue (écrit avant la mesure)

Idée de Mathis : l'image évolue seule entre les tokens ; le token ne dépose qu'un motif qui lui est propre (sans lire l'image) ; c'est l'évolution qui porte le contexte (l'image « se prépare » aux mots possibles, et retravaille plus longtemps si le mot est absurde) ; le transformer lit l'image après évolution.

**Modèle (`DLM`, `src/s2_models.py`).** 6 canaux (3 rapides, 3 lents, comme `V_lam0`). À chaque token : `frame ← rétention ⊙ frame + dépôt(token)` (rétention apprise par canal, indépendante du contexte, initiale 0,8 / 0,99), puis n pas d'évolution `frame ← frame + tanh(réseau(frame))` (le réseau ne voit que l'image). État de repos initial 0,01 ; bruit gaussien 0,01 en entraînement. Frame transportée entre fenêtres, même transformer sans position, mêmes données, même planning (8 000 pas, λ = 0).
- **Dépôt libre** : motif de 6×8×8 valeurs calculé à partir d'un plongement du token.
- **Dépôt en tache** : par canal, 12 amplitudes (bloc 3 lignes × 4 colonnes) posées à une position continue apprise (x, y), répartie entre les pixels voisins par interpolation bilinéaire (amplitudes de tanh ; position bornée pour que la tache reste dans l'image). 14 valeurs par canal, 84 par token. *Écart avec l'idée initiale (3 canaux)* : 6 canaux, pour rester comparable à `V_lam0`.

**Modèles.** `D_free_e1` (dépôt libre, 1 pas d'évolution), `D_blob_e1` (tache, 1 pas), `D_blob_e0` (tache, aucune évolution : témoin), `D_blob_e2` (tache, 2 pas). Références, déjà mesurées : `T_rope` 0,706 bpc, `V_lam0` 0,732 (256 tokens). Paramètres : 5,73 M (libre) et 5,50 M (tache), contre 6,05 M (T) : écart déclaré.

**Critères pré-enregistrés** (bpc de validation à 256 tokens, et tranche 512–1 024 ; une différence n'est lue que si elle dépasse 2 %).
1. *Le dépôt n'a pas besoin de lire l'image* : `D_free_e1` à moins de 3 % de `V_lam0` à 256 tokens.
2. *Compression du dépôt gratuite* : `D_blob_e1` à moins de 2 % de `D_free_e1` à 256 tokens.
3. *L'évolution apporte quelque chose* : `D_blob_e1` bat `D_blob_e0` de plus de 2 % (à 256 tokens ou sur 512–1 024).
4. *Du temps libre supplémentaire aide* : `D_blob_e2` bat `D_blob_e1` de plus de 2 %.
5. *Compétitif* (comme à l'amendement 5) : à moins de 5 % de `T_rope` à 256 tokens.

**Analyses exploratoires après coup, non pré-enregistrées comme critères** : positions apprises des tokens (des tokens proches par le sens ou par la fréquence sont-ils proches sur l'image ?).

**Hors périmètre de cette mesure.** Le pixel « prêt » qui déclenche l'entrée d'un token (durée choisie par l'image), le temps d'évolution appris selon le token ou la surprise, et la sortie en vidéo avec pixel de lecture (CTC). Ici : nombre de pas fixe, sortie en tokens.

**Limites.** Une seule graine ; 8 000 pas (on a vu que des modèles vidéo peuvent apprendre plus lentement) ; nombre de paramètres légèrement différent.

## Amendement 10 (2026-11-18) : étape 3, suivi d'état (écrit avant la mesure)

Contexte : sur TinyStories, l'évolution de l'image n'apporte rien (amendement 9 : `D_blob_e1` ≈ `D_blob_e0` à 256 tokens, et moins bon à l'extrapolation). TinyStories ne demande pas de calcul interne. On teste donc l'idée sur une tâche qui en demande, avec une prédiction précise dans la littérature : les transformers à profondeur fixe échouent à suivre un état non commutatif sur de longues suites, un état récurrent non linéaire peut y réussir.

**Tâche** (`src/s3_tasks.py`). Une suite d'éléments du groupe symétrique S_k (un token par élément, tirés uniformément) ; cible à chaque position : le **produit cumulé**. Trois groupes : S2 (parité, 2 états), S3 (6 états, le plus petit groupe non commutatif), S5 (120 états, non résoluble). Entraînement sur des suites de 24 éléments (perte sur toutes les positions, avec un curriculum : la longueur supervisée passe de 4 à 24 pendant les 40 % premiers pas) ; évaluation sur des suites de **256** éléments : tranches de position [0,24) (dans la longueur d'entraînement), [24,48), [48,96), [96,256) (extrapolation). Pas : 2 000 (S2), 4 000 (S3), 8 000 (S5) ; lot 128 ; AdamW 1e-3, cosinus ; 3 graines ; mêmes données pour tous les modèles à graine égale.

**Modèles.** `T_rope`, `T_nope` (transformers), `V_lam0` (la vidéo où l'écriture lit l'image), `D_blob_e0` (dépôt en tache, sans évolution : intégrateur diagonal), `D_blob_e1` (dépôt en tache, 1 pas d'évolution non linéaire). Même transformer (4 couches, d = 128, sans position pour V et D). Paramètres : ~0,79 M (T) à ~0,86 M (V). Pas de pénalité, pas de bruit, frame non transportée.

**Critères pré-enregistrés** (précision de la mediane des 3 graines ; une tâche est « apprise » si la précision dans [0,24) est ≥ 95 % pour au moins 2 graines sur 3) :
1. *Faisabilité* : au moins un modèle apprend S3 ; sinon la tâche est trop dure pour ce réglage et on ne conclut rien sur S5.
2. *Généralisation en longueur* : un modèle généralise si sa précision dans [96,256) reste ≥ 90 %.
3. *L'évolution non linéaire sert* : `D_blob_e1` dépasse `D_blob_e0` de ≥ 10 points sur S3 ou S5, dans [0,24) ou dans [96,256).
4. *La vidéo bat le transformer* : le meilleur de `V_lam0`, `D_blob_e0`, `D_blob_e1` dépasse `T_rope` de ≥ 10 points dans [96,256) sur S3 ou S5.

**Attentes écrites d'avance** (hypothèses, pas des résultats) : `T_rope` apprend S2 mais n'extrapole pas ; `D_blob_e0` (intégrateur diagonal à rétentions positives) a du mal sur S3 et S5 ; `D_blob_e1` et `V_lam0` ont une chance réelle de mieux faire, sans garantie, car ces tâches sont sensibles au réglage d'entraînement.

**Test de faisabilité déjà vu avant la mesure (CPU, graine 0, S2 seulement, pas une mesure officielle)** : précision dans [0,24) / [24,48) / [48,96) / [96,256) : `T_rope` 99 / 50 / 50 / 50 ; `T_nope` 92 / 64 / 50 / 50 ; `V_lam0` 100 / 98 / 46 / 18 ; `D_blob_e0` 100 / 61 / 50 / 50 (`D_blob_e1` non terminé). La tâche est donc apprenable, et les transformers n'extrapolent pas, comme attendu ; `V_lam0` extrapole jusqu'à ~48 éléments puis échoue (en dessous du hasard : réponses systématiquement inversées). Les critères ci-dessus n'ont pas été modifiés après ce test.

**Limites assumées.** Petits modèles entraînés de zéro ; un seul réglage d'optimisation par tâche (pas de recherche d'hyperparamètres) : un échec peut venir de l'optimisation, pas de l'architecture ; 3 graines seulement ; ces tâches ne sont pas du langage ; pas de témoin « jetons de pause » à calcul égal à cette étape (il viendrait avec la tâche de calcul sans étapes écrites).

### Résultats de l'amendement 10 (19 nov. 2026, `results/st_summary.md`, 3 graines)

Précision médiane [min-max] en % (dans [0,24) | [96,256)) : **S2** : `T_rope` 99 | 50, `T_nope` 96 | 50, **`V_lam0` 100 | 96 [71-100]**, `D_blob_e0` 100 | 50, `D_blob_e1` 100 | 50. **S3** : `T_rope` 76 | 17, `T_nope` 58 | 17, `V_lam0` 63 [61-100] | 17 [17-99], `D_blob_e0` 75 | 17, `D_blob_e1` 61 | 17. **S5** : tous 5 à 9 % dans [0,24) (hasard 0,8 %), 1 % au-delà.
- Critère 1 (S3 apprise par au moins 2 graines sur 3) : **non rempli** (`V_lam0` : 1 graine sur 3, atteignant 100 % à toutes les positions jusqu'à 256 ; `V_lam0` graine 2 passe de 75 % à 1 500 pas à 100 % à 2 500). Pour S5, rien n'est appris : aucune conclusion.
- Critère 2 (généralisation ≥ 90 % sur [96,256)) : rempli par `V_lam0` sur S2 seulement.
- Critère 3 (`D_blob_e1` bat `D_blob_e0` de ≥ 10 points sur S3 ou S5) : **non** (S3, [0,24) : 61 contre 75 %).
- Critère 4 (la vidéo bat `T_rope` de ≥ 10 points sur S3 ou S5 dans [96,256)) : **non** sur ces deux tâches (médianes égales) ; sur S2 (hors critère), `V_lam0` 96 % contre 50 %.
- Lecture : seul `V_lam0`, où l'écriture lit l'image, généralise en longueur (parité) et a résolu S3 une fois ; le dépôt par le token seul (`D_blob_e0` et `D_blob_e1`) échoue au-delà de la longueur d'entraînement. Hypothèse non testée : le suivi d'état demande que la nouvelle valeur dépende à la fois de l'état et du token. Les graines non résolues de `V_lam0` sur S3 montaient encore lentement à 4 000 pas.

## Amendement 11 (2026-11-19) : S3, entraînement long, plus de graines (écrit avant la mesure)

Question : `V_lam0` apprend-il S3 de façon fiable avec un entraînement plus long, ou la graine réussie était-elle un coup de chance ?

**Protocole.** Même tâche S3 et mêmes réglages que l'amendement 10, sauf : **12 000 pas** (3 fois plus ; le curriculum suit : 40 % des pas) et **6 nouvelles graines** (3 à 8, jamais utilisées). Modèles : `T_rope`, `V_lam0`, `D_blob_e1`. Résultat dans `results/stl_summary.md` (par graine).

**Critères pré-enregistrés.** Un modèle « apprend S3 de façon fiable » si, dans [0,24), la précision est ≥ 95 % pour **au moins 4 graines sur 6** ; il « généralise » si ces graines gardent ≥ 90 % dans [96,256).
1. `V_lam0` fiable : ≥ 4/6. 2. `V_lam0` généralise : ≥ 4/6 aussi dans [96,256). 3. Pour `T_rope` et `D_blob_e1`, le même critère de fiabilité (≥ 4/6) pour savoir si la différence est d'apprentissage ou d'extrapolation.

**Limites.** 12 000 pas sont un nouveau réglage d'optimisation, pas la preuve qu'une durée infinie suffirait ; pas de recherche d'hyperparamètres ; ni S5 ni les autres modèles (`T_nope`, `D_blob_e0`) ne sont relancés.

## Amendement 12 (2026-11-19) : étape 4, deux images couplées par deux transformers (écrit avant la mesure)

Idée de Mathis : deux images ; un transformer T1 lit Image1 (entrée) et met à jour Image2 (sortie) ; un second transformer T2 lit Image2 et met à jour Image1 (rétroaction : l'image d'entrée se prépare au token suivant). **Aucun historique** : chaque transformer ne voit que l'image courante, lue en **patchs** (attention entre patchs) ; toute la mémoire est dans les deux images, mises à jour en résidu avec porte d'oubli apprise. Première version où le transformer traite l'image comme une image et n'est pas un hybride récurrence + attention sur l'historique.

**Modèle** (`src/s4_loop.py`). Images 4 canaux × 8×8, patchs 2×2 (16 patchs), d = 128, 2 couches par transformer. Par token : le token dépose un motif sur Image1 ; T1(Image1) → `Image2 ← σ(g+3)·Image2 + tanh(résidu)` ; la prédiction est lue sur Image2 (MLP) ; T2(Image2) → `Image1 ← σ(g+3)·Image1 + tanh(résidu)`. Images de départ apprises.
- `L_loop` : modèle complet (0,90 M paramètres).
- `L_noloop` : T2 supprimé (Image1 ne reçoit que les dépôts et une rétention fixe apprise) : témoin sans rétroaction (0,49 M ; **moins de paramètres**, écart déclaré).
- Références : `T_rope` et `V_lam0` (résultats de l'étape 3, mêmes tâches, graines 0 à 2, mêmes pas).

**Tâches.** S2 (parité) et S3 comme à l'amendement 10 (entraînement sur 24 éléments, évaluation jusqu'à 256, 2 000 et 4 000 pas, graines 0, 1, 2).

**Critères pré-enregistrés.**
1. *Apprend* : `L_loop` ≥ 95 % dans [0,24) sur au moins 2 graines sur 3, pour S2 puis pour S3.
2. *Généralise* : `L_loop` ≥ 90 % dans [96,256) sur ces graines.
3. *La rétroaction sert* : `L_loop` dépasse `L_noloop` de ≥ 10 points dans [0,24) ou [96,256), sur S2 ou S3.
4. *Mieux que la référence* : sur S3, `L_loop` dépasse `V_lam0` de ≥ 10 points dans [96,256) (médiane de `V_lam0` : 17 %).

**Test de plomberie déjà vu (CPU, version réduite : parité, suites de 12, d = 64, 600 pas, 1 graine, non officiel)** : `L_loop` 100 % dans [0,12), [12,24) et [24,48) ; `L_noloop` 89 / 53 / 50 %. Les critères n'ont pas été modifiés après ce test.

**Limites.** Pas de langage : la sortie lue sur Image2 par un MLP est un goulot, cette version n'est pas faite pour TinyStories ; une image de 192 valeurs retenait ~32 caractères à l'étape 1 ; réglages d'optimisation non cherchés ; 3 graines.

### Résultats de l'amendement 12 (19 nov. 2026, `results/st4_summary.md`, 3 graines par tâche)

Précision en % (dans [0,24) | [96,256)), par graine 0 / 1 / 2. **S2** : `L_loop` 100|100, 100|50, 100|100 ; `L_noloop` 83|50, 81|50, 81|50 ; (références : `V_lam0` 100|96, 100|71, 100|100 ; `T_rope` 99|50). **S3** : `L_loop` 30|17, **100|100, 100|100** ; `L_noloop` 53|17, 47|17, 55|17 ; (références : `V_lam0` 61|17, 63|17, 100|99 ; `T_rope` 76|17, 65|17, 80|17).
- Critère 1 (apprend, ≥ 2 graines sur 3) : **rempli** sur S2 (3/3) et S3 (2/3 ; la graine 0 reste à 30 %).
- Critère 2 (généralise ≥ 90 % dans [96,256) « sur ces graines ») : S3 : **rempli** (les deux graines qui apprennent tiennent 100 % jusqu'à 256) ; S2 : 2 graines sur 3 (la graine 1 tombe à 50 %). Lecture stricte de « ces graines » : non rempli sur S2.
- Critère 3 (`L_loop` bat `L_noloop` de ≥ 10 points) : **rempli** (S3 : 100 contre 53 % dans [0,24) ; S2 : 100 contre 81 %). Réserve : `L_noloop` a presque deux fois moins de paramètres et n'a pas de second transformer ; la différence mélange la rétroaction et la capacité, et `L_noloop` n'apprend pas même dans la longueur d'entraînement (il n'est pas un témoin « à capacité égale »).
- Critère 4 (bat `V_lam0` de ≥ 10 points sur S3 dans [96,256)) : **rempli** (médiane 100 contre 17 %).
- Lecture : première architecture du projet qui apprend S3 de façon quasi fiable (2 graines sur 3) et généralise en longueur (jusqu'à 256 pour 256/24 ≈ 10× la longueur d'entraînement), sans aucun historique : la mémoire est entièrement dans les deux images. Non établi : si c'est la rétroaction elle-même ou le second transformer (capacité, calcul par token) qui compte ; si le résultat tient sur S5 ; si l'idée fonctionne sur du langage. Un témoin à capacité égale (T2 présent mais sans lien avec Image1, ou un T1 plus profond) serait nécessaire pour trancher.

## Amendement 13 (2026-11-19) : étape 5, autres tâches d'état pour `L_loop` / `L_noloop` (écrit avant la mesure)

Décision de Mathis : tester le couple avec / sans boucle sur d'autres tâches, avant le témoin à capacité égale. Même architecture, mêmes réglages d'optimisation, mêmes graines 0 à 2 que l'étape 4. Références : `T_rope`, `V_lam0` (re-entraînés sur Dyck et clé-valeur ; S5 : résultats de l'étape 3).

**Tâches** (`src/s5_tasks.py`).
- **Dyck (profondeur)** : suite de parenthèses ouvrantes/fermantes (jamais de profondeur négative, profondeur ≤ 8, marche aléatoire réfléchie) ; cible à chaque position : la profondeur courante (9 classes). Entraînement sur 24 éléments (curriculum 4 → 24 sur 40 % des pas), 3 000 pas ; évaluation jusqu'à 256 ; tranches [0,24), [24,48), [48,96), [96,256).
- **Clé-valeur** : N paires (clé parmi 16 distinctes, valeur parmi 8), jeton de requête, une clé tirée parmi les N ; cible : sa valeur (à la dernière position seulement ; hasard : 1/8, 12,5 %). Entraînement avec N tiré entre 1 et 8 à chaque lot, 8 000 pas ; évaluation à N = 2, 4, 8, 12, 16 (12 et 16 : au-delà de l'entraînement, mesure de capacité en nombre de paires).
- **S5** : comme à l'amendement 10, 8 000 pas (aucun modèle ne l'avait apprise).

**Critères pré-enregistrés.**
1. *Dyck* : « appris » si ≥ 95 % dans [0,24) pour ≥ 2 graines sur 3 ; « généralise » si ≥ 90 % dans [96,256) pour ces graines.
2. *Clé-valeur* : « appris » si ≥ 95 % à N = 8 pour ≥ 2 graines sur 3. La capacité est le plus grand N évalué où la médiane reste ≥ 90 %.
3. *S5* : « appris » si ≥ 95 % dans [0,24) pour ≥ 2 graines sur 3.
4. *La boucle fait la différence* : sur au moins une des trois tâches, `L_loop` dépasse `L_noloop` de ≥ 10 points (Dyck : [0,24) ou [96,256) ; clé-valeur : N = 8 ; S5 : [0,24)). Même réserve qu'à l'amendement 12 : `L_noloop` a moins de paramètres, pas de second transformer ; aucune conclusion causale sur la rétroaction.
5. *Mieux que la référence* : `L_loop` dépasse la meilleure de `T_rope`, `V_lam0` de ≥ 10 points sur au moins une tâche (même mesure).

**Attentes écrites d'avance** (hypothèses) : Dyck est la tâche la plus simple (compteur borné) et `L_loop` devrait l'apprendre et généraliser ; la clé-valeur est limitée par la capacité de l'image (256 valeurs) et on s'attend à une chute à un N modéré ; S5 est incertaine (non apprise par aucun modèle jusqu'ici).

**Limites.** Trois graines ; pas de recherche d'hyperparamètres ; l'effort est inégal entre modèles (temps de calcul par pas : `L_loop` ≈ 0,3 s, `L_noloop` ≈ 0,15 s, `V_lam0` ≈ 0,05 s) ; tâches synthétiques, pas du langage. Test de plomberie CPU (100 pas, non officiel) : aucune information sur la qualité.

## Amendement 14 (2026-11-20) : analyses sans entraînement sur les modèles sauvegardés (exploratoires, non pré-enregistrées)

Script `scripts/analyses_cpu.py`, résultats `results/analyses_cpu.json`. Texte : les 5 passages de 100 tokens écrits à la main (500 tokens), donc des écarts faibles ne sont pas lisibles ; les bpc absolus (0,75 pour `V_lam0`, 0,81 pour `V_lam1_ext`) ne sont pas comparables à ceux de TinyStories. Une seule exécution (bruit : 5 tirages).

**A. Bruit gaussien ajouté à la lecture** (σ en fraction de l'écart-type de chaque canal ; la perte monte de) : `V_lam0` +1 % (0,1), +7 % (0,3), +93 % (1,0) ; `V_lam1_ext` +4 %, +36 %, +157 %. L'hypothèse « la trajectoire continue est plus robuste au bruit » **n'est pas soutenue** : `V_lam1_ext` est plus sensible.

**B. Projection des frames sur leurs k premières directions principales** (par groupe de canaux, 192 valeurs) : bpc avec k = 1, 2, 4, 8, 16, 32 : `V_lam0` 2,46 / 2,34 / 1,95 / 1,55 / 1,15 / 0,89 ; `V_lam1_ext` 2,59 / 2,58 / 2,56 / 2,53 / 2,47 / 1,18 (sans projection : 0,75 et 0,81). Chez `V_lam1_ext`, 16 directions ne retiennent rien d'utile et 32 seulement l'essentiel : bien que la variance soit concentrée sur 2 à 3 directions (amendement des dimensions effectives), **l'information lue par le transformer vit dans des directions de faible variance**. L'idée de compression « garder quelques cellules » est donc **défavorable** à cette mesure : la continuité n'a pas produit un code compact. Limite : la projection est linéaire, sans réentraînement du lecteur ; un lecteur réentraîné sur peu de directions pourrait faire mieux (non testé).

**C. Positions apprises des taches de `D_blob_e1`** (4 096 tokens × 6 canaux × (x, y), comparées à 300 tirages aléatoires de même taille ; distance moyenne dans la classe contre le hasard) : nombres (one … ten, n = 10) 1,61 contre 1,97 (p = 0,003) ; suffixes sans espace (n = 988) 1,95 contre 1,98 (p < 0,003) ; ponctuation (n = 200) 1,94 contre 1,99 (p = 0,04) ; début de mot, majuscules, couleurs, animaux, famille : pas d'écart. Corrélation entre identifiant du token (fréquence BPE) et coordonnées : max |r| = 0,03. Lecture : **aucune organisation spatiale sémantique nette** ; seuls les nombres se rapprochent nettement (un seul groupe de 10 tokens, 8 classes testées sans correction) ; les écarts des suffixes et de la ponctuation sont de 2 à 3 % (réels peut-être, négligeables en taille).

## Artefacts

`results/ae_{bras}_L{L}.json` (courbe, métriques, profil d'oubli, continuité, paramètres), `..._samples.txt` (originaux vs reconstructions, `^` marque les erreurs), `..._latents.png` (bras image et cumul), `..._frames.png` (bras cumul : le film d'un morceau), `results/ae_summary.md` (tableau récapitulatif).

## Amendement 15 : contrôles CPU de la boucle (pré-enregistré après la graine 0 de L_loop/L_noloop, avant les autres)

Config réduite pour le CPU, identique pour tous les modèles : S3, entraînement sur 12 éléments, évaluation jusqu'à 96, d=64, batch 64, 2000 pas, 5 graines (0–4). Script : `scripts/cpu_boucle.py`, résultats `results/st_S3c_*`.

Modèles : `L_loop` (T1+T2, retour vers Image1) ; `L_noloop` (T1 seul) ; `L_wide` (T1 seul à 4 couches, paramètres ≈ L_loop) ; `L_i2` (T2 présent mais écrit dans Image2, pas de retour vers Image1 : mêmes paramètres, sans boucle).

Critères (précision médiane au-delà de la longueur d'entraînement, tranche 24–96) :
- **Boucle causale** si L_loop ≥ 95 % sur ≥ 4/5 graines, et L_wide et L_i2 < 60 % en médiane.
- **Capacité seule** si L_wide ou L_i2 atteint ≥ 95 % sur ≥ 3/5 graines.
- Sinon : indéterminé.
Tests causaux sur les L_loop entraînés (perturbation de T2 à l'inférence, sonde linéaire de l'état) : exploratoires, non critériés.

### Résultat de l'amendement 15
5 graines : L_loop 100 % (entraînement et 24–96) sur 5/5 ; L_noloop et L_wide → 17 % (hasard) au-delà de 24 ; L_i2 28–32 %. **Boucle causale : atteint** ; « capacité seule » : non. Exploratoire : bruit σ=1 sur T2 effondre le modèle (σ ≤ 0,3 sans effet) ; sonde linéaire de l'état 100 % sur Image1 et Image2. Détails : `rapport/4_boucle.md`.

## Amendement 16 : ablations de L_loop (pré-enregistré avant mesure)
Graine 0, config CPU de l'amendement 15 (S3c). Variantes : `A_nogate` (pas de porte d'oubli), `A_readI1` (lecture sur Image1), `A_p4` (patchs 4×4, 4 patchs), `A_1layer` (1 couche par transformer). Une graine : exploratoire. Lecture : un composant est « nécessaire » si la précision au-delà de 24 tombe sous 60 % (L_loop de référence : 100 %).

### Résultat de l'amendement 16
Au-delà de 48 : 1 couche 100 % ; patchs 4×4 31 % ; sans porte 16 % ; lecture Image1 16 %. Nécessaires : porte, lecture sur Image2 (sous 60 %). Une graine.

## Amendement 17 : S3 + mémoire longue (pré-enregistré avant toute mesure)
Tâche `S3M` (hypothèse de Mathis : la boucle garde une mémoire longue dans certains canaux pendant qu'elle suit S3). Séquence : [valeur v∈{0..7}] [n éléments de S3] [requête]. Cibles : état S3 courant à chaque élément ; à la requête, la valeur v. Entraînement n ∈ [4,24] (curriculum), évaluation n = 12, 24, 48, 96. Config CPU de l'amendement 15 (d=64, 3000 pas, batch 64), 3 graines, modèles `L_loop1` (1 couche), `L_noloop1`, `T_rope`.
Critères : (a) **mémoire tenue** si le rappel de L_loop1 ≥ 90 % à n=48 sur ≥ 2/3 graines ET la précision S3 reste ≥ 95 % ; (b) **durée de vie** = plus grand n avec rappel ≥ 90 % ; (c) « la boucle aide » si L_noloop1 et T_rope sont < 60 % de rappel à n=48 pendant que (a) est vrai. Pronostic : rappel fiable à quelques dizaines de pas, dégradé vers 96.
Exploratoire (non critérié) : sonde linéaire de v sur Image1 et Image2 pendant le délai, globale et par canal. Trois scénarios : (i) l'info transite en permanence entre les deux images (lisible sur Image1 et Image2 à chaque pas) ; (ii) elle « patiente » sur Image2 et revient dans Image1 seulement vers la requête (lisible sur Image2 seulement pendant le délai) ; (iii) elle est stockée dans quelques canaux (sonde par canal concentrée).

### Résultat de l'amendement 17
(a) atteint (rappel ≥ 98 % à n=96, S3 100 %, 3/3) ; (b) ≥ 96 ; (c) non atteint à n=48 (L_noloop1 80–96 %, T_rope 9–75 %) mais net à n=96 et sur S3. Sondes : valeur lisible sur Image1 et Image2 tout le délai. Détails : rapport/5_memoire.md.

## Amendement 18 : taille de l'image et distance à partir de laquelle la boucle est nécessaire (pré-enregistré avant mesure)
Hypothèse de Mathis : une image plus petite fait apparaître plus tôt le besoin de la boucle pour la mémoire longue. Tâche S3M, config de l'amendement 17 (3 000 pas, 3 graines). Tailles : 4×8×8 (déjà mesurée, amendement 17), 4×4×4 (64 valeurs), 2×4×4 (32 valeurs) ; modèles `L_loop1_*` et `L_noloop1_*`. Évaluation n = 12, 24, 32, 48, 64, 96 (32 et 64 absents de la référence 4×8×8, comparée aux points communs).
Mesure : n*(noloop) = plus petit n où le rappel médian de L_noloop1 passe sous 90 %. **Hypothèse soutenue** si n* décroît quand l'image rétrécit (n*(2×4×4) < n*(4×8×8)) ET si L_loop1 garde rappel ≥ 90 % à ces n sur ≥ 2/3 graines. Si L_loop1 s'effondre aussi : l'image trop petite limite la mémoire globale (hypothèse non testable ainsi). Sinon : réfutée.

### Résultat de l'amendement 18
n*(noloop) : 4×8×8 entre 48 et 96 ; 4×4×4 et 2×4×4 : 48. L_loop1 ≥ 90 % partout sauf 1 graine à 84 % (2×4×4, n=96). Hypothèse soutenue faiblement (moyenne à n=48 : 89/78/67 %, non monotone à n=96, forte variance). Détails : rapport/5_memoire.md.

## Amendement 19 : petit modèle de langage sur corpus synthétique d'histoires (pré-enregistré avant mesure)
Pas de TinyStories dans le bac à sable CPU : corpus généré (`scripts/cpu_langage.py`), vocabulaire de 29 mots. Histoire : 3 personnes (parmi 6) tiennent chacune un objet (parmi 6) et sont dans un lieu (parmi 4) ; E événements (« A swaps with B . » échange les objets ; « A goes to P . » déplace) ; 3 questions finales (« who has the O ? X . », « where is A now ? P . »). Longueur 51+5E tokens. Entraînement E ∈ [4,10] ; évaluation E = 8 (dans la distribution), 16, 32 (extrapolation).
Entraînement : perte de langage sur tous les tokens, poids ×5 sur le token-réponse (choix de conception, déclaré). Modèles : `L_loop1`, `T_rope`, `V_lam0` ; d=64, 3000 pas, batch 32, **1 graine** (exploratoire, 2e graine si le résultat est net).
Métrique : précision sur les tokens-réponse (hasard : 1/3 « who », 1/4 « where »), détaillée par type.
Critères : « la boucle aide » si L_loop1 ≥ 90 % de précision-réponse à E=32 ET T_rope < 60 % à E=32. « Équivalent » si T_rope ≥ 90 % à E=32 aussi. Pronostic : L_loop1 ≥ 90 % à E=8 et dégradation douce à 32 ; T_rope bon à E=8, s'effondre à 32 (comme S3).
Limite assumée : langage synthétique, pas un test du langage naturel.

### Amendement 19b : v2 de l'entraînement (pré-enregistré après l'échec de la v1, avant mesure)
Résultat v1 (3000 pas, E ∈ [4,10], poids ×5) : les trois modèles à 33–43 % de réponses justes à E=8, hasard : **aucun verdict sur les architectures** : cause identifiée après coup = bug du générateur (les histoires d'un lot de longueurs différentes étaient tronquées à la plus courte, donc la plupart perdaient leurs questions finales). v2 : correction du bug (un seul E par lot) et curriculum sur E (E_max passe de 1 à 10 pendant les 40 premiers % des pas, comme en S3), poids ×20 sur le token-réponse, 4000 pas. Critères et modèles inchangés (amendement 19). Si un modèle < 60 % à E=8 après v2 : non convergé, aucun verdict pour lui.

### Résultat de l'amendement 19b
T_rope et L_loop1 < 60 % à E=8 : non convergés, aucun verdict. V_lam0 : « où » 100 % (E=8) → 64 % (E=32), « qui » 32 %. Détails : rapport/6_langage.md.

## Amendement 20 : modèle de langage réel sur CPU : Wikitext-2 (pré-enregistré avant mesure)
Corpus : Wikitext-2 nettoyé (2,9 M de tokens BPE 4096 d'entraînement, 0,29 M de validation). Modèles : `T_rope`, `V_lam0`, `L_loop1` (image 4×8×8, 1 couche), d=64, 2 couches pour T/V, **fenêtres d'entraînement de 64 tokens**, batch 32, 2000 pas, 1 graine, aucune frame transportée entre fenêtres. Métrique : bits par caractère (bpc) sur la validation, par tranche de position dans des fenêtres de 256 tokens : [0,64) (entraînement), [64,128), [128,256) (extrapolation). Référence : unigramme (entropie des fréquences de tokens).
Critères : (a) un modèle « apprend » s'il bat l'unigramme d'au moins 20 % en [0,64) ; (b) « la boucle est compétitive » si bpc(L_loop1) ≤ 1,05 × bpc(T_rope) en [0,64) ; (c) « la boucle extrapole mieux » si, en [128,256), bpc(L_loop1) < bpc(T_rope) ET bpc(L_loop1) ≤ 1,10 × bpc(L_loop1 en [0,64)). Pronostic : T_rope meilleur dans la longueur d'entraînement ; V et L_loop1 dégradent moins au-delà de 64 ; la boucle n'est pas compétitive en bpc (état de 256 valeurs).
Limites : 1 graine, budget minuscule (2000 pas ≈ 2 M de tokens vus), Wikitext-2 petit.

### Résultat de l'amendement 20
bpc [0,64) : T_rope 2,086, L_loop1 2,139, V_lam0 2,204 (unigramme 2,747). (a) V_lam0 échoue de justesse, (b) atteint, (c) non atteint (2,102 vs 2,096 en [128,256)). Pronostic partiellement infirmé : la boucle est compétitive, T_rope ne dégrade pas au-delà de 64. Détails : rapport/6_langage.md.

## Amendement 21 : temps de réflexion : la boucle utilise-t-elle ses transitions pour raisonner ? (pré-enregistré avant toute mesure, 2026-11-21)

**Idée (Mathis).** Les images sont des mémoires, les transformers des transitions entre elles ; raisonner serait laisser l'activité circuler dans ce réseau avant de répondre, sans produire de tokens de pensée. Premier test, le plus petit : laisser la boucle à deux images tourner **k tours sans entrée nouvelle** avant sa réponse, et voir si plus de tours donne de meilleures réponses sur un problème qui demande plusieurs étapes **après** la lecture.

**Tâche : poursuite de pointeurs** (`scripts/cpu_reflexion.py`). Une permutation aléatoire f de 6 symboles, donnée par ses 6 paires (a, f(a)) dans un ordre aléatoire (12 tokens), puis un jeton de requête, un jeton de nombre de sauts h, un symbole de départ x, puis k jetons de pause, et la cible f^h(x) (lue à la dernière position, seule position supervisée). La requête arrive à la fin : le calcul ne peut pas être fait pendant la lecture. Hasard : 1/6 (16,7 %). Vocabulaire : 6 symboles, 6 jetons de sauts (h = 1 à 6), requête, pause.

**Entraînement.** h tiré dans 1..4 (curriculum : h_max passe de 1 à 4 pendant les 40 premiers % des pas), k tiré uniformément dans 0..8 par lot. 4 000 pas, lot 64, d = 64, mêmes réglages d'optimisation que `cpu_memoire.py`, 3 graines (0, 1, 2), CPU.

**Modèles.**
- `L_loop1_k` : boucle à une couche (image 4×8×8), réflexion = k jetons de pause (un jeton appris comme les autres : il se dépose sur l'image 1 puis T1 et T2 tournent normalement).
- `L_loop1_k0` : même modèle, entraîné avec k = 0 toujours (témoin sans temps de réflexion, mêmes paramètres).
- `L_noloop1_k` : témoin sans retour T2, avec pauses.
- `T_rope_k` : transformer 2 couches, d = 64, avec les mêmes jetons de pause (« pause tokens »).

**Évaluation** (512 suites par case, graine d'évaluation fixe) : précision sur la grille h ∈ {1, …, 6} × k ∈ {0, 1, 2, 4, 6, 8, 12}. h = 5, 6 et k = 12 sont hors entraînement.

**Critères pré-enregistrés.**
- (0) *Pas de verdict* pour un modèle dont la précision à h = 1, k = 8 reste < 60 % sur au moins 2 graines sur 3 (non convergé).
- (a) *La boucle apprend la tâche* : `L_loop1_k` ≥ 90 % pour h = 1 à 4 à k = 8, sur ≥ 2 graines sur 3.
- (b) **Signature du raisonnement** : pour `L_loop1_k`, à h = 3 et à h = 4, précision(k = 8) − précision(k = 0) ≥ 20 points, sur ≥ 2 graines sur 3.
- (c) *La réflexion aide* : à h = 4, `L_loop1_k` (k = 8) dépasse `L_loop1_k0` de ≥ 20 points sur ≥ 2 graines sur 3.
- (d) *C'est la boucle* : à h = 4, k = 8, `L_loop1_k` dépasse `L_noloop1_k` de ≥ 20 points sur ≥ 2 graines sur 3.
- (e) *Extrapolation du raisonnement* (exploratoire) : `L_loop1_k` ≥ 60 % à h = 5 ou 6 avec k = 12.
- `T_rope_k` est une référence : rapporté, sans critère.

**Pronostic (écrit avant).** (a) incertain : il faut aussi stocker 6 paires dans 256 valeurs, ce que la boucle n'a jamais fait (mémoire d'une seule valeur en partie 5). Si (a) est atteint, on attend (b) et (c), car un saut demande au moins un tour de T1 → T2. (d) attendu. (e) peu probable. T_rope_k devrait bien faire à h petit (il relit les paires par attention) et décrocher à h grand.

**Limites.** Une seule tâche, très petite ; 3 graines ; un jeton de pause appris n'est pas exactement « aucune entrée » ; pas de choix de chemin (toutes les transitions tournent à chaque tour) : on teste seulement l'ingrédient « temps de réflexion ». Test de plomberie autorisé (≤ 200 pas, une graine), non rapporté.

## Amendement 22 : mêmes critères que l'amendement 20 sur cinq autres corpus (pré-enregistré avant mesure)
*Numérotation : ce texte a été écrit avant les mesures dans une copie de travail où il portait le numéro 21 ; il est renuméroté 22 car l'amendement 21 (temps de réflexion) a été ajouté entre-temps sur `main` par un autre contributeur.*

Protocole identique à l'amendement 20 (T_rope, V_lam0, L_loop1 ; d=64 ; fenêtres de 64 tokens ; 2000 pas ; 1 graine ; évaluation sur fenêtres de 256 tokens par tranche [0,64) / [64,128) / [128,256) ; critères (a), (b), (c) inchangés, seuils recalculés sur l'unigramme de chaque corpus). Corpus : `code_python`, `gutenberg_contes`, `ptb`, `gutenberg_tout`, `shakespeare`.
Pronostics (écrits avant mesure) : sur le code Python, la boucle n'est pas compétitive ((b) non atteint) mais extrapole mieux ; sur `gutenberg_contes` (0,38 M tokens, ≈ 10 époques en 2000 pas), tous surapprennent ; ailleurs, le schéma de Wikitext-2 se répète (boucle à ~2–5 % de T_rope, pas d'extrapolation). 1 graine : les écarts < 5 % ne sont pas interprétés.

### Résultat de l'amendement 22
bpc [0,64) / [128,256) :

| corpus | unigramme | T_rope | V λ=0 | L_loop1 |
|---|---|---|---|---|
| code_python | 2,752 | 1,739 / 1,844 | 1,787 / 1,817 | 1,795 / 1,791 |
| gutenberg_contes | 2,758 | 2,051 / 2,199 | 2,104 / 2,059 | 2,071 / 2,016 |
| ptb | 2,255 | 1,622 / 1,635 | 1,675 / 1,656 | 1,604 / 1,578 |
| gutenberg_tout | 2,828 | 2,190 / 2,494 | 2,267 / 2,318 | 2,229 / 2,278 |
| shakespeare | 3,075 | 2,338 / 2,603 | 2,433 / 2,416 | 2,425 / 2,414 |

Critères pour L_loop1 : (a) apprend, (b) compétitive (≤ 1,05 × T_rope) et (c) extrapole mieux sont **atteints sur les cinq corpus**. Pour V λ=0 : (a) manqué de peu sur gutenberg_tout (2,267 contre un seuil à 2,262) ; (c) manqué sur ptb. Pronostics : « la boucle n'est pas compétitive sur le code » **infirmé** (+3,2 %), « extrapole mieux sur le code » confirmé ; « pas d'extrapolation ailleurs » **infirmé** sur 4 corpus ; « surapprentissage sur les contes » non constaté (la validation reste correcte). Une graine, 2000 pas : lire les écarts < 5 % avec prudence. Rapport : partie 6.3.

### Résultat de l'amendement 21 (2026-11-21)
12 runs (4 modèles × 3 graines, 4 000 pas). Détail : `results/rf_summary.md`, `results/rf_*.json`, script `scripts/cpu_reflexion.py`.
- **Critère (0) : aucun modèle ne converge**, à h = 1, k = 8, tous restent au hasard (L_loop1_k 16–18 %, L_loop1_k0 15–18 %, L_noloop1_k 24–29 %, T_rope_k 14–20 %). **Pas de verdict** sur (a) à (e). Pour mémoire : L_loop1_k reste au hasard partout (perte ≈ log 6) ; écarts k = 8 − k = 0 entre −3 et +6 points.
- **Défaut de la tâche (erreur de conception, pas du code).** Pour une permutation aléatoire de n symboles, un point donné est dans un cycle de longueur L avec probabilité 1/n pour chaque L ; donc P(f^h(x) = x) = (nombre de diviseurs de h ≤ 6) / 6 : 17 / 33 / 33 / 50 / 33 / 67 % pour h = 1 … 6. L_noloop1_k et T_rope_k reproduisent exactement ces fréquences (≈ 31 / 32 / 48 % à h = 2 / 3 / 4, 3 graines quasi identiques) : ils ont appris « répondre x », pas la poursuite de pointeurs. Même le saut simple (h = 1) n'est appris par aucun modèle.
- **Leçon** : vérifier qu'une tâche n'a pas de raccourci et qu'un transformer de référence l'apprend, avant de tester la boucle. Suite prévue : amendement 21b avec une fonction aléatoire (pas une permutation) sur davantage de symboles, après un test de faisabilité sur T_rope.

## Amendement 23 : réseau de mémoires, expérience 1 : temps de réflexion sur une liste circulaire (pré-enregistré avant toute mesure de l'expérience, 2026-11-21)

**Étape A (faisabilité, déjà mesurée avant cet amendement, à titre de conception).** Tâche : *poursuite de pointeurs sur une liste circulaire* (`scripts/reseau_tache.py`). f est une permutation cyclique unique de n symboles, donnée par ses n paires (a, f(a)) en ordre aléatoire (2n tokens), puis un jeton de requête, x, un jeton h, puis k jetons de pause ; la réponse est f^h(x), lue à la dernière position. *Raccourcis* (précision maximale d'un prédicteur qui n'enchaîne pas les h sauts, 200 000 tirages) : **0,00 à tout h pour la liste circulaire** (n = 8 à 32) ; pour une fonction aléatoire, 0,38–0,54 à n = 12 et 0,17–0,26 à n = 32 (à h ≥ 2) : une fonction aléatoire est écartée. *Faisabilité* (T_rope, 2 couches, d = 64, 1 graine) : h = 1 appris à 100 % pour n = 6 et 87 % pour n = 8 en 6 000 pas, **pas appris à n = 12 en 1 500 pas** (hasard) ; à 2 couches h = 2 reste au hasard (16 %), même avec 8 pauses et 12 000 pas ; à 4 couches h = 2 atteint 48 % en 12 000 pas. La boucle (n = 8, h = 1) atteint 66 % en 4 000 pas. Décision : **n = 6**.

**Tâche de l'expérience.** n = 6, liste circulaire, h ∈ {1, 2, 3} à l'entraînement (curriculum : h_max passe de 1 à 3 pendant les 40 premiers % des pas), k tiré uniformément dans 0..8 par lot (pour les modèles « avec réflexion »). Évaluation : grille h ∈ {1, …, 5} × k ∈ {0, 2, 4, 8, 12}, 512 suites par case ; h = 4, 5 et k = 12 sont hors entraînement ; h ≤ 5 car la liste reboucle à h = 6.

**Modèles** (d = 64, une couche par transformer, 8 000 pas, lot 64, 3 graines 0–2) : `L_loop1_k` (boucle à deux images 4×8×8, k jetons de pause), `L_loop1_k0` (même modèle entraîné avec k = 0 toujours), `L_noloop1_k` (sans retour T2, avec pauses), `T_rope_k` (2 couches, jetons de pause ; référence, sans critère). Script : `scripts/reseau_exp.py`, résultats `results/reseau_*_s*.json`.

**Critères** (« ≥ 2/3 » = au moins 2 graines sur 3) :
- (0) *Pas de verdict* pour un modèle dont la précision à h = 1, k = 8 reste < 60 % sur ≥ 2/3 graines (non convergé).
- (a) *La boucle apprend* : `L_loop1_k` ≥ 90 % à h = 1, 2, 3 avec k = 8, sur ≥ 2/3.
- (b) **Signature de la réflexion** : pour `L_loop1_k`, à h = 2 et h = 3, précision(k = 8) − précision(k = 0) ≥ 20 points, sur ≥ 2/3.
- (c) *La réflexion aide* : à h = 3, `L_loop1_k` (k = 8) dépasse `L_loop1_k0` de ≥ 20 points, sur ≥ 2/3.
- (d) *C'est la boucle* : à h = 3, k = 8, `L_loop1_k` dépasse `L_noloop1_k` de ≥ 20 points, sur ≥ 2/3.
- (e) *Extrapolation* (exploratoire) : `L_loop1_k` ≥ 60 % à h = 4 ou 5 avec k = 12.

**Pronostic (écrit avant).** La boucle apprend h = 1 (probable) ; (a) incertain, ma probabilité est d'environ 40 % à 8 000 pas, vu les 66 % à h = 1 pour n = 8 en 4 000 pas. Si (a) est atteint, je donne environ 50 % à (b) et (c) : une boucle qui chaîne un saut par tour doit profiter des pauses, mais rien n'oblige les transitions à apprendre ce schéma. T_rope_k : h = 1 à ~100 %, h ≥ 2 proche du hasard (profondeur limitée), sans effet des pauses. Si (a) échoue, le résultat est « pas de verdict », pas « pas de raisonnement ».

**Limites.** Une tâche, très petite (6 symboles) ; un jeton de pause appris n'est pas exactement « aucune entrée » ; toutes les transitions tournent à chaque tour (pas de choix de chemin). L'expérience 2 (réseau à trois mémoires) n'est lancée que si (b) et (c) sont atteints, et fera l'objet d'un amendement séparé avec contrôle à nombre de valeurs et de paramètres égaux.

## Amendement 24 : bAbI : une tâche classique de questions-réponses (pré-enregistré avant toute mesure, 2026-11-21)

**Question.** Sur un banc d'essai classique de mémoire et de raisonnement en plusieurs étapes, où se place la boucle par rapport à un transformer, à la vidéo et au témoin sans boucle ? Et, si elle peine, le temps de réflexion (k tours sans entrée) aide-t-il là où l'amendement 21 n'a rien pu dire ?

**Données.** bAbI v1.2, en-10k, tâches 1 (1 fait utile), 2 (2 faits), 3 (3 faits), 6 (oui/non), `data/babi/`. Tokens = mots en minuscules + « . » et « ? » ; une suite = l'histoire jusqu'à la question (phrases précédentes, sans les autres questions ni les numéros), puis la question ; réponse = un mot, prédite à la dernière position (seule position supervisée, sur les mots-réponses possibles). Les suites plus longues que **128 tokens** sont coupées à gauche (garde les 128 derniers) ; mesuré avant : les faits utiles sont dans les 128 derniers tokens pour 99,6 % des questions de la tâche 2 et 96,7 % de la tâche 3 (≈ 100 % pour 1 et 6). Remplissage à gauche par un jeton PAD.

**Étape 0, faisabilité (avant les runs officiels, rapportée).** `T_rope` sur la tâche 1 seule, 1 000 pas : doit atteindre ≥ 90 % en test. Sinon, arrêt et révision de la tâche (pas de runs officiels).

**Étape 1, sans réflexion.** Entraînement **joint** sur les 4 tâches (une tâche tirée par lot), lot 32, 4 000 pas, d = 64, réglages d'optimisation de `cpu_memoire.py`, 3 graines. Modèles : `T_rope` (2 couches), `V_lam0`, `L_loop1` (image 4×8×8, 1 couche), `L_noloop1`. Évaluation : précision sur les 1 000 questions de test de chaque tâche.
- (a) une tâche est **réussie** par un modèle si sa précision médiane ≥ 95 % (seuil usuel de bAbI) ;
- (b) **la boucle est compétitive** sur une tâche si sa médiane est à ≤ 5 points de `T_rope` ;
- (c) **la boucle aide** sur une tâche si elle dépasse `L_noloop1` de ≥ 10 points (médianes), attendu sur 2 et 3 ;
- références rapportées sans critère : hasard (tâches 1–3 : 1/6 des lieux ; tâche 6 : 50 %).

**Étape 2, temps de réflexion (conditionnelle).** Déclenchée pour **chaque tâche, y compris la tâche 1**, où la médiane de `L_loop1` est < 95 % à l'étape 1. Modèle `L_loop1_k` : même entraînement joint, avec k jetons de pause après la question, k tiré uniformément dans 0..8 par lot ; 3 graines. Évaluation des tâches déclenchées à k ∈ {0, 2, 4, 8, 12}.
- (d) **signature du raisonnement** : précision(k = 8) − précision(k = 0) ≥ 10 points sur ≥ 2 graines sur 3 ;
- (e) **la réflexion aide** : `L_loop1_k` à k = 8 dépasse `L_loop1` (étape 1) de ≥ 10 points (médianes).

**Pronostic (écrit avant).** Tâches 1 et 6 réussies par T_rope et la boucle ; tâche 2 plus dure pour la boucle (il faut retenir qui tient quoi et où est chacun dans 256 valeurs) ; tâche 3 ratée par tout le monde à ce budget. Si l'étape 2 est déclenchée, (d) incertain.

**Limites.** Modèles minuscules, pas de pré-entraînement, budget CPU (≈ 1 h par run de boucle) ; coupure à 128 tokens ; entraînement joint sur 4 tâches seulement.

### Résultat de l'amendement 24, étape 0 (2026-11-21)
**Faisabilité non atteinte : arrêt avant les runs officiels.** `T_rope` sur la tâche 1 seule, 1 000 pas : 51,4 % en test (seuil 90 %) ; `results/babi_T_rope_s0_t1.json`, script `scripts/cpu_babi.py`.
Diagnostic exploratoire (hors protocole, validation = 1 000 dernières questions du train, une graine) : le plafond ne vient pas du budget ni de la taille. d = 64 sur 4 000 et 8 000 pas, d = 128 (2 couches, lot 64), lr 3·10⁻³ : tous plafonnent entre 46 et 52 %, même quand la perte d'entraînement baisse (d = 128 : perte 0,70, validation 47 %). Une règle triviale, « répondre le dernier lieu cité dans l'histoire », fait 52,7 % sur le test de la tâche 1 : c'est très probablement ce que le modèle apprend (la vraie règle, « dernier lieu de la dernière phrase qui nomme la personne », fait 100 %). Pas d'étape 1 ni 2. Suite à décider.

## Amendement 25 : ListOps : évaluer des expressions imbriquées (pré-enregistré avant toute mesure de modèle, 2026-11-21)

**Question.** Sur une tâche classique de raisonnement en plusieurs étapes (ListOps, Nangia et Bowman 2018, où les transformers sont notoirement faibles), où se place la boucle ? Généralise-t-elle à des profondeurs d'imbrication jamais vues ? Si elle peine, le temps de réflexion aide-t-il ?

**Tâche** (`scripts/listops_data.py`, générée). Expressions sur les chiffres 0–9 avec MAX, MIN, MED (médiane basse), SM (somme mod 10), 2 à 4 arguments par opérateur, profondeur exactement d, ≤ 64 tokens ; vocabulaire de 17 tokens ; réponse = un chiffre, prédite à la dernière position (seule position supervisée, sur les 10 chiffres). Remplissage à gauche.
**Références triviales mesurées avant** (20 000 expressions par profondeur ; `python scripts/listops_data.py`) : la meilleure règle simple (opérateur extérieur appliqué à ses seuls arguments chiffres) fait 100 / 32 / 25 / 25 / 25 % aux profondeurs 1 à 5 ; « dernier chiffre » 34 / 22 / 18 / 17 / 19 % ; classe majoritaire 12–14 %. Pas de raccourci au-delà de la profondeur 1.

**Entraînement.** Profondeur tirée dans 1..d_max par expression, d_max passant de 1 à 3 pendant les 40 premiers % des pas (curriculum), puis 3. Lot 32, 4 000 pas, d = 64, réglages d'optimisation de `cpu_memoire.py`, `T_rope` à 2 couches, boucles à 1 couche, image 4×8×8. **Évaluation** : 1 000 expressions par profondeur 1 à 5 (4 et 5 jamais vues), générateur à graine fixe.

**Étape 0, faisabilité.** `T_rope`, une graine. Passe si : ≥ 70 % à la profondeur 2 **et** ≥ 45 % à la profondeur 3 (meilleure règle triviale + 20 points). Sinon, `L_loop1`, une graine, mêmes seuils : si elle passe, la tâche est jugée apprenable (l'échec des transformers sur ListOps est connu) et l'étape 1 a lieu ; si aucun ne passe, arrêt.

**Étape 1.** `T_rope`, `V_lam0`, `L_loop1`, `L_noloop1`, graines 0, 1, 2 (les runs de l'étape 0 comptent comme graine 0).
- (a) *appris* : médiane ≥ 90 % à chaque profondeur 1 à 3 ;
- (b) *extrapolation en profondeur* : médiane ≥ 70 % aux profondeurs 4 et 5 ;
- (c) *la boucle aide* : `L_loop1` dépasse `L_noloop1` de ≥ 10 points (médianes) à la profondeur 3 ;
- (d) *la boucle bat le transformer* : `L_loop1` dépasse `T_rope` de ≥ 10 points à la profondeur 3 ; rapporté aussi aux profondeurs 4–5.

**Étape 2, temps de réflexion (conditionnelle).** Déclenchée si la médiane de `L_loop1` est < 90 % à la profondeur 3 ou < 70 % à la profondeur 4 ou 5. `L_loop1_k` : même entraînement, k jetons de pause à la fin, k tiré dans 0..8 par lot ; 3 graines ; évaluation aux profondeurs 1 à 5 pour k ∈ {0, 2, 4, 8, 12}.
- (e) *signature du raisonnement* : précision(k = 8) − précision(k = 0) ≥ 10 points à la profondeur 3 ou 4, sur ≥ 2 graines sur 3 ;
- (f) *la réflexion aide* : `L_loop1_k` à k = 8 dépasse `L_loop1` (étape 1) de ≥ 10 points à la profondeur 3 ou 4 (médianes).

**Pronostic (écrit avant).** `T_rope` apprend les profondeurs 1–2, faible à 3, rien au-delà. La boucle est récurrente comme une pile : (a) possible, (b) incertain (il faut garder jusqu'à 5 niveaux d'arguments en attente dans 256 valeurs). Le témoin sans boucle échoue dès la profondeur 2. Si l'étape 2 a lieu, (e) peu probable : la tâche se calcule pendant la lecture, la réflexion à la fin a peu à faire.

**Limites.** Expressions courtes (≤ 64 tokens, contre jusqu'à 2 000 dans Long Range Arena) ; une seule taille de modèle ; à la profondeur 5, la limite de longueur favorise les opérateurs à peu d'arguments.

### Résultat de l'amendement 25, étape 0 (2026-11-21)
**Faisabilité non atteinte : arrêt.** 4 000 pas, graine 0, 1 000 expressions par profondeur (`results/listops_T_rope_s0.json`, `results/listops_L_loop1_s0.json`, script `scripts/cpu_listops.py`) :

| modèle | paramètres | d1 | d2 | d3 | d4 | d5 |
|---|---|---|---|---|---|---|
| T_rope | 102 k | 80,0 | 51,8 | 38,7 | 36,1 | 33,9 |
| L_loop1 | 154 k | 80,1 | 53,5 | 38,6 | 40,6 | 39,1 |

Seuils (d2 ≥ 70 % et d3 ≥ 45 %) manqués par les deux ; étapes 1 et 2 non lancées. Les deux modèles dépassent la meilleure règle triviale (32 / 25 / 25 / 25 % de d2 à d5) de 13 à 16 points, et la boucle se dégrade un peu moins aux profondeurs jamais vues (40–39 % contre 36–34 %, une graine : non concluant). Hypothèse non testée : à la profondeur 1, où la règle triviale fait 100 %, les deux plafonnent à 80 %, ce qui correspond à trois opérateurs appris et la somme modulo 10 au hasard (75 + 25 × 10 % ≈ 77,5 %) ; l'arithmétique, plus que l'imbrication, bloquerait à ce budget.

## Amendement 26 : double boucle : deux images d'entrée, une image de sortie, quatre transformers (pré-enregistré avant toute mesure, 2026-11-22)
Idée de Mathis. `L_dloop1` : Image1 et Image2 (2×8×8 chacune, 128 valeurs) reçoivent chaque token par **deux read-in distincts** ; Image3 (4×8×8, comme l'image actuelle) est l'image de sortie, la prédiction est lue dessus. Quatre transformers à 1 couche, d=44 : A1 (Image1→3), A2 (Image2→3), B1 (3→Image1), B2 (3→Image2). Un cycle par token, pas de routage, pas de tours supplémentaires. Mémoire totale 512 valeurs = celle de `L_loop1` (2×4×8×8) : un éventuel gain ne vient pas d'une image plus grande, donc pas de témoin à grosse image. Paramètres : 149 775 contre 153 395 pour `L_loop1` (−2,4 %, défavorable au double loop). Calcul par token égal (4 applications à demi-poids contre 2 à plein poids). Code : `src/s6_double.py`, `scripts/cpu_double.py`.
Tâche S3M, config de l'amendement 17 (d_model 64, 3 000 pas, batch 64, curriculum n∈[4,24], évaluation n = 12…96), 3 graines (0, 1, 2). Référence : `results/st_S3M_L_loop1_s{0,1,2}.json` (même config, mêmes graines, déjà mesurées).
Critères, sur le rappel à n=96 et la précision S3 :
(a) **égal à L_loop1** si |rappel médian(L_dloop1) − rappel médian(L_loop1)| ≤ 5 points à n=96 ET S3 ≥ 95 % sur ≥ 2/3 graines ; **gain** si rappel(L_dloop1) ≥ rappel médian(L_loop1) + 5 points sur ≥ 2/3 graines ; **perte** si ≤ −5 points sur ≥ 2/3 graines.
(b) **read-in répartis** si, à n=24, annuler le dépôt sur Image1 puis sur Image2 (à l'inférence) coûte chacun ≥ 5 points de rappel ou de S3 sur ≥ 2/3 graines ; **un read-in inutile** si l'un des deux coûte < 2 points sur ≥ 2/3 graines. **Read-in redondants** si le cosinus absolu moyen entre les dépôts d'un même token (Image1/Image2) ≥ 0,8 sur ≥ 2/3 graines.
(c) Déclencheur de la variante `L_dloop1_dec` (pénalité de décorrélation 0,1·cos² entre les dépôts, 3 graines, mêmes critères) : si « perte », ou « un read-in inutile », ou « read-in redondants ». Sinon elle n'est pas lancée.
Exploratoire (non critérié) : sonde linéaire de la valeur v sur Image1, Image2, Image3 pendant le délai.
Pronostic : parité à ±5 points (le modèle a autant de mémoire et de calcul) ; read-in non redondants à l'initialisation (cos ≈ 0,1 sur la graine de test). Le routage (MLP qui choisit l'arête suivante, plusieurs cycles par token) sera pré-enregistré dans un amendement 26b avant sa mesure, après lecture de ces résultats.

### Résultat de l'amendement 26 (2026-11-22)
Critères appliqués tels qu'écrits. (a) **égal à L_loop1** : rappel à n=96 = 0,988 / 0,914 / 0,984 (médiane 0,984) contre 0,980 / 1,000 / 0,977 (médiane 0,980) ; S3 = 100 % partout. (b) **read-in répartis** et non redondants : annuler le dépôt 1 ou le dépôt 2 coûte à chaque fois ≥ 5 points sur 3/3 graines ; cosinus absolu moyen entre dépôts 0,08 / 0,12 / 0,05 (≪ 0,8). (c) variante `L_dloop1_dec` **non déclenchée**.
Constats hors critères : (i) la tâche est saturée (S3M ≥ 98 % pour les deux modèles sur 5 points sur 6), donc « égal » veut dire « pas de régression », pas « la topologie aide » ; (ii) les deux read-in **se spécialisent spontanément** sur 2 graines et en sens opposés (graine 0 : le dépôt 2 porte la valeur à retenir, rappel 2 % sans lui, et le dépôt 1 porte l'état S3, S3 19 % sans lui ; graine 1 : rôles inversés) ; graine 2 : rôles partagés (rappel 41/47 % sans l'un ou l'autre) ; (iii) la valeur reste lisible à 97–100 % par sonde sur les trois images pendant tout le délai ; (iv) 2× plus lent à entraîner (1 210 s contre 667 s pour L_loop1, 149 775 contre 153 395 paramètres). Résultats : `results/dl_S3M_L_dloop1_s{0,1,2}.json`. Le routage (amendement 26b) reste à pré-enregistrer, sur une tâche non saturée.

## Amendement 27 : S4 : une tâche de suivi d'état non saturée pour la boucle et le double loop (pré-enregistré avant toute mesure, 2026-11-22)
Motif : S3 et S3M sont saturées (amendement 26, §constats) ; S5 n'est apprise par personne (partie 3). S4 (24 états, hasard 4,2 %) est entre les deux, `L_loop1` y est inconnue. Tâche : prédire à chaque pas le produit partiel d'une séquence d'éléments de S4 ; entraînement n ∈ [4, 24] (curriculum 40 %), évaluation n = 12, 24, 48, 96, 192 (précision moyenne sur toutes les positions). Config CPU de l'amendement 17 (d_model 64, batch 64, 4 000 pas), modèles `T_rope`, `L_loop1`, `L_dloop1` (149 775 paramètres, amendement 26). Code : `scripts/cpu_s4.py`.
Étape 0 (une graine, graine 0). **Tâche exploitable** si au moins un des trois modèles a une précision ≥ 50 % à n=24 ; sinon arrêt, aucun changement de protocole. **Tâche non saturée** si aucun modèle n'a ≥ 95 % à la fois à n=96 et à n=192. Si exploitable et non saturée : étape 1 = graines 1 et 2 pour les modèles qui dépassent 50 % à n=24 ; amendement 26b (routage) écrit ensuite. Si l'un des modèles est ≥ 95 % à n=96 et 192, la tâche est saturée pour lui : on passe à S5 réduit ou à S3M plus long (à décider, nouvel amendement).
Pronostic : T_rope apprend n ≤ 24 et s'effondre au-delà (Liu et al., 2023) ; `L_loop1` apprend plus lentement que sur S3 ; `L_dloop1` à peu près comme `L_loop1`.

### Résultat de l'amendement 27, étape 0 (2026-11-22)
Précision à n = 12 / 24 / 48 / 96 / 192 (hasard 4,2 %), graine 0, 4 000 pas : `T_rope` 21 / 13 / 8,5 / 6,6 / 5,3 % ; `L_loop1` 29 / 19 / 13 / 11 / 9,7 % ; `L_dloop1` 29 / 16 / 10 / 7,4 / 5,7 %. Aucun modèle n'atteint 50 % à n=24 : **tâche non exploitable à ce budget, arrêt, étapes suivantes non lancées, aucun changement de protocole.** Les trois modèles sont au-dessus du hasard à n=12 mais loin de l'avoir appris ; les pertes d'entraînement baissaient encore lentement (L_loop1 : 2,16 à 4 000 pas contre 3,18 pour le hasard), donc le budget (4 000 pas, ~100 k paramètres) est une explication au moins aussi plausible que l'échec de l'architecture. Écart L_loop1 > T_rope (+6 points à n=24) : une graine, non interprétable. Résultats : `results/s4_*_s0.json`.

## Amendement 27b : S4 avec un budget de 15 000 pas (pré-enregistré avant mesure, 2026-11-22)
Suite de l'étape 0 non atteinte (4 000 pas). Mêmes tâche, config, modèles (`T_rope`, `L_loop1`, `L_dloop1`) et graine 0 ; seul le nombre de pas change : 15 000 (le curriculum 4→24 reste sur les 40 % premiers pas). Mêmes critères que l'amendement 27 : **exploitable** si au moins un modèle ≥ 50 % à n=24 ; **non saturée** si aucun modèle n'a ≥ 95 % à la fois à n=96 et n=192. Sinon arrêt, sans autre changement de protocole. Si exploitable et non saturée : graines 1 et 2 pour les modèles ≥ 50 % à n=24, puis amendement 26b (routage). Pronostic : `L_loop1` apprend (≥ 50 % à n=24) ; `T_rope` plafonne à n ≤ 24 et s'effondre au-delà ; `L_dloop1` à peu près comme `L_loop1`. Sorties : `results/s4_*_s0_p15000.json`.

### Résultat de l'amendement 27b (2026-11-22)
Graine 0, 15 000 pas, précision à n = 12 / 24 / 48 / 96 / 192 (hasard 4,2 %) : `T_rope` 73 / 40 / 22 / 13 / 8,5 % ; **`L_loop1` 100 / 100 / 100 / 100 / 100 %** ; `L_dloop1` 57 / 41 / 33 / 29 / 27 %. Critères appliqués tels qu'écrits : tâche **exploitable** (L_loop1 ≥ 50 % à n=24) ; **saturée pour `L_loop1`** (≥ 95 % à n=96 et n=192), non saturée pour `T_rope` et `L_dloop1`. `L_loop1` a atteint 100 % à n=12 et 24 avant 7 000 pas (perte 0,000) ; `L_dloop1` plafonne vers 41 % à n=24 dès ~8 000 pas (perte ≈ 0,9–1,1 jusqu'à la fin). Une graine : l'écart L_loop1 100 % contre L_dloop1 41 % est grand mais pourrait être un plateau d'optimisation propre à cette graine ; non interprété. Temps : 441 s (T_rope), 3 156 s (L_loop1), 4 998 s (L_dloop1). Résultats : `results/s4_*_s0_p15000.json`. Étape 1 (graines 1 et 2 de `L_dloop1`, et de `L_loop1` pour confirmer) et amendement 26b : à décider.

### Résultat de l'amendement 23 (2026-11-22)
12 runs (4 modèles × 3 graines, 8 000 pas, n = 6, liste circulaire). Tableau : `python scripts/reseau_resume.py` ; données `results/reseau_*_s*.json` ; rapport : `rapport/9_reflexion.md`.
- **(0)** `L_loop1_k`, `L_noloop1_k` : convergés (h = 1, k = 8 ≥ 60 % sur 3/3 graines). `T_rope_k` : h = 1 à 41, 91 et 94 % (une graine sous 60 %). `L_loop1_k0` : lecture du critère à k = 0, le seul k qu'il voit à l'entraînement (à k = 8 il est hors distribution) ; il converge sur 2 graines sur 3 (100 %), la graine 1 reste à ≈ 15–23 % (perte ≈ 1,8).
- **(a) atteint** : `L_loop1_k` = 100 % à h = 1, 2, 3 avec k = 8, 3/3 graines (pronostic : ≈ 40 %, trop pessimiste).
- **(b) non atteint (0/3)** : la précision à k = 0 est déjà 100 % à h = 2 et 3 ; k = 8 − k = 0 = 0 point. **Pas de signature de la réflexion.**
- **(c) non atteint** : lecture littérale 1/3, mais ce « 1 » est la graine où le témoin `L_loop1_k0` n'a pas convergé (≈ 20 % à k = 0), donc un écart d'optimisation et non un effet des pauses (lecture a posteriori, non prévue par le texte pré-enregistré) ; sur les 2 graines où le témoin converge, l'écart est de 0 point. Pronostic : (b) et (c) ≈ 50 % si (a) : infirmés.
- **(d) atteint (3/3)** : à h = 3, k = 8, `L_loop1_k` (100 %) dépasse `L_noloop1_k` (33–37 %) de 63 à 67 points ; sans T2, h = 1 = 96 %, h = 2 = 50–58 %.
- **(e) non atteint (0/3)** : h = 4 → 0–15 %, h = 5 → 0–6 % avec k = 12, hors entraînement.
- `T_rope_k` (référence) : h = 1 à 41–94 %, h ≥ 2 au hasard, les pauses ne changent rien (pronostic confirmé).

## Amendement 28 : Silverbox : la boucle contre un GRU, un LSTM et un transformer sur des données réelles (pré-enregistré avant toute mesure, 2026-11-22)

**Données.** Silverbox (circuit électronique à oscillateur de Duffing, benchmark d'identification non linéaire, 131 072 échantillons à ≈ 610 Hz), `SNLS80mV.csv` récupéré d'une copie publique (le téléchargement officiel est bloqué ici) ; V1 = entrée, V2 = sortie. Découpage : **test = échantillons 0–39 999** (entrée en « flèche » d'amplitude croissante : |u| atteint 0,149 dans le dernier quart contre 0,123 au maximum de l'entraînement), **entraînement = 40 000–121 071**, **validation = 121 072–131 071** ; moyennes et écarts-types calculés sur l'entraînement.
**Tâche.** Simulation : prédire y_t à partir des seules entrées passées (jamais de y en entrée). Entraînement sur fenêtres aléatoires de 96 pas, perte (EQM) sur les 64 derniers pas (les 32 premiers servent d'échauffement de l'état). Évaluation : fenêtres de 96 pas avec recouvrement de 32 (chaque échantillon prédit avec ≥ 32 pas d'historique), erreur en mV sur les échantillons t ≥ 96, en général et par quart de la plage de test. **Ces chiffres ne sont pas comparables à ceux publiés pour le benchmark** (simulation complète depuis un état initial connu).
**Modèles** (lot 32, lr 1e-3 cosinus avec échauffement de 100 pas, 1 graine, `scripts/silverbox.py`) : `lineaire` (moindres carres sur 32 et 96 entrées passées, référence), `GRU` et `LSTM` (une couche, 256 unités, ≈ 200 k et 265 k paramètres, 2 000 pas), `GRU_20k` et `LSTM_20k` (20 000 pas : calcul comparable à celui de la boucle), `T_rope` (2 couches, d = 64, ≈ 100 k paramètres, 2 000 pas), `L_loop1` (boucle à deux images 4×8×8, ≈ 159 k paramètres, 2 000 pas), `L_noloop1` (sans retour T2, ≈ 105 k paramètres, 2 000 pas). Entrée scalaire projetée par une couche linéaire à la place de l'embedding, sortie scalaire ; temps de calcul rapportés.
**Critères** (RMSE de test ; « meilleur RNN » = le plus bas de GRU et LSTM) :
- (a) un modèle *apprend la non-linéarité* si sa RMSE est < 0,5 × celle de la régression linéaire à 96 retards.
- (b) la boucle est *compétitive* si RMSE(L_loop1) ≤ 1,5 × meilleur RNN à nombre de pas égal (2 000).
- (c) la boucle *bat les RNN* si RMSE(L_loop1) < RMSE de chacun de GRU, LSTM (2 000 pas) **et** de GRU_20k, LSTM_20k (calcul comparable).
- (d) le *retour T2 compte* si RMSE(L_loop1) ≤ 0,8 × RMSE(L_noloop1).
- (e) exploratoire, sans critère : écart entre le premier quart (faible amplitude) et le dernier (amplitude hors de la plage d'entraînement).
**Pronostic (écrit avant).** (a) atteint par GRU et LSTM (≈ 85 %), par la boucle (≈ 60 %), par T_rope (≈ 40 %, fenêtre courte). (b) ≈ 40 %. (c) ≈ 10 % : un GRU entraîné sur ce type de signal lisse et peu bruité est un très bon modèle. (d) ≈ 50 %.
**Limites.** Une graine ; un budget minuscule ; fenêtres de 96 pas, donc un modèle ne peut pas exploiter plus de 96 pas d'historique ; comparaison à nombre de paramètres seulement approximatif (100–265 k) ; Silverbox est un seul système, assez lisse, sans l'état discret qui a motivé la boucle (S3).

## Amendement 29 : double loop à branche principale pleine : dilution du budget ou perturbation par la structure ? (pré-enregistré avant mesure, 2026-11-22)
Motif (idée de Mathis) : sur S4 à 15 000 pas, `L_loop1` atteint 100 % et `L_dloop1` plafonne à 41 % (amendement 27b, une graine). Deux explications : (H1) **dilution** : la tâche a besoin de beaucoup d'information et d'effort *au même endroit*, et le double loop les répartit sur deux branches à demi-taille ; (H2) **perturbation** : avoir deux entrées et une fusion (deux écritures dans Image3) gêne l'apprentissage, quel que soit le budget. Modèle `L_dloop1_full` : la branche 1 est exactement la boucle normale (Image1 4×8×8, transformers A1 et B1 à d=64, 1 couche, comme T1/T2 de `L_loop1`), à laquelle s'ajoute une branche annexe (Image2 2×8×8, A2 et B2 à d=44, 1 couche) ; Image3 4×8×8 comme avant ; le token se dépose sur Image1 et Image2 par deux read-in distincts ; prédiction lue sur Image3. Paramètres : 211 688 contre 154 844 pour `L_loop1` (surensemble de la boucle : le double loop a *plus* de budget, pas moins). Code : `src/s6_double.py` (arguments `c1, c2, d_edge2`), `scripts/cpu_s4.py L_dloop1_full`.
Protocole : S4, config de l'amendement 27b (15 000 pas, curriculum 4→24 sur 40 %), graines 0 et 1. Références : `L_loop1` graine 0 = 100 % partout, `L_dloop1` graine 0 = 41 % à n=24 (27b).
Critères (précision à n=24 et à n=192) : **H1 soutenue (dilution)** si les deux graines ont ≥ 90 % à n=24 ET ≥ 90 % à n=192 ; **H2 soutenue (perturbation)** si les deux graines ont ≤ 60 % à n=24 ; **non tranché** dans les autres cas (une graine réussit, l'autre non, ou résultats intermédiaires) : alors `L_loop1` graine 1 est lancée pour savoir si la graine 1 est difficile pour tous.
Exploratoire (non critérié) : si H1, annuler le dépôt sur Image2 à l'inférence (`dep_scale`) pour voir si la branche annexe sert à quelque chose.
Pronostic : H1 (≈ 55 %), H2 (≈ 25 %), non tranché (≈ 20 %). Limites : deux graines ; le plateau de `L_dloop1` n'a été vu que sur une graine ; la fusion Image3 reste identique à celle de `L_dloop1`.

### Résultat de l'amendement 29 (2026-11-22)
`L_dloop1_full` (211 688 paramètres), S4, 15 000 pas, précision à n = 12 / 24 / 48 / 96 / 192 : **graine 0 : 100 / 100 / 100 / 100 / 100 %** (100 % dès ~8 000 pas, perte 0,000) ; **graine 1 : 68 / 59 / 54 / 52 / 51 %** (plateau vers 57–59 % à n=24 depuis ~9 000 pas). Critères tels qu'écrits : H1 (dilution) non soutenue (graine 1 sous 90 %) ; H2 (perturbation) non soutenue (graine 0 à 100 %) ; **non tranché**. Constat factuel : avec la branche principale pleine, une graine sur deux apprend S4 parfaitement, alors que le double loop demi-taille (une graine) restait à 41 % ; la branche annexe n'empêche donc pas d'apprendre. La graine 1 plateau à ~59 % (au-dessus des 41 % de `L_dloop1` et des 40 % de `T_rope`) : peut-être une graine difficile pour toutes les architectures, ce que le suivi de l'amendement dit de vérifier.
**Suite, décidée avant mesure (2026-11-22)** : conformément à l'amendement 29, `L_loop1` graine 1 (15 000 pas) est lancée pour savoir si la graine 1 est difficile pour la boucle elle-même. Ajout, pour départager si le résultat reste ambigu : `L_dloop1_full` graine 2. Lecture prévue : si `L_loop1` graine 1 < 90 % à n=24, la variabilité entre graines est une propriété de S4 à ce budget et ne dit rien sur le double loop ; si `L_loop1` graine 1 ≥ 90 %, la branche annexe rend l'apprentissage moins fiable (1 graine sur 2 au lieu de 2 sur 2 pour la boucle, dans la limite d'un si petit échantillon). Résultats : `results/s4_L_dloop1_full_s{0,1}_p15000.json`.

### Complément à l'amendement 29 (écrit avant la mesure de `L_loop1` graine 2, 2026-11-22)
Pour avoir 3 graines pour chaque modèle (comparaison à graines égales), `L_loop1` graine 2 (S4, 15 000 pas, même config) est lancée pendant que `L_dloop1_full` graine 2 termine. Mesure déjà connue avant ce complément : `L_loop1` graine 1 = 88 / 69 / 61 / 55 / 52 % (n = 12 / 24 / 48 / 96 / 192), c'est-à-dire **< 90 % à n=24** ; `L_dloop1_full` graine 2 est à 100 % à n=12 et 24 sur ses évaluations intermédiaires (9 500 pas), résultat final non encore lu. Lecture prévue : « apprend S4 » = ≥ 90 % à n=24 et n=192 ; on rapporte le nombre de graines qui apprennent, pour chaque modèle, sans test statistique (3 graines).

### Résultat final de l'amendement 29 et de son complément (2026-11-22)
S4, 15 000 pas, précision à n = 12 / 24 / 48 / 96 / 192 (%). `L_loop1` : graine 0 = 100 / 100 / 100 / 100 / 100 ; graine 1 = 88 / 69 / 61 / 55 / 52 ; graine 2 = 99,8 / 100 / 99,4 / 100 / 99,9. `L_dloop1_full` : graine 0 = 100 partout ; graine 1 = 68 / 59 / 54 / 52 / 51 ; graine 2 = 100 partout. Référence : `L_dloop1` demi-taille graine 0 = 57 / 41 / 33 / 29 / 27 ; `T_rope` graine 0 = 73 / 40 / 22 / 13 / 8,5.
Avec « apprend S4 » = ≥ 90 % à n=24 et n=192 : **`L_loop1` 2 graines sur 3, `L_dloop1_full` 2 graines sur 3** (mêmes graines 0 et 2 ; la graine 1 échoue pour les deux, à ~52–55 % à n=96). La lecture prévue s'applique : `L_loop1` graine 1 est < 90 % à n=24, donc la variabilité entre graines est une propriété de S4 à ce budget et ne vient pas de la branche annexe. **La branche annexe ne perturbe pas l'apprentissage** (H2 non soutenue, même taux de réussite que la boucle). Constat sur H1 : le double loop demi-taille a échoué sur sa seule graine (41 %, graine 0 qui réussit pour `L_loop1` et `L_dloop1_full`), ce qui est compatible avec la dilution, mais **une graine ne suffit pas à l'établir** ; non tranché formellement (critère H1 écrit pour deux graines à ≥ 90 %). Les graines 1 et 2 de `L_dloop1` demi-taille permettraient de conclure. Moments d'apprentissage : décrochage entre ~5 000 et ~14 000 pas selon la graine (la graine 2 de `L_loop1` n'a décroché qu'à ~14 000 pas), donc 15 000 pas est un budget juste. Résultats : `results/s4_{L_loop1,L_dloop1_full}_s{0,1,2}_p15000.json`.

### Résultat de l'amendement 28 (2026-11-22)
RMSE de simulation (mV) test / validation, 1 graine : linéaire 96 retards 14,0 / 5,5 (32 retards 20,4 / 14,4) ; GRU 9,56 / 6,69 ; LSTM 9,62 / 6,68 ; GRU_20k 10,07 / 6,53 ; LSTM_20k 10,19 / 6,55 ; T_rope 15,40 / 9,66 ; L_noloop1 41,96 / 33,70 ; L_loop1 10,51 / 6,94 (1 862 s contre 131 s pour le GRU).
(a) **non** (aucun modèle sous 7 mV ; pronostic GRU/LSTM ≈ 85 % : infirmé) ; (b) **atteint** (10,51 ≤ 14,3) ; (c) **non** ; (d) **atteint** (10,51 ≤ 0,8 × 41,96). (e) l'erreur du dernier quart est 6 à 7 fois celle du premier pour tous les réseaux. En plage d'entraînement, le linéaire fait mieux que tous les réseaux sur la validation (5,5 contre 6,5–6,9). Rapport : `rapport/11_silverbox.md`.

## Amendement 30 : GRU et LSTM sur les tâches de mémoire de la boucle, et graines supplémentaires sur Silverbox (pré-enregistré avant mesure, 2026-11-22)
**Question.** Les RNN classiques réussissent-ils S3, S3 + mémoire longue (S3M) et la poursuite de pointeurs sur liste circulaire, tâches que la boucle réussit et que ses témoins sans retour T2 échouent ? Et l'écart de ≈ 10 % entre GRU et boucle sur Silverbox résiste-t-il à 3 graines ?
**Montage.** `scripts/rnn_taches.py` : GRU et LSTM à 1 couche, 256 unités, plongement 64 (≈ 250 à 330 k paramètres), mêmes données, pas, optimiseur, curriculum et évaluations que la boucle (S3 : entraînement 12, 2 000 pas, éval jusqu'à 96 ; S3M : 3 000 pas ; liste circulaire : n = 6, h ≤ 3, k aléatoire ≤ 8, 8 000 pas, éval h = 1…5). Graines 0, 1, 2. Silverbox : graines 1 et 2 pour GRU, LSTM, T_rope et L_loop1 (graine 0 déjà faite).
**Pronostics (avant mesure) et critères.**
- (S3) précision aux positions 48–96 ≥ 0,95 sur au moins 2 graines sur 3 : GRU/LSTM **80 %**.
- (S3M) rappel de la valeur à n = 96 ≥ 0,90 sur au moins 2 graines sur 3 : **55 %** (les RNN sont bons en mémoire courte, la longueur 96 est 4 fois l'entraînement).
- (PTR) précision à h = 3, k = 8 ≥ 0,90 sur au moins 2 graines sur 3 : **30 %** ; précision à h = 4 > 0,5 : **3 %**.
- (Silverbox) sur 3 graines, la RMSE moyenne du GRU est inférieure à celle de la boucle, et l'écart des moyennes dépasse l'écart-type maximal des deux : **60 %**.
Les écarts avec ces seuils seront rapportés tels quels, y compris si GRU/LSTM font aussi bien que la boucle.

### Résultat de l'amendement 30 (2026-11-22)
GRU (249 k params) et LSTM (332 k) sur 3 graines : S3 précision 48–96 = 0,9998 / 0,996 / 1,0 (GRU), 1,0 ×3 (LSTM) ; S3M rappel et S3 à n = 96 = 1,0 ×3 pour les deux ; liste circulaire h = 1, 2, 3 (k = 8) = 1,0 ×3 pour les deux, h = 4 : 0,02–0,12, h = 5 : 0,01–0,03. Durées 30–60 s (S3, S3M), 175–200 s (liste).
(S3) **atteint** (80 %) ; (S3M) **atteint** (55 %) ; (PTR h = 3 ≥ 0,9) **atteint** (pronostic 30 % : sous-estimé) ; (PTR h = 4 > 0,5) **non** (3 %). Silverbox, 3 graines, RMSE test : GRU 9,56 ± 0,02, LSTM 9,60 ± 0,03, boucle 10,82 ± 0,56, T_rope 15,29 ± 0,79 ; écart GRU-boucle 1,26 > 0,56 : **atteint** (60 %). Rapports : `rapport/11_silverbox.md` (11.4), `rapport/12_rnn_memoire.md`.

## Amendement 31 : S5 avec un réseau d'images en couches 2-3-1 (version légère) (pré-enregistré avant mesure, 2026-11-22)
**Question.** Un réseau de 6 images (couches de 2, 3 et 1 images), avec un transformer par transition orientée entre couches voisines (aucune arête au sein d'une couche), apprend-il S5 (120 états, hasard 0,8 %), que ni les transformers seuls (5 à 9 %) ni les images couplées n'ont appris ?
**Montage.** `src/s7_graphe.py`, `scripts/cpu_s5.py`. `G231` : images 4×4×4, patchs 2×2, transformers à 1 couche, d = 32, 2 têtes ; 18 transformers (6 entrée→intermédiaire, 3 intermédiaire→sortie, 3 retour sortie→intermédiaire, 6 retour intermédiaire→entrée), calculés par groupes (4 appels par token), ≈ 288 k paramètres. Par token : dépôt sur les 2 images d'entrée, balayage avant, lecture sur l'image de sortie, balayage retour. Témoins, mêmes données, pas et optimiseur : `G231_ff` (sans balayage retour, 160 k paramètres), `L_loop1` (boucle à 2 images), `GRU` (256 unités), `T_rope`. S5 : entraînement sur longueur 12 (curriculum de 4 à 12), 8 000 pas, batch 64, évaluation jusqu'à 96, **graine 0 seulement** (réplication seulement si un critère est atteint). Version dense non testée ici.
**Pronostics (avant mesure) et critères** (précision de n ≤ 12 puis de 48 à 96) :
- (a) `G231` apprend S5 : ≥ 0,90 à n ≤ 12 **et** ≥ 0,90 de 48 à 96 : **8 %** ; ≥ 0,90 à n ≤ 12 seulement : **20 %**.
- (b) `G231` dépasse `L_loop1` de ≥ 0,2 de 48 à 96 : **8 %**.
- (c) le retour compte : `G231_ff` < 0,5 × `G231` de 48 à 96 (si `G231` > 0,1) : **85 %** conditionnellement à (a).
- (d) `GRU` ≥ 0,90 à n ≤ 12 : **35 %** (pour situer, pas pour juger).
- (e) `T_rope` < 0,2 de 48 à 96 : **95 %**.
Un résultat nul (tous à moins de 0,2) sera rapporté comme tel, avec la remarque que 8 000 pas CPU sont un budget faible.

### Résultat de l'amendement 31 (2026-11-22)
S5, graine 0, 8 000 pas, précision à n ≤ 12 / 12–24 / 24–48 / 48–96 (hasard 0,8 %) : `G231` (288 k paramètres, 2 459 s) 0,090 / 0,008 / 0,009 / 0,008 ; `G231_ff` (160 k) 0,110 / 0,010 / 0,009 / 0,008 ; `L_loop1` (170 k) 0,091 / 0,008 / 0,009 / 0,008 ; `GRU` (286 k, 123 s) 0,127 / 0,008 / 0,009 / 0,008 ; `T_rope` (215 k) 0,096 / 0,012 / 0,010 / 0,008. La perte reste à 4,25–4,39 (ln 120 = 4,79) : seule la première position (triviale) est apprise (≈ 1/12 de précision sur n ≤ 12).
(a) **non** (pronostics 8 % et 20 %) ; (b) **non** ; (c) sans objet (G231 ≤ 0,1) ; (d) **non** (pronostic 35 %) ; (e) **atteint**. **Résultat nul : aucun modèle, GRU compris, n'apprend S5 à ce budget.** Les 8 000 pas CPU sont un budget manifestement trop faible (S4 n'a décroché qu'après 5 000–6 000 pas avec 15 000 au total) ; la conclusion sur l'architecture en couches reste ouverte.

## Amendement 32 : réseau d'images 2-3-1 sur d'autres tâches : S4 et pointeurs jusqu'à h = 5 (pré-enregistré avant mesure, 2026-11-22)
**Contexte.** L'amendement 31 (S5) est un résultat nul (aucun modèle n'apprend, GRU compris) : budget trop faible. On teste le même réseau `G231` (`src/s7_graphe.py`, 288 k paramètres) sur deux tâches où des références existent.
**(1) S4**, 15 000 pas, graine 0, `scripts/cpu_s4.py G231 0 15000`. Références existantes (amendement 27b) : `L_loop1` 100 % jusqu'à n = 192, `T_rope` 40 % à n = 24, double loop 41 %. Pronostics : `G231` ≥ 0,90 à n = 24 : **35 %** ; ≥ 0,90 à n = 192 : **25 %**.
**(2) Liste circulaire** (n = 6), entraînement jusqu'à **h = 5** (curriculum sur 40 % des pas), sans pauses, 8 000 pas, graine 0, `scripts/reseau_h5.py` ; évaluation h = 1…8 ; modèles `G231`, `L_loop1`, `GRU`. Hypothèse testée : la profondeur de calcul par token limite le nombre de sauts. Pronostics : `L_loop1` ≥ 0,90 à h = 5 : **25 %** ; `G231` ≥ 0,90 à h = 5 : **30 %** ; `GRU` ≥ 0,90 à h = 5 : **70 %** ; `G231` ≥ 0,5 à h = 6 (au-delà de l'entraînement) : **5 %**.
**Critères.** (a) `G231` ≥ 0,90 à h = 5 ; (b) `G231` dépasse `L_loop1` de ≥ 0,2 à h = 5 ; (c) `G231` ≥ 0,90 à n = 24 sur S4. Un échec sera rapporté tel quel.

## Amendement 33 : double loop sur d'autres tâches : S3 + deux valeurs, et liste circulaire (pré-enregistré avant toute mesure, 2026-11-22)
Demande de Mathis : essayer le double loop sur d'autres tâches. Modèle `L_dloop1_full` (amendement 29 : branche principale = boucle, branche annexe Image2 2×8×8 ; 211–215 k paramètres), comparé à `L_loop1`. On ne reprend pas S4 (amendement 32, autre conversation).
**(1) S3 + deux valeurs (`S3M2`)**, `scripts/cpu_s3m2.py`. Séquence [v1][v2] puis n éléments de S3 puis une requête Q1 ou Q2 (tirée au hasard) ; cibles : état S3 à chaque élément, et à la requête v1 (Q1) ou v2 (Q2). Config de l'amendement 17 (d_model 64, 3 000 pas, batch 64, n d'entraînement ∈ [4, 24]) ; évaluation n = 12, 24, 48, 96 ; **3 graines (0, 1, 2) pour `L_loop1` et pour `L_dloop1_full`**. Pourquoi : deux valeurs à garder en parallèle, ce qui correspond naturellement à deux images d'entrée. « Apprend » (par graine) = S3 ≥ 95 % et rappel ≥ 90 % à n=48. **Double loop meilleur** si `L_dloop1_full` a au moins une graine de plus qui apprend OU un rappel médian à n=96 supérieur de ≥ 5 points ; **égal** si les deux comptes sont identiques et l'écart de rappel médian à n=96 est dans ±5 points ; **moins bon** sinon. Exploratoire : annuler le dépôt 1 ou 2 à l'inférence (spécialisation éventuelle d'une image par valeur), rappel séparé pour Q1 et Q2.
**(2) Liste circulaire de 6 symboles** (amendement 23, `scripts/reseau_exp.py L_dloop1_full_k0`), jamais de pause, 8 000 pas, graines 0 et 1 (le calcul est ≈ 2 h pour deux graines en parallèle). Références : `L_loop1_k0` 100 % à h = 1, 2, 3 sur 3 graines sur 3. **Égal** si `L_dloop1_full_k0` ≥ 95 % à h = 1, 2, 3 sur les 2 graines ; sinon rapporté tel quel. Exploratoire : h = 4, 5 (hors entraînement, la boucle fait 11 et 3 %).
**Pronostics (avant mesure).** (1) égal : 55 %, meilleur : 25 %, moins bon : 20 %. (2) égal : 60 %. Ordre des runs : (1) d'abord (≈ 1 h), puis (2). Un échec ou un résultat nul sera rapporté tel quel.

### Résultat de l'amendement 32 (2026-11-22)
**S4, 15 000 pas, graine 0, `G231` (272 k paramètres, 4 634 s)** : précision 0,387 / 0,320 / 0,291 / 0,271 / 0,255 à n = 12 / 24 / 48 / 96 / 192 : plateau comparable à `T_rope` (40 % à n = 24) et au double loop (41 %), très en dessous de `L_loop1` (100 %). Critère (c) **non** (pronostic 35 %).
**Liste circulaire, entraînement jusqu'à h = 5, sans pauses, 8 000 pas, graine 0**, précision à h = 1…8 : `GRU` (252 k) 1,0 / 1,0 / 1,0 / 1,0 / 1,0 / 0,00 / 0,01 / 0,10 ; `L_loop1` (162 k) 0,99 / 0,65 / 1,00 / 0,65 / 0,96 / 0,15 / 0,08 / 0,27 ; `G231` (271 k) 0,13–0,18 partout (hasard : la perte reste à ln 6, y compris à h = 1). Critères (a) **non**, (b) **non**. Pronostics : `GRU` ≥ 0,9 à h = 5 atteint (70 %) ; `L_loop1` ≥ 0,9 à h = 5 **atteint** (25 %, sous-estimé), mais l'apprentissage n'est pas stabilisé (h = 2 et h = 4 à 0,65) ; `G231` à h = 6 : non (5 %). Lecture a posteriori : avec n = 6, f⁶ est l'identité et f⁷, f⁸ valent f, f² ; h = 6, 7, 8 ne sont donc pas des tests d'extrapolation propres, mais aucun modèle n'en tire parti (≤ 0,27).
Conclusion : le réseau 2-3-1 léger n'apporte rien sur ces tâches à ce budget et à une graine ; il n'apprend pas du tout la liste circulaire, alors que la boucle à 2 images et le GRU l'apprennent. Hypothèse (non testée) : l'optimisation de 6 images avec 18 transformers, sans initialisation ni gating adaptés à cette profondeur, est plus difficile ; la comparaison reste à graine unique.

## Amendement 33b : diagnostic du réseau en couches et stabilité de la boucle à h = 5 (pré-enregistré avant mesure, 2026-11-22)
**Contexte.** `G231` n'apprend pas la liste circulaire (même pas h = 1) alors que la boucle à 2 images l'apprend, bien qu'il en soit en principe une généralisation. Raisons possibles avancées avant mesure : (i) images 4×4×4 et d = 32 (plus petits que la boucle) ; (ii) bruit des arêtes supplémentaires dû à l'initialisation ; (iii) calcul plus profond par token.
**(1) Échelle de croissance** (`scripts/diag_graphe.py`, `GraphLM2`), même taille d'image 4×4×4 et d = 32 pour tous, liste circulaire n = 6, h ≤ 3, k = 0, 6 000 pas, graine 0 : `1-1` (la boucle à 2 images, petite), `2-1`, `1-1-1`, `2-2-1`, `2-3-1`, et `2-3-1z` (sortie des transformers initialisée à zéro). On note le premier échelon où la précision à h = 3 passe sous 0,5. **Pronostics** : `1-1` ≥ 0,9 à h = 3 : **65 %** ; `2-3-1` ≥ 0,9 à h = 3 : **10 %** ; `2-3-1z` ≥ 0,9 à h = 3 : **30 %**. Si `1-1` échoue, l'échec vient de la petite taille (i) et la comparaison avec `G231` ne dit rien sur les arêtes.
**(2) Stabilité de la boucle à h = 5** : `scripts/reseau_h5.py L_loop1` graines 1 et 2 (graine 0 : h = 1…5 à 0,99 / 0,65 / 1,00 / 0,65 / 0,96). **Critère** : précision ≥ 0,90 à h = 1, 2, 3, 4 et 5 sur chacune des 2 graines. **Pronostic : 25 %** (les h = 2 et 4 à 0,65 de la graine 0 suggèrent un apprentissage non terminé).

### Résultat de l'amendement 33, partie (1) S3 + deux valeurs (2026-11-22)
Précision S3 / rappel (%) à n = 12 / 24 / 48 / 96. `L_loop1` : graine 0 = 100/100, 100/100, 100/100, 100/98 ; graine 1 = 100/100, 100/100, 100/100, 100/95 ; graine 2 = 100/72, 100/68, 100/57, 95/56 (S3 appris, mémoire des deux valeurs non). `L_dloop1_full` : graine 0 = 30/55, 23/54, 20/43, 19/18 (**S3 lui-même non appris**) ; graine 1 = 100/100, 100/100, 100/100, 98/100 ; graine 2 = 99/100, 100/100, 100/100, 99/96. Critères tels qu'écrits (« apprend » = S3 ≥ 95 % et rappel ≥ 90 % à n=48) : `L_loop1` **2/3** graines, `L_dloop1_full` **2/3** ; rappel médian à n=96 : 95,3 % contre 95,7 % → **égal** (pronostic « égal » 55 % : confirmé). Les graines qui échouent sont différentes (graine 2 pour la boucle, graine 0 pour le double loop) : à 3 graines la fiabilité est la même, mais les échecs n'ont pas la même nature (mémoire contre S3). Exploratoire : annuler un dépôt fait chuter S3 et au moins un rappel à chaque fois (les deux read-in servent), mais la spécialisation n'est pas nette (graine 1 : sans dépôt 2, rappel Q1 = 0 % ; sans dépôt 1, rappel Q1 = 100 % mais S3 = 16 % et Q2 = 17 % ; graine 2 : tout se dégrade dans les deux cas). Pas de gain à avoir deux images d'entrée pour deux valeurs. Coût : 750–780 s (`L_loop1`) contre 1 340–1 390 s (double loop). Résultats : `results/s3m2_*`. La partie (2) (liste circulaire) est en cours.

### Résultat de l'amendement 33b (2026-11-22)
**(1) Échelle de croissance** (images 4×4×4, d = 32, h ≤ 3, 6 000 pas, graine 0), précision h = 1 / 2 / 3 / 4 / 5 : `1-1` 0,51 / 0,26 / 0,21 / 0,20 / 0,19 ; `2-1` 0,66 / 0,28 / 0,27 / 0,21 / 0,18 ; `1-1-1`, `2-2-1`, `2-3-1`, `2-3-1z` : 0,15–0,18 partout (hasard). Aucun ≥ 0,9 à h = 3 : `1-1` **non** (pronostic 65 % : faux), `2-3-1` **non** (10 %), `2-3-1z` **non** (30 %). Le premier échelon sous 0,5 à h = 3 est `1-1` lui-même : la petite taille casse la boucle, donc l'échec de `G231` n'est pas imputable aux arêtes supplémentaires ; l'initialisation à zéro n'y change rien.
**(2) Boucle à h = 5, graines 1 et 2** (graine 0 : 0,99 / 0,65 / 1,00 / 0,65 / 0,96) : graine 1 : 0,19 / 0,20 / 0,20 / 0,22 / 0,18 (hasard) ; graine 2 : 0,47 / 0,25 / 0,25 / 0,26 / 0,46. Critère (≥ 0,9 de h = 1 à 5 sur chaque graine) **non atteint** (pronostic 25 %). Le résultat de la graine 0 n'est pas reproductible ; la remarque de l'amendement 32 sur la partie 9 est corrigée en 12.5.

### Résultat de l'amendement 33, partie (2) liste circulaire (2026-11-22)
`L_dloop1_full_k0` (8 000 pas, jamais de pause), précision à k = 0 : **graine 0 : h = 1, 2, 3 = 100 / 100 / 100 %, h = 4 = 2 %, h = 5 = 1 %** ; **graine 1 : 100 / 100 / 100 %, h = 4 = 1 %, h = 5 = 3 %**. Critère tel qu'écrit (≥ 95 % à h = 1, 2, 3 sur les 2 graines) : **égal, atteint** (pronostic 60 % : confirmé) ; pas d'extrapolation à h = 4, 5, comme la boucle (0–4 %). **Correction d'une erreur de ce pré-enregistrement** : j'y ai écrit que la référence `L_loop1_k0` faisait 100 % à h = 1, 2, 3 sur 3 graines sur 3. C'est faux : cette référence est 100 / 100 / 100 sur les graines 0 et 2, mais **15 / 19 / 20 % sur la graine 1** (la valeur 3 sur 3 est celle de `L_loop1_k`, entraînée avec pauses). La référence correcte est donc 2 graines sur 3 ; le critère (« double loop ≥ 95 % sur ses 2 graines ») n'est pas modifié, son verdict ne change pas, mais « égal à la boucle » est à lire « au moins aussi bon : 2/2 graines contre 2/3 ». Temps : 3 675 s environ par graine. Résultats : `results/reseau_L_dloop1_full_k0_s{0,1}.json`.

## Amendement 34 : échelle de croissance à la taille de la boucle (pré-enregistré avant mesure, 2026-11-23)
**Question.** L'amendement 33b a montré que la petite taille (images 4×4×4, d = 32) casse même la boucle à 2 images. À la taille de la boucle (images 4×8×8, d = 64, 4 têtes, 1 couche ; `DIAG_SIZE=full`), ajouter une image d'entrée ou une couche intermédiaire casse-t-il l'apprentissage ?
**Montage.** `scripts/diag_graphe.py`, `GraphLM2`, liste circulaire n = 6, h ≤ 3, k = 0, 6 000 pas, graine 0 ; configurations `1-1` (équivalente à `L_loop1`), `2-1` (deux entrées), `1-1-1` (une couche intermédiaire). `2-3-1` à cette taille est renvoyé au GPU.
**Pronostics** (précision ≥ 0,90 à h = 1, 2 et 3) : `1-1` **75 %** ; `2-1` **50 %** ; `1-1-1` **40 %**. Le premier échelon qui échoue sera rapporté ; un échec de `1-1` signifierait que 6 000 pas sans pauses ne suffisent pas à cette taille, et la suite ne serait pas interprétable.

## Amendement 35 : tâche de langage réelle : étiquetage grammatical (UPOS) sur UD English-EWT (pré-enregistré avant toute mesure de modèle, 2026-11-23)
Demande de Mathis : une tâche « plutôt transformer » sur données réelles, pas un signal ni une tâche récurrente. Tâche : pour chaque mot d'une phrase anglaise réelle (web : blogs, e-mails, avis, forums), prédire sa catégorie grammaticale (17 UPOS) en ne voyant que **les mots à gauche et le mot lui-même** (modèles causaux). Données : UD English Web Treebank (CC BY-SA 4.0), `data/ud_ewt/{train,dev,test}.tsv` (12 544 / 2 001 / 2 077 phrases ; 204 578 / 25 148 / 25 094 mots), `scripts/cpu_ud.py`. Vocabulaire : 5 000 mots les plus fréquents du train (casse exacte), le reste = UNK ; pendant l'entraînement 5 % des mots deviennent UNK au hasard. Phrases d'entraînement tronquées à 48 mots, évaluation sur phrases entières.
Modèles (d_model 64, 3 000 pas, lot 32, lr 1e-3 cosinus, échauffement 100, **3 graines : 0, 1, 2**) : `T_rope` (2 couches, 421 k paramètres dont 320 k d'embeddings), `L_loop1` (boucle, 1 couche par transformer, 313 k dont 160 k d'embeddings), `L_dloop1_full` (double loop à branche principale pleine, 370 k dont 160 k). Les tailles d'embeddings (vocabulaire de 5 000 mots) dominent : la comparaison de paramètres n'est pas à capacité égale. Référence sans réseau (déjà calculée, descriptive) : étiquette la plus fréquente de chaque mot, mots inconnus → étiquette la plus fréquente des mots rares : **86,1 % sur dev** (54,9 % sur les 3 811 mots hors vocabulaire), 86,2 % sur test.
Critères (précision dev moyenne sur 3 graines ; test rapporté mais pas critérié) :
(a) **tâche exploitable** si `T_rope` dépasse la référence de ≥ 3 points (le contexte est utilisé).
(b) **boucle compétitive** si précision(L_loop1) ≥ précision(T_rope) − 1 point ; **boucle meilleure** si précision(L_loop1) ≥ précision(T_rope) + 1 point sur ≥ 2 graines sur 3 ; sinon **moins bonne**.
(c) **double loop égal à la boucle** si |précision(L_dloop1_full) − précision(L_loop1)| ≤ 0,5 point ; meilleur/moins bon au-delà.
Exploratoire (non critérié) : précision sur les mots hors vocabulaire (références : 54,9 % pour la table de fréquences).
**Pronostics (écrits avant).** `T_rope` ≈ 91–92 % sur dev (a : oui, 85 %) ; `L_loop1` ≈ 90–91 % (compétitive 55 %, meilleure 10 %) ; `L_dloop1_full` à ±0,5 point de la boucle (50 %). Limites : étiquetage de gauche à droite (pas de contexte droit : un étiqueteur bidirectionnel ferait 95 %+), budget minuscule, vocabulaire fermé, trois graines (pas de test statistique), embeddings dominants.

### Résultat de l'amendement 34 (2026-11-23)
Taille de la boucle (images 4×8×8, d = 64), liste circulaire h ≤ 3, k = 0, 6 000 pas, graine 0, précision h = 1 / 2 / 3 / 4 / 5 : `1-1` (152 k paramètres, 1 469 s) 0,98 / 0,50 / 0,36 / 0,17 / 0,14 ; `2-1` (268 k, 2 819 s) 0,98 / 0,54 / 0,62 / 0,22 / 0,08 ; `1-1-1` (260 k, 2 885 s) 0,16 / 0,18 / 0,16 / 0,15 / 0,16 (hasard).
Critère ≥ 0,90 à h = 1, 2, 3 : `1-1` **non** (pronostic 75 % : faux), `2-1` **non** (50 %), `1-1-1` **non** (40 %). Comme prévu au protocole, l'échec de `1-1` (la boucle seule) signifie que 6 000 pas sans pauses ne suffisent pas à cette taille, et l'échelle n'est pas interprétable de façon stricte. Lecture a posteriori, à confirmer sur d'autres graines : avec deux entrées (`2-1`) l'apprentissage va au moins aussi loin qu'avec une seule ; avec une couche intermédiaire (`1-1-1`) il ne démarre pas du tout, alors que `1-1` et `2-1` démarrent à h = 1 : ce serait l'ajout d'une couche intermédiaire, c'est-à-dire de la profondeur par token, qui bloque l'optimisation.

### Résultat de l'amendement 35 (2026-11-23)
Précision UPOS sur dev (%), graines 0 / 1 / 2 et moyenne : `T_rope` 86,35 / 86,40 / 86,69 → **86,48** (421 k paramètres, 66–81 s) ; `L_loop1` 86,42 / 86,29 / 86,54 → **86,42** (313 k, ≈ 1 350 s) ; `L_dloop1_full` 86,46 / 86,65 / 86,45 → **86,52** (370 k, ≈ 2 530 s). Test (moyennes) : 86,76 / 86,63 / 86,65. Mots hors vocabulaire (dev, moyenne) : 49,0 / 48,9 / 49,5 % (table de fréquences : 54,9 %, qui connaît plus de mots que les 5 000 du vocabulaire des modèles). Référence : table de fréquences 86,13 % (dev).
Critères tels qu'écrits : (a) **non atteint** : `T_rope` dépasse la référence de 0,35 point seulement (< 3) : la tâche n'est pas exploitable à ce budget, aucun réseau n'utilise le contexte à gauche mieux qu'une table de fréquences de mots ; (b) `L_loop1` **compétitive** (−0,06 point par rapport à `T_rope`), pas meilleure ; (c) `L_dloop1_full` **égal** à la boucle (+0,10 point). Les trois modèles sont à 0,2 point les uns des autres, bien dans le bruit entre graines (écart 0,1–0,4 point). Pronostics : `T_rope` ≈ 91–92 % (a : 85 %) **infirmé** (86,5 %) ; `L_loop1` ≈ 90–91 % infirmé ; double loop à ±0,5 point de la boucle confirmé. Cause probable, non testée : le plafond vient de la tâche (étiquetage de gauche à droite sans contexte droit, vocabulaire fermé de 5 000 mots, aucun trait de caractères) et pas de l'architecture ; le `T_rope` est à 86,5 % en une minute, comme le double loop en 42. Résultats : `results/ud_*_s{0,1,2}.json`.

## Amendement 35b : échelle de croissance à la taille de la rémanence sur GPU XPU (pré-enregistré avant mesure, 2026-11-23)
**Question.** Reprise de l'amendement 34 avec un budget plus grand (batch 256 au lieu de 64, plus de pas) : les configurations `1-1`, `2-1`, `1-1-1`, `2-3-1` (images 4×8×8, d = 64, 4 têtes) apprennent-elles la liste circulaire à h ≤ 3 (k = 0) ? 
**Montage.** `scripts/xpu_diag.py`, graine 0, appareil de l'utilisateur (Intel XPU), **même nombre de pas pour toutes les configurations**, choisi après une mesure de temps (`--bench`) pour que les quatre tiennent en ≈ 30 minutes ; batch 256 (changement par rapport aux amendements 33b et 34, à garder en tête pour toute comparaison), curriculum de h sur 40 % des pas, évaluation à h = 1…5.
**Pronostics et critère** (précision ≥ 0,90 à h = 1, 2 et 3) : `1-1` **60 %** ; `2-1` **45 %** ; `1-1-1` **25 %** ; `2-3-1` **15 %**. Si `1-1` échoue encore, le budget reste insuffisant et la suite n'est pas interprétable (comme à l'amendement 34).

### Résultat de l'amendement 35b (2026-11-23)
Numérotation : « 35 » était déjà pris par l'autre IA (UD English-EWT) ; celui-ci devient 35b. Appareil : XPU de l'utilisateur, 600 pas (calculés par la mesure de temps pour tenir en ≈ 30 min), batch 256, graine 0. Précision h = 1 / 2 / 3 / 4 / 5 : `1-1` 0,22 / 0,20 / 0,18 / 0,19 / 0,20 (415 s) ; `2-1` 0,19 / 0,19 / 0,20 / 0,21 / 0,20 (352 s) ; `1-1-1` 0,14 / 0,19 / 0,19 / 0,16 / 0,15 (418 s) ; `2-3-1` non lancé (temps). Critère ≥ 0,90 : aucun (`1-1` 60 %, `2-1` 45 %, `1-1-1` 25 % : pronostics faux).
**Non interprétable** : 600 pas est trop peu (sur CPU, `1-1` ne décolle de h = 1 qu'après 2 000 à 2 500 pas à batch 64) et même la rémanence seule reste au hasard, comme le protocole le prévoyait. Constat pratique : sur ce XPU, `1-1` coûte ≈ 0,7 s par pas contre ≈ 0,25 s sur mes CPU : l'architecture, séquentielle sur les tokens avec beaucoup de petits calculs, est limitée par le temps de lancement des opérations et non par les flops, et ce GPU ne l'accélère pas sans changement (grand batch, `torch.compile`, graphes d'opérations).

## Amendement 36 : TinyStories : la rémanence contre T_rope sur XPU (pré-enregistré avant mesure, 2026-11-23)
**Question.** Sur un langage d'enfant simple (TinyStories), la rémanence (`L_loop1`) fait-elle aussi bien que T_rope (amendement 20 : −1 à +4 % à 64 tokens sur 6 corpus ; ≥ 3 % mieux au-delà de 64 tokens sur 4 corpus sur 5) ?
**Montage.** `scripts/ts_remanence.py` (notebook `notebooks/tinystories_remanence.ipynb`) : 80 Mo du début de `TinyStories-train.txt`, validation sur les 5 premiers Mo de `TinyStories-valid.txt`, BPE 4 096 appris sur le train ; fenêtres d'entraînement de 64 tokens, batch 256 (réduit si la mémoire manque), même nombre de pas pour les deux modèles (calculé pour que la rémanence tienne en 30 min), 1 graine ; évaluation en bits par caractère sur 96 fenêtres de 256 tokens, par tranches de position (0-16, 16-32, 32-64, 64-128, 128-256) ; deux histoires générées par modèle, jugées à la lecture seulement. Différences avec l'amendement 20 : batch 256 au lieu de 32, plus de pas (≈ 600 à 1 000 selon l'appareil), corpus TinyStories.
**Pronostics et critères.** (a) rémanence dans ±5 % de T_rope sur 0-64 : **65 %** ; (b) rémanence ≥ 3 % mieux que T_rope sur 128-256 : **45 %** ; (c) les deux modèles produisent des phrases grammaticales reconnaissables : **50 %** (le budget est de l'ordre de 20 M tokens vus au plus). Une seule graine : un écart inférieur à 3 % sera rapporté comme indiscernable.

### Résultat de l'amendement 36 (2026-11-23)
XPU de l'utilisateur ; batch 64 (le batch 256 a fait planter le pilote, `DEVICE_LOST`), 1 950 pas pour les deux modèles (≈ 8 M tokens vus), graine 0. Bits par caractère sur la validation, positions 0-16 / 16-32 / 32-64 / 64-128 / 128-256 : rémanence (950 k paramètres, 2 252 s) 1,330 / 1,210 / 1,259 / 1,225 / 1,254 ; T_rope (624 k, 87 s) 1,330 / 1,211 / 1,257 / 1,235 / 1,481.
(a) dans ±5 % sur 0-64 : **atteint** (écarts < 0,2 %, pronostic 65 %). (b) ≥ 3 % mieux sur 128-256 : **atteint** (−15 %, pronostic 45 %) ; lecture : T_rope n'a vu que des positions < 64 à l'entraînement et se dégrade au-delà, alors que la rémanence n'a pas de position et reste stable ; sur 64-128 l'écart (0,8 %) est indiscernable. (c) phrases grammaticales : **atteint** pour les deux (phrases correctes localement, histoire incohérente). Coût : la rémanence est ≈ 26 fois plus lente sur ce XPU. Une seule graine.

## Amendement 37 : contrôle de longueur d'entraînement, puis Wikipédia simple et PG-19 (pré-enregistré avant mesure, 2026-11-23)
**Contexte.** À l'amendement 36 (TinyStories, fenêtres d'entraînement de 64 tokens), T_rope se dégrade au-delà de 64 (1,48 contre 1,25 pour la rémanence sur 128-256). Cet avantage peut venir de la seule longueur d'entraînement de T_rope.
**(1) Contrôle** : TinyStories, fenêtres d'entraînement de **256** tokens (batch 16, soit les mêmes 4 096 tokens par pas), évaluation jusqu'à 512 tokens (`--window 256 --eval-len 512 --tag _w256`), mêmes pas pour les deux modèles (déterminés par la durée), graine 0. **Critère** : la rémanence reste ≥ 5 % meilleure que T_rope sur 256-512 (hors de la longueur d'entraînement). **Pronostic : 40 %** (si T_rope voit des positions jusqu'à 256, il devrait se dégrader moins vite au-delà de 256, mais pas nécessairement plus que la rémanence).
**(2) Wikipédia simple** (`wikimedia/wikipedia`, `20231101.simple`, 80 Mo de train, 3 Mo de validation), fenêtres de 256, évaluation jusqu'à 512 : **pronostic** rémanence dans ±5 % de T_rope sur 0-256 : **60 %**.
**(3) PG-19** (`deepmind/pg19`, 80 Mo de livres, validation 3 Mo), fenêtres de 256, évaluation jusqu'à 1 024 : critère : la rémanence ≥ 3 % meilleure que T_rope sur 512-1 024 : **35 %**.
Dans les trois cas : 1 graine, bits par caractère par tranche de position ; un écart inférieur à 3 % est rapporté comme indiscernable.

### Résultat de l'amendement 37 (2026-11-23)
XPU de l'utilisateur, fenêtres de 256, batch 16, graine 0, 35 min visées pour la rémanence. **Seuls (1) et (2) ont tourné ; PG-19 n'a pas été lancé.**
**(1) Contrôle TinyStories** : **400 pas** (≈ 1,6 M tokens, contre 1 950 pas à 64 tokens en amendement 36). Bits par caractère 0-16 / 16-32 / 32-64 / 64-128 / 128-256 / 256-512 : rémanence (3 610 s) 2,27 / 2,25 / 2,07 / 2,16 / 2,12 / 2,12 ; T_rope (37 s) 1,85 / 1,78 / 1,57 / 1,65 / 1,63 / 1,64. Critère (rémanence ≥ 5 % meilleure sur 256-512) : **non atteint**, elle est 29 % moins bonne (pronostic 40 % : faux). Les histoires de la rémanence ne sont pas encore des phrases (mots fonctionnels en désordre), celles de T_rope en sont proches.
**(2) Wikipédia simple** : **200 pas**. Rémanence (2 180 s) 3,22 / 3,25 / 3,29 / 3,28 / 3,19 / 3,17 ; T_rope (16 s) 3,16 / 3,27 / 3,29 / 3,30 / 3,22 / 3,23. Écart ≤ 2 % partout : critère « ±5 % » **atteint** de façon triviale, car les deux modèles sont à ≈ 3,2 bits par caractère, le niveau d'une statistique de fréquences : **aucun n'a appris**.
**Lecture** : non interprétable. 400 et 200 pas sont très en dessous des 1 950 pas où les deux modèles s'égalaient (amendement 36) ; à pas égaux et faibles, la rémanence apprend plus lentement que T_rope (≈ 9 s par pas contre 0,1 s à 256 tokens sur ce XPU). Le contrôle de la longueur d'entraînement n'est donc pas réalisé ; il demande environ 5 fois plus de pas, soit plus de 4 h de XPU.

## Amendement 38 : version rapide de la rémanence (optimisation sans changement de modèle, 2026-11-23)
`src/s8_rapide.py` (`FastLoop`) : mêmes paramètres et mêmes calculs que `LoopLM`, avec les images gardées en patchs, une couche de transformer écrite à la main (q, k, v fusionnés) et la lecture faite une fois après la boucle. **Critère d'équivalence** : écart sur les sorties < 1e-5 et écart relatif sur les gradients < 1e-4. **Résultat** : sorties 7e-7, gradients 1e-6 (loop True) et 5e-7 (loop False) : équivalent. **Vitesse sur CPU** (2 cœurs, batch 16, 64 tokens, pas complet avant+arrière) : origine 0,406 s, rapide 0,288 s (×1,41), rapide + `torch.compile` 0,225 s (×1,8). Sur XPU : à mesurer (`scripts/bench_rapide.py`) ; options `--fast` et `--compile` dans `scripts/ts_remanence.py`. **Apprentissage** (`scripts/verif_rapide.py`, liste circulaire h ≤ 3, 4 000 pas, graines 0, 1, 2, même graine pour les deux versions) : les trajectoires divergent malgré des calculs identiques (amplification des arrondis pendant le plateau). Précision à h = 3 finale, origine / rapide : graine 0 0,23 / **1,00**, graine 1 **0,92** / 0,31, graine 2 0,22 / 0,23 : une graine sur trois réussit dans chaque version, pas la même ; temps ≈ 13 % plus court (806-815 s contre 924-932 s). Aucune différence détectable avec 3 graines. Voir `rapport/14_courbes.md`, section 14.4.

## Amendement 39 : T1 et T2 initialisés par un transformer à patchs pré-entraîné (pré-enregistré avant mesure, 2026-11-23)
**Idée de l'utilisateur.** Prendre un gros transformer à patchs déjà entraîné (ViT de `timm`), retirer son plongement de patchs et sa tête, brancher nos images (`src/s9_preentraine.py`, `PreLoop`) : T1 et T2 sont chacun une copie des blocs du ViT, avec une projection d'entrée (16 → d), un plongement de position et une projection de sortie (d → résidu + oubli) neufs. Images, portes, dépôts et tête de lecture inchangés. Modes : `ln` (blocs gelés sauf leurs LayerNorm, défaut), `frozen`, `all` (blocs à lr × 0,1).
**Montage.** `scripts/ts_remanence.py --backbone vit_small_patch16_224` (21,7 M de paramètres par copie, d = 384, 12 blocs), TinyStories, fenêtres de 128 tokens, batch 16, même nombre de pas que T_rope, 1 graine ; comparaison avec la rémanence de départ (`L_loop1`, 0,95 M de paramètres) à mêmes pas dans la partie 13 (1 950 pas, fenêtres de 64) et avec T_rope. Les poids pré-entraînés viennent d'images naturelles : rien n'indique qu'ils servent à des images apprises ; l'hypothèse est qu'ils fournissent un calcul générique et un meilleur point de départ d'optimisation (cf. 1-1-1, qui n'apprenait pas).
**Pronostics et critères** (bits par caractère sur 64-128 et 128-256) : (a) le run se termine sans plantage et la perte baisse en dessous de 4 nats : **70 %** ; (b) à pas égaux, `PreLoop` meilleure que `L_loop1` de ≥ 3 % : **30 %** ; (c) `PreLoop` meilleure que T_rope de ≥ 3 % sur 128-256 : **25 %**. Un résultat où le pré-entraînement n'aide pas (ou nuit) sera rapporté tel quel ; un contrôle avec `--no-pretrained` (même architecture, poids aléatoires) est nécessaire pour attribuer un gain au pré-entraînement : **à lancer ensuite**.

### Résultat de l'amendement 39 (2026-11-24)
XPU de l'utilisateur, `vit_small_patch16_224` (43,5 M de paramètres dont 0,93 M entraînés, mode `ln`), TinyStories, fenêtres de 128, batch 16, **600 pas** (≈ 1,2 M tokens vus), graine 0. La rémanence pré-entraînée a mis 8 965 s (≈ 15 s par pas, comme estimé), T_rope 19 s. Bits par caractère 0-16 / 16-32 / 32-64 / 64-128 / 128-256 : `PreLoop` 2,068 / 2,031 / 2,044 / 2,020 / 2,034 ; T_rope 1,635 / 1,536 / 1,578 / 1,544 / 1,573. À 500 pas (sauvegarde partielle) : 2,10 et 1,59 sur 128-256.
(a) le run se termine et la perte baisse (bpc ≈ 2) : **atteint** (pronostic 70 %). (c) `PreLoop` ≥ 3 % mieux que T_rope sur 128-256 : **non**, elle est 29 % moins bonne (pronostic 25 %). (b) `PreLoop` ≥ 3 % mieux que `L_loop1` à pas égaux : **non évaluable**, aucune `L_loop1` à 600 pas sur ces fenêtres n'a été entraînée ; la seule référence (400 pas, fenêtres de 256, amendement 37) donne 2,12 sur 128-256 contre 2,03 ici, mais les budgets diffèrent. Les histoires générées ne sont pas des phrases (mots fonctionnels en désordre), comme pour la rémanence de départ à 400 pas. Le contrôle `--no-pretrained` n'a pas été lancé : on ne peut donc pas attribuer le niveau atteint au pré-entraînement. Lecture : la rémanence apprend lentement dans les premières centaines de pas, pré-entraîné ou non ; à ce budget, rien ne permet de conclure.

## Amendement 40 : la taille de l'image (la mémoire) contre un GRU de même état, sur du langage (pré-enregistré avant mesure, 2026-11-24) : PRIORITAIRE
**Question.** L'état de la rémanence (deux images 4×8×8, 256 nombres lus) est minuscule pour du texte. Grossir l'image (la mémoire) permet-il à la rémanence de rattraper T_rope, et en profite-t-elle plus qu'un GRU dont on grossit l'état autant ? C'est aussi le premier test d'un GRU sur du langage.
**Montage.** `scripts/suite.py taille` (Colab T4 ou XPU) : TinyStories puis Wikitext-2, fenêtres de 128, batch 32, **2 000 pas pour tous**, évaluation jusqu'à 512 tokens, graine 0. Modèles : rémanence `R_4x8x8` (256 nombres par image), `R_4x16x16` (1 024), GRU de même état `GRU_256`, `GRU_1024`, et T_rope ; avec `--grand` : `R_16x16x16` (4 096) et `GRU_4096`. Mesure : bits par caractère par tranche de position (0-128 dans la longueur d'entraînement, 128-512 au-delà).
**Pronostics et critères.**
- (a) `GRU_256` ≤ `R_4x8x8` + 3 % sur 0-128 (le GRU égale ou bat la rémanence de même état) : **65 %**.
- (b) passer de 256 à 1 024 améliore la rémanence de ≥ 3 % sur 0-128 : **55 %** ; et le GRU de ≥ 3 % : **50 %**.
- (c) gain de taille plus grand pour la rémanence que pour le GRU (écart ≥ 2 points de pourcentage) : **30 %**.
- (d) `R_4x16x16` dans ±3 % de T_rope sur 0-128 : **45 %**.
Une graine : un écart < 3 % est indiscernable. Si (a) est atteint et (c) ne l'est pas, la rémanence n'est pas meilleure qu'un RNN sur du langage, et les directions 41-43 sont à lire dans ce sens.

### Résultats de l'amendement 40 (XPU, 2 000 pas, fenêtres de 128, graine 0, lr 1e-3 non réglé ; bpc 0-128 / 128-512)
| modèle | paramètres | TinyStories | Wikitext-2 | durée (Wikitext-2) |
|---|---|---|---|---|
| T_rope | 624 k | 1,296 / 1,390 | 2,100 / 2,185 | 1,4 min |
| R_4x8x8 | 950 k | 1,302 / 1,250 | 2,139 / 2,156 | 2 h 52 |
| R_4x16x16 | 1,11 M | 2,525 / 2,506 (**n'a pas démarré**, perte 6,8) | **2,024 / 2,040** | 4 h 40 |
| GRU_256 | 1,56 M | 1,281 / 1,227 | 2,085 / 2,091 | 3,6 min |
| GRU_1024 | 7,8 M | **1,095 / 1,041** | **1,930 / 1,932** | 4,8 min |
- (a) **atteint** sur les deux jeux : GRU_256 égale ou bat R_4x8x8 (−1,6 % TinyStories, −2,5 % Wikitext-2).
- (b) rémanence 256 → 1 024 : **atteint sur Wikitext-2 (−5,4 %), raté sur TinyStories** (4×16×16 n'a pas quitté le plateau de départ). GRU 256 → 1 024 : **atteint sur les deux** (−14,5 % et −7,4 %).
- (c) **raté sur les deux jeux** : le GRU gagne plus de sa taille que la rémanence (Wikitext-2 : 7,4 % contre 5,4 %).
- (d) **raté, dans le bon sens à Wikitext-2** : R_4x16x16 est 3,6 % *meilleur* que T_rope (hors de ±3 %) ; raté sur TinyStories (+95 %, non démarré).
**Lecture.** (1) Un GRU fait mieux que la rémanence à état comparable et apprend 30 à 3 000 fois plus vite : sur du langage, la mémoire en images ne bat pas un RNN. (2) Sur Wikitext-2, la grande image aide la rémanence (−5,4 %) et la place devant T_rope, mais à paramètres inégaux (1,11 M contre 0,62 M) : le test CPU de l'amendement 44 donne T_96x2 (1 M) ≈ 5 % meilleur que T_64x2, ce qui l'amènerait au niveau de R_4x16x16 ; T_rope à paramètres égaux n'a pas été mesuré à ce montage. (3) Au-delà de 128 tokens, T_rope se dégrade (+4 % Wikitext-2, +7 % TinyStories) et la rémanence reste stable ; le GRU aussi. (4) Le démarrage échoue pour la grande image sur TinyStories : l'effet de la taille n'est pas reproductible d'un jeu à l'autre avec une graine. (5) Les rémanences coûtent 100 à 200 fois T_rope en temps. Limites : une graine, lr non réglé (l'amendement 44 montre que 2e-3 aurait été meilleur pour tous), 2 000 pas.

**Complément de l'amendement 40 (T_rope à paramètres égaux, lancé après la lecture ci-dessus, 2026-11-25).** `T_96x2` (1,01 M), même montage sur Wikitext-2 : bpc 0-128 **1,997**, 128-512 2,060 (75 s). Contre `R_4x16x16` (1,11 M) : 2,024 / 2,040. À paramètres comparables, T_rope est 1,3 % meilleur dans la longueur d'entraînement et 1,0 % moins bon au-delà : égalité à une graine près. La lecture (2) ci-dessus (« la grande image place la rémanence devant T_rope ») ne vaut que contre le T_rope plus petit ; elle est retirée.

## Amendement 41 : mémoire longue à mémoire égale : PG-19 (pré-enregistré avant mesure, 2026-11-24)
`scripts/suite.py memoire` : PG-19 (livres), fenêtres de 256, batch 16, 2 000 pas, évaluation jusqu'à 1 024 tokens ; `R_4x8x8`, `GRU_1024`, `Twin_64` (T_rope à fenêtre glissante de 64 tokens : mémoire bornée comme les deux autres), T_rope (mémoire complète jusqu'à 256). **Critère** : la rémanence bat `Twin_64` d'au moins 3 % sur 256-1 024 : **40 %** ; elle bat `GRU_1024` de 3 % : **20 %**.

## Amendement 42 : la rémanence comme mémoire d'un transformer (pré-enregistré avant mesure, 2026-11-24)
`src/s10_hybride.py`, `scripts/suite.py hybride` : `H_64` = transformer à fenêtre glissante de 64 tokens, dont chaque position reçoit en plus une projection de l'image 2 d'une rémanence qui lit les mêmes tokens (état après le token précédent). TinyStories puis Wikitext-2, fenêtres de 256, batch 16, 2 000 pas, évaluation jusqu'à 1 024, contre `Twin_64` (même transformer sans mémoire), `R_4x8x8` et T_rope. **Critères** : `H_64` meilleur que `Twin_64` de ≥ 3 % sur 64-256 (la mémoire sert au-delà de la fenêtre) : **50 %** ; `H_64` dans ±3 % de T_rope sur 0-256 : **40 %** ; `H_64` meilleur que T_rope de ≥ 3 % sur 256-1 024 : **35 %**.

## Amendement 43 : que contiennent les images ? Sondes linéaires sur les histoires synthétiques (pré-enregistré avant mesure, 2026-11-24)
`scripts/sondes_histoires.py`, `scripts/suite.py sondes` : on entraîne la rémanence (`L_loop1`), un `GRU_256` et on garde une rémanence non entraînée (`L_loop1_init`) sur les histoires synthétiques (partie 6, 3 000 pas, graines 0 et 1), puis on lit l'état (I1 et I2, ou l'état caché du GRU) au moment de chaque question, et des sondes linéaires prédisent, pour chaque personne, son lieu actuel (4 classes) et l'objet qu'elle tient (6 classes). **Critères** : rémanence entraînée ≥ 0,8 de précision des sondes sur les lieux : **55 %** ; ≥ 0,1 au-dessus du témoin non entraîné : **75 %** ; au moins aussi bonne que le GRU (−0,03) : **50 %**.

## Amendement 44 : dynamique de la taille d'image sur CPU, contre T_rope à paramètres comparables (exploratoire, pré-enregistré avant mesure, 2026-11-24)
**Question.** En attendant la suite GPU (amendement 40) : sur un petit montage CPU, la rémanence se rapproche-t-elle de T_rope quand on grossit l'image, **à nombre de paramètres comparable** ? Sortie voulue : deux courbes (bpc en fonction des paramètres), une pour T_rope agrandi (largeur 64, 96, 128, 192, 2 couches), une pour la rémanence (images 4×8×8, 4×16×16, 8×16×16).
**Montage.** `scripts/ts_remanence.py --dataset wikitext2` (copie pytorch/examples, BPE 4096), fenêtres de 64, batch 16, **1 500 pas pour tous**, graine 0, évaluation sur 256 tokens (0-64 dans la longueur d'entraînement, 64-256 au-delà). Paramètres (vocabulaire 4096) : T_64x2 624 k, T_96x2 1 010 k, T_128x2 1 445 k, T_192x2 2 463 k ; R_4x8x8 950 k, R_4x16x16 1 107 k, R_8x16x16 1 315 k. Une graine, entraînement court : exploratoire, un écart < 3 % est indiscernable ; la fenêtre de 64 limite l'intérêt d'une grande mémoire (le test long est le 41).
**Pronostics et critères (bpc sur 0-64).**
- (a) à paramètres égaux, T_96x2 (1,01 M) bat R_4x8x8 (0,95 M) d'au moins 3 % : **60 %**.
- (b) R_4x16x16 améliore R_4x8x8 d'au moins 3 % : **45 %**.
- (c) R_4x16x16 est dans ±3 % de T_96x2 (paramètres comparables) : **25 %**.
- (d) la pente (bpc par doublement de paramètres) de la courbe rémanence est plus forte que celle de T_rope : **35 %**.
- (e) sur 64-256, la rémanence est meilleure que T_rope de même taille : **55 %** (T_rope n'a jamais vu de positions > 64).
Lecture : si (b) et (d) échouent, grossir l'image ne rattrape pas T_rope dans ce régime ; si (b) réussit sans (c), l'image aide mais ne suffit pas.

### Résultats de l'amendement 44 (CPU, Wikitext-2, fenêtre 64, 1 500 pas, une graine ; `scripts/courbe_a44.py`, figure `figures/p44_taille_vs_trope.png`)
| modèle | paramètres | bpc 0-64 | bpc 64-256 |
|---|---|---|---|
| T_64x2 | 624 k | 2,102 | 2,132 |
| T_96x2 | 1 010 k | 1,989 | 2,008 |
| T_128x2 | 1 445 k | 1,921 | 1,942 |
| T_192x2 | 2 463 k | 1,827 | 1,890 |
| R_4x8x8 | 950 k | 2,129 | 2,122 |
| R_4x16x16 | 1 107 k | 2,146 | 2,138 |
| R_8x16x16 | 1 315 k | 2,995 | 3,070 (n'a pas quitté le plateau : perte 6,6 à 1 500 pas) |
- (a) **atteint** : T_96x2 bat R_4x8x8 de 6,6 % à paramètres égaux.
- (b) **raté** : R_4x16x16 est 0,8 % *moins bon* que R_4x8x8 (courbe de perte plus lente jusqu'au pas 500, rattrapée ensuite, finale 4,71 contre 4,67).
- (c) **raté** : R_4x16x16 est 7,9 % derrière T_96x2.
- (d) **raté** : la courbe de la rémanence est plate (puis s'effondre pour 8×16×16), celle de T_rope descend d'environ 3-4 % par ×1,5 de paramètres.
- (e) **raté** : sur 64-256, T_96x2 reste meilleur que la rémanence (2,008 contre 2,122), malgré la longueur d'entraînement de 64.
**Lecture.** Dans ce régime (1 500 pas, fenêtre 64), grossir l'image n'aide pas la rémanence et ne la rapproche pas de T_rope ; à paramètres comparables, T_rope est meilleur de 6 à 8 %. Le plus grand (8×16×16) n'a pas démarré (même phénomène que les plateaux d'amorçage déjà vus, une graine). Limites : entraînement très court, la rémanence démarre plus tard que T_rope ; l'amendement 40 (2 000 pas, 128 tokens, GPU) reste le test plus informatif.

### Suite de l'amendement 44 : entraînement long (6 000 pas, pré-enregistré avant mesure, 2026-11-24)
Même montage, 6 000 pas (≈ 2,3 époques de Wikitext-2, surapprentissage possible : on lit aussi l'écart avec la validation), modèles `R_4x8x8`, `T_96x2` (paramètres égaux), puis `R_4x16x16` si le temps le permet. Pronostics (bpc 0-64) : (f) l'écart R_4x8x8 / T_96x2 passe de 6,6 % à moins de 3 % : **20 %** ; (g) il reste ≥ 3 % : **80 %** ; (h) R_4x16x16 meilleur que R_4x8x8 d'au moins 3 % : **35 %**.

## Amendement 45 : entraînement alterné images / transformers : est-ce la raison du démarrage tardif ? (pré-enregistré avant mesure, 2026-11-24)
Idée de Mathis : les images (la façon dont les tokens s'y écrivent et s'y lisent) et les transformers (la dynamique entre images) se gênent peut-être en apprenant en même temps, d'où le décrochage tardif observé sur S4 (amendement 27b : 5 500, jamais, 12 500 pas selon la graine pour `L_loop1`). On entraîne donc **les images seules, puis les transformers seuls, en alternance, puis tout ensemble plus tard**. « Lenteur » est lu comme **lenteur à démarrer l'apprentissage (en pas)** ; le temps par pas ne change pas (la rétropropagation traverse toujours tout le modèle), ce n'est donc pas un test de vitesse de calcul.
Modèle : `L_loop1`, S4, config de l'amendement 27b (15 000 pas, curriculum 4→24 sur les 40 % premiers pas, lot 64, lr cosinus). Deux groupes de paramètres (`scripts/cpu_alt.py`) : **images** = `emb`, `dep` (token → image), `i1`, `i2` (images initiales), `ret1`, `head` (lecture de l'image) → 46 k paramètres ; **transformers** = `t1`, `t2` → 109 k. Schéma `alt` : phases de 250 pas alternant images seules (en premier) puis transformers seuls pendant les 6 000 premiers pas (= la durée du curriculum), puis tout ensemble jusqu'à 15 000. Le groupe figé ne reçoit aucune mise à jour (pas d'Adam, pas de décroissance de poids). Arrêt anticipé dès ≥ 99 % à n=24 sur 2 évaluations de suite (le schéma de lr et de curriculum reste celui de 15 000 pas).
Graines 0, 1, 2 du schéma `alt`, **comparées par paire** aux runs `joint` déjà faits (`results/s4_L_loop1_s{0,1,2}_p15000.json`, même graine donc même initialisation et mêmes lots) : décrochage = premier pas d'évaluation (tous les 500) avec ≥ 90 % à n=24 ; référence `joint` : 6 000, jamais (69 % à 15 000), 13 000.
Critères : **alterné plus rapide** si, sur ≥ 2 graines sur 3, le décrochage de `alt` est strictement plus tôt que celui de `joint` (un « jamais » compte comme 15 001) ; **alterné plus lent** si, sur ≥ 2 graines sur 3, il est strictement plus tard ou absent alors que `joint` a décroché ; **pas de différence** sinon. Exploratoire : précision finale à n = 192, nombre de graines qui apprennent (réf. 2 sur 3), moment du décrochage par rapport à la fin de l'alternance (6 000 pas).
**Pronostic (écrit avant).** Plus rapide 30 %, pas de différence 40 %, plus lent 30 %. Raisons de prudence : (i) trois graines seulement et un décrochage très variable (5 500 à 13 000 pas) ; (ii) figer les transformers pendant que les images apprennent revient à les laisser en initialisation aléatoire pendant 125 pas à la fois, ce qui peut aussi retarder ; (iii) le plateau sur S4 est peut-être lié à la tâche (state tracking sur 24 états) et pas à une interférence images/transformers. Un résultat nul sera rapporté tel quel. Si « plus rapide », étape suivante : graines 3 et 4 pour `alt` et `joint`.

**Amendement 44, suite, résultats partiels, 6 000 pas (R_4x8x8 et T_96x2 ; R_4x16x16 en cours).** bpc 0-64 : R_4x8x8 **1,815**, T_96x2 **1,746** (écart 4,0 %, contre 6,6 % à 1 500 pas) ; bpc 64-256 : 1,809 contre 1,808 (égalité). (f) **raté** (écart ≥ 3 %), (g) **atteint**. L'écart se réduit avec l'entraînement (la rémanence démarrait plus tard), et au-delà de la longueur d'entraînement la rémanence égale T_rope. Pas de mesure de perte d'entraînement comparable à la validation (perte d'entraînement par token, bruitée) : le surapprentissage n'est pas évalué.

### Amendement 44, balayage du learning rate (pré-enregistré avant mesure, 2026-11-24)
**Question.** Le learning rate unique (1e-3, jamais réglé) désavantage-t-il la rémanence ? Montage de l'amendement 44 (Wikitext-2, fenêtre 64, 1 500 pas, graine 0), `--lr` 5e-4 et 2e-3 pour `R_4x8x8` et `T_96x2` ; la valeur 1e-3 est déjà mesurée (R 2,129 / T 1,989 sur 0-64, écart 6,6 %).
**Pronostics.** (i) le meilleur learning rate de la rémanence est 2e-3 (plus haut que 1e-3) : **45 %** ; (ii) celui de T_rope est 1e-3 ou 2e-3 : **75 %** ; (iii) l'écart rémanence/T_rope au meilleur learning rate de chacun passe sous 3 % : **15 %**. Un écart < 3 % entre deux learning rates est indiscernable (une graine).

**Résultats du balayage (bpc 0-64 ; perte d'entraînement au pas 1 500).** 
| lr | R_4x8x8 | T_96x2 | écart |
|---|---|---|---|
| 5e-4 | 2,332 (5,19) | 2,149 (4,70) | 8,5 % |
| 1e-3 | 2,129 (4,67) | 1,989 (4,30) | 7,0 % |
| 2e-3 | 2,018 (4,38) | 1,865 (4,00) | 8,2 % |
(i) **atteint** (meilleur = 2e-3, mais c'est le bord de la grille) ; (ii) **atteint** ; (iii) **raté** (écart 8,2 % au meilleur lr testé). Les deux modèles gagnent à un lr plus haut (≈ 5 % de 1e-3 à 2e-3 pour la rémanence, 6 % pour T_rope) : l'optimum n'est pas encadré, 4e-3 reste à tester. À 1 500 pas, le lr de 1e-3 n'explique donc pas l'écart : il se maintient à tous les lr testés (le 7,0 % ici, calculé sur les bpc 0-64 affichés, remplace le 6,6 % cité plus haut, qui était le même calcul arrondi différemment).

### Résultat de l'amendement 45 (2026-11-24)
`L_loop1`, S4, 15 000 pas, schéma alterné (images seules / transformers seuls, 250 pas chacun, jusqu'à 6 000 pas, puis tout ensemble). Précision à n=24 / n=192 en fin de run : graine 0 = **21 % / 10 %** (classique : 100 % / 100 %, décrochage à 6 000 pas) ; graine 1 = **36 % / 26 %** (classique : 69 % / 52 %, jamais ≥ 90 %) ; graine 2 = **36 % / 26 %** (classique : 100 % / 100 %, décrochage à 13 000 pas). Aucune des trois graines alternées n'atteint 90 % à n=24. Critères tels qu'écrits : **alterné plus lent** (pour les 2 graines où le classique a décroché, 0 et 2, l'alterné n'a jamais décroché ; sur la graine 1 le classique n'a pas décroché mais l'alterné est aussi plus bas). Pronostic : « plus rapide » 30 % (infirmé), « pas de différence » 40 % (infirmé), « plus lent » 30 % (confirmé). Constat factuel : la précision de l'alterné augmente lentement pendant et après l'alternance (graine 2 : 18 % à 6 000 pas, 35 % à 12 000), sans le saut brutal du classique ; il se bloque vers 35 % (graines 1 et 2) ou 21 % (graine 0). Les 108 k paramètres des transformers ne sont donc pas gênés par les images qui apprennent en même temps : apprendre ensemble est nécessaire, pas nuisible, à ce budget. Limites : trois graines, une seule configuration d'alternance (phases de 250 pas, 6 000 pas d'alternance) ; d'autres rythmes (phases plus longues, alternance plus courte) ne sont pas testés ; la lenteur par pas de calcul n'est pas concernée. Résultats : `results/alt_alt_s{0,1,2}.json`.

## Amendement 46 : profondeur par token (plusieurs passes T1/T2, à la Ouro/Huginn) + état d'image entre tokens, sur S4 (exploratoire, pré-enregistré avant mesure, 2026-11-25)
**Idée.** Les LoopLM récents (Ouro, Huginn) réappliquent les mêmes couches plusieurs fois par token (profondeur) ; la rémanence garde un état entre tokens (mémoire). Les combiner : `L_loop1_pK` = K passes T1→T2 par token (mêmes poids, aucun paramètre en plus), lecture après la K-ième mise à jour de T1. K=1 est identique à `L_loop1` (vérifié : écart de sortie 0,0).
**Montage.** `scripts/cpu_s4.py L_loop1_p1|p2|p4 GRAINE 4000` (S4, config de l'amendement 27, 4 000 pas, **pas** les 15 000 de 27b : on regarde la vitesse de démarrage, un budget plus long coûterait 1 h par run et par passe). Graines 0 et 1. Coût ≈ K × 14 min. Mesure : précision à n=12 et n=24 à 4 000 pas, et le pas où la précision à n=24 dépasse 0,30 (hasard 0,042) s'il existe. Pas de T_rope ni de GRU ici : la question est « plus de passes aide-t-il la rémanence ? », à mêmes pas et non à même calcul.
**Pronostics et critères.**
- (a) `p2` meilleur que `p1` d'au moins 0,10 de précision absolue à n=24 sur les deux graines : **30 %**.
- (b) `p4` meilleur que `p2` sur les deux graines (même seuil) : **20 %**.
- (c) aucun écart ≥ 0,10 entre p1, p2, p4 (la profondeur n'aide pas à ce budget) : **50 %**.
- (d) `p4` pire que `p1` sur au moins une graine (démarrage plus lent, instabilité) : **25 %**.
**Règle d'arrêt.** Si (c) : on n'explore pas la profondeur davantage sur CPU. Une graine de plateau escapée tôt ou tard fait varier le résultat de ±0,2 (vu en 27b) : un écart < 0,15 sur une seule graine est indiscernable.

### Résultats de l'amendement 46 (S4, 4 000 pas, `scripts/cpu_s4.py`, précision finale à n=12 / n=24 ; hasard 4,2 %)
| passes | graine 0 | graine 1 | durée du run |
|---|---|---|---|
| p1 | 0,293 / 0,191 | 0,285 / 0,166 | 12-14 min |
| p2 | 0,283 / 0,164 | 0,291 / 0,165 | 25 min |
| p4 | 0,279 / 0,163 | 0,277 / 0,158 | 50 min |
- (a) **raté** : p2 n'est pas meilleur que p1 (−0,03 sur la graine 0, 0,00 sur la graine 1). (b) **raté**. (c) **atteint** : l'écart maximal entre profondeurs est 0,03 (n=24), bien sous 0,10. (d) **raté** (p4 est un peu plus bas que p1 sur les deux graines, mais de 0,03 et 0,01 : indiscernable).
- Aucune des six courbes n'a quitté le plateau (≈ 0,16-0,19 à n=24) à 4 000 pas. Ce test ne mesure donc que le début de l'apprentissage, pas la tâche résolue (S4 est résolue à 15 000 pas par `L_loop1`, amendement 27b).
**Lecture.** À ce budget, plus de passes par token (même poids) ne change pas la vitesse de démarrage sur S4 et coûte K fois plus de calcul. Pas d'effet visible, mais le test est court et ne dit rien de ce que donnerait la profondeur après le décrochage, ni à plus grande échelle (Ouro, Huginn) ni sur une tâche où la profondeur par token compte (arithmétique, raisonnement multi-étapes), que S4 n'est peut-être pas. Règle d'arrêt (c) : on ne pousse pas ce test plus loin sur CPU.
