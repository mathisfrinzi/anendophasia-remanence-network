# 2 — Protocole pré-enregistré

> Verrouillé le 2026-11-17 **avant toute mesure** (règle maison : critères GO/NO-GO poussés avant les expériences). Rien dans ce fichier ne sera modifié après le premier pas d'entraînement ; toute déviation sera consignée dans les résultats.

## 1. Données

- TinyStories (roneneldan/TinyStories), split train, en streaming, **dans l'ordre** : les 306 000 premières histoires.
- 300 000 pour l'entraînement, les 6 000 suivantes pour la validation.
- Tokenisation **niveau caractère** ; vocabulaire = caractères triés par point de code, construits sur train + val, rien filtré.
- Tâche : prédire le caractère suivant ; fenêtres de 256 caractères.

## 2. Les bras

Même interface pour tous : forward (B, T) → logits (B, T, V) ; pertes d'entropie croisée sur le caractère suivant.

| Bras | Implémentation | Rôle |
|:---|:---|:---|
| 1 | Transformer decode-only (pré-LN, tête liée à l'embedding) | baseline standard |
| 2 | GRU à 3 couches (cudnn) | témoin récurrent simple |
| 3 | SSM plat sélectif (Mamba simplifié : A diagonal réel, scan séquentiel exact, gating SiLU, conv causale depthwise) | témoin « état plat, gates modernes » |
| 4 | Canvas 1D (V0) : bande C×N, écriture (patch + masque par canal), decay input-dépendant, interactions locales par conv depthwise et lissage | l'échillon de l'idée |
| 5 | Canvas 2D (V3) : plan C×H×W, mêmes mécanismes en 2D | le pari |
| 6 | Vidéo (version littérale) : encodeur appris vers vidéo couleur, SSM sur les frames, décodeur symétrique | la version de départ de l'idée, à armes égales avec les transformers (deux tâches) |

## 3. Égalité de budget (critère pré-enregistré)

- Comptage des paramètres entraînables par src/budget.py **avec le vocab réel, avant tout entraînement**.
- Critère : **ratio max/min ≤ 1,15** entre les bras.
- Comptages attendus (vocab 100) : bras1 ≈ 4,04 M · bras2 ≈ 3,71 M · bras3 ≈ 3,56 M · bras4 ≈ 3,70 M · bras5 ≈ 3,70 M (ratio attendu ≈ 1,13).
- Mêmes tokens vus : 8 000 pas × lot 16 × fenêtre 256 = **32 768 000 caractères par bras**.
- Lots identiques entre bras : RNG numpy seed 42, même séquence de fenêtres.

## 4. Hyperparamètres identiques pour tous

Adam, lr 1e-3, betas (0,9 ; 0,999), warmup linéaire 100 pas, clip de gradient 1,0, lot 16, fenêtre 256, 8 000 pas. **Aucun réglage par bras, aucun tuning post hoc.**

## 5. Métrique de performance

- Perte de validation en **nats/caractère**, moyenne sur des lots fixes (seed 123, 64 lots), mesurée tous les 500 pas.
- Rapportée aussi en bits/caractère (bpc) et ppl = exp(perte).
- La « val finale » = mesure après le pas 8 000.

## 6. Critères de structure (bras 4 et 5 uniquement)

Passe de 200 000 caractères de la validation ; pour chaque caractère lu, la position d'écriture (argmax du masque moyenné sur les canaux, aplatie) et l'embedding d'écriture (sortie de tronc) sont collectés.

- **S1** (l'endroit où un caractère est écrit prédit-il le caractère ?) : F d'ANOVA à un facteur sur les positions d'écriture des 50 caractères les plus fréquents (compte ≥ 1000), p-value par permutation (1 000 permutations, seed 0). Seuil : **p < 0,001**.
- **S2** (la géométrie suit-elle les écritures ?) : Spearman entre distance canvas (1D : distance sur la bande ; 2D : euclidienne des positions moyennes) et distance cosinus des embeddings d'écriture moyens, sur les paires des 50 caractères les plus fréquents, bootstrap 1 000 (IC 95 %). Seuil : **r > 0,2 et borne basse de l'IC > 0**.

## 7. GO / NO-GO

- **G1 (performance)** : min(val finale des bras canvas) ≤ 1,10 × min(val finale des baselines).
- **G2 (structure)** : au moins un bras canvas satisfait S1 **et** S2.
- **GO global = G1 et G2.** Tout le reste = NO-GO, consigné avec la même rigueur qu'un succès. Un NO-GO sur G1 avec G2 réussi sera consigné comme « structure sans performance » : piste ouverte pour les idées suivantes.

## 8. Artefacts

- results/brasN_metrics.json (avec courbe) ; brasN_samples.txt ; brasN_structure.json + .npz (bras 4-5) ; brasN_ckpt.pt (non versionné, dans le zip de session Colab).
- results/comparaison.md + courbes.png + heatmaps : générés par le rapport CPU (scripts/run_report.ps1).

## 9. Matériel et durées estimées (T4 Colab)

bras 1-2 : < 30 min chacun · bras 3 : 2-3 h · bras 4-5 : 1-2 h chacun · bras 6 : ~6-8 h (deux étages, cf. section 11) · rapport : CPU local, ~2 min.

## 10. Configs exactes

| Bras | Config |
|:---|:---|
| 1 | d=256, 5 blocs, 4 têtes, pos-emb 256, tête liée |
| 2 | emb 128, caché 480, 3 couches |
| 3 | d=256, 8 blocs, N=16, dt_rank=16, conv 4, expand 2, SiLU, pré-LN |
| 4 | emb 128, C=96, N=96, tronc 192, decay conv 1×1, depthwise 5, lissage α (init −3) |
| 5 | emb 128, C=64, 12×12, tronc 192, depthwise 3×3, lissage α (init −3) |
| 6 | emb 128, frames couleur 3×8×8, tronc 1152, d=192, 11 blocs SSM (N=16, dt_rank=16, conv 4, expand 2), depthwise 3×3, lecture 512, requêtes positionnelles 256×128 (étage 1) |

## 11. Amendement pré-enregistré (2026-11-17, avant toute mesure du bras 6)

> Ajout d'un sixième bras après réception des résultats des bras 1-2. **Aucune mesure du bras 6 n'a été lancée à la date de cet amendement** : les critères ci-dessous sont donc **pré-enregistrés avant mesure**, conformément à la règle maison. Les sections 1-10 ci-dessus restent inchangées. La présente rédaction remplace la première version (un seul étage) de cette section, elle aussi avant toute mesure.

**Bras 6 — Vidéo (version littérale de l'idée d'origine)** : le texte est encodé en **vidéo couleur** (une frame par caractère : image 3×8×8 persistante où chaque caractère est ajouté par une tête d'écriture apprise, content + masque softmax par canal, decay ∈ [0,5 ; 1], lissage α) ; un **SSM sélectif** (Mamba, mêmes blocs que le bras 3) **appliqué aux frames** (une frame = un pas du scan) transforme la vidéo d'entrée en **vidéo de sortie** (résiduel : la sortie part de l'entrée) ; un **décodeur symétrique de l'encodeur** relit le texte depuis la vidéo de sortie (attention sur les pixels + MLP → vocab).

- Config exacte (config.py, VIDEO) : emb 128, frames couleur 3×8×8, tronc 1152, d=192, 11 blocs SSM (N=16, dt_rank=16, conv 4, expand 2, SiLU, pré-LN), depthwise 3×3 (écriture + pré-mix des frames), lecture 512. **3 731 861 paramètres avec le vocab réel (V=127)**, dont 32 768 de requêtes positionnelles ; à revérifier par budget.report avant lancement (critère ratio ci-dessous).
- **Deux entraînements, pré-enregistrés** (remplacent la spec à un seul étage ; motivation : les deux tâches, auto-encodage et prédiction, au même budget de **stockage**, à armes égales avec les transformers dont les vectorisations sont faites sur mesure) :
  - **Étage 1 — autoencodeur visuel, paires (texte, texte)** : **sans récurrence** (le SSM n'intervient pas). Le texte entier (fenêtre de 256 caractères) est **relu depuis la frame finale** de la vidéo d'entrée par le décodeur, avec une **requête positionnelle apprise par position de sortie** (nn.Embedding(256, 128), une par position) ; cible = le texte lui-même, entropie croisée. But : diagnostique de la **capacité du canal image** (192 valeurs continues pour relire 256 caractères : la reconstruction parfaite n'est **pas attendue**, c'est précisément la mesure). Artefacts dédiés : results/bras6_video_ae_metrics.json (perte de reconstruction recon_val_loss_final, recon_bpc, recon_ppl), bras6_video_ae_samples.txt (textes originaux vs reconstruits), bras6_video_ae_progress.json. **Ne compte pas dans G1.**
  - **Étage 2 — paires (texte_entree, texte_sortie)** : pipeline complet (écriture → SSM sur les frames → lecture), la sortie à la position t prédit le caractère t+1, la requête du décodeur est l'embedding du caractère courant. **Part des poids de l'étage 1 et tous les paramètres continuent d'apprendre** (fine-tune global, pas de freeze). **G1 porte sur la val finale de cet étage uniquement.**
  - **Compute** : chaque étage = 8 000 pas × lot 16 × fenêtre 256 = **32 768 000 tokens**, soit **2 × 32 768 000 au total** pour le bras 6. **Déviation de compute par rapport aux bras 1-5, pré-enregistrée ici ; budget de paramètres inchangé.**
- **Budget** : le critère ratio max/min ≤ 1,15 s'étend aux **six bras** (src/budget.py), vérifié avant lancement avec le vocab réel (actuellement transformer 4 047 360 / SSM 3 575 808 = 1,132 ; bras 6 = 1,04 × SSM).
- Mêmes données, mêmes hyperparamètres, même métrique que les bras 1-5 (val nats/car, lots fixes seed 123, mesurée tous les 500 pas, après 8 000 pas par étage).
- **G1 s'applique au groupe canvas élargi {4, 5, 6}** contre les baselines {1, 2, 3} (seuil inchangé : 1,10 × min baselines), sur la val finale de l'étage 2.
- **S1/S2 sont collectés aussi pour le bras 6** (l'encodeur écrit sur une image : mêmes analyses, mêmes seuils, mêmes 200 000 caractères), consignés à titre comparatif ; le G2 au sens strict reste celui de la section 7 (bras 4-5 verrouillés).
- **Monitoring visuel pré-enregistré** : à chaque évaluation (tous les 500 pas), les **4 images de « the cat is black »** (une par mot : fins de mots aux indices 0-based 2, 6, 9, 15) sont affichées et sauvegardées — **étage 1** dans `results/bras6_viz/ae_step_XXXXXX.png` (rangée du haut : vidéo d'entrée, l'image où le texte s'accumule ; rangée du bas : carte d'attention du décodeur, où il lit la frame finale pour reconstruire chaque position ; titre : nombre de caractères reconstruits correctement sur 16) ; **étage 2** dans `results/bras6_viz/step_XXXXXX.png` (rangée du haut : vidéo d'entrée ; rangée du bas : vidéo de sortie ; échelle fixe [−1,5 ; +1,5] pour rester comparable au fil de l'entraînement). **Purement qualitatif : aucun critère GO/NO-GO ne porte sur ces images.**
- Durée estimée : **~6-8 h sur T4** (deux fois 8 000 pas ; reprise automatique par checkpoints results/bras6_video_ae_ckpt.pt puis results/bras6_video_ckpt.pt, comme les autres bras).

Correction de bug préalable (aucune mesure du bras 3 n'ayant été lancée à cette date) : `ssm.SSMLM.step` renvoyait des logits 3D (B, 1, V) alors que `train.generate` appelle `torch.multinomial` (2D obligatoire, convention des bras 1-2) ; corrigé en (B, V). Aucun impact sur forward ni sur les autres bras.

## 12. Amendement pré-enregistré (2026-11-17, avant toute mesure S1/S2 des bras canvas)

> Contexte, consigné honnêtement : première exécution de `structure.collect_and_save` après l'entraînement du bras 4 → **crash ZeroDivisionError**. Cause identifiée dans le code (pas dans le run) : dans les bras canvas (4, 5, 6), le masque d'écriture ne dépend que de l'embedding du caractère (pas du canvas), donc **la position d'écriture est une fonction déterministe du caractère** : chaque caractère écrit toujours à la même position, la variance intra-groupe de l'ANOVA S1 est nulle par construction (ssw = 0) et le F standard est indéfini (division par zéro). **Aucune mesure S1/S2 n'existe à ce jour** — le crash du bras 4 n'a produit ni JSON ni npz (la statistique n'a jamais été calculée ni observée), et les bras 5 et 6 n'ont pas encore été mesurés — : le présent amendement est donc **pré-enregistré avant toute mesure S1/S2**. Les sections 1-11 restent inchangées ; les seuils et critères des sections 6 et 7 ne sont **pas modifiés**.

- **Règle pour le cas dégénéré (fix de src/structure.py, commit d145f05)** : application de la limite mathématique exacte du test par permutation de la section 6 —
  - ssw = 0 et ssb > 0 (des caractères fréquents écrivent à des positions différentes) → la position prédit parfaitement le caractère : **F = +∞, p = 0** (tout F issu d'une permutation est fini, donc jamais ≥ +∞) ;
  - positions toutes identiques, ou un seul caractère fréquent, ou aucun caractère à ≥ 1000 occurrences → aucune information : **F = 0, p = 1, échec consigné** ;
  - S2 à distances constantes (positions moyennes toutes identiques, ou embeddings d'écriture équidistants) → corrélation non définie : **échec consigné** (valeurs non finies stockées explicitement avec note, jamais silencieusement).
- **Diagnostic ajouté aux artefacts** : nombre de **positions d'écriture distinctes** parmi les caractères fréquents (`S1_n_pos_distinct`), imprimé à l'écran et stocké dans structure.json, consigné dans le rapport. Il répond à « la tête d'écriture est-elle diverse ou effondrée sur quelques positions ? ».
- **Lecture scientifique consignée** : par construction, S1 ne teste pas un placement contextuel mais « des caractères différents écrivent-ils à des positions différentes » ; il est trivialement satisfait dès que la tête d'écriture discrimine les caractères. **S2 (géométrie : la disposition des positions suit-elle les distances d'embeddings) et le diagnostic de diversité restent les critères discriminants** pour G2, inchangés.
- Le bras 4 sera re-mesuré (structure uniquement) avec ce code : ce sera la **première** mesure S1/S2 pour ce bras ; les bras 5 et 6 seront mesurés avec ce code également.
