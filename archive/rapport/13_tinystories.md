[Sommaire](README.md) · [← Partie 12](12_rnn_memoire.md)

# Partie 13 — TinyStories sur le GPU de l'utilisateur (amendement 36)

> **Ce qu'il faut retenir**
>
> - Sur TinyStories (1 950 pas, batch 64, ≈ 8 M tokens, 1 graine), la rémanence et T_rope sont **identiques sur les 64 premiers tokens** (1,33 / 1,21 / 1,26 bits par caractère).
> - **Au-delà de la longueur d'entraînement (64 tokens)**, la rémanence reste stable (1,25 sur 128-256) alors que T_rope se dégrade (1,48), soit **15 % d'avantage** ; c'est le même constat qu'en partie 6, plus net.
> - Les deux modèles écrivent des phrases grammaticales mais pas des histoires cohérentes. La rémanence est ≈ 26 fois plus lente.

| Positions | 0-16 | 16-32 | 32-64 | 64-128 | 128-256 |
|---|---|---|---|---|---|
| Rémanence (`L_loop1`) | 1,330 | 1,210 | 1,259 | 1,225 | **1,254** |
| T_rope | 1,330 | 1,211 | 1,257 | 1,235 | 1,481 |

Exemple (rémanence) : « Once upon a time, there was a little boy named Timmy. [...] He was very excited to take a nap. » Histoires complètes : `results/ts_samples_*`.

**Prudence** : l'avantage au-delà de 64 tient surtout à la faiblesse de T_rope hors des positions vues ; un transformer entraîné sur des fenêtres plus longues ne le montrerait probablement pas. Une graine. Batch 64 au lieu de 256 (plantage du pilote XPU).

Code : `scripts/ts_remanence.py`. Protocole : amendement 36.
