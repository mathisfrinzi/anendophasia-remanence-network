[Version française](RESULTATS.md)

# Detailed results

All figures come from `results/*.json`. "A12" refers to amendment 12 of [PROTOCOLE.md](preenregistrements/PROTOCOLE.md) (in French), which holds the full setup, the criteria and predictions written before the measurement, and the raw results. Accuracy in %, bpc = bits per character (lower is better), "seed" = random initialization. Chance level is given in parentheses.

The model. Two 4×8×8 images (256 numbers each) make up the entire memory; there is no attention over the past and no position. For each token: (1) the token is deposited on image 1; (2) T1, a small transformer over the 16 patches of image 1, writes into image 2 through a learned forget gate; (3) a head reads image 2 and predicts; (4) T2 reads image 2 and rewrites image 1 (same gate). Code: [`src/s4_loop.py`](../src/s4_loop.py) (`LoopLM`), equivalent fast version [`src/s8_rapide.py`](../src/s8_rapide.py).

## Contents

1. [State: does the Remanence Network track a state?](#1-state-does-the-remanence-network-track-a-state)
2. [Memory: does it retain a value?](#2-memory-does-it-retain-a-value)
3. [Against recurrent networks](#3-against-recurrent-networks)
4. [Language](#4-language)
5. [Image size and equal parameters](#5-image-size-and-equal-parameters)
6. [Architecture variants](#6-architecture-variants)
7. [Learning speed, seeds, cost](#7-learning-speed-seeds-cost)
8. [Preliminary steps: text → image, video](#8-preliminary-steps-text--image-video)
9. [Corrections made along the way](#9-corrections-made-along-the-way)
10. [What remains open](#10-what-remains-open)

---

## 1. State: does the Remanence Network track a state?

Verdict: yes, solid on S3 (5 seeds out of 5, controls at equal capacity).

S3: the model reads a sequence of permutations of 3 elements and must output their composition at each step (6 states, chance 16.7%). Trained on sequences of ≤ 12, tested up to 96. CPU, 2,000 steps, d = 64 (A15).

| model | parameters | 0-12 | 12-24 | 24-48 | 48-96 |
|---|---|---|---|---|---|
| **remanence** | 252 k | 100 ×5 | 100 ×5 | 100 ×5 | 100 ×5 |
| control: T2 writes into image 2 (no feedback) | 252 k | 64-79 | 32-35 | 29-33 | 27-32 |
| control: widened T1, no T2 | 248 k | 59-66 | 16-20 | 16-17 | 16-17 |
| no T2 | 148 k | 58-66 | 17-18 | 16-17 | 16-17 |

- The T2 → image 1 feedback is what creates the state: at equal parameters, both controls stay at chance outside the training range.
- Ablations (A16, 1 seed): without the forget gate → 17-22%; reading from image 1 instead of image 2 → 17%; a single layer per transformer → still 100%.
- On GPU (trained on 24, tested up to 256, A12): 2 seeds out of 3 at 100% up to 256; T_rope 17% beyond 48.
- S4 (24 states, chance 4.2%, 15,000 steps, A27b): accuracy at n = 24 / 192: remanence 100 / 100, 70 / 52, 100 / 100 (3 seeds); T_rope 40 / 8.5 (1 seed). At 4,000 steps, no model takes off.
- S5 (120 states, A10, A31): nobody learns in 8,000 steps, GRU included (≈ 9-13% within the training length, chance beyond).

## 2. Memory: does it retain a value?

Verdict: yes, solid (3 seeds out of 3).

S3M: a value among 8 is given at the start, then n S3 elements, then the value is queried (chance 12.5%). Trained on n ≤ 24, 3,000 steps (A17).

| value recall at n = | 48 | 96 |
|---|---|---|
| **remanence** | 100 ×3 | 98.0 / 100 / 97.7 |
| no T2 | 80.5 / 91.0 / 96.5 | 6.2 / 55.5 / 42.6 |
| T_rope | 14.8 / 9.0 / 74.6 | 1.2 / 0.0 / 22.3 |
| GRU, LSTM | 100 ×3 | 100 ×3 |

- With images 8 times smaller (2×4×4 = 32 numbers), the remanence keeps 84 to 100% at n = 96 (A18).
- Linear probes: the value is readable in both images at 94-100% throughout the delay, spread across all channels.
- Two values to retain (A33): 2 seeds out of 3 succeed.

## 3. Against recurrent networks

Verdict: the remanence does no better than a GRU or an LSTM (A30, A28, 3 seeds).

| task | GRU | LSTM | remanence |
|---|---|---|---|
| S3, accuracy 48-96 | 99.6-100 (≈ 30 s) | 100 (≈ 31 s) | 100 (≈ 675 s) |
| S3M, recall at n = 96 | 100 (≈ 55 s) | 100 | 97.7-100 (≈ 670 s) |
| circular list, 3 hops | 100 (≈ 190 s) | 100 | 100 (≈ 2,400 s) |
| circular list, 4 and 5 hops (never seen) | 1-12 | 1-9 | 0-13 |
| Silverbox, test error (mV) | **9.56** ± 0.02 | 9.60 ± 0.03 | 10.82 ± 0.56 |

- Silverbox (A28): identification of a nonlinear electronic circuit. T_rope 15.29 mV, linear regression 14.03, remanence without T2 41.96. Without T2, the error is almost 4 times larger: the feedback matters here too. No network does better than the linear model within the training range.
- Circular list (A23): follow h pointers in a list of 6. The remanence succeeds at 3 hops on 3 seeds out of 3; T_rope plateaus at 21-26% at 3 hops. "Thinking pauses" before the answer do not help. Trained up to 5 hops (A32), the remanence manages it for only 1 seed out of 3; the GRU manages it (1 seed tested).

## 4. Language

Verdict: the remanence is on par with a transformer of the same size within the training length, more stable beyond it, and behind a GRU. One seed per configuration.

Six corpora on CPU (A20, A22; windows of 64, 2,000 steps, ≈ 1 M parameters; bpc 0-64 / 128-256):

| corpus | T_rope | remanence |
|---|---|---|
| Wikitext-2 | **2.086** / 2.096 | 2.139 / 2.102 |
| Python code | **1.739** / 1.844 | 1.795 / **1.791** |
| Penn Treebank | 1.622 / 1.635 | **1.604 / 1.578** |
| Gutenberg tales | **2.051** / 2.199 | 2.071 / **2.016** |
| full Gutenberg | **2.190** / 2.494 | 2.229 / **2.278** |
| Shakespeare | **2.338** / 2.603 | 2.425 / **2.414** |

Within the training window, the remanence is at −1 / +4% of T_rope; beyond it, T_rope degrades by 6 to 14% on 4 corpora and the remanence overtakes it. The control that would train T_rope on longer windows (A37) could not be interpreted (budgets too short).

TinyStories and Wikitext-2 on GPU (A40; 2,000 steps, windows of 128; bpc 0-128 / 128-512):

| model | parameters | TinyStories | Wikitext-2 | duration (Wikitext-2) |
|---|---|---|---|---|
| T_rope | 0.62 M | 1.296 / 1.390 | 2.100 / 2.185 | 82 s |
| T_rope (width 96) | 1.01 M | – | **1.997** / 2.060 | 75 s |
| remanence 4×8×8 | 0.95 M | 1.302 / 1.250 | 2.139 / 2.156 | 2 h 52 |
| remanence 4×16×16 | 1.11 M | 2.525 / 2.506 (did not take off) | 2.024 / **2.040** | 4 h 40 |
| GRU 256 | 1.56 M | 1.281 / 1.227 | 2.085 / 2.091 | 3.6 min |
| GRU 1024 | 7.8 M | **1.095 / 1.041** | **1.930 / 1.932** | 4.8 min |

- At comparable parameters (≈ 1 M), T_rope is 1.3% better than the remanence within the window and 1% worse beyond it: a tie to within one seed.
- GRU 256 matches or beats the remanence 4×8×8 on both corpora.
- Other language tasks, non-discriminating: UD part-of-speech tagging (86.4% for all, A35), bAbI and ListOps (A24, A25: the transformer does not reach the required level, stopped).

## 5. Image size and equal parameters

Verdict: enlarging the image does not help reproducibly; the transformer benefits more from its size.

- A40 (GPU): going from 4×8×8 to 4×16×16 improves the remanence by 5.4% on Wikitext-2, but on TinyStories the 4×16×16 stays on the plateau (loss 6.8). The GRU gains 7 to 15% going from 256 to 1,024.
- A44 (CPU, Wikitext-2, windows of 64, 1,500 steps):

| model | parameters | bpc 0-64 |
|---|---|---|
| T_rope 64×2 | 0.62 M | 2.102 |
| T_rope 96×2 | 1.01 M | 1.989 |
| T_rope 128×2 | 1.45 M | 1.921 |
| T_rope 192×2 | 2.46 M | 1.827 |
| remanence 4×8×8 | 0.95 M | 2.129 |
| remanence 4×16×16 | 1.11 M | 2.146 |
| remanence 8×16×16 | 1.32 M | 2.995 (did not take off) |

- Longer (6,000 steps): the gap between remanence 4×8×8 and T_rope 96×2 goes from 7.0 to 4.0% (1.815 versus 1.746); beyond 64 tokens, a tie (1.809 versus 1.808).
- Learning rate (5e-4, 1e-3, 2e-3): both models prefer 2e-3, and the gap stays at 7 to 8.5%. The single lr of 1e-3 does not explain the shortfall.

## 6. Architecture variants

None beats the base remanence.

| variant | idea | result | source |
|---|---|---|---|
| double loop | 2 input images, 4 transformers | equal on S3M, S4, lists; 2× slower | A26, A29, A33 |
| 2-3-1 image network | 6 images in layers, 18 transformers | does not take off (S4, S5, lists); adding an intermediate layer blocks learning | A31-A35b |
| pretrained ViT | T1, T2 = blocks of `vit_small` (43 M) | inconclusive (600 steps, no control) | A39 |
| alternating training | images, then transformers | slower (S4: 0 seeds out of 3 versus 2 out of 3) | A45 |
| several passes per token | T1→T2 repeated 2 or 4 times (Ouro / Huginn style) | no effect at 4,000 steps on S4, cost ×2 and ×4 | A46 |
| fast version | same computations, fewer operations | outputs identical to 1e-6, ×1.4 to ×1.8 on CPU | A38 |

## 7. Learning speed, seeds, cost

- Plateau then drop: all remanence models first stay at chance level, then abruptly break away. The GRU breaks away 2 to 4 times earlier (S3M: ≤ 500 steps versus 750 to 1,500; lists: 1,500 steps versus 2,500 to 3,500). On S4, the remanence breaks away at 6,000 and 13,000 steps, and one seed never does. Several null results (S5, 2-3-1 network, passes per token) may be plateaus not crossed for lack of budget.
- Seeds: the remanence fails on 1 seed out of 3 in several tasks (S3 GPU, S4, 5-hop lists); image sizes 4×16×16 and 8×16×16 do not take off on some datasets. Single-seed results should be read with this variance in mind.
- Cost: 8× T_rope on CPU (Wikitext-2), 26× on Intel GPU (TinyStories), 49 to 224× on Intel GPU (A40, depending on corpus and size); 12 to 50× a GRU. The loop is limited by the launch of many small operations, not by computation: a GPU barely speeds it up.

## 8. Preliminary steps: text → image, video

- Text in an image (A1-A3): an image of 192 numbers, written character by character like a continuous video, is read back at 99.9% for 16 characters, 97.9% for 32; 53% at 64 (87.9% in 30,000 steps, whereas 32 characters reach 99.6%).
- Video of images read by a transformer without position (A5-A7, TinyStories, 1 seed): on par with T_rope within the training length (0.645 versus 0.625 bpc), much better far beyond (0.846 versus 1.164 at 512-1,024 tokens).
- Video alone is not enough to track a state (A10, A11): S3 learned by 1 seed out of 3. Hence the addition of the T2 feedback: the Remanence Network.
- Readable images? (A14) No: no visible structure, and the imposed continuity brings neither noise robustness nor compactness.

## 9. Corrections made along the way

- The permutation of 6 (A21) left a shortcut ("answer x"): task redone as a circular list (A23).
- A bug in the synthetic story generator (A19) invalidated the first series.
- "The 3-hop limit comes from training" (A32, 1 seed): withdrawn after 2 more seeds (A33b).
- "The large image puts the remanence ahead of the transformer" (A40): true only against a smaller transformer; at equal parameters, a tie.
- The implicit hypothesis of the early days, "RNNs will fail on memory tasks", was refuted (A30): the project went from "the only architecture that learns S3" to "the only transformer without access to the past that learns S3".

## 10. What remains open

1. Scale: everything is ≤ 8 M parameters, ≤ 15,000 steps. Recent LoopLMs (Ouro, Huginn) only work at billions of tokens, and they loop in depth, not in memory.
2. Startup: why the remanence stays on the plateau for so long, and why large images do not take off. A Parcae-type stabilization (constraints on state injection) has not been tried.
3. Very long memory (PG-19, A41) and hybrid transformer + remanence (A42): prepared, not launched.
4. Transformer control trained on long windows: not done properly (A37 not interpretable).
5. What the images contain on language (probes, A43): prepared, not launched.
