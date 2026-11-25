# Configuration pre-enregistree (voir PROTOCOLE.md). Ne pas modifier apres le lancement d'une mesure.

# --- Modeles de langage de reference (bras 1-2, deja mesures) ---
COMMON = dict(
    seed=42,
    eval_seed=123,
    n_train_stories=300_000,
    n_val_stories=6_000,
    seq_len=256,
    batch_size=16,
    steps=8_000,
    lr=1e-3,
    warmup=100,
    grad_clip=1.0,
    eval_every=500,
    eval_batches=64,
    sample_prompts=8,
    sample_prompt_len=100,
    sample_len=300,
    sample_temp=0.8,
)
TRANSFORMER = dict(d_model=256, n_layer=5, n_head=4)
GRU = dict(d_emb=128, d_hidden=480, n_layers=3)

# --- Autoencodeur texte <-> image (etape 1 de la nouvelle direction) ---
AE = dict(
    chunk_sizes=[8, 16, 32, 64],   # caracteres par image
    arms=["cumul", "image", "vector", "none"],   # cumul = video continue (bras principal) ; les 3 autres = encodage d'un seul coup
    latent_c=3, latent_h=8, latent_w=8,   # image 3x8x8 = 192 valeurs ; le temoin vecteur a 192 valeurs aussi
    d_model=128, n_layers=2, n_heads=4,   # encodeur et decodeur de texte
    cumul_tok_ch=8,                        # canaux (par pixel) portant le caractere dans le bras cumul
    spatial_ch=32,                         # canaux de l'etage convolutif (bras image)
    noise_std=0.1,                         # bruit gaussien sur le latent (latent normalise a variance 1) : rend la capacite mesurable
    n_train_stories=50_000, n_val_stories=2_000,
    seed=42, eval_seed=123,
    batch_size=128, steps=8_000, lr=1e-3, warmup=200, grad_clip=1.0,
    eval_every=500, eval_batches=16,
    save_model=False,                      # sauvegarder le modele (results/..._model.pt) pour les analyses
    step_penalty=0.0,                      # lambda de la penalite sur l'indice de continuite (bras cumul) ; 0 = aucune
    success_char_acc=0.99,                 # critere de reussite : >= 99 % des caracteres relus correctement
)

# Mesures supplementaires 1 (PROTOCOLE.md, amendement 2) : DEJA MESUREES le 17 nov. (resultats dans PROTOCOLE.md / discussion).
AE_COMPLEMENT_1 = [
    dict(tag="bruit03", kind="cumul", L=16, steps=8_000, over=dict(noise_std=0.3)),                        # bruit plus fort sur l'image avant le decodeur
    dict(tag="img16", kind="cumul", L=16, steps=8_000, over=dict(latent_h=16, latent_w=16)),               # image 3x16x16 = 768 valeurs
    dict(tag="img16", kind="cumul", L=64, steps=8_000, over=dict(latent_h=16, latent_w=16)),               # la memoire est-elle limitee par la taille de l'image ?
    dict(tag="long30k", kind="cumul", L=32, steps=30_000, over={}),                                        # sous-entrainement ou limite dure ?
    dict(tag="long30k", kind="cumul", L=64, steps=30_000, over={}),
]

# Mesures supplementaires 2 (PROTOCOLE.md, amendement 3) : DEJA MESUREES le 17 nov. (penalite sur la taille des pas, L = 16).
AE_COMPLEMENT_2 = [
    dict(tag="pas03", kind="cumul", L=16, steps=8_000, over=dict(step_penalty=0.3)),
    dict(tag="pas1", kind="cumul", L=16, steps=8_000, over=dict(step_penalty=1.0)),
    dict(tag="pas3", kind="cumul", L=16, steps=8_000, over=dict(step_penalty=3.0)),
]

# Mesures 3 (PROTOCOLE.md, amendement 4) : memes reglages qu'avant, avec sauvegarde du modele puis analyses de structure cachee.
AE_COMPLEMENT = [
    dict(tag="an_base", kind="cumul", L=16, steps=8_000, over=dict(save_model=True), analyze=True),                       # sans penalite
    dict(tag="an_pas1", kind="cumul", L=16, steps=8_000, over=dict(step_penalty=1.0, save_model=True), analyze=True),     # penalite lambda = 1
    dict(tag="an_image", kind="image", L=16, steps=8_000, over=dict(save_model=True), analyze=True),                      # temoin : encodage d'un seul coup
]

# --- Etape 2 : transformer sur tokens (T) contre transformer sur la video continue (V) ---
# Pre-enregistre dans PROTOCOLE.md (amendement 5). Tokens BPE ; metrique en bits par caractere (bpc).
S2 = dict(
    n_train_stories=COMMON["n_train_stories"], n_val_stories=COMMON["n_val_stories"],   # memes donnees que les bras 1-2
    vocab_size=4096,
    ctx=256,                       # tokens par fenetre d'entrainement (~1 000 caracteres)
    batch_size=32, steps=8_000, lr=1e-3, warmup=200, lr_min_frac=0.1, wd=0.01, grad_clip=1.0,
    episode_windows=8,             # fenetres contigues par flux d'entrainement (frame transportee, gradient coupe entre fenetres)
    stream_windows=8, stream_batch=16,   # evaluation en flux : 8 fenetres contigues
    penalty_ramp=1_000,            # la penalite de continuite monte lineairement de 0 a lambda pendant ces pas
    d_model=256, n_layer=5, n_head=4,
    d_tok=128, latent_c=3, latent_h=8, latent_w=8, spatial_ch=32, cumul_tok_ch=8,    # video : meme mecanisme que le bras cumul
    eval_every=1_000, eval_windows=32, ckpt_every=1_000,
    extrap_ctx=1024, extrap_windows=16, buckets=[0, 256, 512, 1024],
    seed=42, eval_seed=123,
    models=[
        dict(name="T_rope", kind="T", pos="rope", penalty=0.0),                 # transformer ordinaire, position RoPE
        dict(name="V_lam0", kind="V", scales=2, penalty=0.0),                   # video continue transportee, 2 echelles de temps, sans penalite
        dict(name="V_lam1", kind="V", scales=2, penalty=1.0),                   # idem, penalite lambda = 1
        dict(name="T_nope", kind="T", pos="none", penalty=0.0),                 # transformer sans aucune position (temoin dur)
        dict(name="V1_lam0", kind="V", scales=1, penalty=0.0),                  # une seule echelle (ablation des deux echelles)
    ],
)


# Prolongation de V_lam1 (amendement 7) : repart des poids a 8 000 pas, optimiseur neuf, jusqu'a 24 000 pas.
_EXT = dict(name="V_lam1_ext", kind="V", scales=2, penalty=1.0, init_from="s2_V_lam1_model.pt", start_step=8_000)
S2_EXT = dict(
    S2, steps=24_000, ext_lr=1e-3, ext_lr_min=1e-4, ext_warmup=200, summary_name="s2x_summary.md",
    models=[S2["models"][0], S2["models"][1], S2["models"][2], _EXT],    # lignes de reference + la prolongation
    run=[_EXT],
)


# Image qui evolue (amendement 9) : depot par le token seul (libre ou tache de 12 amplitudes a position continue) puis n pas d'evolution.
S2_DYN = dict(
    S2, rest_noise=0.01, summary_name="s2d_summary.md",
    models=[S2["models"][0], S2["models"][1],                                                   # references : T_rope, V_lam0 (8 000 pas)
            dict(name="D_free_e1", kind="D", deposit="free", evolve=1, penalty=0.0),
            dict(name="D_blob_e1", kind="D", deposit="blob", evolve=1, penalty=0.0),
            dict(name="D_blob_e0", kind="D", deposit="blob", evolve=0, penalty=0.0),
            dict(name="D_blob_e2", kind="D", deposit="blob", evolve=2, penalty=0.0)],
)
S2_DYN["run"] = S2_DYN["models"][2:]
