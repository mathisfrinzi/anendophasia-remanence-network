import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import data as D


def evaluate(model, val_ids, device):
    cfg = config.COMMON
    model.eval()
    rng = np.random.default_rng(cfg["eval_seed"])
    tot = 0.0
    with torch.no_grad():
        for _ in range(cfg["eval_batches"]):
            x, y = D.batch_from(rng, val_ids, cfg["batch_size"], cfg["seq_len"], device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            tot += float(loss)
    model.train()
    return tot / cfg["eval_batches"]


def generate(model, val_ids, vocab, device):
    cfg = config.COMMON
    rng = np.random.default_rng(777)
    span = cfg["sample_prompt_len"] + cfg["sample_len"] + 2
    offsets = rng.integers(0, len(val_ids) - span, size=cfg["sample_prompts"])
    model.eval()
    outs = []
    with torch.no_grad():
        for o in offsets:
            prompt = val_ids[o : o + cfg["sample_prompt_len"]].astype(np.int64)
            state = model.init_state(1)
            x = torch.from_numpy(prompt).view(1, -1).to(device)
            logits = None
            for t in range(x.shape[1]):
                logits, state = model.step(x[:, t : t + 1], state)
            gen = []
            for _ in range(cfg["sample_len"]):
                p = F.softmax(logits / cfg["sample_temp"], dim=-1)
                nxt = torch.multinomial(p, 1)
                gen.append(int(nxt))
                logits, state = model.step(nxt, state)
            outs.append(D.decode(vocab, prompt) + "[[...]]" + D.decode(vocab, gen))
    model.train()
    return outs


def ckpt_path(arm, root=".", suffix=""):
    return os.path.join(root, "results", arm + suffix + "_ckpt.pt")


def save_ckpt(arm, model, opt, rng, step, curve, elapsed, root=".", vocab=None, suffix=""):
    """Checkpoint complet : permet de reprendre l'entrainement apres un crash,
    et de recharger le modele ailleurs (viz.py) sans retelecharger le corpus.
    suffix="_ae" pour l'etage 1 du bras 6 (checkpoint separe)."""
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "rng": rng.bit_generator.state,
            "curve": curve,
            "elapsed": elapsed,
            "vocab": vocab,
        },
        ckpt_path(arm, root, suffix),
    )


def _load_torch(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch ancien sans weights_only
        return torch.load(path, map_location=device)


def train_arm(arm, model, train_ids, val_ids, vocab, device, root=".", save_every_s=3600.0, on_save=None, on_eval=None):
    """Entrainement d'un bras (protocole 2_protocole.md).

    Reprise automatique : si results/ARM_ckpt.pt existe (fourni par le zip
    televerse au debut du notebook), l'entrainement repart du dernier pas sauvegarde, avec le meme
    etat du generateur de lots (les pas restants sont identiques a un run non interrompu).
    Toutes les save_every_s secondes : checkpoint + results/ARM_progress.json,
    puis appel de on_save(step) (telechargement du zip results/ depuis le notebook).
    A chaque evaluation (tous les eval_every pas) : appel de on_eval(step, model)
    (ex. : images de monitoring du bras 6, sans effet sur l'entrainement)."""
    cfg = config.COMMON
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    model = model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
    rng = np.random.default_rng(cfg["seed"])
    curve = []
    start_step = 0
    base_elapsed = 0.0
    p = ckpt_path(arm, root)
    if os.path.exists(p):
        try:
            ck = _load_torch(p, device)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            rng.bit_generator.state = ck["rng"]
            curve = ck["curve"]
            start_step = int(ck["step"])
            base_elapsed = float(ck.get("elapsed", 0.0))
            print("Reprise du checkpoint : pas {}/{} ({:.0f} s deja ecoulees).".format(start_step, cfg["steps"], base_elapsed))
        except Exception as e:
            print("Checkpoint illisible, depart a zero :", e)
            curve = []
            start_step = 0
            base_elapsed = 0.0
    tag = os.path.join(root, "results", arm)
    t_start = time.time()
    last_save = t_start
    for step in range(start_step + 1, cfg["steps"] + 1):
        lr = cfg["lr"] * min(1.0, step / cfg["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = D.batch_from(rng, train_ids, cfg["batch_size"], cfg["seq_len"], device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            vl = evaluate(model, val_ids, device)
            curve.append({"step": step, "val_loss": vl, "train_loss": float(loss)})
            print("pas {:5d} | train {:.4f} | val {:.4f} nats/car".format(step, float(loss), vl))
            with open(tag + "_progress.json", "w") as f:
                json.dump(
                    {
                        "arm": arm,
                        "step": step,
                        "steps_total": cfg["steps"],
                        "val_loss_last": curve[-1]["val_loss"],
                        "elapsed_s": round(base_elapsed + (time.time() - t_start), 1),
                        "curve": curve,
                    },
                    f,
                    indent=2,
                )
            if on_eval is not None:
                try:
                    on_eval(step, model)
                except Exception as e:
                    print("Visualisation (on_eval) ratee (l'entrainement continue) :", e)
        now = time.time()
        if now - last_save >= save_every_s or step == cfg["steps"]:
            elapsed = base_elapsed + (now - t_start)
            save_ckpt(arm, model, opt, rng, step, curve, elapsed, root, vocab=vocab)
            base_elapsed = elapsed
            last_save = now
            if on_save is not None:
                try:
                    on_save(step)
                except Exception as e:
                    print("Sauvegarde intermediaire (on_save) ratee (l'entrainement continue) :", e)
    final = evaluate(model, val_ids, device)
    elapsed = base_elapsed + (time.time() - t_start)
    metrics = {
        "arm": arm,
        "n_params": n_params,
        "val_loss_final": final,
        "val_loss_best": min(c["val_loss"] for c in curve) if curve else final,
        "bpc": final / math.log(2),
        "ppl": math.exp(final),
        "tokens_seen": cfg["steps"] * cfg["batch_size"] * cfg["seq_len"],
        "seconds": round(elapsed, 1),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "curve": curve,
        "hp": {k: cfg[k] for k in ["steps", "batch_size", "seq_len", "lr", "seed"]},
    }
    with open(tag + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    save_ckpt(arm, model, opt, rng, cfg["steps"], curve, elapsed, root, vocab=vocab)
    with open(tag + "_samples.txt", "w", encoding="utf-8") as f:
        f.write("\n\n=====\n\n".join(generate(model, val_ids, vocab, device)))
    model.train()
    return metrics
