"""Etape 2 : entrainement et evaluation (voir PROTOCOLE.md, amendement 5).

Usage Colab : s2_run.run_all(root) ; test local : python src/s2_run.py --quick --root <dossier avec data/train.txt, val.txt>
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import data as D
from s2_models import build, continuity_index

LN2 = math.log(2)


def get_tokenizer(root, train_text, vocab_size):
    from tokenizers import ByteLevelBPETokenizer, Tokenizer
    path = os.path.join(root, "data", "bpe_%d.json" % vocab_size)
    if not os.path.exists(path):
        print("Entrainement du tokenizer BPE (%d)..." % vocab_size)
        bt = ByteLevelBPETokenizer()
        bt.train_from_iterator(train_text.split("\n\n"), vocab_size=vocab_size, min_frequency=2, special_tokens=["<|endoftext|>"])
        bt.save(path)
    return Tokenizer.from_file(path)


def encode_text(tok, text, cache):
    if os.path.exists(cache):
        return np.load(cache)
    stories = text.split("\n\n")
    sep = np.array(tok.encode("\n\n").ids, dtype=np.uint16)
    parts = []
    for i in range(0, len(stories), 20000):
        for e in tok.encode_batch(stories[i : i + 20000]):
            parts.append(np.array(e.ids, dtype=np.uint16))
            parts.append(sep)
    ids = np.concatenate(parts)
    np.save(cache, ids)
    return ids


def load_data(root, cfg):
    D.prepare(root, cfg["n_train_stories"], cfg["n_val_stories"])      # telecharge si absent (fichiers texte)
    default = (cfg["n_train_stories"], cfg["n_val_stories"]) == (config.COMMON["n_train_stories"], config.COMMON["n_val_stories"])
    tag = "" if default else "_%d_%d" % (cfg["n_train_stories"], cfg["n_val_stories"])
    tr_txt = open(os.path.join(root, "data", "train%s.txt" % tag), encoding="utf-8").read()
    va_txt = open(os.path.join(root, "data", "val%s.txt" % tag), encoding="utf-8").read()
    tok = get_tokenizer(root, tr_txt, cfg["vocab_size"])
    V = tok.get_vocab_size()
    tr = encode_text(tok, tr_txt, os.path.join(root, "data", "ids_train%s_%d.npy" % (tag, V)))
    va = encode_text(tok, va_txt, os.path.join(root, "data", "ids_val%s_%d.npy" % (tag, V)))
    tok_chars = np.array([max(1, len(tok.decode([i]))) for i in range(V)], dtype=np.float64)   # caracteres par token
    return V, tr, va, tok_chars


def windows(ids, rng, B, n):
    off = rng.integers(0, len(ids) - n - 1, size=B)
    return torch.from_numpy(np.stack([ids[o : o + n + 1] for o in off]).astype(np.int64))


def stream_windows(ids, starts, j, n):
    """Fenetre j (n tokens + 1) de flux contigus commencant en `starts`."""
    return torch.from_numpy(np.stack([ids[o + j * n : o + j * n + n + 1] for o in starts]).astype(np.int64))


@torch.no_grad()
def evaluate_stream(model, val_ids, tok_chars, n, n_win, B, seed, device):
    """Flux de n_win fenetres contigues. Un modele video transporte sa frame d'une fenetre a la suivante ; un
    transformer sur tokens ne voit que la fenetre courante. Retourne le bpc par numero de fenetre."""
    model.eval()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(val_ids) - n_win * n - 1, size=B)
    state, out = None, []
    for j in range(n_win):
        w = stream_windows(val_ids, starts, j, n).to(device)
        x, y = w[:, :-1], w[:, 1:]
        logits = model(x, state=state)
        state = getattr(model, "state_out", None)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").cpu().numpy()
        out.append(float(nll.sum() / LN2 / tok_chars[y.cpu().numpy()].sum()))
    model.train()
    return out


@torch.no_grad()
def evaluate(model, val_ids, tok_chars, n, B, seed, device, buckets=None):
    """bpc de validation sur B fenetres de n tokens (+ par tranche de position si buckets)."""
    model.eval()
    w = windows(val_ids, np.random.default_rng(seed), B, n).to(device)
    x, y = w[:, :-1], w[:, 1:]
    logits = model(x)
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="none").view(B, n).cpu().numpy()
    chars = tok_chars[y.cpu().numpy()]
    out = dict(bpc=float(nll.sum() / LN2 / chars.sum()), nats_per_token=float(nll.mean()))
    if buckets:
        for a, b in zip(buckets[:-1], buckets[1:]):
            b = min(b, n)
            if b > a:
                out["bpc_%d_%d" % (a, b)] = float(nll[:, a:b].sum() / LN2 / chars[:, a:b].sum())
    model.train()
    return out


def lr_at(step, cfg, start=0):
    if start > 0:                  # prolongation d'un modele deja entraine : nouvel echauffement puis nouveau cosinus
        s_, tot = step - start, cfg["steps"] - start
        if s_ < cfg["ext_warmup"]:
            return cfg["ext_lr"] * s_ / cfg["ext_warmup"]
        p = (s_ - cfg["ext_warmup"]) / max(1, tot - cfg["ext_warmup"])
        return cfg["ext_lr_min"] + (cfg["ext_lr"] - cfg["ext_lr_min"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    if step < cfg["warmup"]:
        return cfg["lr"] * step / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * (cfg["lr_min_frac"] + (1 - cfg["lr_min_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def run_model(spec, V, tr, va, tok_chars, device, root=".", cfg=None, log=print, stop_after=None):
    """Entraine un modele ; ecrit results/s2_NAME.json (+ _model.pt). Reprise automatique depuis results/s2_NAME_ckpt.pt."""
    cfg = cfg or config.S2
    name = spec["name"]
    res = os.path.join(root, "results")
    os.makedirs(res, exist_ok=True)
    final_path = os.path.join(res, "s2_%s.json" % name)
    if os.path.exists(final_path):
        log("[%s] deja mesure, saute." % name)
        return json.load(open(final_path))
    torch.manual_seed(cfg["seed"])
    model = build(spec, V, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.95), weight_decay=cfg["wd"])
    ckpt = os.path.join(res, "s2_%s_ckpt.pt" % name)
    step0, curve, elapsed = 0, [], 0.0
    start = spec.get("start_step", 0)
    if spec.get("init_from"):      # prolongation : poids d'un modele termine, optimiseur neuf
        sd = torch.load(os.path.join(res, spec["init_from"]), map_location=device, weights_only=False)
        model.load_state_dict(sd["state"])
        step0 = start
        log("[%s] depart des poids de %s (pas %d), jusqu'au pas %d" % (name, spec["init_from"], start, cfg["steps"]))
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step0, curve, elapsed = s["step"], s["curve"], s["elapsed"]
        log("[%s] reprise au pas %d" % (name, step0))
    lam = spec.get("penalty", 0.0)
    K = cfg["episode_windows"]
    assert cfg["ckpt_every"] % K == 0, "les points de reprise doivent tomber en debut d'episode"
    groups = spec.get("scales", 2) if spec["kind"] in ("V", "D") else 1
    state, starts = None, None
    t0 = time.time() - elapsed
    for step in range(step0 + 1, cfg["steps"] + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg, start)
        # flux contigus : un "episode" = K fenetres consecutives par ligne du lot ; la frame est transportee, sans gradient
        ep, j = divmod(step - 1, K)
        if j == 0:
            starts = np.random.default_rng(cfg["seed"] * 1_000_003 + ep).integers(0, len(tr) - K * cfg["ctx"] - 1, cfg["batch_size"])
            state = None
        w = stream_windows(tr, starts, j, cfg["ctx"]).to(device)
        x, y = w[:, :-1], w[:, 1:]
        logits, frames = model(x, return_frames=True, state=state)
        state = getattr(model, "state_out", None)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if lam > 0 and frames is not None:
            ramp = min(1.0, step / cfg["penalty_ramp"]) if cfg.get("penalty_ramp") else 1.0   # penalite introduite progressivement
            loss = loss + lam * ramp * continuity_index(frames, groups)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            m = evaluate(model, va, tok_chars, cfg["ctx"], cfg["eval_windows"], cfg["eval_seed"], device)
            m.update(step=step, train_loss=loss.item())
            curve.append(m)
            log("[%s] pas %d/%d  perte %.3f  val %.3f bpc  (%.0fs)" % (name, step, cfg["steps"], loss.item(), m["bpc"], time.time() - t0))
            with open(os.path.join(res, "s2_%s_progress.json" % name), "w") as f:
                json.dump(curve, f)
        if step % cfg["ckpt_every"] == 0 and step < cfg["steps"]:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, curve=curve, elapsed=time.time() - t0), ckpt)
        if stop_after and step >= stop_after:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, curve=curve, elapsed=time.time() - t0), ckpt)
            return None
    # evaluation finale : longueur d'entrainement puis extrapolation jusqu'a extrap_ctx
    fin = evaluate(model, va, tok_chars, cfg["ctx"], cfg["eval_windows"], cfg["eval_seed"], device)
    ext = evaluate(model, va, tok_chars, cfg["extrap_ctx"], cfg["extrap_windows"], cfg["eval_seed"] + 1, device, cfg["buckets"])
    stream = evaluate_stream(model, va, tok_chars, cfg["ctx"], cfg["stream_windows"], cfg["stream_batch"], cfg["eval_seed"] + 3, device)
    cont = None
    if spec["kind"] in ("V", "D"):
        with torch.no_grad():
            model.eval()
            wv = windows(va, np.random.default_rng(cfg["eval_seed"] + 2), 32, cfg["ctx"]).to(device)
            _, fr = model(wv[:, :-1], return_frames=True)
            cont = float(continuity_index(fr, groups))
    out = dict(name=name, kind=spec["kind"], pos=spec.get("pos"), penalty=lam, n_params=n_params, steps=cfg["steps"],
               seconds=round(time.time() - t0, 1), device=str(device), final=fin, extrapolation=ext, stream_bpc=stream, continuity=cont, curve=curve)
    torch.save(dict(state=model.state_dict(), spec=spec, cfg=cfg, V=V), os.path.join(res, "s2_%s_model.pt" % name))
    if os.path.exists(ckpt):
        os.remove(ckpt)
    with open(final_path, "w") as f:
        json.dump(out, f, indent=1)
    return out


def summarize(root=".", cfg=None):
    cfg = cfg or config.S2
    ed = cfg["buckets"]
    lines = ["# Etape 2 : transformer sur tokens contre transformer sur la video continue", "",
             "Bits par caractere (bpc) de validation, plus bas = mieux. Fenetres d'entrainement de %d tokens ; au-dela, extrapolation (fenetres de %d tokens, par tranche de position)." % (cfg["ctx"], cfg["extrap_ctx"]), "",
             "| modele | params | bpc a %d tokens | %s | continuite | duree |" % (cfg["ctx"], " | ".join("tokens %d-%d" % (a, b) for a, b in zip(ed[:-1], ed[1:]))),
             "|:--|--:|--:|%s--:|--:|" % ("--:|" * (len(ed) - 1))]
    for spec in cfg["models"]:
        p = os.path.join(root, "results", "s2_%s.json" % spec["name"])
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        b = [("%.3f" % r["extrapolation"]["bpc_%d_%d" % (a, c)]) for a, c in zip(ed[:-1], ed[1:])]
        lines.append("| %s | %d | %.3f | %s | %s | %.0f min |" % (
            r["name"], r["n_params"], r["final"]["bpc"], " | ".join(b), ("%.2f" % r["continuity"]) if r["continuity"] is not None else "-", r["seconds"] / 60))
    nw = cfg["stream_windows"]
    sel = sorted({0, 1, nw // 2 - 1, nw - 1})
    lines += ["", "Flux de %d fenetres contigues (%d tokens) : bpc par numero de fenetre. Le modele video transporte sa frame ; le transformer sur tokens n'a aucune memoire d'une fenetre a l'autre." % (nw, nw * cfg["ctx"]), "",
              "| modele | %s |" % " | ".join("fenetre %d" % (i + 1) for i in sel), "|:--|%s" % ("--:|" * len(sel))]
    for spec in cfg["models"]:
        p = os.path.join(root, "results", "s2_%s.json" % spec["name"])
        if os.path.exists(p):
            r = json.load(open(p))
            if "stream_bpc" in r:
                lines.append("| %s | %s |" % (r["name"], " | ".join("%.3f" % r["stream_bpc"][i] for i in sel)))
    if any(sp.get("init_from") for sp in cfg["models"]):
        lines += ["", "Les modeles *_ext sont la prolongation d'un modele de 8 000 pas jusqu'a %d pas (budget inegal avec les autres lignes ; la duree indiquee est celle de la prolongation seule)." % cfg["steps"]]
    lines += ["", "Reference (niveau caractere, bras 1-2, 300 000 histoires) : transformer 1,24 bpc ; GRU 1,14 bpc."]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(root, "results", cfg.get("summary_name", "s2_summary.md")), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def run_all(root=".", cfg=None, models=None, on_done=None, log=print):
    cfg = cfg or config.S2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V, tr, va, tok_chars = load_data(root, cfg)
    log("device %s | vocab %d | train %d tokens (%.1f car./token) | val %d tokens" % (
        device, V, len(tr), tok_chars[tr[:1_000_000]].mean(), len(va)))
    for spec in (models or cfg["models"]):
        run_model(spec, V, tr, va, tok_chars, device, root, cfg, log)
        summarize(root, cfg)
        if on_done:
            on_done()
    txt = summarize(root, cfg)
    log(txt)
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--quick", action="store_true", help="test rapide : tout petit modele, quelques pas")
    a = ap.parse_args()
    if a.quick:
        cfg = dict(config.S2, vocab_size=300, ctx=32, batch_size=8, steps=20, warmup=5, episode_windows=2, stream_windows=4, stream_batch=4, d_model=64, n_layer=2, n_head=4,
                   d_tok=32, eval_every=10, eval_windows=4, ckpt_every=10, extrap_ctx=64, extrap_windows=4, buckets=[0, 32, 64])
        run_all(a.root, cfg)
    else:
        run_all(a.root)
