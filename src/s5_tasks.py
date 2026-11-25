"""Etape 5 : autres taches d'etat pour le couple L_loop / L_noloop (amendement 13) : Dyck (profondeur) et cle-valeur.
S5 est lancee par s3_tasks (memes reglages que l'etape 3). Usage : s5_tasks.run_all(root)."""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import s3_tasks as S3

DEPTH = 8                                  # profondeur maximale des parentheses
DY_V = DEPTH + 1
KEYS, VALS = 16, 8                         # cle-valeur : 16 cles, 8 valeurs
KV_Q = KEYS + VALS                         # jeton de requete
KV_V = KV_Q + 1
KV_TRAIN_PAIRS = (1, 8)                    # nombre de paires a l'entrainement (tire par lot)
KV_EVAL_PAIRS = [2, 4, 8, 12, 16]

TASKS5 = {"DYCK": 3000, "KV": 8000}
ST5 = dict(S3.ST_LOOP, summary_name="st5_summary.md", seeds=[0, 1, 2])


def dyck_batch(rng, B, n):
    x = np.zeros((B, n), dtype=np.int64)
    d = np.zeros((B, n), dtype=np.int64)
    cur = np.zeros(B, dtype=np.int64)
    for t in range(n):
        u = rng.random(B) < 0.5
        op = np.where(cur == 0, True, np.where(cur >= DEPTH, False, u))
        cur = cur + np.where(op, 1, -1)
        x[:, t], d[:, t] = np.where(op, 0, 1), cur
    return torch.from_numpy(x), torch.from_numpy(d)


def kv_batch(rng, B, N):
    keys = np.argsort(rng.random((B, KEYS)), axis=1)[:, :N]
    vals = rng.integers(KEYS, KEYS + VALS, (B, N))
    q = rng.integers(0, N, B)
    seq = np.zeros((B, 2 * N + 2), dtype=np.int64)
    seq[:, 0:2 * N:2], seq[:, 1:2 * N:2] = keys, vals
    seq[:, 2 * N] = KV_Q
    seq[:, 2 * N + 1] = keys[np.arange(B), q]
    return torch.from_numpy(seq), torch.from_numpy(vals[np.arange(B), q])


def lr_at(step, steps, st):
    return S3.lr_at(step, steps, st)


@torch.no_grad()
def evaluate(task, model, st, device, seqs):
    model.eval()
    rng = np.random.default_rng(st["eval_seed"])
    out = {}
    if task == "DYCK":
        x, y = dyck_batch(rng, seqs, st["eval_len"])
        acc = (model(x.to(device)).argmax(-1).cpu() == y).float().mean(0).numpy()
        b = st["buckets"]
        out = {"acc_%d_%d" % (a, c): float(acc[a:c].mean()) for a, c in zip(b[:-1], b[1:])}
    else:
        for N in KV_EVAL_PAIRS:
            x, y = kv_batch(rng, seqs, N)
            out["acc_N%d" % N] = float((model(x.to(device))[:, -1].argmax(-1).cpu() == y).float().mean())
    model.train()
    return out


def run_one(task, spec, seed, steps, st, device, root, log=print):
    res = os.path.join(root, "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "st_%s_%s_s%d.json" % (task, spec["name"], seed))
    if os.path.exists(path):
        return json.load(open(path))
    V = DY_V if task == "DYCK" else KV_V
    torch.manual_seed(seed)
    model = S3.build(spec, V, S3.model_cfg(st)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=st["lr"], betas=(0.9, 0.95), weight_decay=st["wd"])
    t0, curve = time.time(), []
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, st)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        if task == "DYCK":
            n = st["train_len"]
            x, y = dyck_batch(rng, st["batch"], n)
            L = min(n, st["curriculum_start"] + int((n - st["curriculum_start"]) * min(1.0, step / (st["curriculum_frac"] * steps))))
            logits = model(x.to(device))
            loss = F.cross_entropy(logits[:, :L].reshape(-1, V), y.to(device)[:, :L].reshape(-1))
        else:
            N = int(rng.integers(KV_TRAIN_PAIRS[0], KV_TRAIN_PAIRS[1] + 1))
            x, y = kv_batch(rng, st["batch"], N)
            loss = F.cross_entropy(model(x.to(device))[:, -1], y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), st["grad_clip"])
        opt.step()
        if step % st["eval_every"] == 0 or step == steps:
            m = evaluate(task, model, dict(st, eval_len=st["train_len"], buckets=[0, st["train_len"]]), device, st["eval_seqs"])
            key = "acc_0_%d" % st["train_len"] if task == "DYCK" else "acc_N8"
            curve.append(dict(step=step, loss=loss.item(), acc=m[key]))
            log("[%s %s s%d] pas %d/%d  perte %.3f  prec. %.3f  (%.0fs)" % (task, spec["name"], seed, step, steps, loss.item(), m[key], time.time() - t0))
    final = evaluate(task, model, st, device, st["eval_seqs"])
    out = dict(task=task, model=spec["name"], seed=seed, steps=steps, n_params=n_params, seconds=round(time.time() - t0, 1), final=final, curve=curve)
    json.dump(out, open(path, "w"), indent=1)
    return out


def summarize(root=".", st=None):
    st = st or ST5
    lines = ["# Etape 5 : Dyck et cle-valeur (mediane [min-max] sur les graines, en %)", ""]
    dk = ["acc_%d_%d" % (a, c) for a, c in zip(st["buckets"][:-1], st["buckets"][1:])]
    for task, keys, head in (("DYCK", dk, "profondeur des parentheses : tranches de position (entrainement sur %d)" % st["train_len"]),
                             ("KV", ["acc_N%d" % n for n in KV_EVAL_PAIRS], "cle-valeur : nombre de paires dans la suite (entrainement sur 1 a %d paires)" % KV_TRAIN_PAIRS[1])):
        lines += ["## %s" % head, "", "| modele | %s |" % " | ".join(k.split("_", 1)[1].replace("_", "-") for k in keys), "|:--|%s" % ("--:|" * len(keys))]
        for spec in st["models"]:
            rs = [json.load(open(p)) for s in st["seeds"] for p in [os.path.join(root, "results", "st_%s_%s_s%d.json" % (task, spec["name"], s))] if os.path.exists(p)]
            if rs:
                lines.append("| %s | %s |" % (spec["name"], " | ".join("%.0f [%.0f-%.0f]" % (np.median(v), min(v), max(v)) for v in [[100 * r["final"][k] for r in rs] for k in keys])))
        lines.append("")
    txt = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    open(os.path.join(root, "results", st["summary_name"]), "w", encoding="utf-8").write(txt)
    return txt


def run_all(root=".", st=None, tasks=None, on_done=None, log=print):
    st = st or ST5
    tasks = tasks or TASKS5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st = dict(st, models=[m for m in st["models"] if m["name"] in ("L_loop", "L_noloop", "T_rope", "V_lam0")])
    for task in tasks:
        for seed in st["seeds"]:
            for spec in st["models"]:
                run_one(task, spec, seed, tasks[task], st, device, root, log)
                summarize(root, st)
                if on_done:
                    on_done()
    return summarize(root, st)
