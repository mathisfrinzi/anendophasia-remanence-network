"""Etape 3 : suivi d'etat (composition de permutations). Voir PROTOCOLE.md, amendement 10.

Entree : une suite d'elements g_1..g_n du groupe symetrique S_k (un token par element). Cible a chaque position t :
le produit cumule g_1 o ... o g_t. Entrainement sur des suites de 24 elements ; evaluation jusqu'a 256 (extrapolation en longueur).
Usage Colab : s3_tasks.run_all(root) ; test local : python src/s3_tasks.py --quick --root /tmp/st
"""
import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
from s2_models import build as build_s2
from s4_loop import LoopLM


def build(spec, vocab, cfg):
    if spec["kind"] == "L":
        return LoopLM(vocab, dict(cfg, loop_layers=spec.get("layers", cfg.get("loop_layers", 2)), loop_p=spec.get("p", 2), loop_c=spec.get("c", 4), loop_h=spec.get("h", 8), loop_w=spec.get("w", 8), loop_passes=spec.get("passes", 1), no_gate=spec.get("no_gate", False), readout=spec.get("readout", "i2")), spec["loop"])
    return build_s2(spec, vocab, cfg)

TASKS = {                      # nom : (k points, nombre de pas d'entrainement)
    "S2": (2, 2000),           # parite
    "S3": (3, 4000),           # plus petit groupe non commutatif
    "S5": (5, 8000),           # groupe non resoluble
}
ST_LOOP = None      # defini plus bas
TASKS_LONG = {"S3L": (3, 12000)}            # S3, entrainement 3 fois plus long (amendement 11)
ALL_TASKS = {**TASKS, **TASKS_LONG}
ST = dict(
    train_len=24, eval_len=256, batch=128, lr=1e-3, warmup=200, lr_min_frac=0.1, wd=0.01, grad_clip=1.0,
    curriculum_frac=0.4, curriculum_start=4,      # la longueur supervisee passe de 4 a 24 pendant les 40 premiers % des pas
    eval_every=500, eval_seqs=512, buckets=[0, 24, 48, 96, 256], seeds=[0, 1, 2], eval_seed=9999,
    d_model=128, n_layer=4, n_head=4, d_tok=64, spatial_ch=16, rest_noise=0.0,
    models=[
        dict(name="T_rope", kind="T", pos="rope"),
        dict(name="T_nope", kind="T", pos="none"),
        dict(name="V_lam0", kind="V", scales=2, penalty=0.0),
        dict(name="D_blob_e0", kind="D", deposit="blob", evolve=0),
        dict(name="D_blob_e1", kind="D", deposit="blob", evolve=1),
    ],
)


ST_LONG = dict(ST, seeds=[3, 4, 5, 6, 7, 8], eval_every=1000, summary_name="stl_summary.md",
               models=[m for m in ST["models"] if m["name"] in ("T_rope", "V_lam0", "D_blob_e1")])


def model_cfg(st):
    return dict(config.S2, d_model=st["d_model"], n_layer=st["n_layer"], n_head=st["n_head"], d_tok=st["d_tok"],
                spatial_ch=st["spatial_ch"], rest_noise=st["rest_noise"])


def group(k):
    """Table de multiplication de S_k : mult[a, b] = a o b (b d'abord)."""
    perms = list(itertools.permutations(range(k)))
    idx = {p: i for i, p in enumerate(perms)}
    G = len(perms)
    mult = np.zeros((G, G), dtype=np.int64)
    for a, pa in enumerate(perms):
        for b, pb in enumerate(perms):
            mult[a, b] = idx[tuple(pa[pb[i]] for i in range(k))]
    return G, mult


def make_batch(rng, G, mult, B, n):
    g = rng.integers(0, G, (B, n))
    p = np.zeros_like(g)
    p[:, 0] = g[:, 0]
    for t in range(1, n):
        p[:, t] = mult[p[:, t - 1], g[:, t]]
    return torch.from_numpy(g), torch.from_numpy(p)


def lr_at(step, steps, st):
    if step < st["warmup"]:
        return st["lr"] * step / st["warmup"]
    pr = (step - st["warmup"]) / max(1, steps - st["warmup"])
    return st["lr"] * (st["lr_min_frac"] + (1 - st["lr_min_frac"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, pr))))


@torch.no_grad()
def evaluate(model, G, mult, n, B, seed, device, buckets):
    model.eval()
    x, y = make_batch(np.random.default_rng(seed), G, mult, B, n)
    x, y = x.to(device), y.to(device)
    acc = (model(x).argmax(-1) == y).float().mean(0).cpu().numpy()      # precision par position
    model.train()
    out = {"acc_%d_%d" % (a, min(b, n)): float(acc[a:min(b, n)].mean()) for a, b in zip(buckets[:-1], buckets[1:]) if a < n}
    return out


def run_one(task, spec, seed, steps, st, device, root, log=print, steps_override=None):
    res = os.path.join(root, "results")
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "st_%s_%s_s%d.json" % (task, spec["name"], seed))
    if os.path.exists(path):
        return json.load(open(path))
    k, _ = ALL_TASKS[task]
    G, mult = group(k)
    torch.manual_seed(seed)
    model = build(spec, G, model_cfg(st)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=st["lr"], betas=(0.9, 0.95), weight_decay=st["wd"])
    n, t0, curve = st["train_len"], time.time(), []
    for step in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, steps, st)
        x, y = make_batch(np.random.default_rng(seed * 1_000_003 + step), G, mult, st["batch"], n)
        x, y = x.to(device), y.to(device)
        L = min(n, st["curriculum_start"] + int((n - st["curriculum_start"]) * min(1.0, step / (st["curriculum_frac"] * steps))))
        logits = model(x)
        loss = F.cross_entropy(logits[:, :L].reshape(-1, G), y[:, :L].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), st["grad_clip"])
        opt.step()
        if step % st["eval_every"] == 0 or step == steps:
            m = evaluate(model, G, mult, n, st["eval_seqs"], st["eval_seed"], device, [0, n])
            curve.append(dict(step=step, loss=loss.item(), acc_train_len=m["acc_0_%d" % n], supervised_len=L))
            log("[%s %s s%d] pas %d/%d  perte %.3f  prec. (<=%d) %.3f  (%.0fs)" % (task, spec["name"], seed, step, steps, loss.item(), n, m["acc_0_%d" % n], time.time() - t0))
    final = evaluate(model, G, mult, st["eval_len"], st["eval_seqs"], st["eval_seed"], device, st["buckets"])
    out = dict(task=task, model=spec["name"], seed=seed, steps=steps, n_params=n_params, seconds=round(time.time() - t0, 1), chance=1.0 / G, final=final, curve=curve)
    json.dump(out, open(path, "w"), indent=1)
    return out


def summarize(root=".", st=None, tasks=None):
    st = st or ST
    tasks = tasks or TASKS
    keys = ["acc_%d_%d" % (a, min(b, st["eval_len"])) for a, b in zip(st["buckets"][:-1], st["buckets"][1:])]
    lines = ["# Etape 3 : suivi d'etat (precision par position, mediane [min-max] sur les graines)", "",
             "Entrainement sur des suites de %d elements ; tranches de position : %s. Les tranches au-dela de %d mesurent l'extrapolation en longueur." % (
                 st["train_len"], ", ".join(k[4:].replace("_", "-") for k in keys), st["train_len"]), ""]
    for task in tasks:
        lines += ["## %s (hasard : %.1f %%)" % (task, 100.0 / group(tasks[task][0])[0]), "",
                  "| modele | params | %s | graines >= 95 %% dans la longueur d'entrainement |" % " | ".join("pos. " + k[4:].replace("_", "-") for k in keys), "|:--|--:|%s--:|" % ("--:|" * len(keys))]
        for spec in st["models"]:
            rs = []
            for s in st["seeds"]:
                p = os.path.join(root, "results", "st_%s_%s_s%d.json" % (task, spec["name"], s))
                if os.path.exists(p):
                    rs.append(json.load(open(p)))
            if not rs:
                continue
            cells = []
            for k in keys:
                v = np.array([r["final"][k] for r in rs]) * 100
                cells.append("%.0f [%.0f-%.0f]" % (np.median(v), v.min(), v.max()))
            ok = sum(r["final"][keys[0]] >= 0.95 for r in rs)
            lines.append("| %s | %d | %s | %d / %d |" % (spec["name"], rs[0]["n_params"], " | ".join(cells), ok, len(rs)))
        lines.append("")
        lines.append("Par graine (precision en %% dans la longueur d'entrainement | au-dela de %d) :" % st["buckets"][2])
        lines.append("")
        for spec in st["models"]:
            row = []
            for s in st["seeds"]:
                p = os.path.join(root, "results", "st_%s_%s_s%d.json" % (task, spec["name"], s))
                if os.path.exists(p):
                    f = json.load(open(p))["final"]
                    row.append("s%d : %.0f | %.0f" % (s, 100 * f[keys[0]], 100 * f[keys[-1]]))
            if row:
                lines.append("- %s : %s" % (spec["name"], " ; ".join(row)))
        lines.append("")
    txt = "\n".join(lines) + "\n"
    open(os.path.join(root, "results", st.get("summary_name", "st_summary.md")), "w", encoding="utf-8").write(txt)
    return txt


def run_all(root=".", st=None, tasks=None, on_done=None, log=print):
    st = st or ST
    tasks = tasks or TASKS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("device", device)
    for task in tasks:
        for seed in st["seeds"]:
            for spec in st["models"]:
                run_one(task, spec, seed, tasks[task][1], st, device, root, log)
                summarize(root, st, tasks)
                if on_done:
                    on_done()
    txt = summarize(root, st, tasks)
    log(txt)
    return txt


# Etape 4 (amendement 12) : deux images couplees. Memes taches, graines, pas que l'etape 3 ; references reprises des resultats existants.
_L = dict(ST, summary_name="st4_summary.md", seeds=[0, 1, 2], d_model=128, n_head=4,
          models=[m for m in ST["models"] if m["name"] in ("T_rope", "V_lam0")] +
                 [dict(name="L_loop", kind="L", loop=True), dict(name="L_noloop", kind="L", loop=False)])
ST_LOOP = _L
TASKS_LOOP = {"S2": TASKS["S2"], "S3": TASKS["S3"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        st = dict(ST, train_len=12, eval_len=48, batch=32, steps=0, eval_every=20, eval_seqs=64, buckets=[0, 12, 24, 48], seeds=[0], d_model=64, n_layer=2, d_tok=32, spatial_ch=8, warmup=5)
        run_all(a.root, st, {"S2": (2, 60), "S3": (3, 60)})
    else:
        run_all(a.root)

