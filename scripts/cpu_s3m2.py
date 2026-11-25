"""S3 + DEUX valeurs a retenir (amendement 33). Usage : python scripts/cpu_s3m2.py MODELE GRAINE [PAS]   MODELE in {L_loop1, L_dloop1, L_dloop1_full}
Sequence : [v1] [v2] [n elements de S3] [Q1 ou Q2]. Cibles : etat S3 courant a chaque element ; a la requete, v1 (Q1) ou v2 (Q2).
Config de l'amendement 17 (d_model 64, 3 000 pas, batch 64, curriculum n in [4,24]) ; evaluation n = 12, 24, 48, 96."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from s6_double import DoubleLoopLM

torch.set_num_threads(1)
G, mult = s.group(3); NV = 8; Q1, Q2, V = 14, 15, 16        # elements 0..5, valeurs 6..13, requetes 14 et 15
ST = dict(s.ST_LOOP, batch=64, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1)


def build(name):
    cfg = s.model_cfg(ST)
    if name == "L_loop1": return s.build(dict(name="L_loop1", kind="L", loop=True, layers=1), V, cfg)
    if name == "L_dloop1": return DoubleLoopLM(V, cfg, d_edge=44, layers=1)
    return DoubleLoopLM(V, cfg, d_edge=64, d_edge2=44, c1=4, c2=2, layers=1)


def batch(rng, B, n):
    g, p = s.make_batch(rng, G, mult, B, n)
    v = torch.from_numpy(rng.integers(0, NV, (B, 2))); q = torch.from_numpy(rng.integers(0, 2, B))
    x = torch.cat([(6 + v), g, (Q1 + q)[:, None]], 1)
    tgt = 6 + v.gather(1, q[:, None])
    y = torch.cat([torch.full((B, 2), -100), p, tgt], 1)
    return x, y, q


@torch.no_grad()
def evaluate(model, n, B=256, seed=9999):
    model.eval(); x, y, q = batch(np.random.default_rng(seed + n), B, n)
    pr = model(x).argmax(-1); model.train(); ok = (pr[:, -1] == y[:, -1]).float()
    return dict(s3=(pr[:, 2:n + 2] == y[:, 2:n + 2]).float().mean().item(), recall=ok.mean().item(), recall_q1=ok[q == 0].mean().item(), recall_q2=ok[q == 1].mean().item())


@torch.no_grad()
def ablation(model, ns=(24, 48)):
    out = {}
    for tag, sc in [("base", (1.0, 1.0)), ("sans_depot1", (0.0, 1.0)), ("sans_depot2", (1.0, 0.0))]:
        model.dep_scale = sc; out[tag] = {str(n): evaluate(model, n) for n in ns}
    model.dep_scale = (1.0, 1.0); return out


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    path = os.path.join(os.path.dirname(__file__), "..", "results", "s3m2_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    torch.manual_seed(seed); model = build(name)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        nmax = 4 + int(20 * min(1.0, step / (0.4 * steps))); n = int(rng.integers(4, nmax + 1))
        x, y, _ = batch(rng, ST["batch"], n)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 250 == 0:
            e = evaluate(model, 24, 128); print("[S3M2 %s s%d] pas %d/%d perte %.3f s3 %.2f rappel %.2f (q1 %.2f q2 %.2f) (%.0fs)" % (name, seed, step, steps, loss.item(), e["s3"], e["recall"], e["recall_q1"], e["recall_q2"], time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), final={str(n): evaluate(model, n) for n in [12, 24, 48, 96]})
    if name != "L_loop1": res["ablation"] = ablation(model)
    json.dump(res, open(path, "w"), indent=1); print(res["final"], flush=True)
