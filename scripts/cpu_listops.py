"""Amendement 25 : ListOps.
Usage : python scripts/cpu_listops.py MODELE GRAINE [PAS]
MODELE in {T_rope, V_lam0, L_loop1, L_noloop1, L_loop1_k}. Variable d'env LISTOPS_OUT = dossier de sortie (defaut results/)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
import listops_data as D

torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(__file__), "..")
K_TRAIN, KS_EVAL, N_EVAL, DEPTHS = 8, (0, 2, 4, 8, 12), 1000, (1, 2, 3, 4, 5)
SPECS = dict(T_rope=dict(name="T_rope", kind="T", pos="rope"), V_lam0=dict(name="V_lam0", kind="V", scales=2, penalty=0.0),
             L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), L_noloop1=dict(name="L_noloop1", kind="L", loop=False, layers=1),
             L_loop1_k=dict(name="L_loop1_k", kind="L", loop=True, layers=1))
ST = dict(s.ST_LOOP, batch=32, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1)


def logits(model, x):
    return model(x)[:, -1][:, D.DIGIT0:D.DIGIT0 + 10]


def eval_sets(n):
    """Memes expressions pour tous les modeles : (seqs sans pauses, y) par profondeur."""
    out = {}
    for d in DEPTHS:
        x, y, _ = D.batch(np.random.default_rng(777 + d), [d], n)
        out[d] = (x, y)
    return out


def add_pause(x, k):
    return x if k == 0 else np.concatenate([x, np.full((len(x), k), D.PAUSE, dtype=np.int64)], 1)


@torch.no_grad()
def acc(model, sets, d, n=None, k=0, B=250):
    model.eval(); x, y = sets[d]; N = len(x) if n is None else n; ok = 0
    for i in range(0, N, B):
        xb = torch.from_numpy(add_pause(x[i:min(N, i + B)], k)); yb = torch.from_numpy(y[i:min(N, i + B)])
        ok += (logits(model, xb).argmax(-1) == yb).sum().item()
    model.train(); return ok / N


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    out = os.environ.get("LISTOPS_OUT", os.path.join(ROOT, "results")); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "listops_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    torch.manual_seed(seed)
    cfg = s.model_cfg(dict(ST, n_layer=2) if name.startswith("T_") else ST)
    model = s.build(SPECS[name], D.V, cfg)
    final = eval_sets(N_EVAL); mid = eval_sets(200)
    if os.environ.get("LISTOPS_UNTRAINED"):
        print({d: acc(model, final, d, n=200) for d in DEPTHS}); sys.exit()
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time(); curve = []
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        dmax = 1 + round(2 * min(1, step / (0.4 * steps)))
        k = int(rng.integers(0, K_TRAIN + 1)) if name == "L_loop1_k" else 0
        x, y, _ = D.batch(rng, list(range(1, dmax + 1)), ST["batch"], k=k)
        loss = F.cross_entropy(logits(model, torch.from_numpy(x)), torch.from_numpy(y))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0 or step == steps:
            c = dict(step=step, loss=round(loss.item(), 4), **{"d%d" % d: acc(model, mid, d) for d in (1, 2, 3)})
            curve.append(c); print("[listops %s s%d] %s (%.0fs)" % (name, seed, c, time.time() - t0), flush=True)
        elif step in (10, 50) and os.environ.get("LISTOPS_VERBOSE"): print("pas %d %.1fs" % (step, time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0),
               test={"d%d" % d: acc(model, final, d) for d in DEPTHS})
    if name == "L_loop1_k":
        res["test_k"] = {"k%d" % k: {"d%d" % d: acc(model, final, d, k=k) for d in DEPTHS} for k in KS_EVAL}
        res["test"] = res["test_k"]["k0"]
    res["curve"] = curve
    json.dump(res, open(path, "w"), indent=1); print(res["test"], flush=True)
