"""S3 + memoire longue (amendement 17). Usage : python scripts/cpu_memoire.py MODELE GRAINE [PAS]"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s

torch.set_num_threads(1)
G, mult = s.group(3)
NV, V, Q = 8, 15, 14                      # 8 valeurs (ids 6..13), requete = 14
SPECS = dict(L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), L_noloop1=dict(name="L_noloop1", kind="L", loop=False, layers=1),
             **{"L_%s_%s" % (k, z): dict(name="L_%s_%s" % (k, z), kind="L", loop=(k == "loop1"), layers=1, c=c, h=h, w=h) for k in ("loop1", "noloop1") for z, (c, h) in {"4x4x4": (4, 4), "2x4x4": (2, 4)}.items()},
             T_rope=dict(name="T_rope", kind="T", pos="rope"))
ST = dict(s.ST_LOOP, batch=64, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1)


def batch(rng, B, n):
    g, p = s.make_batch(rng, G, mult, B, n)
    v = torch.from_numpy(rng.integers(0, NV, B))
    x = torch.cat([(6 + v)[:, None], g, torch.full((B, 1), Q)], 1)
    y = torch.cat([torch.full((B, 1), -100), p, (6 + v)[:, None]], 1)
    return x, y


@torch.no_grad()
def evaluate(model, n, B=256, seed=9999):
    model.eval(); x, y = batch(np.random.default_rng(seed + n), B, n)
    pr = model(x).argmax(-1); model.train()
    return dict(s3=(pr[:, 1:n + 1] == y[:, 1:n + 1]).float().mean().item(), recall=(pr[:, -1] == y[:, -1]).float().mean().item())


def probes(model, n=48):
    """Sonde lineaire de la valeur stockee sur Image1 / Image2 (globale et par canal) a plusieurs pas du delai."""
    model.eval(); x, y = batch(np.random.default_rng(777), 800, n)
    with torch.no_grad():
        _, fr = model(x, return_frames=True)          # (B, N, 2, C, H, W)
    v = y[:, -1] - 6; Y = F.one_hot(v, NV).float(); out = {}
    def probe(X):
        X = X - X.mean(0); A = torch.cat([X, torch.ones(len(X), 1)], 1)
        Wt = torch.linalg.lstsq(A[:600], Y[:600]).solution
        return round(((A[600:] @ Wt).argmax(1) == v[600:]).float().mean().item(), 3)
    for t in [1, 5, 15, 30, n]:
        out["t%d" % t] = {nm: probe(fr[:, t, i].flatten(1)) for i, nm in enumerate(["Image1", "Image2"])}
        for i, nm in enumerate(["Image1", "Image2"]):
            out["t%d" % t][nm + "_canaux"] = [probe(fr[:, t, i, c].flatten(1)) for c in range(fr.shape[3])]
    return out


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    root = os.path.join(os.path.dirname(__file__), ".."); path = os.path.join(root, "results", "st_S3M_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    torch.manual_seed(seed); model = s.build(SPECS[name], V, s.model_cfg(ST))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        nmax = 4 + int(20 * min(1.0, step / (0.4 * steps))); n = int(rng.integers(4, nmax + 1))
        x, y = batch(rng, ST["batch"], n)
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 250 == 0:
            e = evaluate(model, 24, 128); print("[S3M %s s%d] pas %d/%d perte %.3f s3 %.2f rappel %.2f (%.0fs)" % (name, seed, step, steps, loss.item(), e["s3"], e["recall"], time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), final={str(n): evaluate(model, n) for n in [12, 24, 32, 48, 64, 96]})
    if name.startswith("L_loop") and "x" not in name:
        res["sondes"] = probes(model)
    json.dump(res, open(path, "w"), indent=1); print(res["final"], flush=True)
