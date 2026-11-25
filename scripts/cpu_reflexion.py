"""Amendement 21 : temps de reflexion (poursuite de pointeurs + k pauses). Usage : python scripts/cpu_reflexion.py MODELE GRAINE [PAS]
MODELE in {L_loop1_k, L_loop1_k0, L_noloop1_k, T_rope_k}. Variable d'env RF_OUT = dossier de sortie (defaut results/)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s

torch.set_num_threads(1)
NS, Qt, P = 6, 12, 13                     # symboles 0..5, sauts h=1..6 -> ids 6..11, requete 12, pause 13
V = 14
SPECS = dict(L_loop1_k=dict(name="L_loop1_k", kind="L", loop=True, layers=1), L_loop1_k0=dict(name="L_loop1_k0", kind="L", loop=True, layers=1),
             L_noloop1_k=dict(name="L_noloop1_k", kind="L", loop=False, layers=1), T_rope_k=dict(name="T_rope_k", kind="T", pos="rope"))
ST = dict(s.ST_LOOP, batch=64, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1)
H_TRAIN, K_TRAIN = 4, 8


def batch(rng, B, k, hmax=None, h=None):
    """Retourne x (B, 12+3+1+k), cible f^h(x) (B,), h (B,)."""
    f = np.argsort(rng.random((B, NS)), 1)                      # permutation aleatoire par suite
    order = np.argsort(rng.random((B, NS)), 1)                  # ordre aleatoire des paires
    a = order; fa = np.take_along_axis(f, a, 1)
    pairs = np.stack([a, fa], 2).reshape(B, 2 * NS)
    hh = np.full(B, h) if h is not None else rng.integers(1, hmax + 1, B)
    x0 = rng.integers(0, NS, B); cur = x0.copy()
    for i in range(int(hh.max())):
        nxt = f[np.arange(B), cur]; cur = np.where(i < hh, nxt, cur)
    x = np.concatenate([pairs, np.full((B, 1), Qt), (5 + hh)[:, None], x0[:, None], np.full((B, k), P)], 1)
    return torch.from_numpy(x).long(), torch.from_numpy(cur).long(), hh


@torch.no_grad()
def acc(model, B, k, seed, h=None, hmax=None):
    model.eval(); x, y, _ = batch(np.random.default_rng(seed), B, k, hmax=hmax, h=h)
    r = (model(x)[:, -1, :NS].argmax(-1) == y).float().mean().item(); model.train(); return r


def hmax_at(step, steps):
    return 1 + int(round(3 * min(1.0, step / (0.4 * steps))))


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    out = os.environ.get("RF_OUT", os.path.join(os.path.dirname(__file__), "..", "results")); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "rf_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    torch.manual_seed(seed)
    cfg = s.model_cfg(dict(ST, n_layer=2) if name.startswith("T_") else ST)
    model = s.build(SPECS[name], V, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time(); curve = []
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        k = 0 if name == "L_loop1_k0" else int(rng.integers(0, K_TRAIN + 1))
        x, y, _ = batch(rng, ST["batch"], k, hmax=hmax_at(step, steps))
        loss = F.cross_entropy(model(x)[:, -1, :NS], y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0 or step == steps:
            c = dict(step=step, loss=round(loss.item(), 4), acc_h1_k8=acc(model, 256, 8, 9999 + 108, h=1), acc_h4_k8=acc(model, 256, 8, 9999 + 408, h=4))
            curve.append(c); print("[RF %s s%d] %s (%.0fs)" % (name, seed, c, time.time() - t0), flush=True)
    grid = {"h%d_k%d" % (h, k): acc(model, 512, k, 9999 + 100 * h + k, h=h) for h in range(1, 7) for k in (0, 1, 2, 4, 6, 8, 12)}
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), grid=grid, curve=curve)
    json.dump(res, open(path, "w"), indent=1); print({k: round(v, 2) for k, v in grid.items()}, flush=True)
