"""Suivi d'etat S4 (24 etats) sur CPU (amendement 27). Usage : python scripts/cpu_s4.py MODELE GRAINE [PAS] ; MODELE in {T_rope, L_loop1, L_dloop1}.
Variable d'env S4_K (defaut 4) : groupe symetrique S_K. Sortie : results/s4_MODELE_sGRAINE.json (ecrit aussi a chaque evaluation)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from s6_double import DoubleLoopLM
from s7_graphe import GraphLM

torch.set_num_threads(1)
K = int(os.environ.get("S4_K", "4")); G, mult = s.group(K)
SPECS = dict(L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), T_rope=dict(name="T_rope", kind="T", pos="rope"))
ST = dict(s.ST_LOOP, batch=64, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1)


def build(name):
    cfg = s.model_cfg(ST)
    if name.startswith("L_loop1_p"):         # amendement 46 : k passes T1/T2 par token
        return s.build(dict(name=name, kind="L", loop=True, layers=1, passes=int(name[9:])), G, cfg)
    if name in ("G231", "G231_ff"):      # amendement 32 : reseau d'images en couches 2-3-1
        return GraphLM(G, dict(d_tok=32), retour=(name == "G231"))
    if name == "L_dloop1_full":      # amendement 29 : branche 1 = boucle normale (4x8x8, d=64), branche 2 annexe (2x8x8, d=44)
        return DoubleLoopLM(G, cfg, d_edge=64, d_edge2=44, c1=4, c2=2, layers=1)
    return DoubleLoopLM(G, cfg, d_edge=44, layers=1) if name == "L_dloop1" else s.build(SPECS[name], G, cfg)


@torch.no_grad()
def evaluate(model, n, B=256):
    model.eval(); x, y = s.make_batch(np.random.default_rng(9999 + n), G, mult, B, n); pr = model(x).argmax(-1); model.train()
    return round((pr == y).float().mean().item(), 4)


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    path = os.path.join(os.path.dirname(__file__), "..", "results", "s%d_%s_s%d%s.json" % (K, name, seed, "" if steps == 4000 else "_p%d" % steps))
    if os.path.exists(path) and json.load(open(path)).get("done"): sys.exit()
    torch.manual_seed(seed); model = build(name)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time(); hist = []
    npar = sum(p.numel() for p in model.parameters())
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        nmax = 4 + int(20 * min(1.0, step / (0.4 * steps))); n = int(rng.integers(4, nmax + 1))
        x, y = s.make_batch(rng, G, mult, ST["batch"], n)
        loss = F.cross_entropy(model(x).reshape(-1, G), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0:
            e = {str(m): evaluate(model, m, 128) for m in (12, 24)}; hist.append(dict(step=step, loss=round(loss.item(), 4), **e))
            print("[S%d %s s%d] pas %d/%d perte %.3f acc12 %.2f acc24 %.2f (%.0fs)" % (K, name, seed, step, steps, loss.item(), e["12"], e["24"], time.time() - t0), flush=True)
            json.dump(dict(model=name, seed=seed, steps=steps, n_params=npar, hist=hist, done=False), open(path, "w"), indent=1)
    res = dict(model=name, seed=seed, steps=steps, n_params=npar, seconds=round(time.time() - t0), hist=hist, final={str(n): evaluate(model, n) for n in [12, 24, 48, 96, 192]}, done=True)
    json.dump(res, open(path, "w"), indent=1); print(res["final"], flush=True)
