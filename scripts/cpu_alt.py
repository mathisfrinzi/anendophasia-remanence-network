"""Entrainement alterne images / transformers sur S4 (amendement 45). Usage : python scripts/cpu_alt.py SCHEMA GRAINE [PAS]    SCHEMA in {alt, joint}
Modele : L_loop1 (cpu_s4.build). Deux groupes de parametres :
  "images" = emb, dep (token -> image), i1, i2 (images initiales), ret1, head (lecture de l'image)  -> 46 k parametres ;
  "transformers" = t1, t2 (les deux transformers a patchs)                                           -> 109 k parametres.
alt : phases de P pas qui alternent images seules (en premier), puis transformers seuls, pendant les F premiers pas ; ensuite tous ensemble.
joint : tout ensemble depuis le debut (equivalent a cpu_s4.py, meme config, meme evaluation).
Arret anticipe des qu'on a >= 99 % a n=24 sur 2 evaluations de suite (apres l'arret : evaluation finale n = 12..192). Sortie : results/alt_<schema>_s<graine>.json"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
import cpu_s4 as C

torch.set_num_threads(1)
P, FSTOP = 250, 6000
IMG = ("emb", "dep", "i1", "i2", "ret1", "head")


def phase(step, schema):
    """-> 'images', 'transformers' ou 'joint' pour le pas `step` (1 = premier)."""
    if schema == "joint" or step > FSTOP: return "joint"
    return "images" if ((step - 1) // P) % 2 == 0 else "transformers"


if __name__ == "__main__":
    schema, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 15000
    path = os.path.join(os.path.dirname(__file__), "..", "results", "alt_%s_s%d.json" % (schema, seed))
    if os.path.exists(path) and json.load(open(path)).get("done"): sys.exit()
    torch.manual_seed(seed); model = C.build("L_loop1"); ST = C.ST
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time(); hist = []; ok = 0; stop = None
    named = list(model.named_parameters()); npar = sum(p.numel() for _, p in named)
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        nmax = 4 + int(20 * min(1.0, step / (0.4 * steps))); n = int(rng.integers(4, nmax + 1))
        x, y = s.make_batch(rng, C.G, C.mult, ST["batch"], n)
        loss = F.cross_entropy(model(x).reshape(-1, C.G), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); ph = phase(step, schema)
        for nm, p in named:
            if (ph == "images" and nm.split(".")[0] not in IMG) or (ph == "transformers" and nm.split(".")[0] in IMG): p.grad = None   # le groupe fige ne bouge pas (ni Adam, ni decroissance)
        torch.nn.utils.clip_grad_norm_([p for _, p in named if p.grad is not None], ST["grad_clip"]); opt.step()
        if step % 500 == 0:
            e = {str(m): C.evaluate(model, m, 128) for m in (12, 24)}; hist.append(dict(step=step, loss=round(loss.item(), 4), phase=ph, **e))
            print("[S4 alt=%s s%d] pas %d/%d %s perte %.3f acc12 %.2f acc24 %.2f (%.0fs)" % (schema, seed, step, steps, ph, loss.item(), e["12"], e["24"], time.time() - t0), flush=True)
            json.dump(dict(schema=schema, seed=seed, steps=steps, n_params=npar, hist=hist, done=False), open(path, "w"), indent=1)
            ok = ok + 1 if e["24"] >= 0.99 else 0
            if ok >= 2: stop = step; break
    res = dict(schema=schema, seed=seed, steps=steps, n_params=npar, seconds=round(time.time() - t0), hist=hist, stopped_at=stop, final={str(n): C.evaluate(model, n) for n in [12, 24, 48, 96, 192]}, done=True)
    json.dump(res, open(path, "w"), indent=1); print(res["final"], flush=True)
