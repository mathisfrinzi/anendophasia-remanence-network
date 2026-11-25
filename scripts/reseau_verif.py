"""Verification de faisabilite (etape A) : un T_rope (2 couches, d=64) apprend-il la poursuite de pointeurs ?
Usage : python scripts/reseau_verif.py MODELE n hmax_train k pas [graine]"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from reseau_tache import Tache

torch.set_num_threads(1)
SPECS = dict(T_rope=dict(name="T_rope", kind="T", pos="rope"), L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1),
             L_noloop1=dict(name="L_noloop1", kind="L", loop=False, layers=1))
ST = dict(s.ST_LOOP, batch=64, d_model=64, n_layer=int(os.environ.get("NL", 2)), d_tok=64, spatial_ch=8, warmup=100, lr_min_frac=0.1)


def evaluer(model, t, h, k, B=512, seed=777):
    model.eval()
    with torch.no_grad():
        x, y, _ = t.batch(np.random.default_rng(seed + h), B, h, k)
        acc = (model(torch.from_numpy(x))[:, -1].argmax(-1) == torch.from_numpy(y)).float().mean().item()
    model.train(); return acc


if __name__ == "__main__":
    name, n, hmax, k, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]); seed = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    t = Tache(n, 8, "cycle"); torch.manual_seed(seed); model = s.build(SPECS[name], t.V, s.model_cfg(ST))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step); hm = 1 + int((hmax - 1) * min(1.0, step / (0.4 * steps)))
        x, y, _ = t.batch(rng, ST["batch"], rng.integers(1, hm + 1, ST["batch"]), k)
        loss = F.cross_entropy(model(torch.from_numpy(x))[:, -1], torch.from_numpy(y))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % max(1, steps // 5) == 0:
            print("[%s n=%d h<=%d k=%d] pas %d/%d perte %.3f  acc(h=1..%d) %s  (%.0fs)" % (name, n, hmax, k, step, steps, loss.item(), min(hmax + 2, 6), [round(evaluer(model, t, h, k), 2) for h in range(1, min(hmax + 2, 6) + 1)], time.time() - t0), flush=True)
