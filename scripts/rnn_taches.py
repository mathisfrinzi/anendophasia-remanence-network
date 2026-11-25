"""Amendement 30 : GRU / LSTM (1 couche, 256 unites, plongement 64) sur S3, S3M et la liste circulaire, memes conditions que la boucle.
Usage : python scripts/rnn_taches.py TACHE MODELE GRAINE [PAS]   taches : S3 (2000 pas), S3M (3000), PTR (8000, h<=3, k aleatoire <= 8) ; modeles : GRU, LSTM
Sorties : results/rnn_<tache>_<modele>_s<graine>.json"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import s3_tasks as s

torch.set_num_threads(1)


class RNN(nn.Module):
    def __init__(self, kind, V, h=256, e=64):
        super().__init__(); self.emb = nn.Embedding(V, e); self.r = (nn.GRU if kind == "GRU" else nn.LSTM)(e, h, batch_first=True); self.o = nn.Linear(h, V)
    def forward(self, x): return self.o(self.r(self.emb(x))[0])


def lr_cfg(**kw): return dict(s.ST_LOOP, **kw)


def run_s3(kind, seed, steps, rng_fn=None):
    G, mult = s.group(3); st = lr_cfg(train_len=12, eval_len=96, batch=64, warmup=100, lr_min_frac=0.1, buckets=[0, 12, 24, 48, 96])
    torch.manual_seed(seed); model = RNN(kind, G); opt = torch.optim.AdamW(model.parameters(), lr=st["lr"], betas=(0.9, 0.95), weight_decay=st["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, st)
        x, y = s.make_batch(np.random.default_rng(seed * 1_000_003 + step), G, mult, st["batch"], st["train_len"])
        L = min(12, st["curriculum_start"] + int((12 - st["curriculum_start"]) * min(1.0, step / (st["curriculum_frac"] * steps))))
        loss = F.cross_entropy(model(x)[:, :L].reshape(-1, G), y[:, :L].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), st["grad_clip"]); opt.step()
        if step % 500 == 0: print("[S3 %s s%d] pas %d perte %.3f (%.0fs)" % (kind, seed, step, loss.item(), time.time() - t0), flush=True)
    return model, dict(final=s.evaluate(model, G, mult, 96, 512, 9999, torch.device("cpu"), st["buckets"])), t0


def run_s3m(kind, seed, steps):
    import cpu_memoire as cm
    torch.manual_seed(seed); model = RNN(kind, cm.V); st = cm.ST
    opt = torch.optim.AdamW(model.parameters(), lr=st["lr"], betas=(0.9, 0.95), weight_decay=st["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, st)
        rng = np.random.default_rng(seed * 1_000_003 + step); nmax = 4 + int(20 * min(1.0, step / (0.4 * steps))); n = int(rng.integers(4, nmax + 1))
        x, y = cm.batch(rng, st["batch"], n)
        loss = F.cross_entropy(model(x).reshape(-1, cm.V), y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), st["grad_clip"]); opt.step()
        if step % 500 == 0: print("[S3M %s s%d] pas %d perte %.3f (%.0fs)" % (kind, seed, step, loss.item(), time.time() - t0), flush=True)
    return model, dict(final={str(n): cm.evaluate(model, n) for n in [12, 24, 32, 48, 64, 96]}), t0


def run_ptr(kind, seed, steps):
    from reseau_tache import Tache
    from reseau_verif import ST, evaluer
    t = Tache(6, 8, "cycle"); torch.manual_seed(seed); model = RNN(kind, t.V)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step); hm = 1 + int(2 * min(1.0, step / (0.4 * steps))); k = int(rng.integers(0, 9))
        x, y, _ = t.batch(rng, ST["batch"], rng.integers(1, hm + 1, ST["batch"]), k)
        loss = F.cross_entropy(model(torch.from_numpy(x))[:, -1], torch.from_numpy(y))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0: print("[PTR %s s%d] pas %d perte %.3f h=1..3 %s (%.0fs)" % (kind, seed, step, loss.item(), [round(evaluer(model, t, h, 8, 256), 2) for h in (1, 2, 3)], time.time() - t0), flush=True)
    return model, dict(grille={str(h): {str(k): evaluer(model, t, h, k, 512) for k in [0, 2, 4, 8, 12]} for h in [1, 2, 3, 4, 5]}), t0


if __name__ == "__main__":
    task, kind, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]); steps = int(sys.argv[4]) if len(sys.argv) > 4 else dict(S3=2000, S3M=3000, PTR=8000)[task]
    path = os.path.join(os.path.dirname(__file__), "..", "results", "rnn_%s_%s_s%d.json" % (task, kind, seed))
    if os.path.exists(path): sys.exit()
    model, res, t0 = dict(S3=run_s3, S3M=run_s3m, PTR=run_ptr)[task](kind, seed, steps)
    json.dump(dict(task=task, model=kind, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), **res), open(path, "w"), indent=1); print(res, flush=True)
