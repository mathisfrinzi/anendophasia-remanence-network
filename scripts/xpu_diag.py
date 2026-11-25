"""Amendement 35 : echelle de croissance (taille de la remanence, images 4x8x8, d = 64) sur GPU (XPU / CUDA / CPU), liste circulaire n = 6, h <= 3, k = 0.
  python scripts/xpu_diag.py --bench                         # mesure le temps par pas de chaque configuration (~1-2 min)
  python scripts/xpu_diag.py --steps 4000 --batch 256 1-1 2-1 1-1-1 2-3-1 --seed 0     # entraine les configurations dans l'ordre
Sorties : results/xpu_<widths>_s<graine>.json et une ligne 'RESULT {...}' par configuration (a coller si le push est impossible)."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from reseau_tache import Tache
from s7_graphe import GraphLM2

ST = dict(s.ST_LOOP, warmup=100, lr_min_frac=0.1)


def pick(arg):
    if arg != "auto": return torch.device(arg)
    if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(dev):
    if dev.type == "xpu": torch.xpu.synchronize()
    elif dev.type == "cuda": torch.cuda.synchronize()


def build(name, t, dev):
    torch.manual_seed(SEED)
    return GraphLM2(t.V, [int(v) for v in name.rstrip("z").split("-")], zero=name.endswith("z"), h=8, w=8, d=64, nh=4).to(dev)


def acc(model, t, h, dev, B=512):
    model.eval()
    with torch.no_grad():
        x, y, _ = t.batch(np.random.default_rng(777 + h), B, h, 0)
        a = (model(torch.from_numpy(x).to(dev))[:, -1].argmax(-1).cpu() == torch.from_numpy(y)).float().mean().item()
    model.train(); return a


def train(name, steps, batch, dev, bench=False):
    t = Tache(6, 8, "cycle"); model = build(name, t, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    n = 8 if bench else steps
    for step in range(1, n + 1):
        if bench and step == 4: sync(dev); t0 = time.time()
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(SEED * 1_000_003 + step); hm = 1 + int(2 * min(1.0, step / (0.4 * steps)))
        x, y, _ = t.batch(rng, batch, rng.integers(1, hm + 1, batch), 0)
        loss = F.cross_entropy(model(torch.from_numpy(x).to(dev))[:, -1], torch.from_numpy(y).to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if not bench and step % 250 == 0:
            print("[%s] pas %d/%d perte %.3f h=1..3 %s (%.0fs)" % (name, step, steps, loss.item(), [round(acc(model, t, h, dev, 256), 2) for h in (1, 2, 3)], time.time() - t0), flush=True)
    sync(dev)
    if bench: return (time.time() - t0) / 5, model
    return {str(h): acc(model, t, h, dev) for h in range(1, 6)}, model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("configs", nargs="*", default=["1-1", "2-1", "1-1-1", "2-3-1"]); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto"); ap.add_argument("--bench", action="store_true")
    a = ap.parse_args(); SEED = a.seed; dev = pick(a.device); print("appareil :", dev, "| torch", torch.__version__, flush=True)
    for name in a.configs:
        if a.bench:
            dt, _ = train(name, 100, a.batch, dev, bench=True); print("BENCH %s : %.3f s/pas (batch %d) -> 1000 pas = %.1f min" % (name, dt, a.batch, dt * 1000 / 60), flush=True); continue
        t0 = time.time(); res, model = train(name, a.steps, a.batch, dev)
        out = dict(model=name, seed=SEED, steps=a.steps, batch=a.batch, device=str(dev), n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), acc=res)
        json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "xpu_%s_s%d.json" % (name, SEED)), "w"), indent=1); print("RESULT", json.dumps(out), flush=True)
