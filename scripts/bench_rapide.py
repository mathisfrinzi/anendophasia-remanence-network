"""Amendement 38 : temps par pas de la remanence, version d'origine / rapide / rapide + torch.compile, sur l'appareil choisi.
  python scripts/bench_rapide.py --device xpu --batch 16 --window 256"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
import s3_tasks as s, s8_rapide
from s8_rapide import FastLoop

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="auto"); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--window", type=int, default=256); a = ap.parse_args()
dev = torch.device(a.device if a.device != "auto" else ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
sync = lambda: torch.xpu.synchronize() if dev.type == "xpu" else torch.cuda.synchronize() if dev.type == "cuda" else None
ST = dict(s.ST_LOOP, d_model=64, n_layer=2, d_tok=64, spatial_ch=8)
x = torch.randint(0, 4096, (a.batch, a.window), device=dev); y = torch.randint(0, 4096, (a.batch, a.window), device=dev)


def bench(label, mk, n=3):
    try:
        m = mk().to(dev)
        for _ in range(2): F.cross_entropy(m(x).reshape(-1, 4096), y.reshape(-1)).backward()
        sync(); t = time.time()
        for _ in range(n): m.zero_grad(set_to_none=True); F.cross_entropy(m(x).reshape(-1, 4096), y.reshape(-1)).backward()
        sync(); r = (time.time() - t) / n; print("BENCH %-22s %.3f s/pas (appareil %s, batch %d, fenetre %d)" % (label, r, dev, a.batch, a.window), flush=True); return r
    except Exception as e: print("BENCH %-22s ECHEC : %s" % (label, str(e)[:150]), flush=True)


mk = lambda: s.build(dict(name="L", kind="L", loop=True, layers=1), 4096, s.model_cfg(ST))
r0 = bench("origine", mk); r1 = bench("rapide", lambda: FastLoop(mk()))
s8_rapide._t = torch.compile(s8_rapide._t); r2 = bench("rapide + compile", lambda: FastLoop(mk()))
if r0 and r1: print("gain rapide x%.2f" % (r0 / r1) + ("  | rapide + compile x%.2f" % (r0 / r2) if r2 else ""))
