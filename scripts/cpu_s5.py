"""Amendement 31 : S5 (120 etats) pour le reseau d'images en couches 2-3-1 et ses temoins, config CPU reduite (entrainement 12, eval 96).
Usage : python scripts/cpu_s5.py MODELE GRAINE [PAS]   modeles : G231 (reseau en couches, 18 transformers), G231_ff (sans balayage retour), L_loop1, GRU, T_rope
Sortie : results/s5_<modele>_s<graine>.json"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from s7_graphe import GraphLM
from rnn_taches import RNN

torch.set_num_threads(1)
G, mult = s.group(5)
ST = dict(s.ST_LOOP, train_len=12, eval_len=96, batch=64, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1, buckets=[0, 12, 24, 48, 96])
SPECS = dict(L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), T_rope=dict(name="T_rope", kind="T", pos="rope"))


def build(name):
    if name in ("G231", "G231_ff"): return GraphLM(G, dict(d_tok=32), retour=(name == "G231"))
    if name == "GRU": return RNN("GRU", G)
    return s.build(SPECS[name], G, s.model_cfg(ST))


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    path = os.path.join(os.path.dirname(__file__), "..", "results", "s5_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    torch.manual_seed(seed); model = build(name); st = ST; curve = []
    opt = torch.optim.AdamW(model.parameters(), lr=st["lr"], betas=(0.9, 0.95), weight_decay=st["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, st)
        x, y = s.make_batch(np.random.default_rng(seed * 1_000_003 + step), G, mult, st["batch"], st["train_len"])
        L = min(12, st["curriculum_start"] + int((12 - st["curriculum_start"]) * min(1.0, step / (st["curriculum_frac"] * steps))))
        loss = F.cross_entropy(model(x)[:, :L].reshape(-1, G), y[:, :L].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), st["grad_clip"]); opt.step()
        if step % 500 == 0 or step == steps:
            e = s.evaluate(model, G, mult, 12, 256, 9999, torch.device("cpu"), [0, 12]); curve.append(dict(step=step, loss=loss.item(), acc12=e["acc_0_12"], L=L))
            print("[S5 %s s%d] pas %d/%d perte %.3f prec. n<=12 %.3f (%.0fs)" % (name, seed, step, steps, loss.item(), e["acc_0_12"], time.time() - t0), flush=True)
    final = s.evaluate(model, G, mult, 96, 512, 9999, torch.device("cpu"), st["buckets"])
    json.dump(dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), chance=1 / G, final=final, curve=curve), open(path, "w"), indent=1); print(final, flush=True)
