"""Amendement 32 : poursuite de pointeurs sur liste circulaire (n = 6) entrainee jusqu'a h = 5 (sans pauses, k = 0), evaluee h = 1..8.
Usage : python scripts/reseau_h5.py MODELE GRAINE [PAS]   modeles : G231, G231_ff, L_loop1, GRU     Sortie : results/h5_<modele>_s<graine>.json"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from reseau_tache import Tache
from reseau_verif import SPECS, ST, evaluer
from s7_graphe import GraphLM
from rnn_taches import RNN

torch.set_num_threads(1)
if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    path = os.path.join(os.path.dirname(__file__), "..", "results", "h5_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    t = Tache(6, 8, "cycle"); torch.manual_seed(seed)
    model = GraphLM(t.V, dict(d_tok=32), retour=(name == "G231")) if name.startswith("G231") else RNN("GRU", t.V) if name == "GRU" else s.build(SPECS[name], t.V, s.model_cfg(ST))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step); hm = 1 + int(4 * min(1.0, step / (0.4 * steps)))
        x, y, _ = t.batch(rng, ST["batch"], rng.integers(1, hm + 1, ST["batch"]), 0)
        loss = F.cross_entropy(model(torch.from_numpy(x))[:, -1], torch.from_numpy(y))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0: print("[H5 %s s%d] pas %d/%d perte %.3f h=1..6 %s (%.0fs)" % (name, seed, step, steps, loss.item(), [round(evaluer(model, t, h, 0, 256), 2) for h in range(1, 7)], time.time() - t0), flush=True)
    acc = {str(h): evaluer(model, t, h, 0, 512) for h in range(1, 9)}
    json.dump(dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), acc=acc), open(path, "w"), indent=1); print(acc, flush=True)
