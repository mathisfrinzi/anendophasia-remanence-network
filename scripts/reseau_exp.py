"""Experience 1 (amendement 23) : temps de reflexion sur la poursuite de pointeurs (liste circulaire).
Usage : python scripts/reseau_exp.py MODELE GRAINE [PAS]    modeles : L_loop1_k, L_loop1_k0, L_noloop1_k, T_rope_k
Sortie : results/reseau_<modele>_s<graine>.json (precision par (h, k))."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from reseau_tache import Tache
from s6_double import DoubleLoopLM
from reseau_verif import SPECS, ST, evaluer

torch.set_num_threads(1)
N_SYM, H_TRAIN, K_MAX_TRAIN = 6, 3, 8
HS, KS = [1, 2, 3, 4, 5], [0, 2, 4, 8, 12]
MODELES = dict(L_loop1_k=("L_loop1", True), L_loop1_k0=("L_loop1", False), L_noloop1_k=("L_noloop1", True), T_rope_k=("T_rope", True), L_dloop1_full_k0=("L_dloop1_full", False))

if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 8000
    base, avec_k = MODELES[name]; root = os.path.join(os.path.dirname(__file__), ".."); path = os.path.join(root, "results", "reseau_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    t = Tache(N_SYM, 8, "cycle"); torch.manual_seed(seed); model = (DoubleLoopLM(t.V, s.model_cfg(ST), d_edge=64, d_edge2=44, c1=4, c2=2, layers=1) if base == "L_dloop1_full" else s.build(SPECS[base], t.V, s.model_cfg(ST)))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step); hm = 1 + int((H_TRAIN - 1) * min(1.0, step / (0.4 * steps)))
        k = int(rng.integers(0, K_MAX_TRAIN + 1)) if avec_k else 0
        x, y, _ = t.batch(rng, ST["batch"], rng.integers(1, hm + 1, ST["batch"]), k)
        loss = F.cross_entropy(model(torch.from_numpy(x))[:, -1], torch.from_numpy(y))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0:
            print("[%s s%d] pas %d/%d perte %.3f  h=1..3 (k=8) %s  (%.0fs)" % (name, seed, step, steps, loss.item(), [round(evaluer(model, t, h, 8, 256), 2) for h in (1, 2, 3)], time.time() - t0), flush=True)
    grille = {str(h): {str(k): evaluer(model, t, h, k, 512) for k in KS} for h in HS}
    json.dump(dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), grille=grille), open(path, "w"), indent=1)
    print("fin", {h: [round(v, 2) for v in d.values()] for h, d in grille.items()}, flush=True)
