"""Amendement 33b : diagnostic de l'echec du reseau 2-3-1 ; on part de la boucle a 2 images (1-1) et on ajoute des images, meme taille d'image, memes donnees.
Usage : python scripts/diag_graphe.py WIDTHS GRAINE [PAS]   WIDTHS : '1-1', '2-1', '1-1-1', '2-2-1', '2-3-1', ou suffixe 'z' (sortie des aretes initialisee a zero) : '2-3-1z'
Tache : liste circulaire n = 6, h <= 3, k = 0. Variable DIAG_SIZE=full : taille de la boucle. Sortie : results/diag_<widths>[_full]_s<graine>.json (precision h = 1..5)"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from reseau_tache import Tache
from reseau_verif import ST, evaluer
from s7_graphe import GraphLM2

torch.set_num_threads(1)
if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    FULL = os.environ.get("DIAG_SIZE") == "full"      # taille de la boucle : images 4x8x8, d = 64, 4 tetes
    path = os.path.join(os.path.dirname(__file__), "..", "results", "diag_%s%s_s%d.json" % (name, "_full" if FULL else "", seed))
    if os.path.exists(path): sys.exit()
    t = Tache(6, 8, "cycle"); torch.manual_seed(seed); zero = name.endswith("z")
    model = GraphLM2(t.V, [int(v) for v in name.rstrip("z").split("-")], zero=zero, **(dict(h=8, w=8, d=64, nh=4) if FULL else {}))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step); hm = 1 + int(2 * min(1.0, step / (0.4 * steps)))
        x, y, _ = t.batch(rng, ST["batch"], rng.integers(1, hm + 1, ST["batch"]), 0)
        loss = F.cross_entropy(model(torch.from_numpy(x))[:, -1], torch.from_numpy(y))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0: print("[DIAG %s s%d] pas %d/%d perte %.3f h=1..3 %s (%.0fs)" % (name, seed, step, steps, loss.item(), [round(evaluer(model, t, h, 0, 256), 2) for h in (1, 2, 3)], time.time() - t0), flush=True)
    acc = {str(h): evaluer(model, t, h, 0, 512) for h in range(1, 6)}
    json.dump(dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), acc=acc), open(path, "w"), indent=1); print(acc, flush=True)
