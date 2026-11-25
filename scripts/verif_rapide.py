"""Amendement 38 (suite) : la version rapide apprend-elle comme l'originale ? Liste circulaire n = 6, h <= 3, k = 0, meme graine, memes donnees, 4 000 pas.
Sortie : results/verif_rapide_s<graine>.json (precision h = 1..3 tous les 500 pas, pour les deux versions)."""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
from reseau_tache import Tache
from reseau_verif import SPECS, ST, evaluer
from s8_rapide import FastLoop

torch.set_num_threads(1)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
only = sys.argv[3] if len(sys.argv) > 3 else None          # "origine" ou "rapide" : permet de lancer les deux en parallele
out = {}
for version in ((only,) if only else ("origine", "rapide")):
    t = Tache(6, 8, "cycle"); torch.manual_seed(seed); m = s.build(SPECS["L_loop1"], t.V, s.model_cfg(ST)); model = FastLoop(m) if version == "rapide" else m
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time(); curve = []
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step); hm = 1 + int(2 * min(1.0, step / (0.4 * steps)))
        x, y, _ = t.batch(rng, ST["batch"], rng.integers(1, hm + 1, ST["batch"]), 0)
        loss = F.cross_entropy(model(torch.from_numpy(x))[:, -1], torch.from_numpy(y))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0:
            a = [round(evaluer(model, t, h, 0, 256), 3) for h in (1, 2, 3)]; curve.append(dict(step=step, loss=round(loss.item(), 4), acc=a)); print("[%s] pas %d perte %.4f h=1..3 %s (%.0fs)" % (version, step, loss.item(), a, time.time() - t0), flush=True)
    out[version] = dict(seconds=round(time.time() - t0), curve=curve)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "verif_rapide%s_s%d.json" % ("_" + only if only else "", seed)), "w"), indent=1)
