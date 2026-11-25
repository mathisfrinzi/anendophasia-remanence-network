"""Tests causaux/exploratoires sur un L_loop entraine (config de cpu_boucle.py, graine donnee).
1) bruit injecte dans le residu de T2 a l'inference ; 2) sonde lineaire (moindres carres) de l'etat vrai sur Image1 / Image2."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
import s3_tasks as s
from cpu_boucle import ST, SPECS

torch.set_num_threads(2)
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
steps = 2000
root = os.path.join(os.path.dirname(__file__), "..")
G, mult = s.group(3)
ck = os.path.join(root, "results", "cpu_L_loop_s%d_model.pt" % seed)
model = s.build(SPECS["L_loop"], G, s.model_cfg(ST))
if os.path.exists(ck):
    model.load_state_dict(torch.load(ck))
else:
    torch.manual_seed(seed)
    model = s.build(SPECS["L_loop"], G, s.model_cfg(ST))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"])
    import torch.nn.functional as F
    n = ST["train_len"]
    for step in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = s.lr_at(step, steps, ST)
        x, y = s.make_batch(np.random.default_rng(seed * 1_000_003 + step), G, mult, ST["batch"], n)
        L = min(n, ST["curriculum_start"] + int((n - ST["curriculum_start"]) * min(1.0, step / (ST["curriculum_frac"] * steps))))
        loss = F.cross_entropy(model(x)[:, :L].reshape(-1, G), y[:, :L].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
    torch.save(model.state_dict(), ck)
model.eval()
out = {}
ev = ST["eval_len"]
with torch.no_grad():
    for noise in [0.0, 0.1, 0.3, 1.0]:
        model.t2_noise = noise
        m = s.evaluate(model, G, mult, ev, 256, 9999, torch.device("cpu"), ST["buckets"])
        out["bruit_T2_%.1f" % noise] = {k: round(v, 3) for k, v in m.items()}
        print("bruit T2", noise, out["bruit_T2_%.1f" % noise], flush=True)
    model.t2_noise = 0.0
    # sonde lineaire : etat vrai (produit cumule) lu sur Image1 / Image2 a chaque pas
    xs, ys = s.make_batch(np.random.default_rng(4242), G, mult, 600, 48)
    logits, fr = model(xs, return_frames=True)              # fr : (B, N, 2, C, H, W)
    pred_acc = (logits.argmax(-1) == ys).float().mean(0)
    B, N = xs.shape
    res = {}
    for name, sl in [("Image1", fr[:, :, 0]), ("Image2", fr[:, :, 1])]:
        accs = []
        for t in [5, 11, 23, 47]:
            X = sl[:, t].flatten(1); X = X - X.mean(0); Y = torch.nn.functional.one_hot(ys[:, t], G).float()
            tr, te = slice(0, 400), slice(400, 600)
            W = torch.linalg.lstsq(torch.cat([X[tr], torch.ones(400, 1)], 1), Y[tr]).solution
            P = torch.cat([X[te], torch.ones(200, 1)], 1) @ W
            accs.append(round((P.argmax(1) == ys[te, t]).float().mean().item(), 3))
        res[name] = accs
        print("sonde", name, "t=5,11,23,47 :", accs, flush=True)
    out["sonde_lineaire"] = res
    out["acc_modele_aux_memes_pas"] = [round(pred_acc[t].item(), 3) for t in [5, 11, 23, 47]]
json.dump(out, open(os.path.join(root, "results", "cpu_boucle_causal_s%d.json" % seed), "w"), indent=1)
