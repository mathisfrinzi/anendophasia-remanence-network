"""Direction 3 : que contiennent les images ? Sondes lineaires sur l'etat interne, apres entrainement sur les histoires synthetiques de cpu_langage.
Usage : python scripts/sondes_histoires.py MODELE GRAINE [PAS] [--device auto|cuda|xpu|cpu] [--n-stories 4000] [--tag _x]
MODELE : L_loop1 (remanence), GRU_256, L_loop1_init (remanence NON entrainee : temoin). PAS : 3000 par defaut.
Apres l'entrainement : N histoires neuves (graine fixe), etat capture a deux moments : fin des evenements (dernier '.' avant les questions)
et sur chaque '?' (3 par histoire). Etat = I1 et I2 aplaties concatenees (remanence) ou etat cache du GRU. Cibles : pour chacun des 3 personnages
(dans l'ordre de presentation), son lieu actuel (4 classes) et l'objet qu'il tient (6 classes). Sonde = regression logistique multinomiale
(torch), entrainee sur 75 % des histoires, precision sur les 25 % restantes ; hasard et classe majoritaire donnes en reference.
Sortie : results/sondes_<modele>_s<graine><tag>.json"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_nt = torch.get_num_threads()
import cpu_langage as cl
torch.set_num_threads(_nt)
from rnn_taches import RNN
torch.set_num_threads(_nt)
import s3_tasks as s
from ts_remanence import pick

ID, V, NAMES, OBJS, PLACES, ST = cl.ID, cl.V, cl.NAMES, cl.OBJS, cl.PLACES, cl.ST


def story_gt(rng, E):
    """Meme histoire que cpu_langage.story (memes tirages) ; renvoie en plus : positions des '?', fin des evenements, lieux et objets finaux des 3 personnages."""
    ppl = rng.choice(6, 3, replace=False); obj = rng.choice(6, 3, replace=False); plc = rng.integers(0, 4, 3); toks, q = [], []
    def add(ws): toks.extend(ID[w] for w in ws)
    for i in range(3): add([NAMES[ppl[i]], "has", "the", OBJS[obj[i]], "."])
    for i in range(3): add([NAMES[ppl[i]], "is", "in", PLACES[plc[i]], "."])
    for _ in range(E):
        if rng.random() < 0.5:
            i, j = rng.choice(3, 2, replace=False); obj[i], obj[j] = obj[j], obj[i]; add([NAMES[ppl[i]], "swaps", "with", NAMES[ppl[j]], "."])
        else:
            i = rng.integers(3); plc[i] = rng.integers(4); add([NAMES[ppl[i]], "goes", "to", PLACES[plc[i]], "."])
    fin = len(toks) - 1; plc_f, obj_f = plc.copy(), obj.copy()
    for _ in range(3):
        if rng.random() < 0.5:
            i = rng.integers(3); add(["who", "has", "the", OBJS[obj[i]], "?", NAMES[ppl[i]], "."])
        else:
            i = rng.integers(3); add(["where", "is", NAMES[ppl[i]], "now", "?", PLACES[plc[i]], "."])
        q.append(len(toks) - 3)                                    # position du '?'
    return toks, q, fin, plc_f, obj_f


class Dev(nn.Module):                                              # enveloppe : entrees cpu -> appareil, sorties -> cpu (pour cpu_langage.evaluate)
    def __init__(s_, m, dev): super().__init__(); s_.m, s_.dev = m, dev
    def forward(s_, x): return s_.m(x.to(s_.dev)).cpu()


def train(model, seed, steps, dev):
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        x, a = cl.batch(np.random.default_rng(seed * 1_000_003 + step), ST["batch"], (4, 10)); x, a = x.to(dev), a.to(dev)
        w = 1.0 + 4.0 * (a[:, :-1] > 0).float()
        loss = (F.cross_entropy(model(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1), reduction="none") * w.reshape(-1)).sum() / w.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 250 == 0: print("[sondes %s s%d] pas %d/%d perte %.3f (%.0fs)" % (type(model).__name__, seed, step, steps, loss.item(), time.time() - t0), flush=True)


@torch.no_grad()
def etats(model, x, dev):
    """Etat apres chaque token : (B, N, D). Remanence : I1 et I2 aplaties concatenees ; GRU : etat cache."""
    model.eval()
    if isinstance(model, RNN): return model.r(model.emb(x.to(dev)))[0].cpu()
    _, fr = model(x.to(dev), return_frames=True); return fr.flatten(2).cpu()          # frames : (B, N, 2, C, H, W)


def sonde(Xtr, Ytr, Xte, Yte, K, dev, epochs=300, wd=1e-3):
    """Regression logistique multinomiale (Adam, plein lot), caracteristiques standardisees."""
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-6; Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev); Ytr, Yte = Ytr.to(dev), Yte.to(dev)
    lin = nn.Linear(Xtr.shape[1], K).to(dev); opt = torch.optim.Adam(lin.parameters(), lr=0.01)
    for _ in range(epochs):
        loss = F.cross_entropy(lin(Xtr), Ytr) + wd * (lin.weight ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return (lin(Xte).argmax(-1) == Yte).float().mean().item(), (lin(Xtr).argmax(-1) == Ytr).float().mean().item()


def sondes(model, n, dev, E=8, seed=12345):
    rng = np.random.default_rng(seed); S = [story_gt(rng, E) for _ in range(n)]
    X = torch.tensor([st[0] for st in S]); H = torch.cat([etats(model, X[i:i + 250, :-1], dev) for i in range(0, n, 250)])        # (n, N-1, D)
    plc = torch.tensor(np.stack([st[3] for st in S])); obj = torch.tensor(np.stack([st[4] for st in S])); ntr = int(0.75 * n); out = {}
    caps = dict(fin_evenements=([[st[2]] for st in S]), sur_question=[st[1] for st in S])
    for cap, pos in caps.items():
        k = len(pos[0]); feat = torch.stack([H[i, pos[i]] for i in range(n)])                      # (n, k, D)
        Ftr, Fte = feat[:ntr].flatten(0, 1), feat[ntr:].flatten(0, 1); out[cap] = {}
        for typ, Y, K in (("lieu", plc, 4), ("objet", obj, 6)):
            accs, base, per = [], [], []
            for slot in range(3):
                ytr, yte = Y[:ntr, slot].repeat_interleave(k), Y[ntr:, slot].repeat_interleave(k)
                a, atr = sonde(Ftr, ytr, Fte, yte, K, dev); per.append(round(a, 4)); accs.append(a); base.append((yte == torch.bincount(ytr, minlength=K).argmax()).float().mean().item())
            out[cap][typ] = dict(acc=round(float(np.mean(accs)), 4), par_personnage=per, hasard=round(1 / K, 4), majoritaire=round(float(np.mean(base)), 4))
    return out, H.shape[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("seed", type=int); ap.add_argument("steps", type=int, nargs="?", default=3000)
    ap.add_argument("--device", default="auto"); ap.add_argument("--n-stories", type=int, default=4000); ap.add_argument("--tag", default=""); a = ap.parse_args()
    path = os.path.join(os.path.dirname(__file__), "..", "results", "sondes_%s_s%d%s.json" % (a.model, a.seed, a.tag))
    if os.path.exists(path): print("deja fait :", path); sys.exit()
    dev = pick(a.device); print("appareil :", dev, flush=True); torch.manual_seed(a.seed); t0 = time.time()
    model = (RNN("GRU", V, h=int(a.model[4:])) if a.model.startswith("GRU_") else s.build(cl.SPECS["L_loop1"], V, s.model_cfg(ST))).to(dev)
    if a.model in ("L_loop1",) or a.model.startswith("GRU_"): train(model, a.seed, a.steps, dev)
    elif a.model != "L_loop1_init": raise SystemExit("modele inconnu : " + a.model)
    task = {str(E): cl.evaluate(Dev(model, dev), E) for E in [8, 16, 32]}; print("tache", task, flush=True)
    pr, D = sondes(model, a.n_stories, dev)
    res = dict(model=a.model, seed=a.seed, steps=0 if a.model == "L_loop1_init" else a.steps, n_params=sum(p.numel() for p in model.parameters()), dim_etat=D, n_stories=a.n_stories, device=str(dev), seconds=round(time.time() - t0), tache=task, sondes=pr)
    json.dump(res, open(path, "w"), indent=1); print("RESULT", json.dumps(res), flush=True)
