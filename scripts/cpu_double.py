"""Double boucle sur S3+memoire (amendement 26). Usage : python scripts/cpu_double.py MODELE GRAINE [PAS]
MODELE in {L_dloop1, L_dloop1_dec}. Config identique a l'amendement 17 (cpu_memoire.py) ; reference : results/st_S3M_L_loop1_s*.json."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s
import cpu_memoire as M
from s6_double import DoubleLoopLM

torch.set_num_threads(1)
SPECS = dict(L_dloop1=dict(decorr=0.0), L_dloop1_dec=dict(decorr=0.1))


def build(name, cfg):
    return DoubleLoopLM(M.V, cfg, d_edge=44, layers=1, **SPECS[name])


@torch.no_grad()
def ablation(model, ns=(24, 48, 96)):
    """Perte de performance quand on annule le depot sur Image1 ou sur Image2 (inference seulement)."""
    out = {}
    for tag, sc in [("base", (1.0, 1.0)), ("sans_depot1", (0.0, 1.0)), ("sans_depot2", (1.0, 0.0))]:
        model.dep_scale = sc; out[tag] = {str(n): M.evaluate(model, n) for n in ns}
    model.dep_scale = (1.0, 1.0); return out


@torch.no_grad()
def readin_cosine(model):
    """Cosinus moyen entre les depots Image1 / Image2 d'un meme token (|cos| proche de 1 : read-in redondants)."""
    e = model.emb.weight; a, b = torch.tanh(model.dep1(e)), torch.tanh(model.dep2(e))
    return dict(cos_abs=F.cosine_similarity(a, b, dim=-1).abs().mean().item(), cos=F.cosine_similarity(a, b, dim=-1).mean().item())


@torch.no_grad()
def probes(model, n=48):
    """Sonde lineaire de la valeur stockee (S3M) sur Image1, Image2, Image3 a plusieurs pas du delai."""
    model.eval(); x, y = M.batch(np.random.default_rng(777), 800, n)
    _, fr = model(x, return_frames=True)                              # liste de n+2 tuples (I1, I2, I3)
    v = y[:, -1] - 6; Y = F.one_hot(v, M.NV).float(); out = {}
    def probe(X):
        X = X - X.mean(0); A = torch.cat([X, torch.ones(len(X), 1)], 1)
        Wt = torch.linalg.lstsq(A[:600], Y[:600]).solution
        return round(((A[600:] @ Wt).argmax(1) == v[600:]).float().mean().item(), 3)
    for t in [1, 5, 15, 30, n]:
        out["t%d" % t] = {nm: probe(fr[t][i].flatten(1)) for i, nm in enumerate(["Image1", "Image2", "Image3"])}
    model.train(); return out


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    root = os.path.join(os.path.dirname(__file__), ".."); path = os.path.join(root, "results", "dl_S3M_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    ST = M.ST; torch.manual_seed(seed); model = build(name, s.model_cfg(ST))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        nmax = 4 + int(20 * min(1.0, step / (0.4 * steps))); n = int(rng.integers(4, nmax + 1))
        x, y = M.batch(rng, ST["batch"], n)
        loss = F.cross_entropy(model(x).reshape(-1, M.V), y.reshape(-1), ignore_index=-100)
        if model.decorr: loss = loss + model.aux
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 250 == 0:
            e = M.evaluate(model, 24, 128); print("[S3M %s s%d] pas %d/%d perte %.3f s3 %.2f rappel %.2f (%.0fs)" % (name, seed, step, steps, loss.item(), e["s3"], e["recall"], time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0),
               final={str(n): M.evaluate(model, n) for n in [12, 24, 32, 48, 64, 96]}, ablation=ablation(model), readin=readin_cosine(model), sondes=probes(model))
    json.dump(res, open(path, "w"), indent=1); print(res["final"], res["readin"], flush=True)
