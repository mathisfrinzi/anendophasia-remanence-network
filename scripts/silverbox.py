"""Silverbox (amendement 28) : identification non lineaire en simulation (u -> y), GRU / LSTM / T_rope / boucle.
Usage : python scripts/silverbox.py MODELE GRAINE [PAS]   modeles : lineaire, GRU, LSTM, T_rope, L_loop1, L_noloop1  (GRU_20k, LSTM_20k = 20000 pas)
Donnees : data/silverbox/SNLS80mV.csv (V1 = entree, V2 = sortie). Resultat : results/silverbox_<modele>_s<graine>.json"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, pandas as pd, torch, torch.nn as nn
import s3_tasks as s

torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(__file__), "..")
D = pd.read_csv(os.path.join(ROOT, "data", "silverbox", "SNLS80mV.csv")); U, Y = D.V1.values.astype(np.float64), D.V2.values.astype(np.float64)
TEST, TRAIN, VAL = (0, 40000), (40000, 121072), (121072, 131072)
MU, SU, MY, SY = U[TRAIN[0]:TRAIN[1]].mean(), U[TRAIN[0]:TRAIN[1]].std(), Y[TRAIN[0]:TRAIN[1]].mean(), Y[TRAIN[0]:TRAIN[1]].std()
u_n, y_n = ((U - MU) / SU).astype(np.float32), ((Y - MY) / SY).astype(np.float32)
W, BURN, SKIP = 96, 32, 96            # fenetre, echauffement non note a l'entrainement, premiers echantillons ignores a l'evaluation
ST = dict(s.ST_LOOP, batch=32, d_model=64, n_layer=2, d_tok=64, spatial_ch=8, warmup=100, lr_min_frac=0.1)


class ContEmb(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.lin = nn.Linear(1, dim)
    def forward(self, x): return self.lin(x.unsqueeze(-1))


class RNN(nn.Module):
    def __init__(self, kind, h):
        super().__init__(); self.r = (nn.GRU if kind == "GRU" else nn.LSTM)(1, h, batch_first=True); self.o = nn.Linear(h, 1)
    def forward(self, x): return self.o(self.r(x.unsqueeze(-1))[0])


def build(name):
    base = name.split("_")[0] if name.endswith("20k") else name
    if base in ("GRU", "LSTM"): return RNN(base, 256)
    spec = dict(T_rope=dict(kind="T", pos="rope"), L_loop1=dict(kind="L", loop=True, layers=1), L_noloop1=dict(kind="L", loop=False, layers=1))[base]
    m = s.build(dict(spec, name=base), 1, s.model_cfg(ST)); m.emb = ContEmb(m.emb.embedding_dim); return m


def predict(model, idx_u, B=128):
    """idx_u : tableau (nw, W) d'entrees ; renvoie (nw, W) de sorties (standardisees)."""
    out = []
    with torch.no_grad():
        for i in range(0, len(idx_u), B): out.append(model(torch.from_numpy(idx_u[i:i + B])).squeeze(-1).numpy())
    return np.concatenate(out)


def evaluate(model, lo, hi):
    """Simulation par fenetres de W (pas BURN) ; RMSE (mV) sur les echantillons t >= lo + SKIP, par quart de la plage."""
    model.eval(); stride = W - BURN; starts = np.arange(lo, hi - W + 1, stride)
    pred = predict(model, np.stack([u_n[a:a + W] for a in starts])); yhat = np.full(hi - lo, np.nan)
    for a, p in zip(starts, pred): yhat[a - lo + BURN:a - lo + W] = p[BURN:]
    err = (yhat - y_n[lo:hi]) * SY * 1000; ok = np.arange(hi - lo) >= SKIP; ok &= ~np.isnan(err); n = hi - lo
    q = [float(np.sqrt(np.mean(err[ok & (np.arange(n) >= i * n // 4) & (np.arange(n) < (i + 1) * n // 4)] ** 2))) for i in range(4)]
    model.train(); return dict(rmse_mV=float(np.sqrt(np.mean(err[ok] ** 2))), quarts_mV=q)


def lineaire(lags):
    """Reference : regression lineaire (moindres carres) sur `lags` entrees passees ; RMSE (mV) sur t >= SKIP."""
    def feats(lo, hi):
        t = np.arange(lo + SKIP, hi); return np.stack([u_n[t - j] for j in range(lags)], 1), y_n[t], t - lo
    Xt, yt, _ = feats(*TRAIN); beta = np.linalg.lstsq(np.c_[Xt, np.ones(len(Xt))], yt, rcond=None)[0]; res = {}
    for nom, (lo, hi) in dict(test=TEST, val=VAL).items():
        X, y, t = feats(lo, hi); e = (np.c_[X, np.ones(len(X))] @ beta - y) * SY * 1000; n = hi - lo
        res[nom] = dict(rmse_mV=float(np.sqrt(np.mean(e ** 2))), quarts_mV=[float(np.sqrt(np.mean(e[(t >= i * n // 4) & (t < (i + 1) * n // 4)] ** 2))) for i in range(4)])
    return res


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); path = os.path.join(ROOT, "results", "silverbox_%s_s%d.json" % (name, seed))
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else (20000 if name.endswith("20k") else 2000)
    if os.path.exists(path): sys.exit()
    if name == "lineaire":
        r = dict(model=name, **{"lag%d" % L: lineaire(L) for L in (32, 96)}, constante_test_mV=float(np.sqrt(np.mean(((y_n[TEST[0] + SKIP:TEST[1]] - 0) * SY * 1000) ** 2))))
        json.dump(r, open(path, "w"), indent=1); print(json.dumps(r)[:400]); sys.exit()
    torch.manual_seed(seed); model = build(name); rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for g in opt.param_groups: g["lr"] = s.lr_at(step, steps, ST)
        a = rng.integers(TRAIN[0], TRAIN[1] - W, ST["batch"]); x = torch.from_numpy(np.stack([u_n[i:i + W] for i in a])); y = torch.from_numpy(np.stack([y_n[i:i + W] for i in a]))
        loss = ((model(x).squeeze(-1)[:, BURN:] - y[:, BURN:]) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % max(250, steps // 8) == 0:
            print("[%s s%d] pas %d/%d perte %.5f  val %.2f mV  (%.0fs)" % (name, seed, step, steps, loss.item(), evaluate(model, *VAL)["rmse_mV"], time.time() - t0), flush=True)
    r = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), test=evaluate(model, *TEST), val=evaluate(model, *VAL))
    json.dump(r, open(path, "w"), indent=1); print({k: r[k] for k in ("test", "val", "seconds")}, flush=True)
