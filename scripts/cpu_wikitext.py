"""LM reel sur CPU (amendement 20). Usage : python scripts/cpu_wikitext.py MODELE GRAINE [PAS]"""
import sys, os, json, time, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer
import s3_tasks as s

torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(__file__), "..")
SPECS = dict(L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), T_rope=dict(name="T_rope", kind="T", pos="rope"),
             V_lam0=dict(name="V_lam0", kind="V", scales=2, penalty=0.0))
ST = dict(s.ST_LOOP, batch=32, d_model=64, n_layer=2, d_tok=64, spatial_ch=8, warmup=100, lr_min_frac=0.1)
N, LN2 = 64, math.log(2)
CORPUS = os.environ.get("CORPUS", "wikitext2")
tok = Tokenizer.from_file(os.path.join(ROOT, "tokenizer", "bpe_%s_4096.json" % CORPUS)); V = tok.get_vocab_size()
TR = np.load(os.path.join(ROOT, "data", CORPUS, "ids_train_4096.npy")).astype(np.int64)
VA = np.load(os.path.join(ROOT, "data", CORPUS, "ids_val_4096.npy")).astype(np.int64)
CH = np.array([max(1, len(tok.decode([i]))) for i in range(V)], dtype=np.float64)


def windows(ids, rng, B, n):
    off = rng.integers(0, len(ids) - n - 1, size=B)
    return torch.from_numpy(np.stack([ids[o:o + n + 1] for o in off]))


@torch.no_grad()
def evaluate(model, n, B, seed=9999, edges=(0, 64, 128, 256), uni=None):
    model.eval(); w = windows(VA, np.random.default_rng(seed), B, n); x, y = w[:, :-1], w[:, 1:]
    nll = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), reduction="none").reshape(y.shape).numpy() if uni is None else -uni[y.numpy()]
    ch = CH[y.numpy()]; out = {}
    for a, b in zip(edges[:-1], edges[1:]):
        if a < n: out["%d-%d" % (a, min(b, n))] = float(nll[:, a:b].sum() / LN2 / ch[:, a:b].sum())
    model.train(); return out


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    path = os.path.join(ROOT, "results", ("lm_wiki_%s_s%d.json" if CORPUS == "wikitext2" else "lm_" + CORPUS + "_%s_s%d.json") % (name, seed))
    if name == "unigramme":
        p = np.bincount(TR, minlength=V) + 1.0; p = np.log(p / p.sum())
        r = dict(model=name, final={"256": evaluate(type("M", (), {"eval": lambda s: None, "train": lambda s: None})(), 256, 256, uni=p)}); json.dump(r, open(path, "w"), indent=1); print(r); sys.exit()
    if os.path.exists(path): sys.exit()
    torch.manual_seed(seed); model = s.build(SPECS[name], V, s.model_cfg(ST))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        w = windows(TR, np.random.default_rng(seed * 1_000_003 + step), ST["batch"], N)
        loss = F.cross_entropy(model(w[:, :-1]).reshape(-1, V), w[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 250 == 0:
            e = evaluate(model, N, 64); print("[WIKI %s s%d] pas %d/%d perte %.3f bpc val(0-64) %.3f (%.0fs)" % (name, seed, step, steps, loss.item(), e["0-64"], time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), final={"256": evaluate(model, 256, 96)})
    json.dump(res, open(path, "w"), indent=1); print(res["final"], flush=True)
