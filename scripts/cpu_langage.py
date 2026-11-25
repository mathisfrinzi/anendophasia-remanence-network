"""Petit LM sur histoires synthetiques (amendement 19). Usage : python scripts/cpu_langage.py MODELE GRAINE [PAS]"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s

torch.set_num_threads(1)
NAMES = ["tom", "lea", "max", "zoe", "ben", "ana"]; OBJS = ["key", "ball", "hat", "book", "cup", "pen"]; PLACES = ["park", "shop", "home", "school"]
FUNC = ["has", "the", ".", "is", "in", "swaps", "with", "goes", "to", "who", "where", "now", "?"]
VOC = NAMES + OBJS + PLACES + FUNC; ID = {w: i for i, w in enumerate(VOC)}; V = len(VOC)
SPECS = dict(L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), T_rope=dict(name="T_rope", kind="T", pos="rope"),
             V_lam0=dict(name="V_lam0", kind="V", scales=2, penalty=0.0))
ST = dict(s.ST_LOOP, batch=32, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1)


def story(rng, E):
    ppl = rng.choice(6, 3, replace=False); obj = rng.choice(6, 3, replace=False); plc = rng.integers(0, 4, 3)
    toks, ans = [], []
    def add(ws, a=-1):
        toks.extend(ID[w] for w in ws); ans.extend([0] * (len(ws) - 1) + [0]); 
        if a >= 0: ans[-3] = a      # position du '?' : le token suivant est la reponse
    for i in range(3): add([NAMES[ppl[i]], "has", "the", OBJS[obj[i]], "."]); 
    for i in range(3): add([NAMES[ppl[i]], "is", "in", PLACES[plc[i]], "."])
    for _ in range(E):
        if rng.random() < 0.5:
            i, j = rng.choice(3, 2, replace=False); obj[i], obj[j] = obj[j], obj[i]; add([NAMES[ppl[i]], "swaps", "with", NAMES[ppl[j]], "."])
        else:
            i = rng.integers(3); plc[i] = rng.integers(4); add([NAMES[ppl[i]], "goes", "to", PLACES[plc[i]], "."])
    for _ in range(3):
        if rng.random() < 0.5:
            i = rng.integers(3); toks.extend(ID[w] for w in ["who", "has", "the", OBJS[obj[i]], "?", NAMES[ppl[i]], "."]); ans.extend([0, 0, 0, 0, 1, 0, 0])
        else:
            i = rng.integers(3); toks.extend(ID[w] for w in ["where", "is", NAMES[ppl[i]], "now", "?", PLACES[plc[i]], "."]); ans.extend([0, 0, 0, 0, 2, 0, 0])
    return toks, ans


def batch(rng, B, E):
    if not np.isscalar(E):
        E = int(rng.integers(E[0], E[1] + 1))          # un seul E par lot (histoires de meme longueur, pas de troncature)
    T, A = zip(*[story(rng, E) for _ in range(B)])
    return torch.tensor(T), torch.tensor(A)


def fix_marks(x, a):          # a : 1/2 au token '?' (position de prediction) ; on le recalcule proprement
    a = torch.zeros_like(x); q = ID["?"]; pos = (x == q)
    w = torch.zeros_like(x); w[:, 3:] = (x[:, :-3] == ID["who"]).long()      # '?' est 4 tokens apres 'who' (who has the O ?) -> decalage 4
    return pos


@torch.no_grad()
def evaluate(model, E, B=256, seed=9999):
    model.eval(); rng = np.random.default_rng(seed + E); x, a = batch(rng, B, E)
    pr = model(x[:, :-1]).argmax(-1); y = x[:, 1:]; am = a[:, :-1]; model.train()
    out = {}
    for k, nm in [(1, "who"), (2, "where")]:
        m = am == k; out[nm] = (pr[m] == y[m]).float().mean().item()
    m = am > 0; out["reponse"] = (pr[m] == y[m]).float().mean().item()
    return out


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]); steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    V2 = os.environ.get("LM_V2") == "1"
    root = os.path.join(os.path.dirname(__file__), ".."); path = os.path.join(root, "results", "lm_synth%s_%s_s%d.json" % ("_v2" if V2 else "", name, seed))
    if os.path.exists(path): sys.exit()
    torch.manual_seed(seed); model = s.build(SPECS[name], V, s.model_cfg(ST))
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        emax = 1 + int(9 * min(1.0, step / (0.4 * steps))) if V2 else 10
        x, a = batch(np.random.default_rng(seed * 1_000_003 + step), ST["batch"], (1 if V2 else 4, emax))
        logits = model(x[:, :-1]); w = 1.0 + (19.0 if V2 else 4.0) * (a[:, :-1] > 0).float()
        loss = (F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1), reduction="none") * w.reshape(-1)).sum() / w.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 250 == 0:
            e = evaluate(model, 8, 128); print("[LM %s s%d] pas %d/%d perte %.3f reponse %.2f (who %.2f where %.2f) (%.0fs)" % (name, seed, step, steps, loss.item(), e["reponse"], e["who"], e["where"], time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), final={str(E): evaluate(model, E) for E in [8, 16, 32]})
    json.dump(res, open(path, "w"), indent=1); print(res["final"], flush=True)
