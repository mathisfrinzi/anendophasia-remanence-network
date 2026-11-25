"""Etiquetage grammatical (UPOS) de gauche a droite sur UD English-EWT (amendement 35).
Usage : python scripts/cpu_ud.py MODELE GRAINE [PAS]    MODELE in {T_rope, L_loop1, L_dloop1_full, baseline}
Chaque mot (sensible a la casse, 5 000 plus frequents du train, sinon UNK) recoit son UPOS en ne voyant que le contexte a gauche et lui-meme.
Sortie : results/ud_<modele>_s<graine>.json (precision dev et test, tous mots / mots hors vocabulaire)."""
import sys, os, json, time, collections
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import s3_tasks as s
from s6_double import DoubleLoopLM

torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(__file__), "..")
NV, MAXLEN, PAD, UNK, DROP = 5000, 48, 0, 1, 0.05
ST = dict(s.ST_LOOP, batch=32, d_model=64, n_layer=2, d_tok=32, warmup=100, lr_min_frac=0.1)


def load(sp):
    txt = open(os.path.join(ROOT, "data", "ud_ewt", sp + ".tsv"), encoding="utf8").read().strip().split("\n\n")
    return [[tuple(l.split("\t")) for l in b.split("\n")] for b in txt]


TR, DEV, TE = load("train"), load("dev"), load("test")
CNT = collections.Counter(w for sn in TR for w, _ in sn)
VOC = {w: i + 2 for i, (w, _) in enumerate(CNT.most_common(NV))}
TAGS = sorted({t for sn in TR for _, t in sn}); TID = {t: i for i, t in enumerate(TAGS)}; NT = len(TAGS)


def encode(sents, maxlen=None):
    """-> (liste d'ids de mots, liste d'ids d'etiquettes, liste d'indicateurs hors vocabulaire), phrases tronquees a maxlen."""
    out = []
    for sn in sents:
        sn = sn if maxlen is None else sn[:maxlen]
        out.append(([VOC.get(w, UNK) for w, _ in sn], [TID[t] for _, t in sn], [w not in VOC for w, _ in sn]))
    return out


ETR, EDEV, ETE = encode(TR, MAXLEN), encode(DEV), encode(TE)


def pad(batch):
    L = max(len(x) for x, _, _ in batch)
    X = np.zeros((len(batch), L), np.int64); Y = np.full((len(batch), L), -100, np.int64); O = np.zeros((len(batch), L), bool)
    for i, (x, y, o) in enumerate(batch): X[i, :len(x)], Y[i, :len(y)], O[i, :len(o)] = x, y, o
    return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(O)


def build(name):
    cfg = s.model_cfg(ST); V = NV + 2
    if name == "T_rope": m = s.build(dict(name="T_rope", kind="T", pos="rope"), V, cfg); m.head = nn.Linear(cfg["d_model"], NT, bias=False)
    else:
        m = DoubleLoopLM(V, cfg, d_edge=64, d_edge2=44, c1=4, c2=2, layers=1) if name == "L_dloop1_full" else s.build(dict(name="L_loop1", kind="L", loop=True, layers=1), V, cfg)
        m.head[-1] = nn.Linear(m.head[-1].in_features, NT)
    return m


@torch.no_grad()
def evaluate(model, E, B=64):
    model.eval(); order = sorted(range(len(E)), key=lambda i: len(E[i][0])); ok = n = ok_o = n_o = 0
    for i in range(0, len(order), B):
        X, Y, O = pad([E[j] for j in order[i:i + B]]); pr = model(X).argmax(-1); m = Y != -100
        ok += ((pr == Y) & m).sum().item(); n += m.sum().item(); mo = m & O; ok_o += ((pr == Y) & mo).sum().item(); n_o += mo.sum().item()
    model.train(); return dict(acc=ok / n, acc_oov=ok_o / max(1, n_o), n=n, n_oov=n_o)


def baseline():
    """Etiquette la plus frequente par mot (casse exacte) ; mots inconnus : etiquette la plus frequente des mots rares du train."""
    by = collections.defaultdict(collections.Counter)
    for sn in TR:
        for w, t in sn: by[w][t] += 1
    rare = collections.Counter(t for sn in TR for w, t in sn if CNT[w] == 1).most_common(1)[0][0]
    def ev(S):
        ok = n = ok_o = n_o = 0
        for sn in S:
            for w, t in sn:
                p = by[w].most_common(1)[0][0] if w in by else rare; ok += p == t; n += 1
                if w not in VOC: ok_o += p == t; n_o += 1
        return dict(acc=ok / n, acc_oov=ok_o / max(1, n_o), n=n, n_oov=n_o)
    return dict(dev=ev(DEV), test=ev(TE))


if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    path = os.path.join(ROOT, "results", "ud_%s_s%d.json" % (name, seed))
    if os.path.exists(path): sys.exit()
    if name == "baseline":
        r = baseline(); json.dump(dict(model=name, **r), open(path, "w"), indent=1); print(r); sys.exit()
    torch.manual_seed(seed); model = build(name); rng0 = np.random.default_rng(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time()
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step); idx = rng.integers(0, len(ETR), ST["batch"])
        X, Y, _ = pad([ETR[i] for i in idx]); X = torch.where((torch.from_numpy(rng.random(X.shape)) < DROP) & (X > 1), torch.full_like(X, UNK), X)
        loss = F.cross_entropy(model(X).reshape(-1, NT), Y.reshape(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 250 == 0:
            e = evaluate(model, EDEV[:300]); print("[UD %s s%d] pas %d/%d perte %.3f acc(dev 300 phrases) %.3f (%.0fs)" % (name, seed, step, steps, loss.item(), e["acc"], time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0), dev=evaluate(model, EDEV), test=evaluate(model, ETE))
    json.dump(res, open(path, "w"), indent=1); print(res["dev"], res["test"], flush=True)
