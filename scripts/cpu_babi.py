"""Amendement 24 : bAbI (taches 1, 2, 3, 6), entrainement joint.
Usage : python scripts/cpu_babi.py MODELE GRAINE [PAS] [--taches 1,2,3,6] [--eval_k_taches 1,2,3,6]
MODELE in {T_rope, V_lam0, L_loop1, L_noloop1, L_loop1_k}. Variable d'env BABI_OUT = dossier de sortie (defaut results/)."""
import sys, os, re, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, torch, torch.nn.functional as F
import s3_tasks as s

torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "babi")
FILES = {1: "qa1_single-supporting-fact", 2: "qa2_two-supporting-facts", 3: "qa3_three-supporting-facts", 6: "qa6_yes-no-questions"}
ALL_TASKS = [1, 2, 3, 6]
MAXLEN, K_TRAIN, KS_EVAL = 128, 8, (0, 2, 4, 8, 12)
SPECS = dict(T_rope=dict(name="T_rope", kind="T", pos="rope"), V_lam0=dict(name="V_lam0", kind="V", scales=2, penalty=0.0),
             L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), L_noloop1=dict(name="L_noloop1", kind="L", loop=False, layers=1),
             L_loop1_k=dict(name="L_loop1_k", kind="L", loop=True, layers=1))
ST = dict(s.ST_LOOP, batch=32, d_model=64, d_tok=32, warmup=100, lr_min_frac=0.1)
TOK = re.compile(r"\w+|[.?]")


def tokenize(t): return TOK.findall(t.lower())


def parse(path):
    """Retourne une liste de (tokens_histoire_plus_question, mot_reponse) ; histoire sans les questions precedentes."""
    out, story, multi = [], [], 0
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line.strip(): continue
        num, text = line.split(" ", 1)
        if int(num) == 1: story = []
        if "\t" in text:
            q, a, _ = text.split("\t")
            ans = tokenize(a)
            if len(ans) != 1: multi += 1
            out.append((sum(story, []) + tokenize(q), ans[0] if ans else ""))
        else: story.append(tokenize(text))
    return out, multi


def load(tasks=ALL_TASKS):
    raw = {t: {sp: parse(os.path.join(DATA, "%s_%s.txt" % (FILES[t], sp))) for sp in ("train", "test")} for t in tasks}
    return raw


class Data:
    def __init__(self, tasks):
        raw = load(ALL_TASKS)          # vocabulaire sur les 4 taches, quelle que soit l'option
        self.multi = sum(raw[t][sp][1] for t in raw for sp in raw[t])
        words = sorted({w for t in raw for sp in raw[t] for toks, a in raw[t][sp][0] for w in toks + [a]})
        self.itos = ["<PAD>", "<PAUSE>"] + words; self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.PAD, self.PAUSE, self.V = 0, 1, len(self.itos)
        ans = sorted({self.stoi[a] for t in raw for toks, a in raw[t]["train"][0]})
        self.ans_ids = torch.tensor(ans); self.a2i = {a: i for i, a in enumerate(ans)}
        self.tasks = tasks; self.d = {}
        for t in tasks:
            for sp in ("train", "test"):
                seqs = [[self.stoi[w] for w in toks][-MAXLEN:] for toks, a in raw[t][sp][0]]
                ys = [self.a2i.get(self.stoi[a], -1) for toks, a in raw[t][sp][0]]   # -1 : reponse jamais vue a l'entrainement (comptee fausse)
                self.d[(t, sp)] = (seqs, torch.tensor(ys))

    def make(self, t, sp, idx, k=0):
        seqs, ys = self.d[(t, sp)]; L = max(len(seqs[i]) for i in idx)
        x = torch.full((len(idx), L + k), self.PAUSE, dtype=torch.long); x[:, :L] = self.PAD
        for r, i in enumerate(idx): q = seqs[i]; x[r, L - len(q):L] = torch.tensor(q)
        return x, ys[list(idx)]


def logits(model, x, data):
    return model(x)[:, -1][:, data.ans_ids]


@torch.no_grad()
def evaluate(model, data, t, sp="test", n=None, k=0, B=250):
    model.eval(); N = len(data.d[(t, sp)][0]) if n is None else n; ok = 0
    for i in range(0, N, B):
        x, y = data.make(t, sp, list(range(i, min(N, i + B))), k)
        ok += (logits(model, x, data).argmax(-1) == y).sum().item()
    model.train(); return ok / N


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("seed", type=int); ap.add_argument("steps", type=int, nargs="?", default=4000)
    ap.add_argument("--taches", default="1,2,3,6"); ap.add_argument("--eval_k_taches", default=None)
    a = ap.parse_args(); name, seed, steps = a.model, a.seed, a.steps
    tasks = [int(t) for t in a.taches.split(",")]; ek = [int(t) for t in a.eval_k_taches.split(",")] if a.eval_k_taches else tasks
    suffix = "" if tasks == ALL_TASKS else "_t" + "".join(map(str, tasks))
    out = os.environ.get("BABI_OUT", os.path.join(ROOT, "results")); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "babi_%s_s%d%s.json" % (name, seed, suffix))
    if os.path.exists(path): sys.exit()
    data = Data(tasks)
    if data.multi: print("ATTENTION : %d reponses a plusieurs mots" % data.multi, flush=True)
    print("vocab %d, reponses possibles %d" % (data.V, len(data.ans_ids)), flush=True)
    torch.manual_seed(seed)
    cfg = s.model_cfg(dict(ST, n_layer=2) if name.startswith("T_") else ST)
    model = s.build(SPECS[name], data.V, cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"]); t0 = time.time(); curve = []
    for step in range(1, steps + 1):
        for gp in opt.param_groups: gp["lr"] = s.lr_at(step, steps, ST)
        rng = np.random.default_rng(seed * 1_000_003 + step)
        t = tasks[int(rng.integers(0, len(tasks)))]; k = int(rng.integers(0, K_TRAIN + 1)) if name == "L_loop1_k" else 0
        idx = rng.integers(0, len(data.d[(t, "train")][0]), ST["batch"]).tolist()
        x, y = data.make(t, "train", idx, k)
        loss = F.cross_entropy(logits(model, x, data), y)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step()
        if step % 500 == 0 or step == steps:
            c = dict(step=step, loss=round(loss.item(), 4), **{"acc_qa%d" % t: evaluate(model, data, t, n=200, k=(4 if name == "L_loop1_k" else 0)) for t in tasks})
            curve.append(c); print("[babi %s s%d] %s (%.0fs)" % (name, seed, c, time.time() - t0), flush=True)
        elif step in (10, 50) and os.environ.get("BABI_VERBOSE"): print("pas %d %.1fs" % (step, time.time() - t0), flush=True)
    res = dict(model=name, seed=seed, steps=steps, n_params=sum(p.numel() for p in model.parameters()), seconds=round(time.time() - t0),
               test={"qa%d" % t: evaluate(model, data, t) for t in tasks})
    if name == "L_loop1_k": res["test_k"] = {"qa%d" % t: {"k%d" % k: evaluate(model, data, t, k=k) for k in KS_EVAL} for t in ek}
    res["curve"] = curve
    json.dump(res, open(path, "w"), indent=1); print(res["test"], flush=True)
