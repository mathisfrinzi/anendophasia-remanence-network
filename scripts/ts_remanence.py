"""Amendement 36 : TinyStories, remanence (L_loop1) contre T_rope, sur XPU / CUDA / CPU. Meme nombre de pas pour les deux modeles, determine par une mesure de temps de la remanence.
  python scripts/ts_remanence.py --minutes 30 --batch 256          (telecharge ~80 Mo de TinyStories, entraine un BPE 4096, entraine les deux modeles, evalue, genere)
Sorties : results/ts_<modele>_s<graine>.json, results/ts_samples_<modele>_s<graine>.txt"""
import sys, os, json, time, math, argparse, urllib.request
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import faulthandler; faulthandler.enable()      # trace Python meme en cas de plantage natif (pilote)
import numpy as np, torch, torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import s3_tasks as s
from s8_rapide import FastLoop
_nt = torch.get_num_threads()
from rnn_taches import RNN                  # (rnn_taches fixe torch.set_num_threads(1) a l'import : on le retablit)
torch.set_num_threads(_nt)
from s10_hybride import TwinLM, HybridLM
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")
URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-%s.txt"
EOT = "<|endoftext|>"; LN2 = math.log(2)
SPECS = dict(L_loop1=dict(name="L_loop1", kind="L", loop=True, layers=1), T_rope=dict(name="T_rope", kind="T", pos="rope"))
ST = dict(s.ST_LOOP, d_model=64, n_layer=2, d_tok=64, spatial_ch=8, warmup=100, lr_min_frac=0.1)


def pick(arg):
    if arg != "auto": return torch.device(arg)
    if hasattr(torch, "xpu") and torch.xpu.is_available(): return torch.device("xpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync(dev):
    if dev.type == "xpu": torch.xpu.synchronize()
    elif dev.type == "cuda": torch.cuda.synchronize()


def fetch(split, mb, path):
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    req = urllib.request.Request(URL % split, headers={"Range": "bytes=0-%d" % (mb * 1_000_000 - 1)})
    try:
        txt = urllib.request.urlopen(req, timeout=120).read().decode("utf-8", errors="ignore")
    except Exception as e:                                                  # repli : bibliotheque datasets
        print("telechargement direct impossible (%s) -> datasets" % e, flush=True)
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories", split="train" if split == "train" else "validation")
        out, n = [], 0
        for r in ds:
            out.append(r["text"]); n += len(r["text"])
            if n > mb * 1_000_000: break
        txt = EOT.join(out) + EOT
    txt = txt[:txt.rindex(EOT) + len(EOT)] if EOT in txt else txt          # coupe a la fin d'une histoire entiere
    open(path, "w", encoding="utf-8").write(txt)


def fetch_hf(dataset, mb, path_tr, path_va):
    """Wikipedia simple (wikimedia/wikipedia, 20231101.simple) ou PG-19 (deepmind/pg19), en flux, jusqu'a mb Mo de train et 3 Mo de validation."""
    def take(ds, limit):
        out, n = [], 0
        for r in ds:
            out.append(r["text"]); n += len(r["text"])
            if n >= limit: break
        return EOT.join(out) + EOT
    os.makedirs(os.path.dirname(path_tr), exist_ok=True)
    if dataset == "wikitext2":                                              # petit corpus : pas de limite de taille (mb ignore) ; lignes brutes concatenees
        try:                                                                # le nom court "wikitext" n'est plus accepte par huggingface_hub recent
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
            tr = "".join(r["text"] for r in ds["train"]) + EOT; va = "".join(r["text"] for r in ds["validation"]) + EOT
        except Exception as e:                                              # repli : copie GitHub (pytorch/examples, meme texte, version a <unk>)
            print("HF indisponible (%s) -> copie GitHub" % str(e)[:80], flush=True)
            import urllib.request
            base = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"
            tr = urllib.request.urlopen(base + "train.txt", timeout=120).read().decode("utf-8") + EOT; va = urllib.request.urlopen(base + "valid.txt", timeout=120).read().decode("utf-8") + EOT
        open(path_tr, "w", encoding="utf-8").write(tr); open(path_va, "w", encoding="utf-8").write(va); return
    from datasets import load_dataset
    if dataset == "wikipedia":
        ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train", streaming=True)
        txt = take(ds, (mb + 3) * 1_000_000); k = txt.index(EOT, int(len(txt) * mb / (mb + 3))) + len(EOT); tr, va = txt[:k], txt[k:]
    else:
        tr = take(load_dataset("deepmind/pg19", split="train", streaming=True), mb * 1_000_000); va = take(load_dataset("deepmind/pg19", split="validation", streaming=True), 3_000_000)
    open(path_tr, "w", encoding="utf-8").write(tr); open(path_va, "w", encoding="utf-8").write(va)


def data(a):
    d = os.path.join(ROOT, "data", a.dataset); sfx = "_%dmb" % a.data_mb if a.data_mb < 10 and a.dataset != "wikitext2" else ""      # petits essais : fichiers a part (ne pas polluer le cache)
    tr_p, va_p = os.path.join(d, "train%s.txt" % sfx), os.path.join(d, "val%s.txt" % sfx)
    if a.txt: tr_p = va_p = a.txt
    elif a.dataset == "tinystories": fetch("train", a.data_mb, tr_p); fetch("valid", 5, va_p)
    elif not os.path.exists(tr_p): fetch_hf(a.dataset, a.data_mb, tr_p, va_p)
    tr, va = open(tr_p, encoding="utf-8").read(), open(va_p, encoding="utf-8").read()
    if a.txt: k = int(len(tr) * 0.9); tr, va = tr[:k], tr[k:]
    tp = os.path.join(ROOT, "tokenizer", "bpe_%s%s_4096.json" % (a.dataset, sfx))
    if not os.path.exists(tp):
        tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
        tok.train_from_iterator([tr[:30_000_000]], trainers.BpeTrainer(vocab_size=4096, special_tokens=[EOT], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)); tok.save(tp)
    tok = Tokenizer.from_file(tp)
    enc = lambda t: np.array([i for ch in range(0, len(t), 2_000_000) for i in tok.encode(t[ch:ch + 2_000_000]).ids], dtype=np.int64)
    ci = os.path.join(d, "ids_%d.npz" % a.data_mb)
    if os.path.exists(ci) and not a.txt: z = np.load(ci); TR, VA = z["tr"], z["va"]
    else:
        TR, VA = enc(tr), enc(va)
        if not a.txt: np.savez(ci, tr=TR, va=VA)
    print("tokens : train %d, val %d, vocabulaire %d" % (len(TR), len(VA), tok.get_vocab_size()), flush=True); return tok, TR, VA


def windows(ids, rng, B, n):
    off = rng.integers(0, len(ids) - n - 1, size=B)
    return torch.from_numpy(np.stack([ids[o:o + n + 1] for o in off]))


@torch.no_grad()
def evaluate(model, VA, V, CH, dev, n=256, B=96, edges=(0, 16, 32, 64, 128, 256, 512, 1024, 2048)):
    model.eval(); w = windows(VA, np.random.default_rng(9999), B, n); x, y = w[:, :-1].to(dev), w[:, 1:]
    nll = F.cross_entropy(model(x).reshape(-1, V).float().cpu(), y.reshape(-1), reduction="none").reshape(y.shape).numpy(); ch = CH[y.numpy()]; out = {}
    for a, b in zip(edges[:-1], edges[1:]):
        if a < n: out["%d-%d" % (a, min(b, n))] = float(nll[:, a:b].sum() / LN2 / ch[:, a:b].sum())
    model.train(); return out


@torch.no_grad()
def generate(model, tok, dev, prompt, n_new, seed, temp=0.8, topk=40):
    model.eval(); g = torch.Generator().manual_seed(seed); ids = tok.encode(prompt).ids
    for _ in range(n_new):
        lg = model(torch.tensor([ids[-128:]], device=dev))[0, -1].float().cpu() / temp
        v, ix = lg.topk(topk); ids.append(int(ix[torch.multinomial(F.softmax(v, -1), 1, generator=g)]))
    model.train(); return tok.decode(ids)


def a_batch(a): return a.batch


def PFX(a): return "ts" if a.dataset == "tinystories" else a.dataset


def step_fn(model, opt, TR, rng, a, dev, V, lr):
    for gp in opt.param_groups: gp["lr"] = lr * gp.get("scale", 1.0)
    w = windows(TR, rng, a.batch, a.window).to(dev)
    loss = F.cross_entropy(model(w[:, :-1]).reshape(-1, V).float(), w[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), ST["grad_clip"]); opt.step(); return loss


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=30); ap.add_argument("--batch", type=int, default=256); ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto"); ap.add_argument("--data-mb", type=int, default=80); ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--txt", default=""); ap.add_argument("--dataset", default="tinystories", choices=["tinystories", "wikipedia", "pg19", "wikitext2"]); ap.add_argument("--tag", default=""); ap.add_argument("--fast", action="store_true", help="version rapide de la remanence (memes calculs, moins d'operations)"); ap.add_argument("--backbone", default="", help="nom timm d'un ViT pre-entraine (ex. vit_small_patch16_224) : T1 et T2 en sont les blocs"); ap.add_argument("--pre-mode", default="ln", choices=["ln", "frozen", "all"]); ap.add_argument("--no-pretrained", action="store_true"); ap.add_argument("--compile", action="store_true", help="torch.compile sur la version rapide"); ap.add_argument("--eval-len", type=int, default=256); ap.add_argument("--no-gen", action="store_true"); ap.add_argument("--models", nargs="*", default=["L_loop1", "T_rope"])
    ap.add_argument("--lr", type=float, default=0); a = ap.parse_args(); dev = pick(a.device)
    if a.lr: ST["lr"] = a.lr
    print("appareil :", dev, "| torch", torch.__version__, flush=True)
    tok, TR, VA = data(a); V = tok.get_vocab_size(); CH = np.array([max(1, len(tok.decode([i]))) for i in range(V)], dtype=np.float64)
    def make(name):
        torch.manual_seed(a.seed); cfg = s.model_cfg(ST)
        r = re.fullmatch(r"R_(\d+)x(\d+)x(\d+)", name)
        if r:                                                # remanence a images C x H x W (patch 2), toujours en version rapide (memes calculs)
            c, h, w = map(int, r.groups()); import s8_rapide
            if a.compile: s8_rapide._t = torch.compile(s8_rapide._t)
            return FastLoop(s.build(dict(name=name, kind="L", loop=True, layers=1, c=c, h=h, w=w), V, cfg)).to(dev)
        r = re.fullmatch(r"T_(\d+)x(\d+)", name)           # T_rope de largeur d et L couches (courbe performance/parametres)
        if r:
            d_, L_ = map(int, r.groups()); return s.build(SPECS["T_rope"], V, dict(cfg, d_model=d_, n_layer=L_, n_head=max(1, d_ // 32), d_tok=d_)).to(dev)
        if name.startswith("GRU_"): return RNN("GRU", V, h=int(name[4:])).to(dev)
        if name.startswith("Twin_"): return TwinLM(V, cfg, int(name[5:])).to(dev)
        if name.startswith("H_"): return HybridLM(V, cfg, int(name[2:])).to(dev)
        m = s.build(SPECS[name], V, cfg).to(dev)
        if a.backbone and name == "L_loop1":
            from s9_preentraine import PreLoop
            return PreLoop(m, a.backbone, a.pre_mode, pretrained=not a.no_pretrained).to(dev)
        if a.fast and name == "L_loop1":
            import s8_rapide
            if a.compile: s8_rapide._t = torch.compile(s8_rapide._t)
            m = FastLoop(m)
        return m
    steps = a.steps                                          # mesure de temps sur la remanence (le modele le plus lent) ; batch reduit de moitie si memoire insuffisante
    while not steps:
        try:
            m = make("L_loop1"); o = torch.optim.AdamW(m.parameters(), lr=ST["lr"]); rng = np.random.default_rng(0)
            for i in range(8):
                if i == 3: sync(dev); t0 = time.time()
                step_fn(m, o, TR, rng, a, dev, V, ST["lr"])
            sync(dev); dt = (time.time() - t0) / 5; steps = max(100, int(a.minutes * 60 / dt / 50) * 50)
            print("mesure : %.2f s/pas (batch %d, fenetre %d) -> %d pas pour %.0f min" % (dt, a.batch, a.window, steps, a.minutes), flush=True); del m, o
        except RuntimeError as e:
            if "memory" in str(e).lower() and a.batch > 16: a.batch //= 2; print("memoire insuffisante -> batch", a.batch, flush=True)
            else: raise
    for name in a.models:
        model = make(name)
        if a.backbone and name == "L_loop1":
            from s9_preentraine import param_groups
            opt = torch.optim.AdamW(param_groups(model, ST["lr"]), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"])
        else: opt = torch.optim.AdamW(model.parameters(), lr=ST["lr"], betas=(0.9, 0.95), weight_decay=ST["wd"])
        t0 = time.time(); curve = []
        print("debut de l'entrainement de %s : %d pas, batch %d, %d parametres" % (name, steps, a_batch(a), sum(p.numel() for p in model.parameters())), flush=True)
        for step in range(1, steps + 1):
            loss = step_fn(model, opt, TR, np.random.default_rng(a.seed * 1_000_003 + step), a, dev, V, s.lr_at(step, steps, ST))
            if a.backbone and step % 500 == 0 and step < steps:       # sauvegarde partielle (run long, risque de plantage du pilote)
                part = dict(model=name, step=step, steps=steps, bpc=evaluate(model, VA, V, CH, dev, n=a.eval_len, B=max(8, 24576 // a.eval_len)))
                json.dump(part, open(os.path.join(ROOT, "results", "%s_%s_s%d%s_partiel.json" % (PFX(a), name, a.seed, a.tag)), "w"), indent=1); print("PARTIEL", json.dumps(part), flush=True)
            if step in (1, 2, 5, 20) or step % 25 == 0: curve.append((step, round(loss.item(), 4), round(time.time() - t0)))
            if step in (1, 2, 5, 20) or step % 25 == 0: print("[%s] pas %d/%d perte %.3f (%.0fs)" % (name, step, steps, loss.item(), time.time() - t0), flush=True)
        res = dict(curve=curve, dataset=a.dataset, tag=a.tag, eval_len=a.eval_len, model=name, seed=a.seed, steps=steps, batch=a.batch, window=a.window, device=str(dev), n_params=sum(p.numel() for p in model.parameters()), n_train=sum(p.numel() for p in model.parameters() if p.requires_grad), backbone=a.backbone, pre_mode=a.pre_mode, seconds=round(time.time() - t0), bpc=evaluate(model, VA, V, CH, dev, n=a.eval_len, B=max(8, 24576 // a.eval_len)))
        json.dump(res, open(os.path.join(ROOT, "results", "%s_%s_s%d%s.json" % (PFX(a), name, a.seed, a.tag)), "w"), indent=1); print("RESULT", json.dumps(res), flush=True)
        if not a.no_gen:
            txt = "\n\n".join(generate(model, tok, dev, p, 100, a.seed + i) for i, p in enumerate(["Once upon a time", "One day, a little girl named Lily"]))
            open(os.path.join(ROOT, "results", "%s_samples_%s_s%d%s.txt" % (PFX(a), name, a.seed, a.tag)), "w", encoding="utf-8").write(txt); print("--- %s ---\n%s" % (name, txt), flush=True)
