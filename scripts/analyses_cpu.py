# Analyses sans entrainement sur les modeles sauvegardes (amendement 14). Depuis la racine : python scripts/analyses_cpu.py
import json, sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "src")
from tokenizers import Tokenizer
import s2_models as M
torch.manual_seed(0)
tok = Tokenizer.from_file("tokenizer/bpe_4096.json")
src = open("scripts/indice_continuite_textes.py").read().split("for name in")[0]
ns = {}; exec(src, ns)
ids = torch.tensor(ns["ids"])                                    # 5 textes de 100 tokens
chars = np.mean([len(tok.decode([i])) for i in ids.flatten().tolist()])
def load(n):
    ck = torch.load("results/s2_%s_model.pt" % n, map_location="cpu", weights_only=False)
    m = M.build(ck["spec"], ck["V"], ck["cfg"]); m.load_state_dict(ck["state"]); return m.eval()
def read(m, fr):                                                   # reprend VLM.forward apres la video
    B, N1 = fr.shape[:2]; N = N1 - 1
    f = m._norm(fr[:, 1:].reshape(B * N, m.C, m.h, m.w))
    return m.head(m.stack(m.to_d(m.reader(f).reshape(B * N, -1)).view(B, N, -1)))
def bpc(lg):
    nll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.size(-1)), ids[:, 1:].reshape(-1)).item()
    return nll / np.log(2) / chars
out = {}
with torch.no_grad():
    for name in ("V_lam0", "V_lam1_ext"):
        m = load(name); fr = m.frames(ids); base = bpc(read(m, fr)); r = {"base": base}
        # A. bruit a la lecture, relatif a l'ecart-type de chaque canal
        sd = fr[:, 1:].std(dim=(0, 1), keepdim=True)
        for s in (0.1, 0.3, 1.0):
            r["bruit_%.1f" % s] = float(np.mean([bpc(read(m, torch.cat([fr[:, :1], fr[:, 1:] + s * sd * torch.randn_like(fr[:, 1:])], 1))) for _ in range(5)]))
        # B. projection sur les k premieres directions principales, par groupe
        g = fr[:, 1:].reshape(-1, 2, m.c * m.h * m.w)
        for k in (1, 2, 4, 8, 16, 32):
            rec = []
            for gi in range(2):
                a = g[:, gi]; mu = a.mean(0); U, S, Vt = torch.linalg.svd(a - mu, full_matrices=False)
                rec.append(((a - mu) @ Vt[:k].T) @ Vt[:k] + mu)
            f2 = torch.stack(rec, 1).reshape(5, 100, m.C, m.h, m.w)
            r["pca_%d" % k] = bpc(read(m, torch.cat([fr[:, :1], f2], 1)))
        out[name] = r
        print(name, {k: round(v, 3) for k, v in r.items()})
# C. positions apprises des taches (D_blob_e1) : les tokens d'une meme classe sont-ils proches ?
m = load("D_blob_e1"); V = m.tab.num_embeddings
with torch.no_grad(): pos = m.positions(torch.arange(V)).reshape(V, -1).numpy()       # (V, 2C)
strs = [tok.decode([i]) for i in range(V)]
def cls(f): return np.array([i for i in range(V) if f(strs[i], i)])
groups = {
 "ponctuation": cls(lambda s, i: len(s.strip()) > 0 and not any(c.isalnum() for c in s)),
 "debut de mot (espace + lettres)": cls(lambda s, i: s.startswith(" ") and s[1:].isalpha()),
 "suffixe sans espace": cls(lambda s, i: s.isalpha() and i > 300),
 "majuscule": cls(lambda s, i: s.strip()[:1].isupper()),
 "nombres (one, two...)": cls(lambda s, i: s.strip().lower() in "one two three four five six seven eight nine ten".split()),
 "couleurs": cls(lambda s, i: s.strip().lower() in "red blue green yellow pink purple orange black white brown".split()),
 "animaux": cls(lambda s, i: s.strip().lower() in "dog cat bird fish rabbit bear mouse horse cow duck frog".split()),
 "famille": cls(lambda s, i: s.strip().lower() in "mom dad mother father sister brother friend friends grandma".split()),
}
rng = np.random.default_rng(0); res = {}
def meandist(ix): d = np.linalg.norm(pos[ix][:, None] - pos[ix][None], axis=-1); return d[np.triu_indices(len(ix), 1)].mean()
for k, ix in groups.items():
    if len(ix) < 3: continue
    obs = meandist(ix); null = np.array([meandist(rng.choice(V, len(ix), replace=False)) for _ in range(300)])
    res[k] = (len(ix), obs, null.mean(), float((null <= obs).mean())); print("%-34s n=%3d  dist moyenne %.2f  hasard %.2f  p(hasard <= obs) %.3f" % ((k,) + res[k]))
fq = np.arange(V); 
rho = [float(np.corrcoef(fq, pos[:, j])[0, 1]) for j in range(pos.shape[1])]
print("correlation identifiant (frequence BPE) - coordonnees : max |r| = %.2f" % np.abs(rho).max())
json.dump({"modeles": out, "classes": {k: list(map(float, v)) for k, v in res.items()}, "chars_par_token": float(chars)}, open("results/analyses_cpu.json", "w"), indent=1)
