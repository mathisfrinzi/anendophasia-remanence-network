"""Entrainement et evaluation de l'autoencodeur texte <-> image (voir PROTOCOLE.md).

Usage Colab : ae_run.run_all(root) ; local (test rapide) : python src/ae_run.py --quick
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import data as D
from autoencoder import TextImageAE


def chunk_batch(rng, ids, B, L, device):
    off = rng.integers(0, len(ids) - L - 1, size=B)
    x = np.stack([ids[o : o + L] for o in off]).astype(np.int64)
    return torch.from_numpy(x).to(device)


@torch.no_grad()
def evaluate(model, val_ids, device, cfg, by_pos=False):
    model.eval()
    rng = np.random.default_rng(cfg["eval_seed"])
    n = loss = right = exact = 0
    pos_right = torch.zeros(model.L, device=device)
    for _ in range(cfg["eval_batches"]):
        x = chunk_batch(rng, val_ids, cfg["batch_size"], model.L, device)
        logits = model(x)
        loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1), reduction="sum").item()
        ok = logits.argmax(-1) == x
        right += ok.sum().item()
        exact += ok.all(dim=1).sum().item()
        pos_right += ok.float().sum(dim=0)
        n += x.shape[0]
    model.train()
    chars = n * model.L
    out = dict(val_loss=loss / chars, char_acc=right / chars, exact_acc=exact / n)
    if by_pos:  # profil d'oubli : precision selon la position dans le morceau (la derniere est la plus recente)
        out["acc_by_pos"] = [round(v, 4) for v in (pos_right / n).tolist()]
    return out


@torch.no_grad()
def continuity(model, val_ids, device, cfg):
    """Bras cumul : indice de continuite = pas moyen entre deux frames consecutives
    / distance moyenne entre frames de textes differents au meme instant.
    < 1 : les frames voisines se ressemblent plus que des frames sans rapport (video continue) ;
    ~ 1 : les frames sautent comme des images independantes."""
    model.eval()
    rng = np.random.default_rng(cfg["eval_seed"] + 1)
    x = chunk_batch(rng, val_ids, 64, model.L, device)
    fr = model.encode_frames(x).flatten(2)             # (B, L+1, d)
    step = (fr[:, 1:] - fr[:, :-1]).norm(dim=-1).mean().item()
    other = (fr[1:, 1:] - fr[:-1, 1:]).norm(dim=-1).mean().item()
    model.train()
    return dict(step=step, between_texts=other, index=step / other)


@torch.no_grad()
def samples(model, val_ids, vocab, device, n=8):
    model.eval()
    rng = np.random.default_rng(777)
    x = chunk_batch(rng, val_ids, n, model.L, device)
    pred = model(x).argmax(-1)
    out = []
    for a, b in zip(x.cpu().numpy(), pred.cpu().numpy()):
        mark = "".join(" " if i == j else "^" for i, j in zip(a, b))
        out.append("orig : %s\nrecon: %s\n      %s" % (D.decode(vocab, a).replace("\n", "~"), D.decode(vocab, b).replace("\n", "~"), mark))
    model.train()
    return "\n\n".join(out)


def save_latents_png(model, val_ids, device, path, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(777)
    x = chunk_batch(rng, val_ids, n, model.L, device)
    model.eval()
    with torch.no_grad():
        z = model.latent_images(x).cpu().numpy()
    model.train()
    fig, axs = plt.subplots(1, n, figsize=(1.6 * n, 1.8))
    for a, img in zip(axs, z):
        a.imshow(np.clip((img.transpose(1, 2, 0) + 2) / 4, 0, 1), interpolation="nearest")
        a.axis("off")
    fig.suptitle("latent image (3x8x8) de 8 morceaux de %d caracteres" % model.L, fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_frames_png(model, val_ids, vocab, device, path, cols=16):
    """Le film d'un morceau (bras cumul) : la frame apres chaque caractere lu."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(777)
    x = chunk_batch(rng, val_ids, 1, model.L, device)
    model.eval()
    with torch.no_grad():
        fr = model.encode_frames(x)[0].cpu().numpy()      # (L+1, C, H, W)
    model.train()
    chars = [""] + [D.decode(vocab, [i]).replace("\n", "~").replace(" ", "_") for i in x[0].cpu().numpy()]
    n = len(fr)
    rows = (n + cols - 1) // cols
    lim = max(1e-6, float(np.abs(fr).max()))
    fig, axs = plt.subplots(rows, cols, figsize=(1.0 * cols, 1.15 * rows), squeeze=False)
    for i, a in enumerate(axs.flat):
        a.axis("off")
        if i < n:
            a.imshow(np.clip(fr[i].transpose(1, 2, 0) / (2 * lim) + 0.5, 0, 1), interpolation="nearest")
            a.set_title(chars[i] if i else "debut", fontsize=7)
    fig.suptitle("frames successives (echelle commune) apres chaque caractere lu", fontsize=8)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run_one(kind, L, train_ids, val_ids, vocab, device, root=".", cfg=None, steps=None, log=print, tag=""):
    """Entraine un (bras, taille de morceau) ; ecrit results/ae_KIND_LL.json (+ echantillons, PNG).
    Si le JSON final existe deja, la mesure est reprise telle quelle (pas de relance)."""
    cfg = cfg or config.AE
    steps = steps or cfg["steps"]
    res_dir = os.path.join(root, "results")
    os.makedirs(res_dir, exist_ok=True)
    base = os.path.join(res_dir, "ae_%s_L%d%s" % (kind, L, ("_" + tag) if tag else ""))
    if os.path.exists(base + ".json"):
        log("[%s L=%d %s] deja mesure, saute." % (kind, L, tag))
        return json.load(open(base + ".json"))
    torch.manual_seed(cfg["seed"])
    model = TextImageAE(len(vocab), L, kind, cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.999))
    rng = np.random.default_rng(cfg["seed"])
    curve, t0 = [], time.time()
    for step in range(1, steps + 1):
        lr = cfg["lr"] * min(1.0, step / cfg["warmup"])
        for g in opt.param_groups:
            g["lr"] = lr
        x = chunk_batch(rng, train_ids, cfg["batch_size"], L, device)
        lam = cfg.get("step_penalty", 0.0) if kind == "cumul" else 0.0
        if lam > 0:
            logits, frames = model(x, return_frames=True)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1)) + lam * model.continuity_index(frames)
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        if step % cfg["eval_every"] == 0 or step == steps:
            m = evaluate(model, val_ids, device, cfg)
            m.update(step=step, train_loss=loss.item())
            curve.append(m)
            log("[%s L=%d] pas %d/%d  train %.3f  val %.3f  car %.1f%%  morceau exact %.1f%%  (%.0fs)" % (
                kind, L, step, steps, loss.item(), m["val_loss"], 100 * m["char_acc"], 100 * m["exact_acc"], time.time() - t0))
    final = dict(curve[-1], **evaluate(model, val_ids, device, cfg, by_pos=True))
    cont = continuity(model, val_ids, device, cfg) if kind == "cumul" else None
    if cont:
        log("[%s L=%d] indice de continuite %.3f (pas %.3f / entre textes %.3f)" % (kind, L, cont["index"], cont["step"], cont["between_texts"]))
    out = dict(arm=kind, chunk=L, tag=tag, latent=[cfg["latent_c"], cfg["latent_h"], cfg["latent_w"]], noise_std=cfg["noise_std"], step_penalty=cfg.get("step_penalty", 0.0),
               n_params=n_params, steps=steps, seconds=round(time.time() - t0, 1),
               device=str(device), success=bool(final["char_acc"] >= cfg["success_char_acc"]),
               final=final, continuity=cont, curve=curve)
    with open(base + "_samples.txt", "w", encoding="utf-8") as f:
        f.write(samples(model, val_ids, vocab, device))
    if kind in ("image", "cumul"):
        save_latents_png(model, val_ids, device, base + "_latents.png")
    if kind == "cumul":
        save_frames_png(model, val_ids, vocab, device, base + "_frames.png")
    if cfg.get("save_model"):
        torch.save(dict(state=model.state_dict(), kind=kind, L=L, cfg=dict(cfg), vocab=vocab), base + "_model.pt")
    with open(base + ".json", "w") as f:
        json.dump(out, f, indent=1)
    return out


def summarize(root=".", cfg=None):
    """Tableau recapitulatif (results/ae_summary.md) : precision par caractere selon (bras, taille de morceau),
    et le plus grand morceau reussi (>= success_char_acc) par bras."""
    cfg = cfg or config.AE
    rows, best = {}, {}
    for kind in cfg["arms"]:
        for L in cfg["chunk_sizes"]:
            p = os.path.join(root, "results", "ae_%s_L%d.json" % (kind, L))
            if os.path.exists(p):
                r = json.load(open(p))
                rows[(kind, L)] = r
                if r["success"]:
                    best[kind] = max(best.get(kind, 0), L)
    lines = ["# Autoencodeur texte <-> image : resultats", "",
             "Critere de reussite pre-enregistre : >= %.0f %% des caracteres relus correctement (validation)." % (100 * cfg["success_char_acc"]), "",
             "| bras | morceau (car.) | params | car. corrects | morceaux exacts | perte (nats/car) | continuite | reussi |", "|:--|--:|--:|--:|--:|--:|--:|:-:|"]
    for (kind, L), r in sorted(rows.items(), key=lambda kv: (cfg["arms"].index(kv[0][0]), kv[0][1])):
        f = r["final"]
        ci = ("%.2f" % r["continuity"]["index"]) if r.get("continuity") else "-"
        lines.append("| %s | %d | %d | %.1f %% | %.1f %% | %.3f | %s | %s |" % (
            kind, L, r["n_params"], 100 * f["char_acc"], 100 * f["exact_acc"], f["val_loss"], ci, "oui" if r["success"] else "non"))
    lines += ["", "Continuite (bras cumul) : pas entre frames consecutives / distance entre frames de textes differents ; < 1 = video continue.", ""]
    for (kind, L), r in sorted(rows.items(), key=lambda kv: (cfg["arms"].index(kv[0][0]), kv[0][1])):
        ap = r["final"].get("acc_by_pos")
        if ap:  # profil d'oubli : precision au debut, au milieu et a la fin du morceau
            lines.append("- %s L=%d, precision par position (debut / milieu / fin du morceau) : %.0f %% / %.0f %% / %.0f %%" % (
                kind, L, 100 * ap[0], 100 * ap[len(ap) // 2], 100 * ap[-1]))
    lines += ["", "Plus grand morceau reussi par bras : " + (", ".join("%s = %d" % kv for kv in sorted(best.items())) or "aucun")]
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(root, "results", "ae_summary.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def summarize_complement(root="."):
    """Tableau des mesures supplementaires (runs avec tag) : results/ae_complement.md."""
    import glob
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "results", "ae_*_L*_*.json"))):
        r = json.load(open(p))
        if r.get("tag"):
            rows.append(r)
    lines = ["# Mesures supplementaires (hors protocole principal)", "",
             "| tag | bras | morceau | pas | latent | bruit | penalite | params | car. corrects | morceaux exacts | continuite | debut / milieu / fin |",
             "|:--|:--|--:|--:|:--|--:|--:|--:|--:|--:|--:|:--|"]
    for r in sorted(rows, key=lambda r: (r["tag"], r["chunk"])):
        f = r["final"]
        ap = f.get("acc_by_pos") or [0]
        ci = ("%.2f" % r["continuity"]["index"]) if r.get("continuity") else "-"
        lines.append("| %s | %s | %d | %d | %s | %.2f | %.1f | %d | %.1f %% | %.1f %% | %s | %.0f / %.0f / %.0f %% |" % (
            r["tag"], r["arm"], r["chunk"], r["steps"], "x".join(map(str, r["latent"])), r["noise_std"], r.get("step_penalty", 0.0), r["n_params"],
            100 * f["char_acc"], 100 * f["exact_acc"], ci, 100 * ap[0], 100 * ap[len(ap) // 2], 100 * ap[-1]))
    txt = "\n".join(lines) + "\n"
    with open(os.path.join(root, "results", "ae_complement.md"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def run_complement(root=".", on_done=None, log=print):
    """Mesures supplementaires (config.AE_COMPLEMENT), chacune etiquetee d'un tag ; elles ne
    remplacent pas les verdicts a 8 000 pas du protocole principal."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = config.AE
    vocab, train_ids, val_ids = D.prepare(root, base["n_train_stories"], base["n_val_stories"])
    log("device %s | vocab %d | train %d car. | val %d car." % (device, len(vocab), len(train_ids), len(val_ids)))
    for job in config.AE_COMPLEMENT:
        cfg = dict(base, **job["over"])
        run_one(job["kind"], job["L"], train_ids, val_ids, vocab, device, root, cfg, job["steps"], log, tag=job["tag"])
        summarize_complement(root)
        if job.get("analyze"):
            import ae_analysis
            log("[analyse %s] ablation par zone, cartes d'ecriture..." % job["tag"])
            ae_analysis.analyze(root, job["tag"], job["kind"], job["L"], val_ids, device)
            ae_analysis.summarize(root)
        if on_done:
            on_done()
    txt = summarize_complement(root)
    log(txt)
    if any(j.get("analyze") for j in config.AE_COMPLEMENT):
        import ae_analysis
        log(ae_analysis.summarize(root))
    return txt


def run_all(root=".", chunk_sizes=None, arms=None, steps=None, cfg=None, on_done=None, log=print):
    cfg = cfg or config.AE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, train_ids, val_ids = D.prepare(root, cfg["n_train_stories"], cfg["n_val_stories"])
    log("device %s | vocab %d | train %d car. | val %d car." % (device, len(vocab), len(train_ids), len(val_ids)))
    for L in (chunk_sizes or cfg["chunk_sizes"]):
        for kind in (arms or cfg["arms"]):
            run_one(kind, L, train_ids, val_ids, vocab, device, root, cfg, steps, log)
            summarize(root, cfg)
            if on_done:
                on_done()
    txt = summarize(root, cfg)
    log(txt)
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--quick", action="store_true", help="test rapide : 40 pas, morceaux 8 et 16, textes synthetiques si pas de reseau")
    args = ap.parse_args()
    if args.quick:
        cfg = dict(config.AE, eval_every=20, eval_batches=2, batch_size=16, n_train_stories=200, n_val_stories=50)
        run_all(args.root, chunk_sizes=[8, 16], steps=40, cfg=cfg)
    else:
        run_all(args.root)
