"""Rappel de listes (PROTOCOLE.md, amendement 8) : liste de n tokens aleatoires [+ d tokens de distraction] puis la meme liste.

Usage : python src/s2_recall.py --root <depot> [--models T_rope V_lam0 ...]
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from s2_models import build

LOW, HIGH = 300, 2000          # tokens frequents du BPE (pas de vrais mots : pas de tokenizer ici)


@torch.no_grad()
def run(model, n, d, B, seed, device):
    rng = np.random.default_rng(seed + 1000 * n + d)
    lst = rng.integers(LOW, HIGH, (B, n))
    dis = rng.integers(LOW, HIGH, (B, d))
    seq = torch.from_numpy(np.concatenate([lst, dis, lst], axis=1)).long().to(device)
    logits = model(seq[:, :-1])
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1), reduction="none").view(B, -1)
    acc = (logits.argmax(-1) == seq[:, 1:]).float()
    a, b = n + d - 1, n + d + n - 1                  # cibles de la seconde copie
    return dict(first_nll=float(nll[:, : n - 1].mean()), second_nll=float(nll[:, a:b].mean()), second_acc=float(acc[:, a:b].mean()),
                first_acc=float(acc[:, : n - 1].mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--models", nargs="*", default=["T_rope", "T_nope", "V_lam0", "V1_lam0"])
    ap.add_argument("--B", type=int, default=256)
    a = ap.parse_args()
    device = torch.device("cpu")
    out = {}
    for name in a.models:
        ck = torch.load(os.path.join(a.root, "results", "s2_%s_model.pt" % name), map_location=device, weights_only=False)
        m = build(ck["spec"], ck["V"], ck["cfg"]).to(device)
        m.load_state_dict(ck["state"])
        m.eval()
        out[name] = {}
        for d in (0, 64):
            for n in (8, 16, 32, 64):
                out[name]["n%d_d%d" % (n, d)] = run(m, n, d, a.B, 42, device)
        print(name, "ok", flush=True)
    json.dump(out, open(os.path.join(a.root, "results", "s2_recall.json"), "w"), indent=1)
    lines = ["# Rappel de listes (tokens aleatoires, seconde copie)", "",
             "Perte en nats par token (plus bas = meilleur rappel) ; entre parentheses la precision top-1. Un tirage uniforme sur les %d tokens vaudrait %.2f nats, mais ces modeles, entraines sur des histoires, jugent ces tokens aleatoires bien plus improbables que cela : un resultat proche de la premiere copie signifie « aucun rappel »." % (HIGH - LOW, math.log(HIGH - LOW)), ""]
    for d in (0, 64):
        lines += ["Distraction d = %d tokens" % d, "", "| modele | " + " | ".join("n = %d" % n for n in (8, 16, 32, 64)) + " |", "|:--|" + "--:|" * 4]
        for name in a.models:
            lines.append("| %s | %s |" % (name, " | ".join("%.2f (%.0f %%)" % (out[name]["n%d_d%d" % (n, d)]["second_nll"], 100 * out[name]["n%d_d%d" % (n, d)]["second_acc"]) for n in (8, 16, 32, 64))))
        lines.append("")
    lines += ["Premiere copie (temoin : aucun rappel possible) : " + ", ".join("%s %.2f" % (nm, np.mean([v["first_nll"] for v in out[nm].values()])) for nm in a.models) + " nats."]
    open(os.path.join(a.root, "results", "s2_recall.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
