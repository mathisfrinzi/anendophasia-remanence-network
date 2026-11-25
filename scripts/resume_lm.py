"""Tableau recapitulatif des LM CPU : python scripts/resume_lm.py [corpus ...]"""
import json, os, sys, glob
R = os.path.join(os.path.dirname(__file__), "..", "results")
for c in (sys.argv[1:] or ["wikitext2", "code_python", "gutenberg_contes", "ptb", "gutenberg_tout", "shakespeare"]):
    pre = "lm_wiki_" if c == "wikitext2" else "lm_%s_" % c; row = {}
    for m in ["unigramme", "T_rope", "V_lam0", "L_loop1"]:
        f = os.path.join(R, "%s%s_s0.json" % (pre, m))
        if os.path.exists(f): row[m] = json.load(open(f))["final"]["256"]
    if not row: continue
    print(c); print("  %-10s %8s %8s %8s" % ("", "0-64", "64-128", "128-256"))
    for m, v in row.items(): print("  %-10s %8.3f %8.3f %8.3f" % (m, v["0-64"], v["64-128"], v["128-256"]))
    if "unigramme" in row and "T_rope" in row:
        u, t = row["unigramme"]["0-64"], row["T_rope"]["0-64"]
        for m in ("V_lam0", "L_loop1"):
            if m in row:
                v = row[m]; print("  %s : (a) %s  (b) %s  (c) %s" % (m, "oui" if v["0-64"] <= 0.8 * u else "non", "oui" if v["0-64"] <= 1.05 * t else "non", "oui" if (v["128-256"] < row["T_rope"]["128-256"] and v["128-256"] <= 1.10 * v["0-64"]) else "non"))
