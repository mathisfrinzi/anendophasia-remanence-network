"""Point d'entree unique des mesures longues (Colab T4 / Windows XPU). Reprise possible : un run dont le json existe deja est saute.
  python scripts/suite.py PLAN [--device auto|cuda|xpu|cpu] [--fast-test] [--grand]
PLAN : taille | memoire | hybride | sondes | tout (= taille, sondes, hybride, memoire).
--fast-test : essai de fumee (3 pas, fenetre 16, batch 2, eval 32, --data-mb 2), fichiers avec le suffixe _ft.
--grand : ajoute R_16x16x16 et GRU_4096 au plan taille.
Chaque run est un sous-processus (scripts/ts_remanence.py ou scripts/sondes_histoires.py) ; un echec n'arrete pas la suite ; tableau final lu dans results/*.json."""
import sys, os, json, subprocess, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results")


def ts_path(ds, model, seed, tag): return os.path.join(RES, "%s_%s_s%d%s.json" % ("ts" if ds == "tinystories" else ds, model, seed, tag))


def plan_ts(name, a):
    cfg = dict(taille=dict(ds=["tinystories", "wikitext2"], models=["R_4x8x8", "R_4x16x16", "GRU_256", "GRU_1024", "T_rope"] + (["R_16x16x16", "GRU_4096"] if a.grand else []), window=128, batch=32, steps=2000, eval=512, tag="_taille"),
               memoire=dict(ds=["pg19"], models=["R_4x8x8", "GRU_1024", "Twin_64", "T_rope"], window=256, batch=16, steps=2000, eval=1024, tag="_mem"),
               hybride=dict(ds=["tinystories", "wikitext2"], models=["H_64", "Twin_64", "R_4x8x8", "T_rope"], window=256, batch=16, steps=2000, eval=1024, tag="_hyb"))[name]
    if a.fast_test: cfg.update(window=16, batch=2, steps=3, eval=32, tag=cfg["tag"] + "_ft")
    runs = []
    for ds in cfg["ds"]:
        for m in cfg["models"]:
            cmd = [sys.executable, os.path.join(HERE, "ts_remanence.py"), "--dataset", ds, "--models", m, "--window", str(cfg["window"]), "--batch", str(cfg["batch"]), "--steps", str(cfg["steps"]),
                   "--eval-len", str(cfg["eval"]), "--tag", cfg["tag"], "--seed", "0", "--device", a.device] + (["--data-mb", "2", "--no-gen"] if a.fast_test else [])
            runs.append(dict(kind="ts", label="%s %s%s" % (ds, m, cfg["tag"]), path=ts_path(ds, m, 0, cfg["tag"]), cmd=cmd, ds=ds, model=m))
    return runs


def plan_sondes(a):
    tag = "_ft" if a.fast_test else ""; runs = []
    for m in ["L_loop1", "GRU_256", "L_loop1_init"]:
        for seed in [0, 1]:
            cmd = [sys.executable, os.path.join(HERE, "sondes_histoires.py"), m, str(seed), "20" if a.fast_test else "3000", "--device", a.device, "--tag", tag] + (["--n-stories", "300"] if a.fast_test else [])
            runs.append(dict(kind="sonde", label="sondes %s s%d%s" % (m, seed, tag), path=os.path.join(RES, "sondes_%s_s%d%s.json" % (m, seed, tag)), cmd=cmd, model=m))
    return runs


def build(plan, a):
    if plan == "tout": return sum([build(p, a) for p in ("taille", "sondes", "hybride", "memoire")], [])
    return plan_sondes(a) if plan == "sondes" else plan_ts(plan, a)


def table(runs):
    print("\n==== RESUME ====", flush=True)
    for r in runs:
        if not os.path.exists(r["path"]): print("%-45s MANQUANT (echec ou non lance)" % r["label"]); continue
        d = json.load(open(r["path"]))
        if r["kind"] == "ts": print("%-45s params %9d  %5ds  bpc %s" % (r["label"], d["n_params"], d["seconds"], "  ".join("%s:%.3f" % kv for kv in d["bpc"].items())))
        else:
            sd = d["sondes"]; print("%-45s tache %.2f  lieu fin/? %.2f/%.2f  objet fin/? %.2f/%.2f" % (r["label"], d["tache"]["8"]["reponse"], sd["fin_evenements"]["lieu"]["acc"], sd["sur_question"]["lieu"]["acc"], sd["fin_evenements"]["objet"]["acc"], sd["sur_question"]["objet"]["acc"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("plan", choices=["taille", "memoire", "hybride", "sondes", "tout"]); ap.add_argument("--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"])
    ap.add_argument("--fast-test", action="store_true"); ap.add_argument("--grand", action="store_true"); a = ap.parse_args()
    runs = build(a.plan, a); t0 = time.time(); os.makedirs(RES, exist_ok=True)
    for i, r in enumerate(runs, 1):
        if os.path.exists(r["path"]): print("[%d/%d] deja fait : %s" % (i, len(runs), r["label"]), flush=True); continue
        print("\n[%d/%d] %s  (%.0f min ecoulees)\n$ %s" % (i, len(runs), r["label"], (time.time() - t0) / 60, " ".join(r["cmd"])), flush=True)
        try:
            rc = subprocess.call(r["cmd"])
            if rc: print("ECHEC (code %d) : %s -> on continue" % (rc, r["label"]), flush=True)
        except Exception as e: print("ECHEC (%s) : %s -> on continue" % (e, r["label"]), flush=True)
    table(runs)
