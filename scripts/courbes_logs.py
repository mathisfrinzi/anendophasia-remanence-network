"""Rassemble les courbes d'apprentissage des journaux d'entrainement (hors depot) dans results/courbes_logs.json, pour que les figures
de la partie 14 soient reproductibles. Usage : python scripts/courbes_logs.py DOSSIER_JOURNAUX [AUTRE_DOSSIER ...]
Chaque point : {step, total, loss, + val_mV | acc [h=1..] | reponse | bpc_0_64}. Cle : 'fichier|etiquette|graine'."""
import sys, os, re, json, glob
RX = re.compile(r"\[(?P<tag>[^\]]+?) s(?P<seed>\d+)\] pas (?P<step>\d+)(?:/(?P<total>\d+))? perte (?P<loss>[\d.]+)(?P<rest>.*)")
out = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.log"))):
        for line in open(f, encoding="utf-8", errors="ignore"):
            m = RX.search(line)
            if not m: continue
            p = dict(step=int(m["step"]), loss=float(m["loss"])); r = m["rest"]
            if m["total"]: p["total"] = int(m["total"])
            if (v := re.search(r"val ([\d.]+) mV", r)): p["val_mV"] = float(v[1])
            if (v := re.search(r"h=1\.\.\d+ \[([^\]]*)\]", r)): p["acc"] = [float(x) for x in v[1].split(",")]
            if (v := re.search(r"reponse ([\d.]+)", r)): p["reponse"] = float(v[1])
            if (v := re.search(r"bpc val\(0-64\) ([\d.]+)", r)): p["bpc_0_64"] = float(v[1])
            if (v := re.search(r"acc12 ([\d.]+) acc24 ([\d.]+)", r)): p["acc12"], p["acc24"] = float(v[1]), float(v[2])
            out.setdefault("%s|%s|%s" % (os.path.basename(f)[:-4], m["tag"], m["seed"]), []).append(p)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "courbes_logs.json"), "w"), indent=0)
print(len(out), "courbes,", sum(len(v) for v in out.values()), "points")
for k in sorted(out): print(k, len(out[k]), "pas", out[k][-1]["step"])
