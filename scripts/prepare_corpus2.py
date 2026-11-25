"""Corpus supplementaires accessibles : Penn Treebank, Gutenberg (contes / tout), code Python local. Meme format que prepare_corpus.py."""
import os, sys, glob, random, re
sys.path.insert(0, os.path.dirname(__file__))
from prepare_corpus import build, ROOT

G = "/tmp/gutenberg/gutenberg/"
STORIES = ["carroll-alice", "bryant-stories", "burgess-busterbrown", "edgeworth-parents"]


def split_books(names, d):
    os.makedirs(d, exist_ok=True); tr, va = [], []
    for n in names:
        t = re.sub(r"\n{3,}", "\n\n", open(G + n + ".txt", encoding="latin-1").read()); t = re.sub(r"[ \t]+", " ", t); k = t.index("\n", int(len(t) * 0.95))
        tr.append(t[:k + 1]); va.append(t[k + 1:])
    open(os.path.join(d, "train.txt"), "w", encoding="utf-8").write("\n".join(tr)); open(os.path.join(d, "val.txt"), "w", encoding="utf-8").write("\n".join(va))


if __name__ == "__main__":
    build("ptb", ["train.txt", "valid.txt"])          # PTB : deja minuscule, sans ponctuation, <unk> = mot rare
    split_books(STORIES, os.path.join(ROOT, "data", "gutenberg_contes")); build("gutenberg_contes", ["train.txt", "val.txt"])
    allb = sorted(os.path.basename(f)[:-4] for f in glob.glob(G + "*.txt")); split_books(allb, os.path.join(ROOT, "data", "gutenberg_tout")); build("gutenberg_tout", ["train.txt", "val.txt"])
    files = sorted(glob.glob("/usr/lib/python3*/**/*.py", recursive=True)); random.Random(0).shuffle(files); d = os.path.join(ROOT, "data", "code_python"); os.makedirs(d, exist_ok=True)
    tr, va, a, b = [], [], 0, 0
    for f in files:
        t = open(f, encoding="utf-8", errors="ignore").read()
        if len(t) < 500: continue
        if a < 10_000_000: tr.append(t); a += len(t)
        elif b < 1_000_000: va.append(t); b += len(t)
    open(os.path.join(d, "train.txt"), "w", encoding="utf-8").write("\n<|endoftext|>\n".join(tr)); open(os.path.join(d, "val.txt"), "w", encoding="utf-8").write("\n<|endoftext|>\n".join(va)); build("code_python", ["train.txt", "val.txt"])
