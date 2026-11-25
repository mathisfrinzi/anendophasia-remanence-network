"""Prepare des corpus reels accessibles depuis le bac a sable (GitHub raw) : Wikitext-2 et Tiny Shakespeare.
Nettoie, entraine un BPE 4096 par corpus (tokenizer/bpe_<corpus>_4096.json), tokenise (data/<corpus>/ids_*.npy) et affiche les stats."""
import os, re, sys, numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

ROOT = os.path.join(os.path.dirname(__file__), "..")


def clean_wikitext(t):
    t = t.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    t = re.sub(r" ([.,;:!?)\]'])", r"\1", t); t = re.sub(r"([(\[]) ", r"\1", t)
    t = t.replace(" n't", "n't").replace(" 's", "'s").replace("\n \n", "\n\n").replace("\n = ", "\n= ")
    return t.replace("<unk>", "¤")          # marque d'inconnu : un seul caractere (conserve, non special)


def build(name, texts, cleaner=None):
    d = os.path.join(ROOT, "data", name); tr, va = [open(os.path.join(d, f), encoding="utf-8").read() for f in texts]
    if cleaner: tr, va = cleaner(tr), cleaner(va)
    tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator([tr], trainers.BpeTrainer(vocab_size=4096, special_tokens=["<|endoftext|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok.save(os.path.join(ROOT, "tokenizer", "bpe_%s_4096.json" % name))
    for tag, txt in [("train", tr), ("val", va)]:
        open(os.path.join(d, tag + "_clean.txt"), "w", encoding="utf-8").write(txt)
        ids = np.array(tok.encode(txt).ids, dtype=np.uint16); np.save(os.path.join(d, "ids_%s_4096.npy" % tag), ids)
        print("%-10s %-5s %9d caracteres %9d tokens  %.2f car/token" % (name, tag, len(txt), len(ids), len(txt) / len(ids)))


if __name__ == "__main__":
    build("wikitext2", ["train.txt", "valid.txt"], clean_wikitext)
    # Shakespeare : un seul fichier -> 90 % / 10 % (coupe a une limite de ligne)
    p = os.path.join(ROOT, "data", "shakespeare", "input.txt"); t = open(p, encoding="utf-8").read(); k = t.index("\n", int(len(t) * 0.9))
    open(os.path.join(ROOT, "data", "shakespeare", "train.txt"), "w", encoding="utf-8").write(t[:k + 1]); open(os.path.join(ROOT, "data", "shakespeare", "val.txt"), "w", encoding="utf-8").write(t[k + 1:])
    build("shakespeare", ["train.txt", "val.txt"])
