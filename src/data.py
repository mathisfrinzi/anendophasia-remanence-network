import os
import numpy as np
import torch
import config


def prepare(root=".", n_train=None, n_val=None):
    """Charge TinyStories (streaming, dans l'ordre : les n_train premieres histoires pour
    train, les n_val suivantes pour val). Valeurs par defaut = protocole des bras 1-2 ;
    l'autoencodeur en passe de plus petites (les fichiers portent alors le nombre dans leur nom)."""
    n_train = n_train or config.COMMON["n_train_stories"]
    n_val = n_val or config.COMMON["n_val_stories"]
    default = (n_train, n_val) == (config.COMMON["n_train_stories"], config.COMMON["n_val_stories"])
    tag = "" if default else "_%d_%d" % (n_train, n_val)
    tr = os.path.join(root, "data", "train%s.txt" % tag)
    va = os.path.join(root, "data", "val%s.txt" % tag)
    if not (os.path.exists(tr) and os.path.exists(va)):
        os.makedirs(os.path.join(root, "data"), exist_ok=True)
        from datasets import load_dataset
        import itertools
        n = n_train + n_val
        print("Streaming TinyStories :", n, "histoires...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        texts = [s["text"] for s in itertools.islice(ds, n)]
        assert len(texts) == n, "streaming incomplet : " + str(len(texts))
        with open(tr, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts[:n_train]))
        with open(va, "w", encoding="utf-8") as f:
            f.write("\n\n".join(texts[n_train:]))
    train_txt = open(tr, encoding="utf-8").read()
    val_txt = open(va, encoding="utf-8").read()
    vocab = sorted(set(train_txt) | set(val_txt))
    assert len(vocab) <= 512
    train_ids = encode(train_txt, vocab)
    val_ids = encode(val_txt, vocab)
    return vocab, train_ids, val_ids


def encode(txt, vocab):
    codes = np.array([ord(c) for c in vocab], dtype=np.int64)
    ids = np.empty(len(txt), dtype=np.uint16)
    step = 10_000_000
    for i in range(0, len(txt), step):
        chunk = txt[i : i + step]
        cp = np.frombuffer(chunk.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
        idx = np.searchsorted(codes, cp)
        assert (idx < len(codes)).all()
        assert (codes[idx] == cp).all()
        ids[i : i + len(chunk)] = idx.astype(np.uint16)
    return ids


def decode(vocab, ids):
    return "".join(vocab[int(i)] for i in ids)


def batch_from(rng, ids, B, T, device):
    offsets = rng.integers(0, len(ids) - T - 1, size=B)
    chunks = np.stack([ids[o : o + T + 1] for o in offsets]).astype(np.int64)
    x = torch.from_numpy(chunks[:, :T]).to(device)
    y = torch.from_numpy(chunks[:, 1:]).to(device)
    return x, y
