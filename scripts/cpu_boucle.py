"""Controles CPU de l'etape 4 (amendement 15) : config reduite (S3, entrainement 12, eval 96, d=64), memes conditions pour tous les modeles.
Usage : python scripts/cpu_boucle.py MODELE GRAINE [PAS]   (modeles : L_loop, L_noloop, L_wide, L_i2, T_rope, V_lam0)"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch
import s3_tasks as s

torch.set_num_threads(1)
SPECS = dict(L_loop=dict(name="L_loop", kind="L", loop=True), L_noloop=dict(name="L_noloop", kind="L", loop=False),
             L_wide=dict(name="L_wide", kind="L", loop=False, layers=4), L_i2=dict(name="L_i2", kind="L", loop="i2"),
             A_nogate=dict(name="A_nogate", kind="L", loop=True, no_gate=True), A_readI1=dict(name="A_readI1", kind="L", loop=True, readout="i1"),
             A_p4=dict(name="A_p4", kind="L", loop=True, p=4), A_1layer=dict(name="A_1layer", kind="L", loop=True, layers=1),
             T_rope=dict(name="T_rope", kind="T", pos="rope"), V_lam0=dict(name="V_lam0", kind="V", scales=2, penalty=0.0))
ST = dict(s.ST_LOOP, train_len=12, eval_len=96, batch=64, d_model=64, d_tok=32, eval_every=250, eval_seqs=256,
          buckets=[0, 12, 24, 48, 96], warmup=100, summary_name="st_cpu_summary.md")
if __name__ == "__main__":
    name, seed = sys.argv[1], int(sys.argv[2])
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    s.ALL_TASKS["S3c"] = (3, steps)
    root = os.path.join(os.path.dirname(__file__), "..")
    s.run_one("S3c", SPECS[name], seed, steps, ST, torch.device("cpu"), root, lambda *a: print(*a, flush=True))
