"""Bit-identity: W0 via train_v2.py vs the matrix A-MIX wipe seed-0 run (logged losses, step 0 eval)."""
import json, sys
def load(p):
    tr, ev = {}, []
    for l in open(p):
        d = json.loads(l)
        if "train/loss" in d: tr[int(d["step"])] = d["train/loss"]
        if "eval/dev_mse_loss" in d: ev.append((d["eval/dev_mse_loss"], d["eval/train_mse_loss"]))
        if "eval/mse_loss" in d: ev.append(("official", d["eval/mse_loss"]))
    return tr, ev
a_tr, a_ev = load(sys.argv[1]); b_tr, b_ev = load(sys.argv[2])
ks = sorted(set(a_tr) & set(b_tr))
res = {"ref": sys.argv[1], "new": sys.argv[2], "n_common_loss_steps": len(ks), "steps": [ks[0], ks[-1]],
       "n_exact": sum(a_tr[k] == b_tr[k] for k in ks), "max_abs": max(abs(a_tr[k] - b_tr[k]) for k in ks),
       "step0_eval_ref": a_ev[:2], "step0_eval_new": b_ev[:2], "step0_eval_exact": a_ev[:2] == b_ev[:2]}
json.dump(res, open(sys.argv[3], "w"), indent=1); print(json.dumps(res, indent=1))
