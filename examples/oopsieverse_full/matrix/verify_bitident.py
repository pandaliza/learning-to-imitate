"""Verification 3 helpers.

run:      python verify_bitident.py run [--deterministic] -- <script.py> <args...>
          Runs a trainer script unchanged (runpy) after optionally enabling torch deterministic algorithms
          (CUBLAS_WORKSPACE_CONFIG=:4096:8, use_deterministic_algorithms(True), cudnn.deterministic). Used for both the
          reproduced baseline wrapper (baseline/train_amix.py) and train_matrix.py, so the comparison isolates code-path
          differences from GPU atomic-add non-determinism.
compare:  python verify_bitident.py compare <out.json> <ckptdirA> <ckptdirB> [...]
          Exact state-dict and optimizer-state equality at every common step_*.pth, plus logged losses.
"""

import glob
import json
import os
import runpy
import sys


def run(argv):
    det = "--deterministic" in argv
    i = argv.index("--")
    script, rest = argv[i + 1], argv[i + 2:]
    if det:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    sys.argv = [script] + rest
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    runpy.run_path(script, run_name="__main__")


def _losses(run_dir):
    out = {}
    p = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.exists(p):
        return out
    for line in open(p):
        d = json.loads(line)
        if "train/loss" in d:
            out[int(d["step"])] = d["train/loss"]
        if "eval/mse_loss" in d:
            out[f"eval{int(d['step'])}"] = d["eval/mse_loss"]
    return out


def compare(out, dirs):
    import torch
    res = {"runs": dirs, "steps": {}}
    sets = [set(os.path.basename(p) for p in glob.glob(os.path.join(d, "ckpt", "step_*.pth"))) for d in dirs]
    common = sorted(set.intersection(*sets), key=lambda s: int(s.split("_")[1][:-4]))
    ref = dirs[0]
    for name in common:
        a = torch.load(os.path.join(ref, "ckpt", name), map_location="cpu")
        row = {}
        for d in dirs[1:]:
            b = torch.load(os.path.join(d, "ckpt", name), map_location="cpu")
            sa, sb = a["policy_state_dict"], b["policy_state_dict"]
            keys_equal = list(sa) == list(sb)
            maxdiff = max(float((sa[k].double() - sb[k].double()).abs().max()) for k in sa) if keys_equal else None
            exact = keys_equal and all(torch.equal(sa[k], sb[k]) for k in sa)
            oa, ob = a["optimizer_state_dict"]["state"], b["optimizer_state_dict"]["state"]
            opt_exact = all(torch.equal(oa[k][s], ob[k][s]) for k in oa for s in oa[k]) if set(oa) == set(ob) else False
            row[d] = {"params_exact": exact, "params_maxabs": maxdiff, "optimizer_exact": opt_exact}
        res["steps"][name] = row
    la = _losses(ref)
    res["losses"] = {}
    for d in dirs[1:]:
        lb = _losses(d)
        ks = sorted(set(la) & set(lb), key=str)
        res["losses"][d] = {"n_common": len(ks), "n_exact": sum(la[k] == lb[k] for k in ks),
                            "max_abs": max((abs(la[k] - lb[k]) for k in ks), default=None),
                            "first_mismatch": next((k for k in sorted(ks, key=lambda x: (isinstance(x, str), str(x).zfill(8))) if la[k] != lb[k]), None)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2:])
    else:
        compare(sys.argv[2], sys.argv[3:])
