"""Native-evaluator parity check (spec section 4): compare per-replan action chunks dumped by
eval_libero_intent.py --episode-seeding S --dump-actions D  (D/native/*.npy)
with eval_spatial_shifts.py --deltas 0.0 --policy-seeds S --dump-actions D  (D/<arm>/*.npy)
on the same unperturbed resets. Any action/trajectory discrepancy beyond --atol blocks the campaign.

  python examples/openpi/oops/parity_check.py --dump-dir oops_runs/parity/M0 --arm M0
"""
import argparse
import json
import pathlib
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = pathlib.Path(args.dump_dir)
    rows, ok_all = [], True
    native = sorted((d / "native").glob("*.npy"))
    if not native:
        sys.exit(f"no native dumps in {d / 'native'}")
    for pn in native:
        pw = d / args.arm / pn.name
        if not pw.exists():
            rows.append({"episode": pn.stem, "status": "MISSING_WRAPPER_DUMP"})
            ok_all = False
            continue
        a = list(np.load(pn, allow_pickle=True))
        b = list(np.load(pw, allow_pickle=True))
        ta, tb = a.pop(), b.pop()
        n = min(len(a), len(b))
        first_div, max_diff = None, 0.0
        for k in range(n):
            if a[k]["t"] != b[k]["t"]:
                first_div = first_div if first_div is not None else k
                continue
            diff = float(np.max(np.abs(np.asarray(a[k]["actions"]) - np.asarray(b[k]["actions"]))))
            sdiff = float(np.max(np.abs(np.asarray(a[k]["state"]) - np.asarray(b[k]["state"]))))
            max_diff = max(max_diff, diff, sdiff)
            if max(diff, sdiff) > args.atol and first_div is None:
                first_div = k
        same_len = len(a) == len(b)
        ok = first_div is None and same_len and ta["done"] == tb["done"] and ta["steps"] == tb["steps"]
        ok_all &= ok
        rows.append({"episode": pn.stem, "status": "MATCH" if ok else "MISMATCH", "n_replans": [len(a), len(b)],
                     "first_divergent_replan": first_div, "max_abs_diff": max_diff,
                     "native_done_steps": [ta["done"], ta["steps"]], "wrapper_done_steps": [tb["done"], tb["steps"]]})
    for r in rows:
        print(json.dumps(r))
    print("PARITY:", "PASS" if ok_all else "FAIL (blocks the campaign until explained)")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps({"pass": ok_all, "atol": args.atol, "episodes": rows}, indent=2))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
