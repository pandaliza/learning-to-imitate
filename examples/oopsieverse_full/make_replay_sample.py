"""Pre-register the stratified replay sample from STORED health only (spec section 6).

Strata on worst-entity final loss L = max_obj (100 - min_link final health):
  zero: L <= 1e-4; mild: 1e-4 < L <= 5; substantial: L > 5.
Per task: 4 per stratum, split as evenly as possible across feedback conditions
(given availability), chosen with numpy default_rng(20261006) inside each cell.
Writes outputs/oopsieverse_full_study/replay/replay_sample.json. Must run before any replay.
"""

import json
import os
import hashlib
import datetime

import h5py
import numpy as np

from common import (OUT, TASKS, CONDITIONS, MILD_MAX, ZERO_TOL, teleop_path,
                    demo_sort_key, object_final_health, stratum)

PER_STRATUM = 4
SEED = 20261006


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rng = np.random.default_rng(SEED)
    out = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "rule": {
            "worst_entity_loss": "max over tracked objects of (100 - final object health); object health = min over links",
            "zero": f"L <= {ZERO_TOL}", "mild": f"{ZERO_TOL} < L <= {MILD_MAX}", "substantial": f"L > {MILD_MAX}",
            "per_stratum": PER_STRATUM, "allocation": "even split across with/without feedback, remainder to the condition with more episodes; cells sampled with numpy default_rng",
            "seed": SEED, "source": "stored obs/health only (no simulator)",
        },
        "files": {}, "population": {}, "sample": {},
    }
    for task in TASKS:
        cells = {}
        for cond in CONDITIONS:
            p = teleop_path(task, cond)
            out["files"][f"{task}/{cond}"] = {"path": p, "sha256": sha256(p)}
            with h5py.File(p, "r") as f:
                for d in sorted(f["data"], key=demo_sort_key):
                    g = f["data"][d]
                    names = [n.decode() if isinstance(n, bytes) else str(n) for n in g.attrs["health_list_link_names"]]
                    fin = object_final_health(g["obs/health"][-1], names)
                    L = max(100.0 - v for v in fin.values())
                    cells.setdefault((stratum(L), cond), []).append({"demo": d, "cond": cond, "worst_loss": round(L, 4),
                                                                     "final_obj_health": {k: round(v, 4) for k, v in fin.items()}})
        out["population"][task] = {f"{s}/{c}": len(v) for (s, c), v in sorted(cells.items())}
        sel = []
        for s in ["zero", "mild", "substantial"]:
            avail = {c: list(cells.get((s, c), [])) for c in CONDITIONS}
            alloc = {c: 0 for c in CONDITIONS}
            for _ in range(PER_STRATUM):  # round-robin allocation honoring availability
                cands = [c for c in CONDITIONS if alloc[c] < len(avail[c])]
                if not cands:
                    break
                c = min(cands, key=lambda c: (alloc[c], -len(avail[c])))
                alloc[c] += 1
            for c in CONDITIONS:
                pool = avail[c]
                idx = rng.choice(len(pool), size=alloc[c], replace=False) if alloc[c] else []
                for i in sorted(idx):
                    sel.append(dict(pool[i], stratum=s))
        out["sample"][task] = sel
    os.makedirs(os.path.join(OUT, "replay"), exist_ok=True)
    path = os.path.join(OUT, "replay", "replay_sample.json")
    assert not os.path.exists(path), "sample already pre-registered; refusing to overwrite"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    for t, s in out["sample"].items():
        print(t, [(r["stratum"], r["cond"], r["demo"], r["worst_loss"]) for r in s])
    print(out["population"])


if __name__ == "__main__":
    main()
