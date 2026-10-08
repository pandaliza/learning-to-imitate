"""Equivalence of fast_contacts.fast_get_part_contacts with the released method, on damaging demo replays (CPU).

Per control step and per tracked body, both functions are called on the same sim state and their outputs compared
exactly (same length/order, same keys, identical value types, identical float bits). Then a whole replay is run once
with the released method and once with the shim, and the per-step traces (integration state, health, damage_info,
success, ...) are compared bit-exactly. Timing of both is reported.
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_restore as TR  # noqa: E402
import fast_contacts as FC  # noqa: E402


def same(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, float):
        return np.float64(a).tobytes() == np.float64(b).tobytes()
    return a == b


def main(cases, out):
    os.chdir("/edrive1/lizad/oopsieverse_full_study/src/oopsieverse")
    orig = FC.original()
    res = []
    for case in cases:
        task, cond, demo = case.split(":")
        c = TR.load_case(task, cond, demo)
        acts = c["actions"][1:]
        env = TR.make_env_cpu(task)
        TR.setup_demo(env, c)
        n_calls = n_eq = n_nonempty = 0
        for a in acts:
            env.step(a)
            for o in env.get_damageable_objects():
                for ev in o.damage_evaluators:
                    for p in ev._get_damageable_part_names():
                        x = orig(ev, p)
                        y = FC.fast_get_part_contacts(ev, p)
                        n_calls += 1
                        n_eq += same(x, y)
                        n_nonempty += bool(x)
        FC.uninstall()
        TR.setup_demo(env, c)
        t0 = time.time()
        A = TR.run(env, acts)
        t_orig = time.time() - t0
        FC.install()
        TR.setup_demo(env, c)
        t0 = time.time()
        B = TR.run(env, acts)
        t_fast = time.time() - t0
        FC.uninstall()
        eq, comp = TR.compare(A, B)
        r = {"case": case, "steps": len(acts), "per_call": {"n_calls": n_calls, "n_identical": n_eq,
                                                              "n_nonempty_contact_lists": n_nonempty},
             "trace_bit_exact": eq, "trace_max_abs_diff": comp, "damage_events": int((A["damage_info"][:, :, 0] > 0).sum()),
             "s_per_step_released": t_orig / len(acts), "s_per_step_shim": t_fast / len(acts)}
        r["pass"] = bool(n_eq == n_calls and eq)
        res.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "trace_max_abs_diff"}), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1].split(","), sys.argv[2])
