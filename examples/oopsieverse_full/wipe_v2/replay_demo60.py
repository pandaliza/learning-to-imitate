"""Open-loop replay reference on P-demo60 (spec addendum 2026-10-29 23:30 UTC, "reference measurement").

Replays the recorded human actions of all 60 wipe_countertop demonstrations from their exact initial states (the frozen
pools/pools_demo60_v1.json), under normal physics, with no policy. Method = the 40-demo reference (replay.py, mode
"open": actions[1:] stepped once from states[0], demo's own length, success = robocasa predicate at the last step,
primary safe = success and every entity's final health > 95), except that the restore is the P-demo60 one
(build_pool_demo.restore_and_check = rollout_eval.apply_reset + observable refresh), in the evaluator env
(rollout_eval.make_env, render on for EGL, GPU 6 only). Wiped fraction = total_wiped / 50 at the last step.

Frozen code is imported, never edited. Records: runs/wipe_v2/all60/replay_demo60/<reset>.json (atomic, resumable).
Cross-check: per-step wiped trace vs runs/replay/open_teleop/*.npz (rp_wiped) for the same demos.

  python replay_demo60.py run [--limit N]   # run all (or the first N) pool entries
  python replay_demo60.py report            # writes reports/wipe_v2/replay_demo60.md
"""
import json
import math
import os
import sys
import time

os.environ["MUJOCO_EGL_DEVICE_ID"] = "6"  # GPU 6 only (shared box; see the study's GPU rules)

import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wv2_common as C  # noqa: E402
import build_pools_all60 as BA  # noqa: E402
import build_pool_demo as BPD  # noqa: E402
import rollout_eval as RE  # noqa: E402

OUT = os.path.join(C.V2_RUNS, "all60", "replay_demo60")
REPORT = os.path.join(C.V2_REPORTS, "replay_demo60.md")
PRIOR = os.path.join(C.OVS, "runs/replay/open_teleop/wipe_countertop__{cond}__{demo}.npz")
N_MARKERS = 50
GROUPS = ["g00", "g01", "g02", "g03", "g04", "g05", "g06"]


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


def safe_name(reset_id):
    return reset_id.replace("/", "__") + ".json"


def replay_one(env, e):
    dev_info = BPD.restore_and_check(env, e)  # leaves env in the restored state, no steps taken
    dev = {k: v for k, v in dev_info["dev"].items() if not k.startswith("stale")}
    with h5py.File(e["teleop_file"], "r") as f:
        g = f["data"][e["demo"]]
        acts = g["actions"][:]
        rec_eef = g["obs/robot0_eef_pos"][:].astype(np.float64)
        n_states = int(g["states"].shape[0])
    assert len(acts) == n_states, ("actions/states length", len(acts), n_states)
    link_names = list(env.health_list_link_names)
    wiped, succ, eef_err, health_last = [], [], [], None
    assert len(env.markers) == N_MARKERS, len(env.markers)
    for i, a in enumerate(acts[1:]):
        obs, rew, done, info = env.step(a)
        wiped.append(int(info["total_wiped"]))
        succ.append(RE.success_now(env))
        eef_err.append(float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], np.float64) - rec_eef[i + 1])))
        health_last = np.asarray(obs["health"], np.float64)
    ent = {}
    for v, n in zip(health_last, link_names):
        o = n.split("@", 1)[0]
        ent[o] = min(ent.get(o, 100.0), float(v))
    final_success = bool(succ[-1])
    safe = all(v > 95.0 for v in ent.values())
    prior = None
    pf = PRIOR.format(cond=e["cond"], demo=e["demo"])
    if os.path.exists(pf):
        pz = np.load(pf, allow_pickle=True)
        pw = pz["rp_wiped"]
        prior = {"prior_final_wiped": int(pw[-1]), "prior_final_success": bool(pz["rp_success"][-1]),
                 "wiped_trace_equal_to_prior": bool(len(pw) == len(wiped) and np.array_equal(pw, np.asarray(wiped)))}
    return {
        "reset_id": e["reset_id"], "demo_id": f"{e['cond']}/{e['demo']}", "layout_group": e["layout_group"],
        "split": e.get("split"), "success": final_success, "primary_safe_success": final_success and safe,
        "final_wiped_count": wiped[-1], "final_proportion_wiped": wiped[-1] / N_MARKERS, "steps": len(wiped),
        "first_success_step": (int(np.argmax(succ)) + 1) if any(succ) else None,
        "ever_success": bool(any(succ)), "final_min_health_per_entity": ent,
        "restore_max_deviation_vs_recorded_obs0": dev, "restore_dev_max": max(dev.values()) if dev else None,
        "eef_err_first10": eef_err[:10], "eef_err_max": max(eef_err), "eef_err_final": eef_err[-1],
        "wiped_trace": wiped, "cross_check_open_teleop_npz": prior,
        "pool": "P-demo60", "pool_file": BA.DEMO60_FILE, "mode": "open-loop replay, actions[1:], normal physics",
        "evaluator_env": "rollout_eval.make_env(use_camera_obs=False), install_ctx_free, apply_reset + observable refresh",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def cmd_run(limit=None):
    doc = BA.load_pool_doc(BA.DEMO60_FILE)
    ents = doc["entries"][:limit] if limit else doc["entries"]
    todo = [e for e in ents if not os.path.exists(os.path.join(OUT, safe_name(e["reset_id"])))]
    print(f"{len(ents)} entries, {len(todo)} to run, GPU {os.environ['MUJOCO_EGL_DEVICE_ID']}", flush=True)
    if not todo:
        return
    os.chdir(RE.OOPSIE)
    from ctxfix import install_ctx_free
    env = RE.make_env(C.TASK, use_camera_obs=False)
    install_ctx_free(env)
    t0 = time.time()
    for e in todo:
        r = replay_one(env, e)
        atomic_json(os.path.join(OUT, safe_name(e["reset_id"])), r)
        print(r["reset_id"], "succ", int(r["success"]), "safe", int(r["primary_safe_success"]),
              "wiped", r["final_wiped_count"], "steps", r["steps"], "restore_dev", f"{r['restore_dev_max']:.2g}",
              "eef_err_max", f"{r['eef_err_max']:.3g}", "prior_eq", (r["cross_check_open_teleop_npz"] or {}).get("wiped_trace_equal_to_prior"),
              f"{time.time() - t0:.0f}s", flush=True)
    print("RUN_DONE", flush=True)
    os._exit(0)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def pct(x):
    return f"{100 * x:.1f}%"


def cell(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({pct(k / n)}; {pct(lo)}–{pct(hi)})"


def cmd_report():
    doc = BA.load_pool_doc(BA.DEMO60_FILE)
    rows = []
    for e in doc["entries"]:
        p = os.path.join(OUT, safe_name(e["reset_id"]))
        if not os.path.exists(p):
            sys.exit(f"missing record {p}")
        rows.append(json.load(open(p)))
    assert len(rows) == 60, len(rows)
    L = ["# wipe_countertop: open-loop replay of all 60 demonstrations (P-demo60 reference)\n"]
    L.append("Spec addendum 2026-10-29 23:30 UTC (reference measurement; no pass/fail). Pool `runs/wipe_v2/pools/pools_demo60_v1.json` "
             f"(content_sha256 `{doc['content_sha256']}`). Each demo: exact restore (apply_reset + observable refresh), then its recorded "
             "`actions[1:]` stepped once under normal physics, no policy. Success = robocasa predicate at the last replayed step; "
             "primary safe = success and every entity's final health > 95; wiped fraction = total_wiped / 50. One pass. "
             f"Records: `{OUT}/`. Code: `examples/oopsieverse_full/wipe_v2/replay_demo60.py`.\n")

    def summ(rs):
        n = len(rs)
        k = sum(r["success"] for r in rs)
        ks = sum(r["primary_safe_success"] for r in rs)
        fw = [r["final_proportion_wiped"] for r in rs]
        return cell(k, n), ks, float(np.mean(fw)), float(np.median(fw))
    L.append("## Overall and per layout group\n")
    L.append("| Set | n | Task success k/n (rate; Wilson 95%) | Primary safe k | Mean final wiped | Median final wiped |")
    L.append("|---|---|---|---|---|---|")
    c, ks, mu, md = summ(rows)
    L.append(f"| all 60 | {len(rows)} | {c} | {ks} | {mu:.3f} | {md:.3f} |")
    for g in GROUPS:
        rs = [r for r in rows if r["layout_group"] == g]
        if not rs:
            continue
        c, ks, mu, md = summ(rs)
        L.append(f"| {g} | {len(rs)} | {c} | {ks} | {mu:.3f} | {md:.3f} |")
    L.append("")
    L.append("## Per-demo rows\n")
    L.append("| Demo | Layout | Split | Success | Primary safe | Final wiped (count / fraction) | Steps | First success step | Restore dev max | EEF err max vs recorded | Wiped trace = prior npz |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        cc = r["cross_check_open_teleop_npz"]
        eq = "n/a" if cc is None else str(cc["wiped_trace_equal_to_prior"])
        L.append(f"| {r['demo_id']} | {r['layout_group']} | {r['split']} | {int(r['success'])} | {int(r['primary_safe_success'])} | "
                 f"{r['final_wiped_count']} / {r['final_proportion_wiped']:.2f} | {r['steps']} | {r['first_success_step'] or '-'} | "
                 f"{r['restore_dev_max']:.2g} | {r['eef_err_max']:.3g} | {eq} |")
    fails = [r for r in rows if not r["success"]]
    L.append("\n## Failures (not task success)\n")
    L.append("| Demo | Layout | Final wiped fraction | Steps |")
    L.append("|---|---|---|---|")
    for r in fails:
        L.append(f"| {r['demo_id']} | {r['layout_group']} | {r['final_proportion_wiped']:.2f} | {r['steps']} |")
    txt = "\n".join(L) + "\n"
    open(REPORT, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    if sys.argv[1] == "run":
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        cmd_run(lim)
    elif sys.argv[1] == "report":
        cmd_report()
    else:
        sys.exit(__doc__)
