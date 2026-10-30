"""Instrumented wipe_countertop rollouts for the wipe diagnosis (new script; the frozen evaluator files
baseline/rollout_eval.py and matrix/rollout_eval_matrix.py are imported, never edited).

Episodes run through the UNCHANGED baseline run_episode (frozen protocol: 1250-step timeout, 10-step post-success
window, stochastic Euler-16 sampling, full 8-step chunks). We only wrap env.step to log, per control step:
markers wiped, ||ee_force - ee_force_bias||, sponge height above the counter top, eef-sponge distance, gripper qpos;
and optionally save policy-input segmentation (+ RGB, rendered separately) at fixed steps.

Conditions (--cond):
  bank        eval_dev bank resets (first --n), policy LUT exactly as the frozen evaluator builds it.
  bank_alias  same resets/seeds, but the policy LUT maps the env class names produced by a plain env.reset()
              (e.g. "Counter") to the training-vocabulary names ("DamageableCounter"). Diagnostic only.
  demo        demo initial states (ep_meta + model_xml + states[0], as official playback / study replay).
              --demos "with/demo_10,without/demo_0,..."

Placement: inference server on GPU 0 (CUDA_VISIBLE_DEVICES=0); workers CUDA-free, EGL on GPU 3 only.
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for p in (os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import rollout_eval as RE  # noqa: E402
import rollout_eval_matrix as REM  # noqa: E402

FRAME_STEPS = (0, 25, 50, 100, 200, 300, 400, 600, 800, 1000, 1249)
CAMS = ["robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"]


class AliasPolicy(REM.MatrixRemotePolicy):
    def seg_lut(self, seg_id_to_class):
        m = max(int(k) for k in seg_id_to_class)
        lut = np.zeros(m + 1, dtype=np.int64)
        for k, name in seg_id_to_class.items():
            cid = self.class_to_id.get(name, 0)
            if cid == 0 and ("Damageable" + name) in self.class_to_id:
                cid = self.class_to_id["Damageable" + name]
            lut[int(k)] = cid
        return lut


def worker(args):
    wid, jobs, run_dir, out_dir, cond, ckpt, qs, frames_for = args
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "3"
    wt = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wdiag_{os.getpid()}")
    os.makedirs(wt, exist_ok=True)
    os.environ["TMPDIR"] = wt
    os.chdir(RE.OOPSIE)
    import torch
    torch.set_num_threads(1)
    env = RE.make_env("wipe_countertop", use_camera_obs=False)
    from ctxfix import install_ctx_free
    install_ctx_free(env)
    P = AliasPolicy if cond == "bank_alias" else REM.MatrixRemotePolicy
    pol = P(run_dir, ckpt, wid, qs[0], qs[1])
    orig_step = env.step
    done = 0
    for j in jobs:
        out = os.path.join(out_dir, f"{j['id']}__r{j['rollout_seed']}.json")
        if os.path.exists(out):
            continue
        log = {k: [] for k in ("wiped", "fdiff", "sponge_h", "eef_sponge_d", "grip", "eef")}
        frames = {}
        want_frames = j["id"] in frames_for
        state = {"t": 0}

        def snap(obs, t):
            lut = pol.seg_lut(env.seg_id_to_class)
            segs, _ = pol.frame(obs, lut)
            fr = {f"seg__{k.split('/')[1]}": v.astype(np.uint8) for k, v in segs.items()}
            for cam in CAMS:
                fr[f"rgb__{cam}"] = env.sim.render(width=256, height=256, camera_name=cam)[::-1]
            frames[t] = fr

        def logged_step(a):
            obs, r, d, info = orig_step(a)
            state["t"] += 1
            arm = env.robots[0].arms[0]
            f = np.asarray(env.robots[0].ee_force[arm])
            b = np.asarray(env.ee_force_bias.get(arm, np.zeros(3)))
            sp = env.sim.data.body_xpos[env.obj_body_id["sponge"]]
            eef = env.sim.data.site_xpos[env.robots[0].eef_site_id[arm]]
            top = float(env.counter.pos[2] + env.counter.height / 2) if hasattr(env.counter, "height") else 0.0
            log["wiped"].append(int(info.get("total_wiped", -1)))
            log["fdiff"].append(float(np.linalg.norm(f - b)))
            log["sponge_h"].append(float(sp[2] - top))
            log["eef_sponge_d"].append(float(np.linalg.norm(eef - sp)))
            log["grip"].append(float(np.sum(np.abs(obs["robot0_gripper_qpos"]))))
            log["eef"].append([float(x) for x in eef])
            if want_frames and state["t"] in FRAME_STEPS:
                snap(obs, state["t"])
            return obs, r, d, info

        try:
            env.step = logged_step
            reset = j["reset"]
            if want_frames:
                orig_reset_obs = RE.apply_reset

                def apply_reset_snap(e, r):
                    o = orig_reset_obs(e, r)
                    snap(o[0], 0)
                    return o
                RE.apply_reset = apply_reset_snap
            t0 = time.time()
            rec = RE.run_episode(env, pol, "wipe_countertop", reset, j["rollout_seed"], exec_horizon=8,
                                 record_chunks=False)
            if want_frames:
                RE.apply_reset = orig_reset_obs
            arm = env.robots[0].arms[0]
            w = np.asarray(log["wiped"])
            fd = np.asarray(log["fdiff"])
            rec.update({
                "episode_id": j["id"], "cond": cond, "run_dir": run_dir, "arm": pol.arm_meta["arm"],
                "training_seed": pol.arm_meta["seed"], "ckpt_name": ckpt,
                "final_wiped": int(w[-1]) if len(w) else 0, "max_wiped": int(w.max()) if len(w) else 0,
                "final_frac_wiped": float(w[-1]) / 50.0 if len(w) else 0.0,
                "first_wipe_step": int(np.argmax(w > 0)) + 1 if (w > 0).any() else None,
                "wiped_at": {str(s): int(w[min(s, len(w)) - 1]) for s in (100, 200, 300, 400, 600, 800, 1000, 1250) if len(w)},
                "steps_fdiff_gt100": int((fd > 100).sum()), "max_fdiff": float(fd.max()) if len(fd) else 0.0,
                "steps_sponge_lifted_gt2cm": int((np.asarray(log["sponge_h"]) - log["sponge_h"][0] > 0.02).sum()) if log["sponge_h"] else 0,
                "ee_force_bias": np.asarray(env.ee_force_bias.get(arm)).tolist(),
                "demo_len": reset.get("demo_len"), "diag_wall_s": time.time() - t0,
            })
            np.savez_compressed(out[:-5] + "__trace.npz", **{k: np.asarray(v) for k, v in log.items()})
            if frames:
                np.savez_compressed(out[:-5] + "__frames.npz",
                                    **{f"t{t:04d}__{k}": v for t, fr in frames.items() for k, v in fr.items()})
        except Exception:
            rec = {"episode_id": j["id"], "infra_error": traceback.format_exc(), "cond": cond}
        finally:
            env.step = orig_step
        tmp = out + ".tmp"
        json.dump(rec, open(tmp, "w"), default=float)
        os.replace(tmp, out)
        done += 1
        print(f"[w{wid}] {j['id']} r{j['rollout_seed']} steps={rec.get('steps')} wiped={rec.get('final_wiped')} "
              f"succ={rec.get('success')}", flush=True)
    return wid, done


def _entry(args, rq):
    try:
        rq.put(worker(args))
    except Exception:
        rq.put((args[0], traceback.format_exc()))


def build_jobs(cond, n, demos, seeds):
    if cond in ("bank", "bank_alias"):
        rs = REM.dev_resets("wipe_countertop")[:n]
    else:
        from wipe_common import demo_reset
        rs = [demo_reset(*d.split("/")) for d in demos.split(",")]
    return [{"id": r["id"], "reset": r, "rollout_seed": int(s)} for s in seeds.split(",") for r in rs]


def main():
    import multiprocessing as mp
    import threading
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="step_30000.pth")
    ap.add_argument("--cond", required=True, choices=["bank", "bank_alias", "demo"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--demos", default="")
    ap.add_argument("--rollout-seeds", default="0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--frames", type=int, default=2, help="save frames for the first K resets")
    ap.add_argument("--max-mem-gb", type=float, default=10.0)
    a = ap.parse_args()
    assert a.workers <= 9
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = build_jobs(a.cond, a.n, a.demos, a.rollout_seeds)
    frames_for = {j["id"] for j in jobs[: a.frames]}
    W = min(a.workers, len(jobs))
    chunks = [jobs[i::W] for i in range(W)]
    ctx = mp.get_context("spawn")
    req_q, resp = ctx.Queue(), [ctx.Queue() for _ in range(W)]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    srv = ctx.Process(target=REM.inference_server, args=(a.run_dir, a.ckpt, "cuda:0", req_q, resp, 64, None))
    srv.start()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "3"
    os.environ["EGL_DEVICE_ID"] = "3"
    rq = ctx.Queue()
    procs = []
    for i in range(W):
        p = ctx.Process(target=_entry, args=((i, chunks[i], a.run_dir, a.out_dir, a.cond, a.ckpt, (req_q, resp[i]),
                                              frames_for), rq))
        p.start()
        procs.append(p)
    pids = {p.pid for p in procs}
    REM.RENDER_GPU = 3
    stop = threading.Event()
    mon = open(os.path.join(a.out_dir, "_gpu3_monitor.jsonl"), "a")

    def monitor():
        while not stop.is_set():
            try:
                ours, free = REM._gpu3_usage(pids)
                mon.write(json.dumps({"t": time.time(), "ours_mib": ours, "free_mib": free}) + "\n")
                mon.flush()
                if ours > a.max_mem_gb * 1024 or free < 8 * 1024:
                    mon.write(json.dumps({"t": time.time(), "event": "LIMIT_EXCEEDED_ABORT"}) + "\n")
                    mon.flush()
                    for p in procs:
                        p.terminate()
                    os._exit(3)
            except Exception as e:
                mon.write(json.dumps({"t": time.time(), "err": str(e)}) + "\n")
            stop.wait(15)
    threading.Thread(target=monitor, daemon=True).start()
    res = [rq.get() for _ in procs]
    for p in procs:
        p.join()
    stop.set()
    req_q.put(None)
    srv.join()
    print("DIAG_DONE", a.cond, len(jobs), res)


if __name__ == "__main__":
    main()
