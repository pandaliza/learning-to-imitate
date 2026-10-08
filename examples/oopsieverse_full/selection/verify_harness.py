"""Harness + rendered restore verification (GPU-3 EGL rendering, CPU torch inference; no GPU-0 use except V3).

V1 null branch: the harness state machine started at the parent reset reproduces the frozen evaluator
   (rollout_eval.run_episode, same policy object, same per-episode generator) record-for-record;
   V1b: the same with render skipping on.
V2 rendered restore through damage with the policy in the loop: canonical parent reset, policy to t_b, capture
   (incl. the 2-frame seg/proprio history and the torch generator state), continue to the end = reference.
   Then: in-place restore (twice), restore into a fresh env instance, disk round trip in a fresh process.
   Compared: per-chunk policy-input hashes (seg history + proprio), chunks, full health trace hash, final record.
V3 (GPU 0) fixed-shape batch invariance of the inference path: a request decoded at different batch positions with
   different companions gives bit-identical actions/intents (all 4 arms).
V5 timing: per-step env time with full rendering vs render skipping.

  python verify_harness.py v1 --parents 3 --out OUT.json
  python verify_harness.py v2 --out OUT.json
  python verify_harness.py v3 --out OUT.json      (CUDA_VISIBLE_DEVICES=0)
"""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harness as H  # noqa: E402
import snapshot_damage_state as SDS  # noqa: E402
import gpu3_budget  # noqa: E402

RE = H.RE
RUNS = "/edrive1/lizad/oopsieverse_full_study/runs/matrix"
ARMS = {"amix": f"{RUNS}/amix_pick_egg_v1_s0", "imean": f"{RUNS}/imean_pick_egg_v1_s0",
        "ipath": f"{RUNS}/ipath_pick_egg_v1_s0", "ipathg": f"{RUNS}/ipathg_pick_egg_v1_s0"}
CKPT = "step_30000.pth"
WORK = "/edrive1/lizad/oopsieverse_full_study/runs/selection/verify"


def parents(n):
    from rollout_eval_matrix import bank_resets
    return bank_resets("pick_egg", "sel_dev_parents")[:n]


def make_env():
    from ctxfix import install_ctx_free
    env = RE.make_env("pick_egg", use_camera_obs=False)
    install_ctx_free(env)
    return env


def strip(rec):
    drop = {"wall_s", "reset_s", "infer_s", "env_step_s", "ckpt", "sampling", "reset", "rollout_seed", "exec_horizon",
            "nfe", "chunks_log", "reset_obj_pos", "reset_obj_pos_bank_maxabs", "timeout_overridden", "termination"}
    return {k: v for k, v in rec.items() if k not in drop}


def cmp_records(a, b):
    a, b = strip(a), strip(b)
    keys = sorted(set(a) & set(b))
    diff = [k for k in keys if json.dumps(a[k], default=float, sort_keys=True) != json.dumps(b[k], default=float, sort_keys=True)]
    return {"equal": not diff, "differing": diff, "n_compared": len(keys),
            "only_in_one": sorted(set(a) ^ set(b))}


class HashingDecoder:
    """Local CPU decoder that also logs a hash of every policy input and output."""

    def __init__(self, arm):
        self.dec = H.LocalDecoder({arm: ARMS[arm]}, CKPT, "cpu")
        self.log = []

    def decode(self, arm, seg, prop, x0, z):
        a, it = self.dec.decode(arm, seg, prop, x0, z, fixed_batch=1)
        h = hashlib.sha256()
        for k in sorted(seg):
            h.update(seg[k].tobytes())
        h.update(prop.tobytes())
        self.log.append({"in": h.hexdigest()[:16], "out": hashlib.sha256(a.tobytes()).hexdigest()[:16]})
        return a, it


def v1(n_parents, out):
    import torch
    torch.set_num_threads(4)
    os.chdir(RE.OOPSIE)
    env = make_env()
    from rollout_eval_matrix import MatrixPolicy
    pol = MatrixPolicy(ARMS["amix"], ckpt=CKPT, device="cpu")
    frames = H.Frames(ARMS["amix"])
    ainfo = H.arm_info(ARMS["amix"])
    res = []
    for r in parents(n_parents):
        row = {"parent": r["family"]}
        t0 = time.time()
        rec = RE.run_episode(env, pol, "pick_egg", r, 0, exec_horizon=8)
        row["evaluator_wall_s"] = time.time() - t0
        for skip in (False, True):
            dec = H.LocalDecoder({"amix": ARMS["amix"]}, CKPT, "cpu")
            obs = SDS.canonical_reset(env, r["seed"]) or env.get_observations()[0]
            ep = H.new_episode(env, "pick_egg", frames, obs, r)
            lut = frames.lut(env.seg_id_to_class)
            gen = torch.Generator().manual_seed(0 * 1_000_003 + int(r["seed"]))
            t0 = time.time()

            class D:
                def decode(self, arm, seg, prop, x0, z):
                    return dec.decode(arm, seg, prop, x0, z, fixed_batch=1)
            H.run_policy(env, ep, D(), "amix", ainfo, frames, lut, gen, render_skip=skip)
            mine = H.finalize(ep)
            row[f"harness_render_skip_{skip}"] = {"wall_s": time.time() - t0, "steps": mine["steps"],
                                                   **cmp_records(rec, mine)}
        row["evaluator_record"] = {k: rec[k] for k in ("steps", "success", "outcome", "worst_entity_loss", "first_damage_step")}
        res.append(row)
        print(json.dumps(row, default=float), flush=True)
        json.dump(res, open(out, "w"), indent=1, default=float)
    return res


def _gen_state(g):
    return g.get_state().numpy().copy()


def _gen_from(state):
    import torch
    g = torch.Generator()
    g.set_state(torch.from_numpy(state))
    return g


def _health_trace_run(env, ep, dec, ainfo, frames, lut, gen):
    """Continue to the end, logging the per-step health vector hash and damage events."""
    trace = []
    orig = H.step_env

    def step_logged(env_, ep_, a, frames_, lut_, render):
        d = orig(env_, ep_, a, frames_, lut_, render)
        trace.append(hashlib.sha256(np.asarray(ep_["health"]).tobytes() + SDS.integration_state(env_).tobytes()).hexdigest()[:12])
        return d
    H.step_env = step_logged
    try:
        H.run_policy(env, ep, dec, "amix", ainfo, frames, lut, gen, render_skip=True)
    finally:
        H.step_env = orig
    return trace


def v2_child(snap_path, out_path):
    import torch
    torch.set_num_threads(4)
    os.chdir(RE.OOPSIE)
    env = make_env()
    frames = H.Frames(ARMS["amix"])
    ainfo = H.arm_info(ARMS["amix"])
    snap = SDS.load(snap_path)
    ep = SDS.rebuild_and_restore(env, snap)
    lut = frames.lut(env.seg_id_to_class)
    gen = _gen_from(snap["meta"]["gen_state"])
    dec = HashingDecoder("amix")
    tr = _health_trace_run(env, ep, dec, ainfo, frames, lut, gen)
    json.dump({"record": H.finalize(ep), "trace": tr, "dec_log": dec.log}, open(out_path, "w"), default=float)


def v2(out, max_parents=8):
    import torch
    torch.set_num_threads(4)
    os.makedirs(WORK, exist_ok=True)
    os.chdir(RE.OOPSIE)
    env = make_env()
    frames = H.Frames(ARMS["amix"])
    ainfo = H.arm_info(ARMS["amix"])
    results = []
    for r in parents(max_parents):
        obs = SDS.canonical_reset(env, r["seed"]) or env.get_observations()[0]
        ep = H.new_episode(env, "pick_egg", frames, obs, r)
        lut = frames.lut(env.seg_id_to_class)
        gen = torch.Generator().manual_seed(777 + int(r["seed"]))
        dec = HashingDecoder("amix")
        H.run_policy(env, ep, dec, "amix", ainfo, frames, lut, gen, until_t=48, render_skip=True)
        if ep["done"] or ep["first_damage_step"] is not None:
            continue
        S = SDS.capture(env, episode=ep, meta={"parent_seed": r["seed"], "gen_state": _gen_state(gen)})
        ep_r = SDS.restore(env, S)  # immediately after capture: the reference run
        dec_ref = HashingDecoder("amix")
        tr_ref = _health_trace_run(env, ep_r, dec_ref, ainfo, frames, lut, _gen_from(S["meta"]["gen_state"]))
        rec_ref = H.finalize(ep_r)
        row = {"parent": r["family"], "t_b": 48, "reference": {k: rec_ref[k] for k in (
            "steps", "success", "outcome", "worst_entity_loss", "first_damage_step", "entity_final_health")}}
        # first run after capture (env advanced from S without restore) vs restored run
        checks = {}

        def run_check(name, env_, ep_):
            dec_ = HashingDecoder("amix")
            tr = _health_trace_run(env_, ep_, dec_, ainfo, frames, lut, _gen_from(S["meta"]["gen_state"]))
            rec = H.finalize(ep_)
            checks[name] = {"record": cmp_records(rec_ref, rec), "trace_equal": tr == tr_ref,
                            "policy_inputs_equal": [d["in"] for d in dec_.log] == [d["in"] for d in dec_ref.log],
                            "policy_outputs_equal": [d["out"] for d in dec_.log] == [d["out"] for d in dec_ref.log],
                            "n_steps": len(tr), "n_chunks": len(dec_.log)}
            checks[name]["pass"] = bool(checks[name]["record"]["equal"] and checks[name]["trace_equal"]
                                        and checks[name]["policy_inputs_equal"])
        run_check("in_place_restore", env, SDS.restore(env, S))
        SDS.restore(env, S)
        run_check("restore_twice", env, SDS.restore(env, S))
        env2 = make_env()
        SDS.canonical_reset(env2, parents(max_parents)[-1]["seed"])  # a different parent first
        run_check("fresh_env_instance", env2, SDS.rebuild_and_restore(env2, S))
        env2.close()
        del env2
        sp = f"{WORK}/v2_{r['id']}.pkl"
        SDS.save(S, sp)
        op = f"{WORK}/v2_{r['id']}_child.json"
        rc = subprocess.call([sys.executable, os.path.abspath(__file__), "v2child", sp, op],
                             stdout=subprocess.DEVNULL, stderr=open(op + ".log", "w"))
        if rc == 0:
            z = json.load(open(op))
            checks["disk_fresh_process"] = {"record": cmp_records(rec_ref, z["record"]), "trace_equal": z["trace"] == tr_ref,
                                            "policy_inputs_equal": [d["in"] for d in z["dec_log"]] == [d["in"] for d in dec_ref.log]}
            checks["disk_fresh_process"]["pass"] = bool(checks["disk_fresh_process"]["record"]["equal"] and
                                                        checks["disk_fresh_process"]["trace_equal"] and
                                                        checks["disk_fresh_process"]["policy_inputs_equal"])
        else:
            checks["disk_fresh_process"] = {"pass": False, "error": f"rc={rc}"}
        row["checks"] = checks
        row["damaging_continuation"] = rec_ref["first_damage_step"] is not None
        results.append(row)
        print(json.dumps({"parent": row["parent"], "damaging": row["damaging_continuation"],
                          **{k: v["pass"] for k, v in checks.items()}}), flush=True)
        json.dump(results, open(out, "w"), indent=1, default=float)
        if sum(x["damaging_continuation"] for x in results) >= 2 and len(results) >= 3:
            break
    return results


def v3(out):
    import torch
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from rollout_eval_matrix import MatrixPolicy
    rng = np.random.default_rng(0)
    res = {}
    for arm, rd in ARMS.items():
        pol = MatrixPolicy(rd, ckpt=CKPT, device="cuda:0")
        ai = H.arm_info(rd)
        B = 40
        seg = {k: rng.integers(0, 6, size=(B, 2, 128, 128)).astype(np.uint8) for k in H.Frames(rd).seg_keys}
        prop = rng.normal(size=(B, 23)).astype(np.float32)
        x0 = rng.normal(size=(B, 8, ai["action_dim"])).astype(np.float32)
        z = rng.normal(size=(B, ai["n_tok"], ai["tok_dim"])).astype(np.float32) if ai["conditioned"] else None
        a_all, i_all = H.decode_batch(pol, seg, prop, x0, z, "cuda:0", 16)
        ok, n = True, 0
        for j in (0, 5, 17, 39):
            for comp in range(3):
                idx = list(rng.choice([q for q in range(B) if q != j], size=int(rng.integers(0, 15)), replace=False))
                pos = int(rng.integers(0, len(idx) + 1))
                idx.insert(pos, j)
                sl = np.array(idx)
                a, it = H.decode_batch(pol, {k: v[sl] for k, v in seg.items()}, prop[sl], x0[sl],
                                       None if z is None else z[sl], "cuda:0", 16)
                ok &= bool(np.array_equal(a[pos], a_all[j]))
                if it is not None:
                    ok &= bool(np.array_equal(it[pos], i_all[j]))
                n += 1
        # batch-size-1 path (padded to 16) == position in a full batch
        a1, _ = H.decode_batch(pol, {k: v[3:4] for k, v in seg.items()}, prop[3:4], x0[3:4],
                               None if z is None else z[3:4], "cuda:0", 16)
        ok &= bool(np.array_equal(a1[0], a_all[3]))
        # unpadded variable batch for contrast (the evaluator's server): does it differ?
        a_var, _ = pol.decode({k: torch.from_numpy(v[:5]).float().cuda() for k, v in seg.items()},
                              torch.from_numpy(prop[:5]).cuda(), torch.from_numpy(x0[:5]).cuda(),
                              None if z is None else torch.from_numpy(z[:5]).cuda())
        res[arm] = {"bit_identical_across_positions_and_companions": ok, "n_trials": n + 1,
                    "variable_batch5_vs_fixed16_maxabs": float(np.abs(a_var - a_all[:5]).max())}
        print(arm, res[arm], flush=True)
    json.dump(res, open(out, "w"), indent=1)


def v5(out, n_steps=120):
    os.chdir(RE.OOPSIE)
    env = make_env()
    r = parents(1)[0]
    SDS.canonical_reset(env, r["seed"])
    a = np.zeros(12)
    a[6] = -1
    res = {}
    for name, render in (("render_every_step", True), ("no_render", False)):
        env.render_segmentation = render
        t0 = time.time()
        for _ in range(n_steps):
            env.step(a)
        res[name] = (time.time() - t0) / n_steps
    env.render_segmentation = True
    res["render_skip_expected_per_step"] = (2 * res["render_every_step"] + 6 * res["no_render"]) / 8
    print(res)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "v2child":
        v2_child(sys.argv[2], sys.argv[3])
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("which")
    ap.add_argument("--parents", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.which in ("v1", "v2", "v5"):
        assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("MUJOCO_EGL_DEVICE_ID") == "3"
        gpu3_budget.wait_for_slot(1)
    {"v1": lambda: v1(a.parents, a.out), "v2": lambda: v2(a.out), "v3": lambda: v3(a.out),
     "v5": lambda: v5(a.out)}[a.which]()
