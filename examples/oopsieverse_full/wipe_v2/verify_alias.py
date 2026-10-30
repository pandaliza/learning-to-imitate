"""Counter-label alias verification for the wipe v2 evaluator wrapper (Amendment 7.2). One EGL env (one process).

A. Training frames: for train demos of each training layout, restore the demo exactly as official playback does
   (ep_meta, reset, reset_from_xml_string, sim.reset, hide markers, states) and build the policy input with the
   wrapper's alias LUT (eval_wipe_v2.WorkerPolicy.seg_lut path = wv2_common.alias_seg_lut). Compare the model input
   tensors (3 views x 2 frames, vocabulary remap + 128 nearest-exact) with the official PlaybackDataset sample of the
   TRAINING data for the same (demo, t): step path and state path, as baseline/verify_obs_alignment.py does.
B. Evaluation resets (as wipe_diag/probe_equiv.py): reset each pool's first entries through the wrapper's reset path;
   policy seg with alias LUT and with the frozen evaluator LUT; then reload the SAME state through the playback path
   (set_ep_meta(get_ep_meta()) -> reset -> reset_from_xml_string(model xml) -> set_state), which names the
   classes as the training data do, and take the frozen LUT. Report per-view pixel agreement + counter pixels.
C. alias_seg_lut is identical to wipe_diag/diag_rollout.AliasPolicy.seg_lut on every class map met.
  CUDA_VISIBLE_DEVICES= MUJOCO_EGL_DEVICE_ID=3 python verify_alias.py OUT.json
"""
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wv2_common as C  # noqa: E402
import rollout_eval as RE  # noqa: E402

PARTS = os.path.join(C.OVS, "data/playback/parts")
DEMOS = [("with", "demo_10"), ("with", "demo_15"), ("without", "demo_0")]
TS = [1, 2, 60, 150]
RUN = os.path.join(C.OVS, "runs/matrix/amix_wipe_countertop_v1_s0")


def main(out_path):
    import torch
    os.chdir(RE.OOPSIE)
    from learning.dataset import PlaybackDataset
    from path_remap import remap_asset_paths
    from ctxfix import install_ctx_free
    import types
    sys.path.insert(0, os.path.join(C.FULL, "wipe_diag"))
    from diag_rollout import AliasPolicy

    def diag_alias_lut(c2i, m):
        return AliasPolicy.seg_lut(types.SimpleNamespace(class_to_id=c2i), m)
    vf = os.path.join(C.V2_RUNS, "verify", f"_alias_links_pid{os.getpid()}.hdf5")
    with h5py.File(vf, "w") as f:
        g = f.create_group("data")
        for k, (c, d) in enumerate(DEMOS):
            g[f"demo_{k}"] = h5py.ExternalLink(os.path.join(PARTS, f"wipe_countertop__{c}__{d}.hdf5"), f"data/{d}")
    pol = RE.Policy(RUN, ckpt="step_30000.pth", device="cpu", load_model=False)
    tc = pol.meta["trainer_config"]
    ds = PlaybackDataset(data_path=vf, frame_stack=tc["frame_stack"], action_chunk_size=tc["action_chunk"],
                         seg_img_size=(tc["seg_img_size"],) * 2, normalize_action=True,
                         objects_of_interest=tc["objects_of_interest"], obs_keys=tc["obs_keys"])
    assert ds.class_to_id == pol.class_to_id, (ds.class_to_id, pol.class_to_id)
    idx = {(dk, t): i for i, (dk, t) in enumerate(ds.samples)}
    env = RE.make_env(C.TASK, use_camera_obs=False)
    install_ctx_free(env)
    res = {"A_training_frames": [], "B_eval_resets": [], "C_lut_identical_to_wipe_diag": []}

    def segs(obs, lut):
        s, _ = pol.frame(obs, lut)
        return {k.split("/")[1]: v for k, v in s.items()}

    def check_c():
        a = C.alias_seg_lut(pol.class_to_id, env.seg_id_to_class)
        b = diag_alias_lut(pol.class_to_id, env.seg_id_to_class)
        res["C_lut_identical_to_wipe_diag"].append(bool(np.array_equal(a, b)))

    # ── A ──
    TELE = os.path.join(C.OVS, "data/oopsieverse-demos/paper_experiments/teleop_data/wipe_countertop")
    for k, (c, d) in enumerate(DEMOS):
        with h5py.File(os.path.join(TELE, f"{c}_live_feedback.hdf5"), "r") as tf, \
                h5py.File(os.path.join(PARTS, f"wipe_countertop__{c}__{d}.hdf5"), "r") as pf:
            tg = tf["data"][d]
            states = tg["states"][:]
            pacts = pf["data"][d]["action"][:]
            ep_meta = json.loads(remap_asset_paths(tg.attrs["ep_meta"]))
            xml = remap_asset_paths(tg.attrs["model_file"])
        env.set_ep_meta(ep_meta)
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        RE.hide_markers(env)
        lut = C.alias_seg_lut(pol.class_to_id, env.seg_id_to_class)
        lut_frozen = C.frozen_seg_lut(pol.class_to_id, env.seg_id_to_class)
        check_c()

        def at_state(i):
            env.sim.set_state_from_flattened(states[i])
            env.sim.forward()
            RE._sync_vel(env)
            return env.get_observations()[0]

        def step_from(i):
            at_state(i)
            return env.step(pacts[i])[0]

        for t in TS:
            if (f"demo_{k}", t) not in idx:
                continue
            s = ds[idx[(f"demo_{k}", t)]]
            for path in ("step", "state"):
                if path == "step":
                    o_prev = at_state(0) if t - 1 == 0 else step_from(t - 2)
                    o_cur = step_from(t - 1)
                else:
                    o_prev, o_cur = at_state(t - 1), at_state(t)
                hist = [pol.frame(o_prev, lut), pol.frame(o_cur, lut)]
                seg, prop = pol.model_inputs(hist)
                eq = {key.split("/")[1]: float((seg[key][0].cpu() == s["obs"]["extero"][key]).float().mean()) for key in pol.seg_keys}
                cnt = {key.split("/")[1]: int((s["obs"]["extero"][key] == 1).sum()) for key in pol.seg_keys}
                conf, eq_nodirt = {}, {}
                for key in pol.seg_keys:
                    ev, tr = seg[key][0].cpu().numpy().astype(int), s["obs"]["extero"][key].numpy().astype(int)
                    bad = ev != tr
                    for (x, y) in zip(tr[bad].tolist(), ev[bad].tolist()):
                        conf[f"train{x}->eval{y}"] = conf.get(f"train{x}->eval{y}", 0) + 1
                    keep = (ev != 2) & (tr != 2)
                    eq_nodirt[key.split("/")[1]] = float((ev[keep] == tr[keep]).mean())
                with h5py.File(os.path.join(PARTS, f"wipe_countertop__{c}__{d}.hdf5"), "r") as pf2:
                    tw = int(pf2["data"][d]["info/total_wiped"][t - 1])
                row = {"demo": f"{c}/{d}", "t": t, "path": path, "seg_equal_frac_alias_vs_training": eq,
                       "training_counter_px_128x2": cnt, "mismatch_confusion_train_to_eval": conf,
                       "seg_equal_frac_excluding_dirt_pixels": eq_nodirt, "training_total_wiped_at_t": tw,
                       "env_wiped_markers_after_restore": len(env.wiped_markers), "luts_equal_here": bool(np.array_equal(lut, lut_frozen)),
                       "proprio_maxabs": float((prop[0].cpu() - s["obs"]["proprio"]).abs().max())}
                res["A_training_frames"].append(row)
                print(json.dumps(row), flush=True)
    # ── B ──
    pools = C.load_pools()
    for pool, n in (("P-random", 4), ("P-train", 3), ("P-devlayout", 2)):
        for r in C.pool_resets(pool, pools)[:n]:
            obs, _ = RE.apply_reset(env, r)
            check_c()
            names_eval = sorted(set(env.seg_id_to_class.values()))
            alias = segs(obs, C.alias_seg_lut(pol.class_to_id, env.seg_id_to_class))
            frozen = segs(obs, C.frozen_seg_lut(pol.class_to_id, env.seg_id_to_class))
            em, xml = env.get_ep_meta(), env.sim.model.get_xml()
            state = env.sim.get_state().flatten().copy()
            obs2, _ = RE.apply_reset(env, {"seed": r["seed"], "ep_meta": em, "model_xml": xml, "state": state.tolist()})
            check_c()
            pb = segs(obs2, C.frozen_seg_lut(pol.class_to_id, env.seg_id_to_class))
            row = {"pool": pool, "id": r["id"], "state_maxabs_diff_after_reload": float(np.abs(env.sim.get_state().flatten() - state).max()),
                   "eval_class_names": names_eval, "playback_class_names": sorted(set(env.seg_id_to_class.values()))}
            for cam in pb:
                row[cam] = {"alias_vs_playback_agree": float((alias[cam] == pb[cam]).mean()),
                            "frozen_vs_playback_agree": float((frozen[cam] == pb[cam]).mean()),
                            "counter_px_alias": int((alias[cam] == 1).sum()), "counter_px_frozen": int((frozen[cam] == 1).sum()),
                            "counter_px_playback": int((pb[cam] == 1).sum())}
            res["B_eval_resets"].append(row)
            print(json.dumps(row), flush=True)
    A = res["A_training_frames"]
    B = res["B_eval_resets"]
    res["summary"] = {
        "A_min_seg_equal_alias_vs_training": min(min(r["seg_equal_frac_alias_vs_training"].values()) for r in A),
        "A_n": len(A),
        "B_min_alias_vs_playback": {p: min(min(r[c]["alias_vs_playback_agree"] for c in ("robot0_eye_in_hand", "robot0_agentview_right", "robot0_agentview_left"))
                                           for r in B if r["pool"] == p) for p in ("P-random", "P-train", "P-devlayout")},
        "B_min_frozen_vs_playback_eye_in_hand": {p: min(r["robot0_eye_in_hand"]["frozen_vs_playback_agree"] for r in B if r["pool"] == p)
                                                for p in ("P-random", "P-train", "P-devlayout")},
        "C_all_identical": all(res["C_lut_identical_to_wipe_diag"]), "C_n": len(res["C_lut_identical_to_wipe_diag"]),
    }
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
    try:
        os.remove(vf)
    except OSError:
        pass
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1])
