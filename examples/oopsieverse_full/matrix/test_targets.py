"""Verification 1: target extraction on real demos.

Part A (project .venv, legacy repo code, read-only): legacy I-MEAN from mip.datasets.robomimic_dataset.RobomimicDataset
  (intent_type="mean", intent_horizon=8, intent_keys=[robot0_eef_pos, robot0_eef_quat], obs_keys/obs_steps of
  examples/configs/task/oopsieverse_base.yaml) on the TELEOP HDF5 vs our intent_targets.raw_targets on the official
  PLAYBACK HDF5 for the same windows (current obs t, future obs t+1..t+8), normalised with the legacy normaliser.
  Also our extraction applied to the teleop arrays themselves (same source; exactness of the formula).
Part B (oopsie env): quaternion convention vs robosuite.utils.transform_utils; rotation round trips
  (quat<->mat, mat<->6D, relative<->compose) on every train/dev/test frame and window of both tasks; quaternion
  hemisphere statistics for the legacy component-wise mean.

  .venv/bin/python test_targets.py legacy  -> outputs/.../matrix/verification/targets_legacy.json
  oopsie python    test_targets.py geometry -> outputs/.../matrix/verification/targets_geometry.json
"""

import glob
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import intent_targets as IT  # noqa: E402

REPO = "/scr/lizad/personal/learning-to-imitate"
OUT = os.path.join(REPO, "outputs/oopsieverse_full_study/matrix/verification")
TELEOP = "/edrive1/lizad/oopsieverse_full_study/data/oopsieverse-demos/paper_experiments/teleop_data/{task}/{cond}_live_feedback.hdf5"
PARTS = "/edrive1/lizad/oopsieverse_full_study/data/playback/parts/{task}__{cond}__{demo}.hdf5"
OBS_KEYS = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]


def load_playback(task, cond, demo):
    with h5py.File(PARTS.format(task=task, cond=cond, demo=demo), "r") as f:
        g = f["data"][demo]
        return g["obs/proprio"][:], g["action"][:]


def legacy(task="pick_egg"):
    sys.path.insert(0, REPO)
    from mip.datasets.robomimic_dataset import RobomimicDataset
    res = {"task": task, "conditions": {}}
    for cond in ("with", "without"):
        path = TELEOP.format(task=task, cond=cond)
        ds = RobomimicDataset(path, horizon=max(16, 2 + 8), obs_keys=OBS_KEYS, pad_before=1, pad_after=7,
                              abs_action=False, mode="train", val_dataset_percentage=0.0, intent_conditioning=True,
                              obs_steps=2, act_steps=8, intent_horizon=8, intent_type="mean",
                              intent_keys=["robot0_eef_pos", "robot0_eef_quat"])
        assert (ds.intent_start, ds.intent_end) == (14, 21), (ds.intent_start, ds.intent_end)
        nz = ds.normalizer["obs"]["state"]
        leg_norm = IT.MinMax(nz.min[14:21], nz.max[14:21])
        ends = ds.replay_buffer.episode_ends[:]
        starts = np.concatenate([[0], ends[:-1]])
        with h5py.File(path, "r") as f:
            n_demo = len(f["data"])
            teleop = {i: (f[f"data/demo_{i}/obs/robot0_eef_pos"][:], f[f"data/demo_{i}/obs/robot0_eef_quat"][:],
                          f[f"data/demo_{i}/actions"][:]) for i in range(n_demo)}
        rng = np.random.default_rng(0)
        errs_pb, errs_same, n = [], [], 0
        flips = 0
        for idx in range(len(ds)):
            b0, b1, s0, s1 = ds.sampler.indices[idx]
            ep = int(np.searchsorted(ends, b0, side="right"))
            loc = b0 - starts[ep] - s0          # window start (episode-local), may be -1
            t = loc + 1                         # current obs index (obs_steps - 1 = 1)
            Tt = ends[ep] - starts[ep]
            if t < 1 or t + 8 > Tt - 1 or s1 != ds.horizon:  # no padding inside the intent window
                continue
            if rng.random() > 0.25:             # subsample windows for speed (seeded)
                continue
            leg = ds[idx]["intent"].numpy().astype(np.float64)
            # ours on the official playback arrays (playback obs[t] == teleop obs[t], t >= 1)
            prop, act = _cached_pb(task, cond, f"demo_{ep}")
            ours_pb = leg_norm.normalize(IT.raw_targets(prop, act, ts=[t])["imean"][0, 0])
            # ours on the teleop arrays (pseudo-proprio with eef in cols 14:21; pseudo playback action = actions[1:])
            pos, quat, acts = teleop[ep]
            pp = np.zeros((len(pos), 23))
            pp[:, 14:17], pp[:, 17:21] = pos, quat
            ours_same = leg_norm.normalize(IT.raw_targets(pp, acts[1:], ts=[t])["imean"][0, 0])
            errs_pb.append(np.abs(ours_pb - leg).max())
            errs_same.append(np.abs(ours_same - leg).max())
            q = quat[t:t + 9]
            flips += int(np.any(q[1:] @ q[0] < 0))
            n += 1
        res["conditions"][cond] = {
            "n_windows_compared": n, "n_legacy_samples": len(ds),
            "max_abs_err_vs_playback_source (normalised units)": float(np.max(errs_pb)),
            "median_abs_err_vs_playback_source": float(np.median(errs_pb)),
            "max_abs_err_same_teleop_source": float(np.max(errs_same)),
            "windows_with_quat_hemisphere_flip": flips,
            "legacy_normaliser_eef_min": leg_norm.min.tolist(), "legacy_normaliser_eef_max": leg_norm.max.tolist()}
    return res


_PB = {}


def _cached_pb(task, cond, demo):
    k = (task, cond, demo)
    if k not in _PB:
        _PB[k] = load_playback(task, cond, demo)
    return _PB[k]


def geometry():
    import robosuite.utils.transform_utils as TU
    res = {}
    for task in ("pick_egg", "wipe_countertop"):
        r = {"quat_conv_maxerr_vs_robosuite_quat2mat": 0.0, "quat_mat_quat_maxerr": 0.0, "mat_6d_mat_maxerr": 0.0,
             "rel_compose_pos_maxerr_m": 0.0, "rel_compose_rot_maxerr": 0.0, "orthonormality_maxerr": 0.0,
             "det_min": 9.0, "quat_norm_dev_max": 0.0, "n_frames": 0, "n_windows": 0, "windows_quat_flip": 0,
             "imean_quat_norm_min": 9.0, "imean_quat_norm_p01": None, "noisy6d_projection_ok": True,
             "targets_finite": True, "rel_pos_k8_abs_max_m": 0.0, "rel_rot_k8_max_deg": 0.0}
        norms = []
        rs = np.random.default_rng(1)
        for p in sorted(glob.glob(PARTS.format(task=task, cond="*", demo="demo_*"))):
            demo = os.path.basename(p).split("__")[-1][:-5]
            cond = os.path.basename(p).split("__")[1]
            prop, act = load_playback(task, cond, demo)
            prop = prop[1:]  # stale playback obs[0] excluded (alignment_audit.md); re-pad below for indices
            prop = np.concatenate([prop[:1], prop])
            q = prop[:, 17:21].astype(np.float64)
            r["quat_norm_dev_max"] = max(r["quat_norm_dev_max"], float(np.abs(np.linalg.norm(q, axis=1) - 1).max()))
            R = IT.quat_xyzw_to_mat(q)
            Rref = np.stack([TU.quat2mat(qq / np.linalg.norm(qq)) for qq in q[:: max(1, len(q) // 50)]])
            r["quat_conv_maxerr_vs_robosuite_quat2mat"] = max(r["quat_conv_maxerr_vs_robosuite_quat2mat"],
                                                              float(np.abs(R[:: max(1, len(q) // 50)] - Rref).max()))
            q2 = IT.mat_to_quat_xyzw(R)
            qn = q / np.linalg.norm(q, axis=1, keepdims=True)
            qn = qn * np.sign(qn[:, 3:4] + 1e-12)
            r["quat_mat_quat_maxerr"] = max(r["quat_mat_quat_maxerr"], float(np.abs(q2 - qn).max()))
            r["mat_6d_mat_maxerr"] = max(r["mat_6d_mat_maxerr"], float(np.abs(IT.sixd_to_mat(IT.mat_to_6d(R)) - R).max()))
            r["orthonormality_maxerr"] = max(r["orthonormality_maxerr"], float(np.abs(np.swapaxes(R, 1, 2) @ R - np.eye(3)).max()))
            r["det_min"] = min(r["det_min"], float(np.linalg.det(R).min()))
            r["n_frames"] += len(q)
            ts = IT.sample_indices(len(act))
            tg = IT.raw_targets(prop, act, ts=ts)
            r["targets_finite"] &= all(np.isfinite(v).all() for v in tg.values())
            # relative -> compose round trip for every window and every k
            for k in range(1, 9):
                pr, Rr = IT.relative_pose(prop[ts, 14:17], q[ts], prop[ts + k, 14:17], q[ts + k])
                pw, Rw = IT.compose_pose(prop[ts, 14:17], q[ts], pr, Rr)
                r["rel_compose_pos_maxerr_m"] = max(r["rel_compose_pos_maxerr_m"], float(np.abs(pw - prop[ts + k, 14:17]).max()))
                r["rel_compose_rot_maxerr"] = max(r["rel_compose_rot_maxerr"], float(np.abs(Rw - IT.quat_xyzw_to_mat(q[ts + k])).max()))
                if k == 8:
                    r["rel_pos_k8_abs_max_m"] = max(r["rel_pos_k8_abs_max_m"], float(np.abs(pr).max()))
                    ang = np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1)))
                    r["rel_rot_k8_max_deg"] = max(r["rel_rot_k8_max_deg"], float(ang.max()))
            # path token at k=8 == the 6D of R_rel(k=8)
            pr8, Rr8 = IT.relative_pose(prop[ts, 14:17], q[ts], prop[ts + 8, 14:17], q[ts + 8])
            assert np.allclose(tg["ipath"][:, 3, :3], pr8) and np.allclose(tg["ipath"][:, 3, 3:], IT.mat_to_6d(Rr8))
            assert np.allclose(tg["ipathg"][:, :, 9], act[ts[:, None] + np.array([1, 3, 5, 7])[None], 6])
            # noisy 6D (as sampled by a flow) projects to a valid rotation
            noisy = IT.mat_to_6d(Rr8) + 0.3 * rs.standard_normal((len(ts), 6))
            Rn = IT.sixd_to_mat(noisy)
            ok = np.abs(np.swapaxes(Rn, 1, 2) @ Rn - np.eye(3)).max() < 1e-9 and np.linalg.det(Rn).min() > 0.999
            r["noisy6d_projection_ok"] &= bool(ok)
            fut = ts[:, None] + np.arange(1, 9)[None]
            r["windows_quat_flip"] += int(np.sum(np.any(np.einsum("nkd,nd->nk", q[fut], q[ts]) < 0, axis=1)))
            r["n_windows"] += len(ts)
            norms.append(np.linalg.norm(tg["imean"][:, 0, 3:7], axis=1))
        norms = np.concatenate(norms)
        r["imean_quat_norm_min"] = float(norms.min())
        r["imean_quat_norm_p01"] = float(np.percentile(norms, 1))
        res[task] = r
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    out = {t: legacy(t) for t in ("pick_egg", "wipe_countertop")} if mode == "legacy" else geometry()
    json.dump(out, open(os.path.join(OUT, f"targets_{mode}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
