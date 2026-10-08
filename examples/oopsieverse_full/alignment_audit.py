"""Obs/action alignment audit for the OopsieVerse RoboCasa teleop HDF5s (Phase B, task 4).

Question: in the stored teleop data, is obs[t] the observation AFTER actions[t] was applied
(teleop_robocasa.py:116-140 appends env.step(action) output together with that action), or the
observation from which actions[t] was chosen (robomimic convention)?

Modes
  sim     (oopsie env): simulator tests on a sample of episodes
            A  restore states[t-1], step actions[t]   -> compare with states[t], obs[t]     (H_post)
            B  restore states[t],   step actions[t]   -> compare with states[t+1], obs[t+1] (H_pre)
            C  restore states[t], forward, get_observations -> compare with obs[t]
          + data-only lag analysis over all 120 episodes (gripper and arm response to command changes)
  playback (oopsie env): run the OFFICIAL playback_episode (portable copy, collection-time gripper)
          on one episode per task into a scratch HDF5 and check its index contract against the teleop file
  loader  (project .venv): instantiate this repo's RobomimicDataset on a teleop file and show which
          raw indices end up paired in a training sample

Read-only on all inputs. Outputs JSON next to the reset-bank runs on /edrive1.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = "/edrive1/lizad/oopsieverse_full_study/runs/alignment"
HF = ("/home/lizad/.cache/huggingface/hub/datasets--ut-robin-lab--oopsieverse-demos/snapshots/"
      "c0ec29395fdcf898de2aed8dc0b3e786dc2a37d3/paper_experiments/teleop_data")
CMP_KEYS = ["robot0_eef_pos", "robot0_gripper_qpos", "robot0_joint_pos", "object-state"]


def tpath(task, cond):
    return os.path.join(HF, task, f"{cond}_live_feedback.hdf5")


def sim_mode(n_eps, n_steps):
    """Sequential passes per episode (controllers carry internal state between steps, so each
    hypothesis is tested as an in-order replay from a fresh episode setup):
      A (H_post): for t = 1..T-1: restore states[t-1], step actions[t]; compare with states[t] / obs[t]
                  (this is the official playback pairing)
      B (H_pre):  for t = 0..T-2: restore states[t],   step actions[t]; compare with states[t+1] / obs[t+1]
      C: restore states[t], forward, force-refresh observables; compare with obs[t]
    """
    sys.path.insert(0, HERE)
    import h5py
    import replay as R  # official-equivalent setup (path remap, ep_meta, model_file, collection gripper)
    out = {"per_episode": [], "lag": {}}

    def obs_err(o, ref):
        return max(float(np.abs(np.asarray(o[k], dtype=np.float64) - ref[k]).max()) for k in CMP_KEYS)

    for task in ("pick_egg", "wipe_countertop"):
        env = R.get_env(R.TASKS[task])
        for cond in ("with", "without"):
            for demo in [f"demo_{i}" for i in range(n_eps)]:
                row = dict(task=task, cond=cond, demo=demo)
                for hyp in ("A", "B", "C"):
                    with h5py.File(tpath(task, cond), "r") as f:
                        g = f["data"][demo]
                        acts = g["actions"][:]
                        st_obs = {k: g["obs"][k][:].astype(np.float64) for k in CMP_KEYS}
                        states, _ = R.setup_episode(env, g)
                    T = len(states)
                    nq = env.sim.model.nq
                    se, qe, oe = [], [], []
                    rng_t = range(1, T) if hyp == "A" else range(0, T - 1) if hyp == "B" else range(0, T)
                    for t in rng_t:
                        if hyp == "C":
                            env.sim.set_state_from_flattened(states[t]); env.sim.forward()
                            o = env._get_observations(force_update=True)
                            oe.append(obs_err(o, {k: v[t] for k, v in st_obs.items()}))
                            continue
                        src, tgt = (t - 1, t) if hyp == "A" else (t, t + 1)
                        env.sim.set_state_from_flattened(states[src]); env.sim.forward()
                        R.sync_damage_evaluator_velocities(env)
                        o, *_ = env.step(acts[t])
                        s = env.sim.get_state().flatten()
                        se.append(float(np.linalg.norm(s[1:] - states[tgt][1:])))
                        qe.append(float(np.linalg.norm(s[1:1 + nq] - states[tgt][1:1 + nq])))
                        oe.append(obs_err(o, {k: v[tgt] for k, v in st_obs.items()}))
                    row["T"] = T
                    if se:
                        row[f"{hyp}_state_err_median"] = float(np.median(se))
                        row[f"{hyp}_state_err_p95"] = float(np.percentile(se, 95))
                        row[f"{hyp}_qpos_err_median"] = float(np.median(qe))
                    row[f"{hyp}_obs_err_median"] = float(np.median(oe))
                    row[f"{hyp}_obs_err_p95"] = float(np.percentile(oe, 95))
                    row[f"{hyp}_obs_err_max"] = float(np.max(oe))
                out["per_episode"].append(row)
                print(json.dumps(row), flush=True)
    out["lag"] = lag_analysis()
    return out


def lag_analysis():
    """Data-only: where does the response to a command change first appear?

    For each episode, x = first difference of the command, y = first difference of the observed
    quantity. corr(x[t], y[t+k]) for k = -2..3. If obs[t] is post-action, the response to a change in
    actions[t] appears in obs[t] - obs[t-1] (k = 0); if obs[t] is pre-action it can first appear at k = 1.
    Gripper: actions[:, 6] vs finger opening obs robot0_gripper_qpos[:,0]-[:,1].
    Arm: actions[:, 0:3] (OSC delta, base frame) vs robot0_base_to_eef_pos velocity, per axis.
    """
    import h5py
    lags = list(range(-2, 4))
    res = {}
    for task in ("pick_egg", "wipe_countertop"):
        acc = {"gripper": {k: [] for k in lags}, "arm": {k: [] for k in lags}}
        first_resp = []
        for cond in ("with", "without"):
            with h5py.File(tpath(task, cond), "r") as f:
                for demo in f["data"]:
                    g = f["data"][demo]
                    a = g["actions"][:].astype(np.float64)
                    gq = g["obs/robot0_gripper_qpos"][:].astype(np.float64)
                    opening = gq[:, 0] - gq[:, 1]
                    eef = g["obs/robot0_base_to_eef_pos"][:].astype(np.float64)
                    xs = {"gripper": [np.diff(a[:, 6])], "arm": [np.diff(a[:, j]) for j in range(3)]}
                    ys = {"gripper": [np.diff(opening)], "arm": [np.diff(np.diff(eef[:, j])) for j in range(3)]}
                    # arm: action is a velocity-like delta command -> compare its change with the change in eef velocity
                    xs["arm"] = [np.diff(a[1:, j]) for j in range(3)]
                    for kind in ("gripper", "arm"):
                        for x, y in zip(xs[kind], ys[kind]):
                            n = min(len(x), len(y))
                            x, y = x[:n], y[:n]
                            for k in lags:
                                if k >= 0:
                                    xx, yy = x[: n - k], y[k:]
                                else:
                                    xx, yy = x[-k:], y[: n + k]
                                if xx.std() > 0 and yy.std() > 0:
                                    acc[kind][k].append(float(np.corrcoef(xx, yy)[0, 1]))
                    # first-response test on isolated gripper command jumps
                    da = np.diff(a[:, 6])
                    for t in np.where(np.abs(da) > 0.3)[0] + 1:   # command changes at index t
                        if 2 <= t < len(a) - 2 and np.abs(np.diff(a[t - 2:t, 6])).max() < 1e-6:
                            d = np.diff(opening)
                            # opening change at t vs at t-1 (the step before the command change)
                            first_resp.append(dict(sign_cmd=float(np.sign(da[t - 1])),
                                                   d_open_tminus1=float(d[t - 2]), d_open_t=float(d[t - 1]),
                                                   d_open_tplus1=float(d[t])))
        res[task] = {kind: {str(k): dict(mean=float(np.mean(v)), median=float(np.median(v)), n=len(v))
                            for k, v in d.items()} for kind, d in acc.items()}
        fr = first_resp
        if fr:
            s = np.array([r["sign_cmd"] for r in fr])
            res[task]["gripper_isolated_jumps"] = dict(
                n=len(fr),
                # opening should move in the direction of the command (gripper action +1 = close -> opening decreases)
                frac_response_at_t=float(np.mean(np.sign([r["d_open_t"] for r in fr]) == -s)),
                frac_response_at_tminus1=float(np.mean(np.sign([r["d_open_tminus1"] for r in fr]) == -s)),
                median_abs_d_open_tminus1=float(np.median(np.abs([r["d_open_tminus1"] for r in fr]))),
                median_abs_d_open_t=float(np.median(np.abs([r["d_open_t"] for r in fr]))),
                median_abs_d_open_tplus1=float(np.median(np.abs([r["d_open_tplus1"] for r in fr]))),
            )
    return res


def playback_mode():
    sys.path.insert(0, HERE)
    os.environ.setdefault("PYNPUT_BACKEND", "dummy")
    import h5py
    from damagesim.robosuite.io_utils import patch_gripper_for_position_control
    patch_gripper_for_position_control()  # collection-time gripper, as in replay.py
    import playback_robocasa_portable as P
    import replay as R
    os.makedirs(RUN_DIR, exist_ok=True)
    out = {}
    for task in ("pick_egg", "wipe_countertop"):
        env = R.get_env(R.TASKS[task])
        cond, demo = "with", "demo_0"
        pb_path = os.path.join(RUN_DIR, f"official_playback_{task}_{cond}_{demo}.hdf5")
        if os.path.exists(pb_path):
            os.remove(pb_path)
        with h5py.File(tpath(task, cond), "r") as src, h5py.File(pb_path, "w") as dst:
            P.playback_episode(src, demo, env, dst)
        with h5py.File(tpath(task, cond), "r") as f, h5py.File(pb_path, "r") as p:
            g, q = f["data"][demo], p["data"][demo]
            ta, pa = g["actions"][:], q["action"][:]
            pe = q["obs/eef_pos"][:]
            te = g["obs/robot0_eef_pos"][:]
            r = dict(teleop_T=len(ta), playback_action_len=len(pa), playback_obs_len=len(pe),
                     playback_obs_keys=sorted(q["obs"].keys()),
                     action_equals_teleop_actions_shift1=bool(np.array_equal(pa, ta[1:])),
                     action_equals_teleop_actions_shift0=bool(len(pa) == len(ta) and np.array_equal(pa, ta)),
                     obs_eef_vs_teleop_same_index_maxabs=float(np.abs(pe - te).max()),
                     obs_eef_vs_teleop_shift_minus1_maxabs=float(np.abs(pe[1:] - te[:-1]).max()))
            r["contract"] = ("playback obs[t] (len T) = teleop obs[t]; playback action[t] (len T-1) = teleop actions[t+1]; "
                             "PlaybackDataset pairs obs[t] with action[t:t+chunk] -> teleop obs[t] with teleop actions[t+1:...]")
        out[task] = r
        print(task, json.dumps(r), flush=True)
    return out


def loader_mode():
    """Project .venv: this repo's RobomimicDataset on the raw teleop file (read-only, no HF download)."""
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
    import h5py
    from mip.datasets.robomimic_dataset import RobomimicDataset
    obs_keys = ["object-state", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]  # task/oopsieverse_base.yaml
    path = tpath("pick_egg", "with")
    ds = RobomimicDataset(path, horizon=16, obs_keys=obs_keys, pad_before=1, pad_after=7, abs_action=False,
                          mode="train", val_dataset_percentage=0.0, obs_steps=2, act_steps=8)
    with h5py.File(path, "r") as f:
        g = f["data/demo_0"]
        raw_obs = np.concatenate([g["obs"][k][:] for k in obs_keys], -1).astype(np.float32)
        raw_act = g["actions"][:].astype(np.float32)
    rows = []
    for idx in (60, 150, 250):          # windows inside demo_0 (T=344), away from padding
        smp = ds.sampler.sample_sequence(idx)
        o, a = smp["obs"], smp["action"]
        cur = o[1]                      # obs_steps-1: the "current" observation of the window
        hits = np.where((raw_obs == cur).all(1))[0]
        t_obs = int(hits[0]) if len(hits) == 1 else None
        # train_oopsieverse.py offline_eval / agent: executed chunk = action[obs_steps-1 : obs_steps-1+act_steps]
        chunk = a[1:9]
        same_index = t_obs is not None and np.array_equal(raw_act[t_obs:t_obs + 8], chunk)
        next_index = t_obs is not None and np.array_equal(raw_act[t_obs + 1:t_obs + 9], chunk)
        rows.append(dict(sample_index=idx, current_obs_raw_index=t_obs,
                         executed_chunk_equals_raw_actions_from_same_index=bool(same_index),
                         executed_chunk_equals_raw_actions_from_next_index=bool(next_index),
                         chunk_is_distinctive=bool(np.abs(np.diff(raw_act[t_obs:t_obs + 9, :7], axis=0)).max() > 0)))
    r = dict(file=path, obs_keys=obs_keys, windows=rows,
             verdict=("repo pairs current obs[t] with actions[t:t+8] (same raw index)"
                      if all(x["executed_chunk_equals_raw_actions_from_same_index"] and not x["executed_chunk_equals_raw_actions_from_next_index"] for x in rows if x["chunk_is_distinctive"])
                      else "inconclusive"))
    print(json.dumps(r))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sim", "playback", "loader"])
    ap.add_argument("--n-eps", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=25)
    a = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)
    res = {"sim": lambda: sim_mode(a.n_eps, a.n_steps), "playback": playback_mode, "loader": loader_mode}[a.mode]()
    json.dump(res, open(os.path.join(RUN_DIR, f"alignment_{a.mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
