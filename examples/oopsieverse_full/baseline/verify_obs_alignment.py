"""Check that the rollout evaluator builds exactly the training inputs.

For a few demos: construct the evaluator env (rollout_eval.make_env), restore the demo exactly as
official playback does (ep_meta, reset, reset_from_xml_string, marker hiding, states[0]), and
  (a) step path: from states[t-1] step the playback action[t-1] (what playback did) -> obs at t;
  (b) state path: set states[t] directly -> get_observations() (what a branch-restore would see).
Build the evaluator's model inputs (Policy.frame/model_inputs, history [t-1, t]) and compare against
the official PlaybackDataset.__getitem__ sample for the same (demo, t): seg tensors (all 3 views x 2
frames, after vocabulary remap and 128 nearest-exact resize), proprio, and the policy output for a
shared init noise. Also checks use_camera_obs=False gives identical seg/proprio (speed option).
"""
import json
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rollout_eval as R  # noqa: E402
from split_data import PARTS, STUDY  # noqa: E402

TELEOP = "/edrive1/lizad/oopsieverse_full_study/data/oopsieverse-demos/paper_experiments/teleop_data/pick_egg"
DEMOS = [("with", "demo_1"), ("without", "demo_3"), ("with", "demo_2")]
TS = [1, 2, 50, 120, 200]


def main(run_dir, use_cam):
    import torch
    os.chdir(R.OOPSIE)
    from learning.dataset import PlaybackDataset
    vf = "/edrive1/lizad/oopsieverse_full_study/data/playback/_verify_obs.hdf5"
    with h5py.File(vf, "w") as f:
        g = f.create_group("data")
        for k, (c, d) in enumerate(DEMOS):
            g[f"demo_{k}"] = h5py.ExternalLink(os.path.join(PARTS, f"pick_egg__{c}__{d}.hdf5"), f"data/{d}")
    ck = sorted(os.listdir(os.path.join(run_dir, "ckpt")), key=lambda n: (n != "final.pth", -int(n.split("_")[-1].split(".")[0]) if n.startswith("step_") else 0))[0]
    pol = R.Policy(run_dir, ckpt=ck, device="cuda:0" if torch.cuda.is_available() else "cpu")
    tc = pol.meta["trainer_config"]
    ds = PlaybackDataset(data_path=vf, frame_stack=tc["frame_stack"], action_chunk_size=tc["action_chunk"],
                         seg_img_size=(tc["seg_img_size"],) * 2, normalize_action=True,
                         objects_of_interest=tc["objects_of_interest"], obs_keys=tc["obs_keys"])
    assert ds.class_to_id == pol.class_to_id, (ds.class_to_id, pol.class_to_id)
    idx = {(dk, t): i for i, (dk, t) in enumerate(ds.samples)}
    env = R.make_env("pick_egg", use_camera_obs=use_cam)
    _KEEP.append(env)  # freeing the EGL env at function exit kills the interpreter
    rows = []
    for k, (c, d) in enumerate(DEMOS):
        with h5py.File(os.path.join(TELEOP, f"{c}_live_feedback.hdf5"), "r") as tf, \
                h5py.File(os.path.join(PARTS, f"pick_egg__{c}__{d}.hdf5"), "r") as pf:
            tg = tf["data"][d]
            states = tg["states"][:]
            pacts = pf["data"][d]["action"][:]
            ep_meta = json.loads(R_remap(tg.attrs["ep_meta"]))
            xml = R_remap(tg.attrs["model_file"])
            pprop = pf["data"][d]["obs/proprio"][:]
        env.set_ep_meta(ep_meta)
        env.reset()
        env.reset_from_xml_string(xml)
        env.sim.reset()
        R.hide_markers(env)
        lut = pol.seg_lut(env.seg_id_to_class)

        def at_state(i):
            env.sim.set_state_from_flattened(states[i]); env.sim.forward(); R._sync_vel(env)
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
                eq = {key: float((seg[key][0].cpu() == s["obs"]["extero"][key]).float().mean()) for key in pol.seg_keys}
                pd = float((prop[0].cpu() - s["obs"]["proprio"]).abs().max())
                x0 = torch.randn(1, 8, pcfg_dim(pol), generator=torch.Generator().manual_seed(t))
                s_seg = {key: s["obs"]["extero"][key].unsqueeze(0).to(pol.device) for key in pol.seg_keys}
                with torch.no_grad():
                    a_eval = pol.model.generate_action(seg, prop, init_noise=x0.to(pol.device))
                    a_ds = pol.model.generate_action(s_seg, s["obs"]["proprio"].unsqueeze(0).to(pol.device), init_noise=x0.to(pol.device))
                rows.append({"demo": f"{c}/{d}", "t": t, "path": path, "seg_equal_frac": eq,
                             "proprio_maxabs": pd, "proprio_vs_hdf5_maxabs": float(np.abs(prop[0].cpu().numpy() - pprop[t]).max()),
                             "action_maxabs_eval_vs_dataset_input": float((a_eval - a_ds).abs().max()),
                             "keys_order_dataset": list(s["obs"]["extero"].keys()), "keys_order_eval": list(seg.keys())})
                print(rows[-1], flush=True)
    return rows


_KEEP = []


def pcfg_dim(pol):
    return pol.cfg.action_dim


def R_remap(s):
    sys.path.insert(0, os.path.dirname(HERE))
    from path_remap import remap_asset_paths
    return remap_asset_paths(s)


if __name__ == "__main__":
    run_dir = sys.argv[1]
    use_cam = sys.argv[2] == "cam"  # one env per process (a second EGL env in-process crashed)
    res_path = os.path.join(STUDY, "baseline/obs_alignment_check.json")
    out = json.load(open(res_path)) if os.path.exists(res_path) else {}
    out["run_dir"] = run_dir
    out[f"use_camera_obs={use_cam}"] = main(run_dir, use_cam)
    allrows = [r for k in ("use_camera_obs=True", "use_camera_obs=False") for r in out.get(k, [])]
    out["summary"] = {
        "n": len(allrows), "modes": [k for k in ("use_camera_obs=True", "use_camera_obs=False") if k in out],
        "min_seg_equal_frac": {p: min(min(r["seg_equal_frac"].values()) for r in allrows if r["path"] == p) for p in ("step", "state")},
        "max_proprio_absdiff": {p: max(r["proprio_maxabs"] for r in allrows if r["path"] == p) for p in ("step", "state")},
        "max_action_absdiff": {p: max(r["action_maxabs_eval_vs_dataset_input"] for r in allrows if r["path"] == p) for p in ("step", "state")},
        "key_order_match": all(r["keys_order_dataset"] == r["keys_order_eval"] for r in allrows),
    }
    json.dump(out, open(res_path, "w"), indent=1)
    print(json.dumps(out["summary"], indent=1))
    os._exit(0)
