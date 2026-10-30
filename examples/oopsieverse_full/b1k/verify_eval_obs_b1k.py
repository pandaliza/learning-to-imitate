"""Evaluator-equivalence check (B1K): are the evaluation-env model inputs identical to the training samples?

python verify_eval_obs_b1k.py TASK RUN_DIR OUT_JSON
1. In the evaluation env (b1k_common.build_env, cameras on), for 2 training demos: load teleop state s[k+5]
   (shelve_item: + Temperature=23.0 layout shim, as in replay_b1k.py) and env.step(teleop a[k+4]) exactly like
   official playback step k+4, for k in a few indices; compare the evaluator's model inputs (B1KPolicy.frame:
   remapped seg at 256^2 + proprio) with the training part's obs[k] remapped by the official PlaybackDataset.
2. On 5 dev bank resets: fraction of non-'unknown' pixels per view and the set of vocabulary classes visible.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b1k_common as C  # noqa: E402

task, run_dir, out = sys.argv[1:4]
C.kit_args_from_env()
env, cfg, mod, tb = C.build_env(task, cameras=True)
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch as th  # noqa: E402
import omnigibson as og  # noqa: E402
import eval_b1k as E  # noqa: E402

g = int(os.environ["OMNIGIBSON_GPU_ID"])
pol = E.B1KPolicy(run_dir, "final.pth" if os.path.exists(os.path.join(run_dir, "ckpt/final.pth")) else
                  sorted(os.listdir(os.path.join(run_dir, "ckpt")))[-1], f"cuda:{g}")
from learning.dataset import PlaybackDataset  # noqa: E402

meta = pol.meta
res = {"task": task, "run_dir": run_dir, "class_to_id": pol.class_to_id, "demos": [], "bank": []}
D = "/edrive1/lizad/oopsieverse_full_study/data/oopsieverse-demos/paper_experiments/teleop_data"
rt_size = int(og.sim.dump_state(serialized=True).numel())
res["runtime_state_size"] = rt_size
members = meta["train_episodes"][:2]
for m in members:
    t, c, demo = m.split("/")
    part = f"/edrive1/lizad/oopsieverse_full_study/data/b1k_playback/parts/{t}__{c}__{demo}.hdf5"
    ds = PlaybackDataset.__new__(PlaybackDataset)
    ds.data_path, ds.objects_of_interest, ds.seg_mappings = part, meta["trainer_config"]["objects_of_interest"], {}
    ds.obs_keys = pol.obs_keys
    PlaybackDataset._build_class_vocabulary(ds)
    ds.class_to_id = pol.class_to_id  # training vocabulary (mixed train)
    ds.load_seg_mapping = True
    PlaybackDataset._pre_process_global_class_mapping(ds)
    shift = int(h5py.File(part, "r")[f"data/{demo}"].attrs["b1k_shift"])
    tf = h5py.File(f"{D}/{t}/{'with' if c == 'fb' else 'without'}_live_feedback.hdf5", "r")["data"][demo]
    st_all, sz, act = tf["state"][()], tf["state_size"][()], tf["action"][()]
    prop_part = h5py.File(part, "r")[f"data/{demo}/obs/proprio"][()]
    rows = []
    for k in (3, 20, 40):
        s = th.as_tensor(st_all[k + 5][: int(sz[k + 5])])
        if s.numel() == rt_size - 1:
            s = th.cat([s[:-15], th.tensor([23.0]), s[-15:]])
        if s.numel() != rt_size:
            rows.append({"k": k, "skipped": f"state size {s.numel()} != runtime {rt_size}"})
            continue
        og.sim.load_state(s, serialized=True)
        og.sim.load_state(s, serialized=True)
        obs, _, _, _, info = env.step(th.as_tensor(act[k + 4]))
        segs, prop, cover = pol.frame(obs, info["obs_info"])
        r = {"k": k, "proprio_max_abs_diff": float(np.abs(prop - prop_part[k]).max()), "views": {}}
        for key in pol.seg_keys:
            ref = ds.remapped_seg_data[demo][key][k]
            r["views"][key] = {"pixel_agree": float((segs[key] == ref).mean()), "eval_nonunknown": cover[key],
                               "train_nonunknown": float((ref > 0).mean())}
        rows.append(r)
        print(m, r, flush=True)
    res["demos"].append({"episode": m, "shift": shift, "rows": rows})
bank, ents = E.load_bank(task, "dev")
for i in range(5):
    obs, oi, rinfo = E.apply_reset(env, cfg, mod, tb, ents[i], task)
    segs, prop, cover = pol.frame(obs, oi)
    vis = {k: sorted({meta and [n for n, j in pol.class_to_id.items() if j == v][0] for v in np.unique(segs[k])})
           for k in segs}
    res["bank"].append({"index": i, "reset_check": rinfo, "nonunknown": cover, "classes_visible": vis})
    print("bank", i, cover, vis, rinfo, flush=True)
json.dump(res, open(out, "w"), indent=1, default=float)
print("VERIFY_DONE", flush=True)
og.shutdown()
