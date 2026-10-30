"""Per-demo training parts from the official image playback (b1k_diag_protocol.md §1-2).

For every demo of every replay job (runs/b1k_diag/replay_img/<task>_<cond>_<nn>/playback.hdf5) writes
data/b1k_playback/parts/<task>__<cond>__demo_<i>.hdf5 with group data/demo_<i> holding only what the official
PlaybackDataset reads (+ health/task_completion for audit): action, obs/proprio, obs/cam/*/seg, obs/health,
info/obs_info, task_completion. With --shift 1 (alignment rule outcome) the pairing becomes
obs[k] -> action[k+1:...]: action := action[1:], every state-aligned array := [:T] (drops the last state).
No values are modified. Writes parts_manifest_b1k.json (sha256 per part) to the B1K output dir.
"""
import argparse
import glob
import hashlib
import json
import os

import h5py
import numpy as np

R = "/edrive1/lizad/oopsieverse_full_study/runs/b1k_diag/replay_img"
P = "/edrive1/lizad/oopsieverse_full_study/data/b1k_playback/parts"
OUT = "/edrive1/lizad/oopsieverse_full_study/reports/b1k"
STATE_KEYS = ["obs/proprio", "obs/cam/eef/seg", "obs/cam/external_sensor0/seg", "obs/cam/external_sensor1/seg",
              "obs/health", "info/obs_info", "task_completion"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", type=int, choices=[0, 1], required=True)
    ap.add_argument("--task", required=True)
    a = ap.parse_args()
    os.makedirs(P, exist_ok=True)
    man = {"shift": a.shift, "parts": {}}
    for d in sorted(glob.glob(f"{R}/{a.task}_*_[0-9][0-9]")):
        assert os.path.exists(f"{d}/DONE"), d
        tag = os.path.basename(d)
        task, cond = tag.rsplit("_", 2)[0], tag.rsplit("_", 2)[1]
        with h5py.File(f"{d}/playback.hdf5", "r") as f:
            for k in f["data"]:
                g = f["data"][k]
                T = g["action"].shape[0]
                out = f"{P}/{task}__{cond}__{k}.hdf5"
                tmp = out + ".tmp"
                with h5py.File(tmp, "w") as o:
                    og = o.create_group(f"data/{k}")
                    act = g["action"][:]
                    og.create_dataset("action", data=act[1:] if a.shift else act)
                    for key in STATE_KEYS:
                        x = g[key][:]
                        assert x.shape[0] == T + 1, (key, x.shape, T)
                        if a.shift:
                            x = x[:T]
                        if key == "info/obs_info":
                            og.create_dataset(key, data=x, dtype=g[key].dtype)
                        else:
                            og.create_dataset(key, data=x, compression=None)
                    for at, v in g.attrs.items():
                        og.attrs[at] = v
                    og.attrs["b1k_shift"] = a.shift
                    og.attrs["source"] = f"{d}/playback.hdf5:data/{k}"
                os.replace(tmp, out)
                man["parts"][f"{task}/{cond}/{k}"] = {"path": out, "sha256": sha(out), "T_actions": int(T - a.shift)}
                print(task, cond, k, T - a.shift, flush=True)
    body = json.dumps(man, indent=1, sort_keys=True)
    man["manifest_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    json.dump(man, open(f"{OUT}/parts_manifest_b1k_{a.task}.json", "w"), indent=1)
    print("parts", len(man["parts"]))


if __name__ == "__main__":
    main()
