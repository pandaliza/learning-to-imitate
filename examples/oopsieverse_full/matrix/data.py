"""Dataset wrappers for the policy matrix. The official PlaybackDataset (a739f99) is subclassed, not edited.

MatrixPlaybackDataset == PlaybackDataset sample-for-sample (same index, same seg preprocessing, same action
normalisation), with three additions that do not change the native sample:
  1. `_load_trajectory` keeps only the arrays __getitem__ reads (action, proprio). The raw 256^2 seg arrays it
     also cached are never read when the remapped seg cache exists (load_seg_mapping=True, always here); a
     presence marker is kept for each seg key because __getitem__ tests `obs_key in traj` first.
     `preload()` fills the cache before DataLoader workers fork, so workers share it copy-on-write.
  2. `override_norm(...)`: subset arms (A-SAFE, safe-sized mixed control) reuse the mixed-TRAIN action
     normaliser and class vocabulary (splits_v1 rules: shared normaliser across all arms).
  3. Intent arms: the normalised future-motion target is appended to `obs/proprio` (23 + n_tok*tok_dim).
     This lets the official, unmodified CFMTrainer loop carry it to policy.compute_loss(seg, state, action),
     where the intent policy splits it off. Native arms never set an intent, so their samples are unchanged.
"""

import os
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OOPSIE = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
if OOPSIE not in sys.path:
    sys.path.insert(0, OOPSIE)

from learning.dataset import PlaybackDataset  # noqa: E402

import intent_targets as IT  # noqa: E402


class MatrixPlaybackDataset(PlaybackDataset):
    def __init__(self, *args, intent_name=None, **kw):
        self.intent_name = intent_name
        self.intent_norm = None
        self.intent_raw = {}
        super().__init__(*args, **kw)

    # (1) only the arrays __getitem__ reads; identical values
    def _load_trajectory(self, demo_key):
        if demo_key in self.traj_data:
            return self.traj_data[demo_key]
        with h5py.File(self.data_path, "r") as f:
            g = f["data"][demo_key]
            traj = {"action": np.asarray(g["action"][:], dtype=np.float32)}
            for key in self.obs_keys:
                if key not in g["obs"]:
                    continue
                if key == "proprio" or key.endswith("proprio"):
                    traj[f"obs_{key}"] = g["obs"][key][:]
                else:
                    # presence marker only: __getitem__ checks `obs_key in traj`, then slices the remapped cache
                    assert key in self.remapped_seg_data[demo_key]
                    traj[f"obs_{key}"] = None
        self.traj_data[demo_key] = traj
        return traj

    def demo_keys(self):
        return sorted({d for d, _ in self.samples}, key=lambda k: int(k.split("_")[1]))

    def preload(self):
        for d in self.demo_keys():
            self._load_trajectory(d)
        return self

    # (2)
    def override_norm(self, action_min, action_max, class_to_id):
        assert class_to_id == self.class_to_id, ("vocabulary differs from mixed train", class_to_id, self.class_to_id)
        self.action_min = np.asarray(action_min, np.float32)
        self.action_max = np.asarray(action_max, np.float32)

    # (3)
    def raw_demos(self):
        out = []
        for d in self.demo_keys():
            tr = self._load_trajectory(d)
            out.append((tr["obs_proprio"], tr["action"]))
        return out

    def setup_intent(self, norm=None):
        """Compute raw targets for every native sample; fit the normaliser (train) or reuse it (dev)."""
        assert self.intent_name is not None
        demos = dict(zip(self.demo_keys(), self.raw_demos()))
        for d, (p, a) in demos.items():
            self.intent_raw[d] = IT.raw_targets(p, a)[self.intent_name].astype(np.float64)  # row i <-> t = i+1
        self.intent_norm = norm if norm is not None else IT.fit_normalizer(self.intent_name, list(demos.values()))
        self.intent_norm_t = {d: self.intent_norm.normalize(v).astype(np.float32) for d, v in self.intent_raw.items()}
        n_tok, dim = IT.TARGET_SPECS[self.intent_name]
        self.intent_shape = (n_tok, dim)
        return self.intent_norm

    def intent_of(self, demo_key, t):
        return self.intent_norm_t[demo_key][t - (self.frame_stack - 1)]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.intent_name is not None:
            it = torch.from_numpy(self.intent_of(sample["demo_key"], sample["timestep"]).reshape(-1))
            sample["obs"]["proprio"] = torch.cat([sample["obs"]["proprio"], it])
        return sample


def build_subset_hdf5(task, subset, members, out_dir="/edrive1/lizad/oopsieverse_full_study/data/playback"):
    """Like baseline/split_data.build_split_hdf5 but for a splits_v1 `subsets` entry (ExternalLinks)."""
    import json
    from split_data import COND, part_path  # baseline helper
    eps = []
    for eid in members:
        t, c, demo = eid.split("/")
        eps.append((t, COND[c], demo))
    out = os.path.join(out_dir, f"{task}__{subset}__v1.hdf5")
    tmp = out + f".tmp{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        for k, (t, c, demo) in enumerate(eps):
            assert os.path.exists(part_path(t, c, demo)), part_path(t, c, demo)
            g[f"demo_{k}"] = h5py.ExternalLink(part_path(t, c, demo), f"data/{demo}")
        f.attrs["source_ids"] = json.dumps([f"{t}/{c}/{d}" for t, c, d in eps])
        f.attrs["subset"] = subset
    os.replace(tmp, out)
    return out, eps
