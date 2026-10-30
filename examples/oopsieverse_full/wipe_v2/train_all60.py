"""Matched-condition baseline check (wipe_v2_spec.md addendum 2026-10-27): A-MIX and I-PATH on ALL 60 wipe demos.

Runs the UNMODIFIED matrix trainer (matrix/train_matrix.py: main/_train, official CFMTrainer loop) with only the data
source swapped. The module's data helpers are rebound in-process (the file is not edited):
  data_file(task, "A_MIX_train") -> runs/wipe_v2/all60/data/wipe_countertop__all60__v1.hdf5: ExternalLinks to the 60
      playback parts, members = splits_v1 train + dev + test in that order, built by data.build_subset_hdf5 (the
      mechanism used for the frozen subset files). Frozen splits and existing split files are untouched.
  mixed_train_stats -> action min/max + class vocabulary computed by the OFFICIAL PlaybackDataset methods on the
      all-60 file (cached under all60/data/_stats). The intent normaliser (I-PATH) is refitted by the matrix dataset on
      the training file, i.e. the 60 demos.
  dev split (10 dev demos) is unchanged for the dev-MSE logging; note those demos are now IN the training data.
  LEDGER -> runs/wipe_v2/all60/train/run_ledger.csv; code hash adds this file; data fingerprint covers the 60 parts.
--data40: no rebinding of the data source (frozen 40-demo train file, matrix stats); used for the 200-step
bit-identity check against runs/matrix/amix_wipe_countertop_v1_s0. All other flags are train_matrix.py's.

  python train_all60.py [--data40] --arm amix --task wipe_countertop --seed 0 --run-id amix_wipe_countertop_all60_s0 \
      --runs-root /edrive1/.../runs/wipe_v2/all60/train --workers 4 --act-ckpt --save-every 1000
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(os.path.dirname(HERE), "matrix")
BASE = os.path.join(os.path.dirname(HERE), "baseline")
for p in (MATRIX, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)
import train_matrix as TM  # noqa: E402

ALL60 = "/edrive1/lizad/oopsieverse_full_study/runs/wipe_v2/all60"
TASK = "wipe_countertop"
_orig_code_hash = TM.code_hash
_orig_data_file = TM.data_file


def all60_members():
    d = json.load(open(TM.SPLITS))["tasks"][TASK]["splits"]
    m = list(d["train"]["members"]) + list(d["dev"]["members"]) + list(d["test"]["members"])
    assert len(m) == 60 and len(set(m)) == 60
    return m


def data_file(task, which):
    if which != "A_MIX_train":
        return _orig_data_file(task, which)
    assert task == TASK
    import h5py
    from data import build_subset_hdf5
    from split_data import COND
    members = all60_members()
    out = os.path.join(ALL60, "data", f"{task}__all60__v1.hdf5")
    expect = [f"{m.split('/')[0]}/{COND[m.split('/')[1]]}/{m.split('/')[2]}" for m in members]
    if not os.path.exists(out):
        build_subset_hdf5(task, "all60", members, out_dir=os.path.join(ALL60, "data"))
    with h5py.File(out, "r") as f:
        assert json.loads(f.attrs["source_ids"]) == expect
    return out, members


def mixed_train_stats(task, cfg):
    import contextlib
    from learning.dataset import PlaybackDataset
    path, _ = data_file(task, "A_MIX_train")
    cache = os.path.join(ALL60, "data", "_stats", f"{task}_all60_v1.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    obj = PlaybackDataset.__new__(PlaybackDataset)
    obj.data_path, obj.objects_of_interest, obj.seg_mappings = path, cfg.objects_of_interest, {}
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        PlaybackDataset._build_class_vocabulary(obj)
        amin, amax = PlaybackDataset.compute_action_norm_stat(obj)
    out = {"action_min": amin.tolist(), "action_max": amax.tolist(), "class_to_id": obj.class_to_id, "source": path}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + f".tmp{os.getpid()}"
    json.dump(out, open(tmp, "w"))
    os.replace(tmp, cache)
    return out


def code_hash():
    h = hashlib.sha256(_orig_code_hash().encode())
    h.update(open(os.path.abspath(__file__), "rb").read())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    data40 = "--data40" in sys.argv
    if data40:
        sys.argv.remove("--data40")
    else:
        TM.data_file = data_file
        TM.mixed_train_stats = mixed_train_stats
    TM.code_hash = code_hash
    TM.LEDGER = os.path.join(ALL60, "train" if not data40 else "verify", "run_ledger.csv")
    TM.main()
