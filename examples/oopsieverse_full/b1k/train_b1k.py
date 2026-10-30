"""B1K diagnostic-arm trainer: runs matrix/train_matrix.py UNCHANGED (same arms, models.py, data.py,
intent_targets.py, official CFMTrainer loop) with B1K constants swapped in:
  CONFIGS  -> official learning/configs/{add_firewood,shelve_item}.yaml
  SPLITS   -> outputs/oopsieverse_full_study/b1k/splits_b1k_v1_<task>.json (frozen, one per task)
  data     -> ExternalLink split/subset files over prepared per-demo playback parts
              (/edrive1/.../data/b1k_playback/parts/<task>__<cond>__demo_<i>.hdf5, see prepare_b1k_parts.py)
  RUNS / LEDGER -> runs/b1k_diag/train, outputs/.../b1k/run_ledger_b1k.csv
Usage (RoboCasa study env): CUDA_VISIBLE_DEVICES=g $PY train_b1k.py --arm amix --task add_firewood --seed 0 --workers 4
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(os.path.dirname(HERE), "matrix")
sys.path.insert(0, MATRIX)
import train_matrix as TM  # noqa: E402

STORE = "/edrive1/lizad/oopsieverse_full_study"
PARTS = os.path.join(STORE, "data/b1k_playback/parts")
_TASK = sys.argv[sys.argv.index("--task") + 1] if "--task" in sys.argv else "add_firewood"
SPLITS_B1K = f"/edrive1/lizad/oopsieverse_full_study/reports/b1k/splits_b1k_v1_{_TASK}.json"

TM.CONFIGS = {"add_firewood": "learning/configs/add_firewood.yaml", "shelve_item": "learning/configs/shelve_item.yaml"}
TM.SPLITS = SPLITS_B1K
TM.RUNS = os.path.join(STORE, "runs/b1k_diag/train")
TM.LEDGER = "/edrive1/lizad/oopsieverse_full_study/reports/b1k/run_ledger_b1k.csv"


def part_path(task, cond, demo):
    return os.path.join(PARTS, f"{task}__{cond}__{demo}.hdf5")


def split_members(task, which):
    d = json.load(open(SPLITS_B1K))["tasks"][task]
    if which in ("train", "dev", "test"):
        return list(d["splits"][which]["members"])
    return list(d["subsets"][which]["members"])


def data_file(task, which):
    import h5py
    members = split_members(task, which)
    if which == "A_MIX_train":
        assert members == split_members(task, "train")
        which = "train"
    out = os.path.join(STORE, "data/b1k_playback", f"{task}__{which}__b1kv1.hdf5")
    if os.path.exists(out):
        with h5py.File(out, "r") as f:
            assert json.loads(f.attrs["source_ids"]) == members, f"{out} differs from splits_b1k_v1"
        return out, members
    tmp = out + f".tmp{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        for k, m in enumerate(members):
            t, c, demo = m.split("/")
            p = part_path(t, c, demo)
            assert os.path.exists(p), p
            g[f"demo_{k}"] = h5py.ExternalLink(p, f"data/{demo}")
        f.attrs["source_ids"] = json.dumps(members)
        f.attrs["subset"] = which
    os.replace(tmp, out)
    return out, members


def data_fingerprint(paths_members):
    h = hashlib.sha256()
    h.update(json.load(open(SPLITS_B1K))["content_sha256_excluding_this_field"].encode())
    for path, members in paths_members:
        h.update(os.path.basename(path).encode())
        for m in members:
            t, c, d = m.split("/")
            st = os.stat(part_path(t, c, d))
            h.update(f"{m}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


TM.split_members = split_members
TM.data_file = data_file
TM.data_fingerprint = data_fingerprint

_orig_code_hash = TM.code_hash


def code_hash():
    h = hashlib.sha256(_orig_code_hash().encode())
    h.update(open(os.path.abspath(__file__), "rb").read())
    return h.hexdigest()[:16]


TM.code_hash = code_hash

if __name__ == "__main__":
    if "--run-id" not in sys.argv:
        a = sys.argv
        arm, task, seed = a[a.index("--arm") + 1], a[a.index("--task") + 1], a[a.index("--seed") + 1]
        lab = a[a.index("--label") + 1] if "--label" in a else "final"
        ns = a[a.index("--num-steps") + 1] if "--num-steps" in a else "30000"
        sys.argv += ["--run-id", f"{arm}_{task}_b1kv1_s{seed}" + ("" if lab == "final" else f"__{lab}{ns}")]
    TM.main()
