"""B1K add_firewood matched-condition check (b1k_diag_protocol.md, Addendum 2026-10-30): A-MIX on ALL 60 demos.

Runs the UNMODIFIED matrix trainer (matrix/train_matrix.py, official CFMTrainer loop, amix arm) with only the data
source swapped, in-process (the files are not edited; same pattern as wipe_v2/train_all60.py):
  data_file("add_firewood", "A_MIX_train") -> runs/b1k/all60/data/add_firewood__all60__b1kv1.hdf5, ExternalLinks to the
      60 frozen playback parts (parts_manifest_b1k_add_firewood.json, shift 0). Members = the 60 rows of
      replay_labels_add_firewood.csv in row order (splits NOT used). Every part's sha256 is checked against the manifest
      before the file is built; the member list, part hashes and the data-file sha256 go to all60/data/all60_manifest.json.
  mixed_train_stats -> action min/max and class vocabulary of the 60-demo file (official PlaybackDataset methods,
      cached under all60/data/_stats).
  data_fingerprint, code_hash (adds this file), LEDGER -> runs/b1k/all60/train/run_ledger.csv.
  The dev helpers (10 frozen dev demos, now inside the training pool) are the 40-demo B1K ones, via train_b1k (imported,
  not edited). Dev MSE is therefore in-sample.

  python train_b1k_all60.py --build-data-only                       # verify parts, build data file + manifest
  CUDA_VISIBLE_DEVICES=1 python train_b1k_all60.py --arm amix --task add_firewood --seed 0 \
      --run-id amix_add_firewood_all60_s0 --runs-root .../runs/b1k/all60/train --workers 4 --act-ckpt --save-every 1000
"""
import contextlib
import csv
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(os.path.dirname(HERE), "matrix")
for p in (HERE, MATRIX):
    if p not in sys.path:
        sys.path.insert(0, p)
import train_matrix as TM  # noqa: E402
import train_b1k as B  # noqa: E402  (imported, not edited: 40-demo dev helpers and PARTS path)

STORE = "/edrive1/lizad/oopsieverse_full_study"
ALL60 = os.path.join(STORE, "runs/b1k/all60")
TASK = "add_firewood"
PARTS = B.PARTS
LABELS = os.path.join(STORE, "reports/b1k/replay_labels_add_firewood.csv")
MANIFEST = os.path.join(STORE, "reports/b1k/parts_manifest_b1k_add_firewood.json")
DATA_OUT = os.path.join(ALL60, "data", f"{TASK}__all60__b1kv1.hdf5")
MANI_OUT = os.path.join(ALL60, "data", "all60_manifest.json")
_orig_code_hash = TM.code_hash


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def all60_members():
    rows = list(csv.DictReader(open(LABELS)))
    assert len(rows) == 60 and all(r["task"] == TASK for r in rows)
    assert all(r["official_success"] == "True" for r in rows), "successes-only rule"
    m = [f"{r['task']}/{r['cond']}/{r['demo']}" for r in rows]
    assert len(set(m)) == 60
    return m


def part_path(member):
    t, c, d = member.split("/")
    return os.path.join(PARTS, f"{t}__{c}__{d}.hdf5")


def build_all60():
    import h5py
    members = all60_members()
    man = json.load(open(MANIFEST))["parts"]
    if os.path.exists(DATA_OUT) and os.path.exists(MANI_OUT):
        with h5py.File(DATA_OUT, "r") as f:
            assert json.loads(f.attrs["source_ids"]) == members, f"{DATA_OUT} differs from the 60-row member list"
        return DATA_OUT, members
    part_sha = {}
    for m in members:
        p = part_path(m)
        assert os.path.exists(p), p
        part_sha[m] = sha256_file(p)
        assert part_sha[m] == man[m]["sha256"] and man[m]["path"] == p, f"part hash mismatch: {m}"
    tmp = DATA_OUT + f".tmp{os.getpid()}"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        for k, m in enumerate(members):
            demo = m.split("/")[2]
            g[f"demo_{k}"] = h5py.ExternalLink(part_path(m), f"data/{demo}")
        f.attrs["source_ids"] = json.dumps(members)
        f.attrs["subset"] = "all60"
    os.replace(tmp, DATA_OUT)
    manifest = {"task": TASK, "n_demos": len(members), "members": members, "parts_sha256": part_sha,
                "data_file": DATA_OUT, "data_file_sha256": sha256_file(DATA_OUT),
                "labels_csv": LABELS, "labels_csv_sha256": sha256_file(LABELS),
                "parts_manifest": MANIFEST, "parts_manifest_sha256": sha256_file(MANIFEST),
                "splits_used": False, "successes_only_rule": "official_success True for all 60 (checked)"}
    json.dump(manifest, open(MANI_OUT + ".tmp", "w"), indent=1)
    os.replace(MANI_OUT + ".tmp", MANI_OUT)
    return DATA_OUT, members


def data_file(task, which):
    if which != "A_MIX_train":
        return B.data_file(task, which)  # dev: frozen 40-demo dev file (dev demos are inside the training pool)
    assert task == TASK
    return build_all60()


def data_fingerprint(paths_members):
    h = hashlib.sha256()
    h.update(open(MANI_OUT, "rb").read())
    for path, members in paths_members:
        h.update(os.path.basename(path).encode())
        for m in members:
            st = os.stat(part_path(m))
            h.update(f"{m}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def mixed_train_stats(task, cfg):
    """Action min/max + class vocabulary of the 60-demo file, computed by the OFFICIAL PlaybackDataset methods."""
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


TM.data_file = data_file
TM.data_fingerprint = data_fingerprint
TM.mixed_train_stats = mixed_train_stats
TM.code_hash = code_hash
TM.LEDGER = os.path.join(ALL60, "train", "run_ledger.csv")

if __name__ == "__main__":
    if "--build-data-only" in sys.argv:
        path, members = build_all60()
        print("DATA", path, len(members), "members; sha256", sha256_file(path), "manifest", MANI_OUT)
        sys.exit(0)
    a = sys.argv
    if "--runs-root" not in a:
        a += ["--runs-root", os.path.join(ALL60, "train")]
    if "--run-id" not in a:
        seed = a[a.index("--seed") + 1]
        a += ["--run-id", f"amix_{TASK}_all60_s{seed}"]
    TM.main()
