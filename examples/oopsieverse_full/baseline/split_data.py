"""Split handling and split-filtered playback HDF5s for the official PlaybackDataset.

The official trainer takes ONE playback HDF5 (config data_path). We build, per (task, split, split
version), a small HDF5 whose data/demo_k entries are h5py ExternalLinks into the per-demo playback
parts at <storage>/data/playback/parts/<task>__<cond>__demo_<i>.hdf5. No data is copied or changed.
The mapping demo_k -> source id is stored in attrs.

Split sources (episode ids "pick_egg/fb/demo_3" with fb = with_live_feedback, nofb = without):
  - frozen: outputs/oopsieverse_full_study/roster/splits_v1.json  (owned by the roster agent)
  - draft : outputs/oopsieverse_full_study/catalog/splits_draft.json (SMOKE ONLY)
"""

import hashlib
import json
import os

import h5py

REPO = "/scr/lizad/personal/learning-to-imitate"
STUDY = os.path.join(REPO, "outputs/oopsieverse_full_study")
STORAGE = "/edrive1/lizad/oopsieverse_full_study"
PARTS = os.path.join(STORAGE, "data/playback/parts")
SPLIT_FILES = {
    "v1": os.path.join(STUDY, "roster/splits_v1.json"),
    "draft": os.path.join(STUDY, "catalog/splits_draft.json"),
}
COND = {"fb": "with", "nofb": "without", "with": "with", "without": "without"}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _members_from_task_entry(entry, split):
    """Accept the draft format ({groups:[{split, members}]}) and simple variants
    ({splits:{train:[ids]}} or {train:[ids]})."""
    if "groups" in entry:
        out = []
        for g in entry["groups"]:
            if g["split"] == split:
                out.extend(g["members"])
        return out
    if "splits" in entry and split in entry["splits"]:
        v = entry["splits"][split]
        return v["members"] if isinstance(v, dict) else list(v)
    if split in entry:
        v = entry[split]
        return v["members"] if isinstance(v, dict) else list(v)
    raise KeyError(f"cannot find split {split} in split entry keys {list(entry)}")


def load_split(task, split, version):
    p = SPLIT_FILES[version]
    d = json.load(open(p))
    entry = d[task] if task in d else d["tasks"][task]
    ids = _members_from_task_entry(entry, split)
    parsed = []
    for eid in ids:
        t, c, demo = eid.split("/")
        parsed.append((t, COND[c], demo))
    return parsed, {"split_file": p, "split_sha256": sha256_file(p), "version": version,
                    "status": entry.get("status", d.get("status", "?"))}


def part_path(task, cond, demo):
    return os.path.join(PARTS, f"{task}__{cond}__{demo}.hdf5")


def build_split_hdf5(task, split, version, out_dir=os.path.join(STORAGE, "data/playback")):
    eps, meta = load_split(task, split, version)
    out = os.path.join(out_dir, f"{task}__{split}__{version}.hdf5")
    missing = [part_path(*e) for e in eps if not os.path.exists(part_path(*e))]
    assert not missing, f"missing playback parts: {missing[:5]}"
    tmp = out + ".tmp"
    with h5py.File(tmp, "w") as f:
        g = f.create_group("data")
        src_ids = []
        for k, (t, c, demo) in enumerate(eps):
            g[f"demo_{k}"] = h5py.ExternalLink(part_path(t, c, demo), f"data/{demo}")
            src_ids.append(f"{t}/{c}/{demo}")
        f.attrs["source_ids"] = json.dumps(src_ids)
        f.attrs["split_meta"] = json.dumps(meta)
    os.replace(tmp, out)
    return out, eps, meta
