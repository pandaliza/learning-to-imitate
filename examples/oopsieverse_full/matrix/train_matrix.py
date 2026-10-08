"""Policy-matrix trainer (spec section 8/9): one run = (arm, task, seed). Wraps the OFFICIAL CFMTrainer loop.

Arms
  amix      A-MIX       mixed train split (splits_v1 A_MIX_train)          official CFMPolicy
  asafe     A-SAFE      A_SAFE_train subset, mixed-train normaliser        official CFMPolicy
  ssmix     safe-sized mixed control (safe_sized_mixed_control_train)      official CFMPolicy
  imean     I-MEAN      A_MIX_train, legacy mean future EEF pose           IntentCFMPolicy(condition, 1x7)
  ipath     I-PATH      A_MIX_train, 4 ordered current-frame poses (6D)    IntentCFMPolicy(condition, 4x9)
  ipathg    I-PATH-G    A_MIX_train, I-PATH + gripper commands             IntentCFMPolicy(condition, 4x10)
  cmean     canonical mean-pose control (current frame, 6D)                IntentCFMPolicy(condition, 1x9)
  auxpathg  AUX-PATH-G  A_MIX_train, native action flow + I-PATH-G co-prediction  IntentCFMPolicy(aux, 4x10)

The native training math (official CFMTrainer.train loop, AdamW, clip, batch, steps, Beta(1.5,1), Euler-16) is
not modified. For native arms the code path is baseline/train_amix.py's; this is verified bit-identical
(arm_verification.md). Wrapper additions: split/subset data files, shared mixed-train normaliser for subsets,
optional DataLoader worker processes (same sampler, same batches; verified), JSONL logging, dev metrics,
checkpoint sidecar, ledger, run keys, resume.

Usage:
  CUDA_VISIBLE_DEVICES=5 python train_matrix.py --arm ipath --task pick_egg --seed 0 [--num-steps 500 --label smoke]
"""

import argparse
import contextlib
import csv
import datetime
import fcntl
import glob
import hashlib
import json
import os
import random
import socket
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
sys.path.insert(0, HERE)
sys.path.insert(0, BASE)
OOPSIE = "/edrive1/lizad/oopsieverse_full_study/src/oopsieverse"
sys.path.insert(0, OOPSIE)
REPO = "/scr/lizad/personal/learning-to-imitate"
STUDY = os.path.join(REPO, "outputs/oopsieverse_full_study")
SPLITS = os.path.join(STUDY, "roster/splits_v1.json")
STORAGE = "/edrive1/lizad/oopsieverse_full_study"
RUNS = os.path.join(STORAGE, "runs/matrix")
LEDGER = os.path.join(STUDY, "matrix/run_ledger.csv")
CONFIGS = {"pick_egg": "learning/configs/pick_egg.yaml", "wipe_countertop": "learning/configs/wipe_counter.yaml"}
OFFICIAL_SHA = "a739f99e94f32c3229ae6932299e2e0e8c065480"

ARMS = {
    "amix": {"subset": "A_MIX_train", "intent": None, "mode": None, "label": "A-MIX"},
    "asafe": {"subset": "A_SAFE_train", "intent": None, "mode": None, "label": "A-SAFE"},
    "ssmix": {"subset": "safe_sized_mixed_control_train", "intent": None, "mode": None, "label": "safe-sized mixed control"},
    "imean": {"subset": "A_MIX_train", "intent": "imean", "mode": "condition", "label": "I-MEAN"},
    "ipath": {"subset": "A_MIX_train", "intent": "ipath", "mode": "condition", "label": "I-PATH"},
    "ipathg": {"subset": "A_MIX_train", "intent": "ipathg", "mode": "condition", "label": "I-PATH-G"},
    "cmean": {"subset": "A_MIX_train", "intent": "cmean", "mode": "condition", "label": "canonical mean-pose control"},
    "auxpathg": {"subset": "A_MIX_train", "intent": "ipathg", "mode": "aux", "label": "AUX-PATH-G"},
}
LEDGER_FIELDS = ["time", "run_id", "attempt", "event", "arm", "task", "seed", "label", "num_steps", "step",
                 "wall_s", "gpu", "host", "pid", "workers", "run_key", "data_hash", "config_hash", "code_hash",
                 "run_dir", "log", "note"]


# ───────────────────────────── hashing / ledger ─────────────────────────────

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_hash():
    # only files that determine training (evaluation/verification/launcher scripts are excluded)
    files = [os.path.join(HERE, f) for f in ("train_matrix.py", "models.py", "data.py", "intent_targets.py")] + \
        [os.path.join(BASE, "train_amix.py"), os.path.join(BASE, "split_data.py")]
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    h.update(OFFICIAL_SHA.encode())
    return h.hexdigest()[:16]


def ledger_append(row):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    row = {k: row.get(k, "") for k in LEDGER_FIELDS}
    row["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LEDGER, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        w = csv.DictWriter(f, fieldnames=LEDGER_FIELDS)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


# ───────────────────────────── data files ─────────────────────────────

def split_members(task, which):
    d = json.load(open(SPLITS))["tasks"][task]
    if which in ("train", "dev", "test"):
        return list(d["splits"][which]["members"])
    return list(d["subsets"][which]["members"])


def data_file(task, which):
    """Split/subset HDF5 of ExternalLinks. A_MIX_train == the train split file the baseline used."""
    from split_data import build_split_hdf5
    from data import build_subset_hdf5
    members = split_members(task, which)
    if which == "A_MIX_train":
        assert members == split_members(task, "train"), "A_MIX_train must equal the train split"
        which = "train"
    out = os.path.join(STORAGE, "data/playback", f"{task}__{which}__v1.hdf5")
    import h5py
    from split_data import COND
    expect = []
    for m in members:
        t_, c_, d_ = m.split("/")
        expect.append(f"{t_}/{COND[c_]}/{d_}")
    if os.path.exists(out):
        with h5py.File(out, "r") as f:
            have = json.loads(f.attrs["source_ids"])
        assert have == expect, f"{out} source_ids differ from splits_v1"
        return out, members
    if which in ("train", "dev"):
        build_split_hdf5(task, which, "v1")
    else:
        build_subset_hdf5(task, which, members)
    return data_file(task, which)


def data_fingerprint(paths_members):
    """sha256 over splits_v1 content hash, member ids, and each playback part's (size, mtime_ns)."""
    from split_data import part_path, COND
    h = hashlib.sha256()
    h.update(json.load(open(SPLITS))["content_sha256_excluding_this_field"].encode())
    for path, members in paths_members:
        h.update(os.path.basename(path).encode())
        for m in members:
            t, c, d = m.split("/")
            p = part_path(t, COND[c], d)
            st = os.stat(p)
            h.update(f"{m}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def mixed_train_stats(task, cfg):
    """Action min/max + class vocabulary of the mixed TRAIN split, computed by the OFFICIAL PlaybackDataset
    methods (compute_action_norm_stat, _build_class_vocabulary) without the seg preprocessing."""
    from learning.dataset import PlaybackDataset
    path, _ = data_file(task, "train")
    cache = os.path.join(RUNS, "_stats", f"{task}_mixed_train_v1.json")
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


# ───────────────────────────── logging shim ─────────────────────────────

def make_shim(path, policy):
    from train_amix import JsonlWandb

    class Shim(JsonlWandb):
        def log(self, d):
            if "train/loss" in d and hasattr(policy, "pop_component_means"):
                d = dict(d, **policy.pop_component_means())
            super().log(d)
    return Shim(path)


# ───────────────────────────── main ─────────────────────────────

def build_run(args):
    import learning.train_eval.cfm_trainer as T
    from learning.models import CFMPolicy, PolicyConfig
    from data import MatrixPlaybackDataset
    from models import IntentCFMPolicy
    import intent_targets as IT

    arm = ARMS[args.arm]
    raw = T.load_yaml_config(CONFIGS[args.task])
    train_path, train_members = data_file(args.task, arm["subset"])
    dev_path, dev_members = data_file(args.task, "dev")
    raw["data_path"] = train_path
    raw["save_dir"] = os.path.join(args.run_dir, "ckpt")
    raw["use_wandb"] = True
    raw["wandb_run_name"] = args.run_id
    raw["num_steps"] = args.num_steps
    raw["save_every"] = args.save_every
    if args.device:
        raw["device"] = args.device
    cfg = T.trainer_config_from_mapping(raw)
    return T, CFMPolicy, PolicyConfig, MatrixPlaybackDataset, IntentCFMPolicy, IT, arm, cfg, train_path, train_members, dev_path, dev_members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--task", default="pick_egg", choices=sorted(CONFIGS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=10000)
    ap.add_argument("--label", default="final", choices=["final", "smoke", "bench", "verify"])
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes (0 = official)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default=RUNS)
    ap.add_argument("--no-dev-eval", action="store_true")
    ap.add_argument("--dev-intent-batches", type=int, default=5)
    ap.add_argument("--record-batches", type=int, default=0, help="verification: record first N batches' (demo,t)")
    ap.add_argument("--max-wall-s", type=float, default=None, help="bench: stop after this wall time")
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--device", default=None, help="override the yaml device (cpu only for smoke when GPUs are full)")
    ap.add_argument("--act-ckpt", action="store_true",
                    help="activation checkpointing of seg_encoder.conv_layers (memory only; verified bit-identical)")
    ap.add_argument("--adopt-existing", default=None, help="register an existing bit-identical-recipe run dir")
    args = ap.parse_args()
    if args.label == "final":
        assert args.num_steps == 30000, "final runs use the declared 30k budget"
    args.run_id = args.run_id or (f"{args.arm}_{args.task}_v1_s{args.seed}" + ("" if args.label == "final" else f"__{args.label}{args.num_steps}"))
    args.run_dir = os.path.join(args.runs_root, args.run_id)
    os.makedirs(args.run_dir, exist_ok=True)
    lock_f = open(os.path.join(args.run_dir, "LOCK"), "a+")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("LOCKED: another process is running", args.run_id)
        sys.exit(75)
    if not args.adopt_existing:
        os.makedirs(os.path.join(args.run_dir, "ckpt"), exist_ok=True)

    os.chdir(OOPSIE)
    (T, CFMPolicy, PolicyConfig, MatrixPlaybackDataset, IntentCFMPolicy, IT, arm, cfg,
     train_path, train_members, dev_path, dev_members) = build_run(args)

    # ── run key (data/config/code hashes) and resumability ──
    d_hash = data_fingerprint([(train_path, train_members), (dev_path, dev_members)])
    conf = {k: v for k, v in vars(cfg).items() if k not in ("save_dir", "wandb_run_name", "data_path", "save_every")}
    conf.update({"arm": args.arm, "arm_spec": arm, "seed": args.seed, "intent_nfe": 16 if arm["intent"] else None,
                 "intent_loss_weight": 1.0 if arm["intent"] else None})
    c_hash = sha256_bytes(json.dumps(conf, sort_keys=True, default=str).encode())[:16]
    k_hash = code_hash()
    run_key = sha256_bytes(f"{d_hash}|{c_hash}|{k_hash}".encode())[:16]
    if args.adopt_existing:
        adopt(args, cfg, conf, train_path, train_members, run_key, d_hash, c_hash, k_hash)
        return
    done_p = os.path.join(args.run_dir, "DONE.json")
    key_p = os.path.join(args.run_dir, "RUN_KEY")
    if os.path.exists(done_p) and json.load(open(done_p)).get("run_key") == run_key:
        print("ALREADY_DONE", args.run_id, run_key)
        return
    if os.path.exists(key_p) and open(key_p).read().strip() != run_key:
        raise SystemExit(f"REFUSE: {args.run_dir} holds a different run key ({open(key_p).read().strip()} != {run_key}); "
                         "data/config/code changed. Move it aside explicitly.")
    open(key_p, "w").write(run_key)
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt", "step_*.pth")), key=lambda p: int(p.split("_")[-1][:-4]))
    resume_from = None if (args.no_resume or not ckpts) else ckpts[-1]
    attempt = 1 + sum(1 for _ in glob.glob(os.path.join(args.run_dir, "attempt_*.json")))
    base_row = {"run_id": args.run_id, "attempt": attempt, "arm": args.arm, "task": args.task, "seed": args.seed,
                "label": args.label, "num_steps": args.num_steps, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "host": socket.gethostname(), "pid": os.getpid(), "workers": args.workers, "run_key": run_key,
                "data_hash": d_hash, "config_hash": c_hash, "code_hash": k_hash, "run_dir": args.run_dir,
                "log": os.environ.get("MATRIX_LOG", "")}
    json.dump({"attempt": attempt, "start": time.time(), "resume_from": resume_from, "pid": os.getpid()},
              open(os.path.join(args.run_dir, f"attempt_{attempt}.json"), "w"))
    if not args.no_ledger:
        ledger_append(dict(base_row, event="start", note=f"resume_from={os.path.basename(resume_from)}" if resume_from else ""))

    import signal

    def _term(signum, frame):
        raise SystemExit("SIGTERM from launcher")
    signal.signal(signal.SIGTERM, _term)
    t_wall0 = time.time()
    try:
        _train(args, T, CFMPolicy, PolicyConfig, MatrixPlaybackDataset, IntentCFMPolicy, IT, arm, cfg,
               train_path, train_members, dev_path, dev_members, run_key, d_hash, c_hash, k_hash, resume_from)
    except BaseException as e:
        msg = (type(e).__name__ + ": " + str(e))[:300].replace("\n", " ")
        if isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower():
            ev = "infra_retry_oom"      # infrastructure (shared-GPU memory), not a result
        elif isinstance(e, SystemExit) and "SIGTERM" in str(e):
            ev = "preempted"            # stopped by the launcher (GPU headroom rule); resumes from checkpoint
        else:
            ev = "failed"
        if not args.no_ledger:
            ledger_append(dict(base_row, event=ev, wall_s=round(time.time() - t_wall0, 1), note=msg))
        traceback.print_exc()
        if ev == "infra_retry_oom":
            sys.exit(86)
        raise
    if not args.no_ledger:
        ledger_append(dict(base_row, event="completed", step=args.num_steps, wall_s=round(time.time() - t_wall0, 1)))


def _train(args, T, CFMPolicy, PolicyConfig, MatrixPlaybackDataset, IntentCFMPolicy, IT, arm, cfg,
           train_path, train_members, dev_path, dev_members, run_key, d_hash, c_hash, k_hash, resume_from):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    t_data = time.time()
    kw = dict(frame_stack=cfg.frame_stack, action_chunk_size=cfg.action_chunk,
              seg_img_size=(cfg.seg_img_size, cfg.seg_img_size), normalize_action=cfg.normalize_action,
              objects_of_interest=cfg.objects_of_interest, obs_keys=cfg.obs_keys)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        ds = MatrixPlaybackDataset(data_path=cfg.data_path, intent_name=arm["intent"], **kw)
    stats = mixed_train_stats(args.task, cfg)
    if arm["subset"] != "A_MIX_train":
        ds.override_norm(stats["action_min"], stats["action_max"], stats["class_to_id"])
    else:
        assert np.array_equal(ds.action_min, np.asarray(stats["action_min"], np.float32))
        assert np.array_equal(ds.action_max, np.asarray(stats["action_max"], np.float32))
    intent_norm = None
    if arm["intent"]:
        intent_norm = ds.setup_intent()
    ds.preload()
    dev = None
    if not args.no_dev_eval:
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            dev = MatrixPlaybackDataset(data_path=dev_path, intent_name=arm["intent"], **kw)
        assert dev.class_to_id == ds.class_to_id
        dev.action_min, dev.action_max = ds.action_min, ds.action_max
        if arm["intent"]:
            dev.setup_intent(norm=intent_norm)
    t_data = time.time() - t_data

    pcfg = PolicyConfig(
        state_dim=ds.state_dim, action_dim=ds.action_dim, num_seg_views=ds.num_seg_views,
        num_seg_classes=max(ds.num_seg_classes, 1), frame_stack=cfg.frame_stack,
        action_chunk_size=cfg.action_chunk, seg_img_height=cfg.seg_img_size, seg_img_width=cfg.seg_img_size,
        action_min=ds.action_min if cfg.normalize_action else None,
        action_max=ds.action_max if cfg.normalize_action else None)
    if arm["intent"] is None:
        policy = CFMPolicy(pcfg)
    else:
        n_tok, tok_dim = IT.TARGET_SPECS[arm["intent"]]
        # extra modules are initialised under a forked RNG so the global CPU stream (data order) == A-MIX
        policy = IntentCFMPolicy(pcfg, n_tok, tok_dim, arm["mode"], intent_seed=args.seed)
    pc = policy.get_parameter_count()
    if args.act_ckpt:
        enable_activation_checkpointing(policy)

    shim = make_shim(os.path.join(args.run_dir, "metrics.jsonl"), policy)
    T.wandb = shim
    T.WANDB_AVAILABLE = True

    extra_eval = make_intent_dev_eval(policy, dev, args) if (arm["intent"] and dev is not None) else None

    class Trainer(T.CFMTrainer):
        @torch.no_grad()
        def evaluate(self):
            v = super().evaluate()  # official: train set
            if dev is not None:
                saved = self.eval_dataset
                self.eval_dataset = dev
                d = super().evaluate()
                self.eval_dataset = saved
                rec = {"eval/dev_mse_loss": float(d), "eval/train_mse_loss": float(v)}
                if extra_eval is not None:
                    rec.update(extra_eval())
                if self.wandb_run:
                    T.wandb.log(rec)
            return v

    meta = {
        "run_id": args.run_id, "label": args.label, "arm": args.arm, "arm_spec": arm, "task": args.task,
        "seed": args.seed, "run_key": run_key, "data_hash": d_hash, "config_hash": c_hash, "code_hash": k_hash,
        "train_hdf5": train_path, "dev_hdf5": dev_path, "train_episodes": train_members, "dev_episodes": dev_members,
        "splits_v1": SPLITS, "trainer_config": {k: v for k, v in vars(cfg).items()},
        "policy_config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(pcfg).items()},
        "class_to_id": ds.class_to_id, "action_min": ds.action_min.tolist(), "action_max": ds.action_max.tolist(),
        "normaliser_source": "mixed train split (splits_v1 train), shared by all arms",
        "n_train_samples": len(ds), "n_train_episodes": len(train_members), "n_dev_samples": len(dev) if dev else 0,
        "repeat_exposure_epochs": (args.num_steps + 1) * cfg.batch_size / len(ds),
        "intent": None if not arm["intent"] else {
            "name": arm["intent"], "mode": arm["mode"], "n_tok": policy.n_tok, "tok_dim": policy.tok_dim,
            "normaliser": intent_norm.to_dict(), "intent_nfe": 16, "intent_prior": "N(0, I)",
            "loss_weight": 1.0, "teacher_forcing": "always GT intent for the action flow (no sampled-intent mixing)",
            "path_ks": list(IT.PATH_KS), "H": IT.H},
        "parameter_count": pc, "dataset_build_s": t_data, "workers": args.workers,
        "activation_checkpointing": bool(args.act_ckpt),
        "official_code": {"repo": OOPSIE, "sha": OFFICIAL_SHA},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "precision": "fp32 (official; no autocast)", "ema": "none (official)", "resume_from": resume_from,
    }
    meta_p = os.path.join(args.run_dir, "meta.json")
    json.dump(meta, open(meta_p + ".tmp", "w"), indent=1, default=str)
    os.replace(meta_p + ".tmp", meta_p)

    trainer = Trainer(config=cfg, policy=policy, train_dataset=ds, args_dict=meta["trainer_config"])
    if args.workers > 0:
        from torch.utils.data import DataLoader
        trainer.train_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=args.workers,
                                          drop_last=True, persistent_workers=False, multiprocessing_context="fork")
    if args.record_batches:
        install_batch_recorder(trainer, args.record_batches, os.path.join(args.run_dir, "batch_record.json"))
    if args.max_wall_s:
        install_wall_limit(trainer, args.max_wall_s, os.path.join(args.run_dir, "bench.json"))
    resume_step = 0
    if resume_from:
        trainer.load_checkpoint(resume_from)
        resume_step = int(os.path.basename(resume_from).split("_")[-1][:-4]) + 1
    t0 = time.time()
    try:
        trainer.train(resume_step=resume_step)
    except StopBench:
        pass
    meta["train_wall_s"] = time.time() - t0
    meta["max_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    json.dump(meta, open(meta_p + ".tmp", "w"), indent=1, default=str)
    os.replace(meta_p + ".tmp", meta_p)
    if not args.max_wall_s:
        json.dump({"run_key": run_key, "time": time.time(), "train_wall_s": meta["train_wall_s"]},
                  open(os.path.join(args.run_dir, "DONE.json"), "w"))
    print("TRAIN_DONE", args.run_id, meta["train_wall_s"])


def adopt(args, cfg, conf, train_path, train_members, run_key, d_hash, c_hash, k_hash):
    """Register an existing native run (baseline/train_amix.py) whose recipe is identical to this arm's. Checks the
    recorded configuration; bit-identity of the code path is established separately (arm_verification.md)."""
    src = os.path.abspath(args.adopt_existing)
    m = json.load(open(os.path.join(src, "meta.json")))
    assert ARMS[args.arm]["intent"] is None and args.label == "final"
    tc = dict(m["trainer_config"])
    mine = {k: v for k, v in vars(cfg).items()}
    diffs = {k: (tc.get(k), mine[k]) for k in mine if tc.get(k) != mine[k] and k not in ("save_dir", "wandb_run_name", "save_every")}
    assert not diffs, f"recipe differs: {diffs}"
    assert int(m["seed"]) == args.seed and m["task"] == args.task and m["train_hdf5"] == train_path
    assert [e.replace("/with/", "/fb/").replace("/without/", "/nofb/") for e in m["train_episodes"]] == train_members
    assert os.path.exists(os.path.join(src, "ckpt", "final.pth"))
    for step in (10000, 20000, 30000):
        assert os.path.exists(os.path.join(src, "ckpt", f"step_{step}.pth"))
    if os.path.exists(os.path.join(args.run_dir, "DONE.json")):
        print("ALREADY_ADOPTED")
        return
    os.symlink(os.path.join(src, "ckpt"), os.path.join(args.run_dir, "ckpt"))
    os.symlink(os.path.join(src, "metrics.jsonl"), os.path.join(args.run_dir, "metrics.jsonl"))
    meta = dict(m)
    meta.update({"run_id": args.run_id, "label": "final", "arm": args.arm, "arm_spec": ARMS[args.arm], "intent": None,
                 "run_key": run_key, "data_hash": d_hash, "config_hash": c_hash, "code_hash": k_hash,
                 "adopted_from": src, "adoption_note": "recipe identical except save_every (5000 vs 10000; no effect on "
                 "training math); first 640 updates verified bit-identical to the matrix A-MIX path "
                 "(outputs/oopsieverse_full_study/matrix/verification/bitident_vs_existing_run.json)",
                 "n_train_episodes": len(train_members),
                 "repeat_exposure_epochs": (cfg.num_steps + 1) * cfg.batch_size / m["n_train_samples"]})
    json.dump(meta, open(os.path.join(args.run_dir, "meta.json"), "w"), indent=1, default=str)
    open(os.path.join(args.run_dir, "RUN_KEY"), "w").write(run_key)
    json.dump({"run_key": run_key, "time": time.time(), "train_wall_s": m.get("train_wall_s"), "adopted_from": src},
              open(os.path.join(args.run_dir, "DONE.json"), "w"))
    if not args.no_ledger:
        ledger_append({"run_id": args.run_id, "attempt": 1, "event": "adopted", "arm": args.arm, "task": args.task,
                       "seed": args.seed, "label": "final", "num_steps": cfg.num_steps, "step": cfg.num_steps,
                       "wall_s": round(m.get("train_wall_s") or 0, 1), "gpu": m.get("cuda_visible_devices"),
                       "host": socket.gethostname(), "run_key": run_key, "data_hash": d_hash, "config_hash": c_hash,
                       "code_hash": k_hash, "run_dir": args.run_dir, "note": f"adopted existing run {src}"})
    print("ADOPTED", src, "->", args.run_dir)


def enable_activation_checkpointing(policy):
    """Recompute seg_encoder.conv_layers activations in backward instead of storing them. Instance-level forward
    wrap: module structure and state_dict keys unchanged; no RNG inside the CNN; only active with grad enabled."""
    from torch.utils.checkpoint import checkpoint
    seq = policy.seg_encoder.conv_layers
    orig = seq.forward

    def fwd(x):
        if torch.is_grad_enabled() and seq.training:
            return checkpoint(orig, x, use_reentrant=False, preserve_rng_state=True)
        return orig(x)
    seq.forward = fwd


class StopBench(Exception):
    pass


def install_wall_limit(trainer, max_wall_s, out):
    """Benchmark mode: count optimizer steps; stop after max_wall_s (excluding the first 60 s warm-up)."""
    opt_step = trainer.optimizer.step
    st = {"n": 0, "t0": None, "n0": 0, "start": time.time()}

    def step(*a, **k):
        r = opt_step(*a, **k)
        st["n"] += 1
        now = time.time()
        if st["t0"] is None and now - st["start"] > 60:
            st["t0"], st["n0"] = now, st["n"]
        if st["t0"] is not None and now - st["t0"] > max_wall_s:
            rate = (st["n"] - st["n0"]) / (now - st["t0"])
            json.dump({"updates_per_s": rate, "updates": st["n"] - st["n0"], "window_s": now - st["t0"],
                       "max_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}, open(out, "w"))
            raise StopBench()
        return r
    trainer.optimizer.step = step


def install_batch_recorder(trainer, n, out):
    loader = trainer.train_loader
    rec = []

    class Rec:
        def __iter__(self_inner):
            for b in loader:
                if len(rec) < n:
                    rec.append([f"{d}:{int(t)}" for d, t in zip(b["demo_key"], b["timestep"])])
                    if len(rec) == n:
                        json.dump(rec, open(out, "w"))
                yield b

        def __len__(self_inner):
            return len(loader)
    trainer.train_loader = Rec()


def make_intent_dev_eval(policy, dev, args):
    """Teacher-forced vs sampled-intent dev metrics on fixed dev batches with fixed noise (dedicated generators;
    no draw from the global RNG streams)."""
    from torch.utils.data import DataLoader
    from learning.utils import denormalize_action

    def run():
        dev_dev = next(policy.parameters()).device
        g_cpu = torch.Generator().manual_seed(12345)
        loader = DataLoader(dev, batch_size=64, shuffle=False, num_workers=0, generator=g_cpu)
        g = torch.Generator(device=dev_dev).manual_seed(4242)
        was = policy.training
        policy.eval()
        acc = {k: 0.0 for k in ("tf_action_flow", "si_action_flow", "intent_flow", "tf_action_mse", "si_action_mse",
                                "intent_sample_mse", "intent_sample_std")}
        n = 0
        for i, batch in enumerate(loader):
            if i >= args.dev_intent_batches:
                break
            seg = {k: v.to(dev_dev) for k, v in batch["obs"]["extero"].items()}
            state = batch["obs"]["proprio"].to(dev_dev)
            action = batch["action"].to(dev_dev)
            B = action.shape[0]
            s23, it_gt = policy.split_state(state)
            t = torch.rand(B, device=dev_dev, generator=g) ** (1 / 1.5)
            x0 = torch.randn(action.shape, device=dev_dev, generator=g)
            ti = torch.rand(B, device=dev_dev, generator=g) ** (1 / 1.5)
            z0 = torch.randn(it_gt.shape, device=dev_dev, generator=g)
            zs = torch.randn(it_gt.shape, device=dev_dev, generator=g)
            zs2 = torch.randn(it_gt.shape, device=dev_dev, generator=g)
            xa = torch.randn(action.shape, device=dev_dev, generator=g)
            obs_tok = policy.obs_tokens(seg, s23)
            it_s = policy.sample_intent(obs_tok, zs)
            it_s2 = policy.sample_intent(obs_tok, zs2)
            l_tf, l_int = policy.flow_losses(seg, state, action, t=t, x0=x0, ti=ti, z0=z0)
            l_si, _ = policy.flow_losses(seg, state, action, t=t, x0=x0, ti=ti, z0=z0, intent_for_action=it_s)
            a_raw = denormalize_action(action, policy.action_min, policy.action_max)
            a_tf, _ = policy.generate(seg, s23, init_noise=xa, intent=it_gt if policy.mode == "condition" else None)
            a_si, _ = policy.generate(seg, s23, init_noise=xa, intent=it_s if policy.mode == "condition" else None)
            acc["tf_action_flow"] += float(l_tf)
            acc["si_action_flow"] += float(l_si)
            acc["intent_flow"] += float(l_int)
            acc["tf_action_mse"] += float(torch.mean((a_tf - a_raw) ** 2))
            acc["si_action_mse"] += float(torch.mean((a_si - a_raw) ** 2))
            acc["intent_sample_mse"] += float(torch.mean((it_s - it_gt) ** 2))
            acc["intent_sample_std"] += float(torch.mean((it_s - it_s2).abs()))
            n += 1
        if was:
            policy.train()
        return {f"dev_intent/{k}": v / max(n, 1) for k, v in acc.items()}
    return run


if __name__ == "__main__":
    main()
