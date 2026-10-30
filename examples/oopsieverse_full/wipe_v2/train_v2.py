"""wipe_countertop v2 trainer: one run = (arm, seed). Wraps the OFFICIAL CFMTrainer loop exactly as
matrix/train_matrix.py does (same helpers imported from it, unchanged).

Arms (reports/wipe_v2/wipe_v2_spec.md; models: models_v2.py)
  w0  official CFMPolicy (reference; only used for the bit-identity check against the matrix A-MIX wipe s0 run)
  w1  spatial-softmax readout          SpatialSoftmaxCFMPolicy
  w2  8x8 token readout                TokenCFMPolicy
  w3  dirt-targeted intent, W0 readout DirtIntentCFMPolicy (matrix IntentCFMPolicy, condition, 1x5)

Recipe (unchanged from the official config learning/configs/wipe_counter.yaml as used by A-MIX): 30k updates,
batch 64, AdamW lr 3e-4, wd 1e-5, grad clip 5, no EMA, fp32, Beta(1.5,1) times, Euler 16; data = splits_v1 train
(the same 40 demos, wipe_countertop__train__v1.hdf5) with the mixed-train action normaliser; official eval cadence
(eval_every 500: official train-set MSE + dev MSE on the 10 dev demos, as the matrix trainer). Data-loader workers
(same sampler/batches; verified in the matrix) and activation checkpointing of the conv trunk (verified exact) are
throughput-only options. save_every 5000 (IO only).

Wrapper additions: run key = sha(data hash | config hash | code hash); ledger (runs/wipe_v2/train/run_ledger.csv);
meta.json; metrics.jsonl; STATUS.json (step, rate, ETA); DONE.json; resume from the newest checkpoint; a STOP file
in the run dir stops the run right after its next checkpoint (resumable).

  CUDA_VISIBLE_DEVICES=0 python train_v2.py --arm w1 --seed 0 --workers 4 --act-ckpt
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
import signal
import socket
import sys
import time
import traceback

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MATRIX = os.path.join(os.path.dirname(HERE), "matrix")
BASE = os.path.join(os.path.dirname(HERE), "baseline")
for p in (HERE, MATRIX, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)
import train_matrix as TM  # noqa: E402  (helpers only; the module is not modified)

OOPSIE = TM.OOPSIE
STORAGE = TM.STORAGE
TASK = "wipe_countertop"
RUNS = os.path.join(STORAGE, "runs/wipe_v2/train")
LEDGER = os.path.join(RUNS, "run_ledger.csv")
ARMS = {
    "w0": {"readout": "official_avgpool_2x2", "intent": None, "mode": None, "label": "W0 (A-MIX)"},
    "w1": {"readout": "spatial_softmax_32kp", "intent": None, "mode": None, "label": "W1 spatial softmax"},
    "w2": {"readout": "tokens_8x8", "intent": None, "mode": None, "label": "W2 8x8 tokens"},
    "w3": {"readout": "official_avgpool_2x2", "intent": "dirt", "mode": "condition", "label": "W3 dirt intent"},
}


def code_hash():
    files = [os.path.join(HERE, f) for f in ("train_v2.py", "models_v2.py", "dirt_targets.py")] + \
        [os.path.join(MATRIX, f) for f in ("train_matrix.py", "models.py", "data.py", "intent_targets.py")] + \
        [os.path.join(BASE, "train_amix.py"), os.path.join(BASE, "split_data.py")]
    h = hashlib.sha256()
    for p in files:
        h.update(os.path.basename(p).encode())
        h.update(open(p, "rb").read())
    h.update(TM.OFFICIAL_SHA.encode())
    return h.hexdigest()[:16]


def ledger_append(row):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    row = {k: row.get(k, "") for k in TM.LEDGER_FIELDS}
    row["time"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LEDGER, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0, 2)
        w = csv.DictWriter(f, fieldnames=TM.LEDGER_FIELDS)
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def atomic_json(obj, path):
    tmp = path + f".tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, path)


class StopRequested(Exception):
    pass


def make_dataset_class():
    from data import MatrixPlaybackDataset
    import h5py
    import dirt_targets as DT

    class V2PlaybackDataset(MatrixPlaybackDataset):
        """MatrixPlaybackDataset (native samples unchanged); W3 appends the 1x5 dirt target to proprio."""

        def member_of(self):
            with h5py.File(self.data_path, "r") as f:
                ids = json.loads(f.attrs["source_ids"])
            return {f"demo_{k}": ids[k] for k in range(len(ids))}

        def setup_intent(self, norm=None):
            assert self.intent_name == "dirt"
            mem = self.member_of()
            raws, self.intent_src = {}, {}
            for d in self.demo_keys():
                tr = self._load_trajectory(d)
                mpos, ws = DT.load_markers(mem[d])
                raws[d], self.intent_src[d] = DT.raw_dirt_targets(tr["obs_proprio"], tr["action"], mpos, ws,
                                                                   return_src=True)
            self.intent_raw = raws
            self.intent_norm = norm if norm is not None else DT.fit_normalizer(list(raws.values()))
            self.intent_norm_t = {d: self.intent_norm.normalize(v).astype(np.float32) for d, v in raws.items()}
            self.intent_shape = DT.SPEC
            return self.intent_norm
    return V2PlaybackDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-steps", type=int, default=30000)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--label", default="final", choices=["final", "smoke", "verify", "bench"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--act-ckpt", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default=RUNS)
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--max-wall-s", type=float, default=None)
    ap.add_argument("--dev-intent-batches", type=int, default=5)
    args = ap.parse_args()
    if args.label == "final":
        assert args.num_steps == 30000 and args.arm != "w0", "final runs: 30k budget, W1-W3 only (W0 exists)"
    args.run_id = args.run_id or (f"{args.arm}_{TASK}_v2_s{args.seed}" + ("" if args.label == "final" else f"__{args.label}{args.num_steps}"))
    args.run_dir = os.path.join(args.runs_root, args.run_id)
    os.makedirs(os.path.join(args.run_dir, "ckpt"), exist_ok=True)
    lock_f = open(os.path.join(args.run_dir, "LOCK"), "a+")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("LOCKED", args.run_id)
        sys.exit(75)
    os.chdir(OOPSIE)
    import learning.train_eval.cfm_trainer as T
    arm = ARMS[args.arm]
    raw = T.load_yaml_config(TM.CONFIGS[TASK])
    train_path, train_members = TM.data_file(TASK, "A_MIX_train")
    dev_path, dev_members = TM.data_file(TASK, "dev")
    raw.update({"data_path": train_path, "save_dir": os.path.join(args.run_dir, "ckpt"), "use_wandb": True,
                "wandb_run_name": args.run_id, "num_steps": args.num_steps, "save_every": args.save_every})
    cfg = T.trainer_config_from_mapping(raw)

    d_hash = TM.data_fingerprint([(train_path, train_members), (dev_path, dev_members)])
    if arm["intent"]:
        import dirt_targets as DT
        h = hashlib.sha256(d_hash.encode())
        for m in train_members + dev_members:
            mp = DT.MARKERS.format(cond=DT.COND[m.split("/")[1]], demo=m.split("/")[2])
            h.update(TM.sha256_file(mp).encode())
        d_hash = h.hexdigest()[:16]
    conf = {k: v for k, v in vars(cfg).items() if k not in ("save_dir", "wandb_run_name", "data_path", "save_every")}
    conf.update({"arm": args.arm, "arm_spec": arm, "seed": args.seed, "phase": "wipe_v2",
                 "intent_nfe": 16 if arm["intent"] else None, "intent_loss_weight": 1.0 if arm["intent"] else None})
    c_hash = TM.sha256_bytes(json.dumps(conf, sort_keys=True, default=str).encode())[:16]
    k_hash = code_hash()
    run_key = TM.sha256_bytes(f"{d_hash}|{c_hash}|{k_hash}".encode())[:16]
    done_p, key_p = os.path.join(args.run_dir, "DONE.json"), os.path.join(args.run_dir, "RUN_KEY")
    if os.path.exists(done_p) and json.load(open(done_p)).get("run_key") == run_key:
        print("ALREADY_DONE", args.run_id, run_key)
        return
    if os.path.exists(key_p) and open(key_p).read().strip() != run_key:
        raise SystemExit(f"REFUSE: {args.run_dir} holds run key {open(key_p).read().strip()} != {run_key}")
    open(key_p, "w").write(run_key)
    ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt", "step_*.pth")), key=lambda p: int(p.split("_")[-1][:-4]))
    resume_from = ckpts[-1] if ckpts else None
    attempt = 1 + len(glob.glob(os.path.join(args.run_dir, "attempt_*.json")))
    base_row = {"run_id": args.run_id, "attempt": attempt, "arm": args.arm, "task": TASK, "seed": args.seed,
                "label": args.label, "num_steps": args.num_steps, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "host": socket.gethostname(), "pid": os.getpid(), "workers": args.workers, "run_key": run_key,
                "data_hash": d_hash, "config_hash": c_hash, "code_hash": k_hash, "run_dir": args.run_dir,
                "log": os.environ.get("V2_LOG", "")}
    atomic_json({"attempt": attempt, "start": time.time(), "resume_from": resume_from, "pid": os.getpid()},
                os.path.join(args.run_dir, f"attempt_{attempt}.json"))
    if not args.no_ledger:
        ledger_append(dict(base_row, event="start", note=f"resume_from={os.path.basename(resume_from)}" if resume_from else ""))

    def _term(signum, frame):
        raise SystemExit("SIGTERM")
    signal.signal(signal.SIGTERM, _term)
    t0 = time.time()
    try:
        ev = _train(args, T, arm, cfg, train_path, train_members, dev_path, dev_members,
                    run_key, d_hash, c_hash, k_hash, resume_from)
    except BaseException as e:
        msg = (type(e).__name__ + ": " + str(e))[:300].replace("\n", " ")
        ev = "infra_retry_oom" if (isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()) else \
            ("preempted" if isinstance(e, SystemExit) and "SIGTERM" in str(e) else "failed")
        if not args.no_ledger:
            ledger_append(dict(base_row, event=ev, wall_s=round(time.time() - t0, 1), note=msg))
        traceback.print_exc()
        raise
    if not args.no_ledger:
        ledger_append(dict(base_row, event=ev, step=args.num_steps if ev == "completed" else "",
                           wall_s=round(time.time() - t0, 1)))


def _train(args, T, arm, cfg, train_path, train_members, dev_path, dev_members, run_key, d_hash, c_hash, k_hash,
           resume_from):
    from learning.models import PolicyConfig
    import models_v2 as M2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    DS = make_dataset_class()
    t_data = time.time()
    kw = dict(frame_stack=cfg.frame_stack, action_chunk_size=cfg.action_chunk,
              seg_img_size=(cfg.seg_img_size, cfg.seg_img_size), normalize_action=cfg.normalize_action,
              objects_of_interest=cfg.objects_of_interest, obs_keys=cfg.obs_keys)
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        ds = DS(data_path=cfg.data_path, intent_name=arm["intent"], **kw)
    stats = TM.mixed_train_stats(TASK, cfg)
    assert np.array_equal(ds.action_min, np.asarray(stats["action_min"], np.float32))
    assert np.array_equal(ds.action_max, np.asarray(stats["action_max"], np.float32))
    intent_norm = ds.setup_intent() if arm["intent"] else None
    ds.preload()
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        dev = DS(data_path=dev_path, intent_name=arm["intent"], **kw)
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
    policy = M2.build_policy(args.arm, pcfg, args.seed)
    pc = policy.get_parameter_count()
    pc["total_parameters"] = sum(p.numel() for p in policy.parameters())
    if args.act_ckpt:
        TM.enable_activation_checkpointing(policy)

    status_p = os.path.join(args.run_dir, "STATUS.json")
    st = {"t_last": None, "s_last": None}

    from train_amix import JsonlWandb

    class Shim(JsonlWandb):
        def log(self, d):
            if "train/loss" in d and hasattr(policy, "pop_component_means"):
                d = dict(d, **policy.pop_component_means())
            super().log(d)
            if "train/loss" in d and int(d["step"]) % 100 == 0:
                now, s = time.time(), int(d["step"])
                rate = None if st["t_last"] is None else (s - st["s_last"]) / max(now - st["t_last"], 1e-9)
                if st["t_last"] is None or now - st["t_last"] > 60:
                    st["t_last"], st["s_last"] = now, s
                    prev = st.get("rate")
                    st["rate"] = rate if rate is not None else prev
                r = st.get("rate")
                atomic_json({"run_id": args.run_id, "state": "running", "step": s, "num_steps": args.num_steps,
                             "updates_per_s": r, "eta_h": (args.num_steps - s) / r / 3600 if r else None,
                             "train_loss": d["train/loss"], "time": now, "pid": os.getpid(),
                             "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}, status_p)
    shim = Shim(os.path.join(args.run_dir, "metrics.jsonl"))
    T.wandb = shim
    T.WANDB_AVAILABLE = True
    extra_eval = TM.make_intent_dev_eval(policy, dev, args) if arm["intent"] else None

    class Trainer(T.CFMTrainer):
        @torch.no_grad()
        def evaluate(self):
            v = super().evaluate()
            saved = self.eval_dataset
            self.eval_dataset = dev
            dv = super().evaluate()
            self.eval_dataset = saved
            rec = {"eval/dev_mse_loss": float(dv), "eval/train_mse_loss": float(v)}
            if extra_eval is not None:
                rec.update(extra_eval())
            if self.wandb_run:
                T.wandb.log(rec)
            return v

        def save_checkpoint(self, path):
            super().save_checkpoint(path)
            if os.path.basename(path).startswith("step_") and os.path.exists(os.path.join(args.run_dir, "STOP")):
                raise StopRequested(path)

    meta = {
        "run_id": args.run_id, "phase": "wipe_v2", "label": args.label, "arm": args.arm, "arm_spec": ARMS[args.arm],
        "task": TASK, "seed": args.seed, "run_key": run_key, "data_hash": d_hash, "config_hash": c_hash,
        "code_hash": k_hash, "train_hdf5": train_path, "dev_hdf5": dev_path, "train_episodes": train_members,
        "dev_episodes": dev_members, "splits_v1": TM.SPLITS, "trainer_config": {k: v for k, v in vars(cfg).items()},
        "policy_config": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in vars(pcfg).items()},
        "class_to_id": ds.class_to_id, "action_min": ds.action_min.tolist(), "action_max": ds.action_max.tolist(),
        "normaliser_source": "mixed train split (splits_v1 train)", "n_train_samples": len(ds),
        "n_train_episodes": len(train_members), "n_dev_samples": len(dev),
        "num_obs_tokens": int(policy.num_obs_tokens),
        "intent": None if not arm["intent"] else {
            "name": "dirt", "mode": "condition", "n_tok": policy.n_tok, "tok_dim": policy.tok_dim,
            "normaliser": intent_norm.to_dict(), "intent_nfe": 16, "intent_prior": "N(0, I)", "loss_weight": 1.0,
            "teacher_forcing": "always GT intent for the action flow (no sampled-intent mixing)",
            "definition": "dirt_targets.py header", "dims": ["p_x", "p_y", "p_z", "count/50", "has_target"],
            "train_source_counts": {str(k): int(v) for k, v in zip(*np.unique(np.concatenate(list(ds.intent_src.values())), return_counts=True))}},
        "parameter_count": pc, "dataset_build_s": t_data, "workers": args.workers,
        "activation_checkpointing": bool(args.act_ckpt), "official_code": {"repo": OOPSIE, "sha": TM.OFFICIAL_SHA},
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "precision": "fp32 (official; no autocast)",
        "ema": "none (official)", "resume_from": resume_from,
    }
    meta_p = os.path.join(args.run_dir, "meta.json")
    atomic_json(meta, meta_p)

    trainer = Trainer(config=cfg, policy=policy, train_dataset=ds, args_dict=meta["trainer_config"])
    if args.workers > 0:
        from torch.utils.data import DataLoader
        trainer.train_loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=args.workers,
                                          drop_last=True, persistent_workers=False, multiprocessing_context="fork")
    if args.max_wall_s:
        TM.install_wall_limit(trainer, args.max_wall_s, os.path.join(args.run_dir, "bench.json"))
    resume_step = 0
    if resume_from:
        trainer.load_checkpoint(resume_from)
        resume_step = int(os.path.basename(resume_from).split("_")[-1][:-4]) + 1
    t0 = time.time()
    ev = "completed"
    try:
        trainer.train(resume_step=resume_step)
    except TM.StopBench:
        ev = "bench_done"
    except StopRequested as e:
        ev = "stopped"
        atomic_json({"run_id": args.run_id, "state": "stopped", "at": str(e), "time": time.time()}, status_p)
    meta["train_wall_s"] = time.time() - t0
    meta["max_gpu_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    atomic_json(meta, meta_p)
    if ev == "completed":
        atomic_json({"run_key": run_key, "time": time.time(), "train_wall_s": meta["train_wall_s"]}, os.path.join(args.run_dir, "DONE.json"))
        atomic_json({"run_id": args.run_id, "state": "done", "step": args.num_steps, "time": time.time()}, status_p)
    print("TRAIN_END", ev, args.run_id, meta["train_wall_s"])
    return ev


if __name__ == "__main__":
    main()
