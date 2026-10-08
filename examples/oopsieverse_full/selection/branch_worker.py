"""Record-producing code of the selection harness: constants, keyed RNG, state collection, candidates, continuations.
Moved verbatim out of run_selection.py on 2026-10-08 (supervisor-only refactor); the run key hashes this file.
"""

import argparse
import glob
import hashlib
import json
import os
import signal
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OVS = "/edrive1/lizad/oopsieverse_full_study"
REPO = "/scr/lizad/personal/learning-to-imitate"
STUDY = f"{REPO}/outputs/oopsieverse_full_study"
RUNS = f"{OVS}/runs/matrix"
CKPT = "step_30000.pth"
SOURCE_ARMS = ("amix", "imean", "ipath", "ipathg")
PROPOSAL_ARMS = ("amix", "imean", "ipath", "ipathg")
M, R = 16, 4
BRANCH_CHUNKS = (1, 12)          # t_b = 8c, c ~ U{1..12}
MAX_ATTEMPTS = 4
EXPERIMENT_ID = "oopsie_full_v1"
CODE_FILES = ["harness.py", "snapshot_damage_state.py", "branch_worker.py", "fast_contacts.py",
              "../baseline/rollout_eval.py", "../matrix/rollout_eval_matrix.py", "../matrix/models.py",
              "../matrix/fast_native.py", "../matrix/ctxfix.py"]


def variants():
    out = [(a, "primary") for a in PROPOSAL_ARMS]
    out += [(a, "joint") for a in PROPOSAL_ARMS if a != "amix"]
    return out


def keyed():
    sys.path.insert(0, REPO)
    from mip.intent_selection.rng import KeyedRNG
    return KeyedRNG(EXPERIMENT_ID)


def exp_name(task, pool):
    return f"{EXPERIMENT_ID}/selection/{task}/{pool}"


def sha_file(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:n]


def run_dirs(task, seed):
    return {a: f"{RUNS}/{a}_{task}_v1_s{seed}" for a in PROPOSAL_ARMS}


def make_run_key(cfg):
    code = {f: sha_file(os.path.join(HERE, f)) for f in CODE_FILES}
    ck = {a: sha_file(os.path.join(d, "ckpt", CKPT)) for a, d in run_dirs(cfg["task"], cfg["train_seed"]).items()}
    bank = json.load(open(f"{STUDY}/roster/reset_banks_v1.json"))["tasks"][cfg["task"]]["pools"][cfg["pool"]]
    mc = json.load(open(f"{STUDY}/metric_contract.json"))["version"]
    blob = {"cfg": cfg, "code": code, "ckpt": ck, "bank_states_sha256": bank["states_sha256"], "metric": mc}
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:16], blob


def root_dir(task, pool, seed):
    return f"{OVS}/runs/selection/{task}/{pool}/train_seed_{seed}"


def state_path(root, p):
    return f"{root}/states/p{p:04d}"


def group_dir(root, arm, var, p):
    return f"{root}/branches/{arm}__{var}/p{p:04d}"


def atomic_json(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float)
    os.replace(tmp, path)


# ───────────────────────────── worker side ─────────────────────────────

class Worker:
    def __init__(self, wid, cfg, req_q, resp_q, stop_ev):
        import torch
        torch.set_num_threads(1)
        import harness as H
        import snapshot_damage_state as SDS
        import fast_contacts
        fast_contacts.install()
        from ctxfix import install_ctx_free
        self.H, self.SDS, self.torch = H, SDS, torch
        self.cfg, self.wid, self.stop_ev = cfg, wid, stop_ev
        os.chdir(H.RE.OOPSIE)
        t0 = time.time()
        self.env = H.RE.make_env(cfg["task"], use_camera_obs=False)
        install_ctx_free(self.env)
        self.env_build_s = time.time() - t0
        self.dec = H.RemoteDecoder(wid, req_q, resp_q)
        self.rd = run_dirs(cfg["task"], cfg["train_seed"])
        self.frames = H.Frames(self.rd["amix"])
        for a, d in self.rd.items():
            f = H.Frames(d)
            assert (f.seg_keys, f.class_to_id, f.frame_stack) == (self.frames.seg_keys, self.frames.class_to_id, self.frames.frame_stack)
        self.ainfo = {a: H.arm_info(d) for a, d in self.rd.items()}
        self.ck = {a: sha_file(os.path.join(d, "ckpt", CKPT)) for a, d in self.rd.items()}
        self.kr = keyed()
        self.exp = exp_name(cfg["task"], cfg["pool"])
        self.root = root_dir(cfg["task"], cfg["pool"], cfg["train_seed"])
        self.loaded_parent = None   # parent seed whose compiled model the env holds
        self.snaps = {}

    # ---- env holding ----
    def restore(self, snap):
        t = time.time()
        if self.loaded_parent != snap["meta"]["parent_seed"]:
            ep = self.SDS.rebuild_and_restore(self.env, snap)
            self.loaded_parent = snap["meta"]["parent_seed"]
        else:
            ep = self.SDS.restore(self.env, snap)
        return ep, time.time() - t

    # ---- state collection ----
    def collect(self, p, parent):
        H, SDS, torch = self.H, self.SDS, self.torch
        sp = state_path(self.root, p)
        if os.path.exists(sp + ".json"):
            return {"unit": f"collect:{p}", "skipped": True}
        k = self.cfg["train_seed"]
        seed = self.kr.derive_seed("state_collection", experiment=self.exp, checkpoint_pair=f"train_seed_{k}", state_id=p)
        rng = np.random.default_rng(seed)
        src = SOURCE_ARMS[int(rng.integers(len(SOURCE_ARMS)))]
        attempts = []
        chosen = None
        t0 = time.time()
        for att in range(MAX_ATTEMPTS):
            c = int(rng.integers(BRANCH_CHUNKS[0], BRANCH_CHUNKS[1] + 1))
            tb = 8 * c
            nseed = int(rng.integers(2 ** 31 - 1))
            SDS.canonical_reset(self.env, parent["seed"])
            self.loaded_parent = parent["seed"]
            obs = self.env.get_observations()[0]
            ep = H.new_episode(self.env, self.cfg["task"], self.frames, obs, parent)
            lut = self.frames.lut(self.env.seg_id_to_class)
            gen = torch.Generator().manual_seed(nseed)
            H.run_policy(self.env, ep, self.dec, src, self.ainfo[src], self.frames, lut, gen, until_t=tb)
            a = {"attempt": att, "t_b": tb, "branch_chunk": c, "source_noise_seed": nseed, "t_reached": ep["t"]}
            if ep["invalid"]:
                a["outcome"] = "invalid_action_before_t_b"
            elif ep["done"]:
                a["outcome"] = f"episode_ended_before_t_b ({ep['termination']})"
            elif ep["first_succ"] is not None:
                a["outcome"] = "success_before_t_b"
            elif not np.all(ep["health"] >= 100.0) or ep["first_damage_step"] is not None:
                a["outcome"] = f"damage_before_t_b (step {ep['first_damage_step']})"
            else:
                a["outcome"] = "eligible"
            attempts.append(a)
            if a["outcome"] == "eligible":
                snap = SDS.capture(self.env, episode=ep, meta={
                    "parent_seed": parent["seed"], "parent_id": parent["family"], "parent_index": p,
                    "training_seed": k, "source_arm": src, "t_b": tb, "attempt": att, "state_seed": seed})
                SDS.save(snap, sp + ".pkl")
                chosen = {"attempt": att, "t_b": tb, "snapshot_sha": SDS.snapshot_digest(snap)[:16],
                          "snapshot_bytes": os.path.getsize(sp + ".pkl")}
                break
        rec = {"parent_index": p, "parent_id": parent["family"], "parent_seed": parent["seed"], "pool": self.cfg["pool"],
               "task": self.cfg["task"], "training_seed": k, "state_collection_seed": seed, "source_arm": src,
               "source_ckpt_sha16": self.ck[src], "attempts": attempts, "eligible": chosen is not None,
               "chosen": chosen, "wall_s": time.time() - t0, "branch_window_t": [8 * BRANCH_CHUNKS[0], 8 * BRANCH_CHUNKS[1]],
               "eligibility": "no link health loss (all == 100), no success, no invalid action, episode running at t_b"}
        atomic_json(rec, sp + ".json")
        return {"unit": f"collect:{p}", "eligible": chosen is not None}

    # ---- candidates ----
    def candidates(self, p, arm, var, snap, gdir):
        H, torch = self.H, self.torch
        path = f"{gdir}/candidates.npz"
        if os.path.exists(path):
            return dict(np.load(path, allow_pickle=False))
        ai = self.ainfo[arm]
        k = self.cfg["train_seed"]
        cp = f"train_seed_{k}"
        cseeds = [self.kr.derive_seed("candidate", experiment=self.exp, arm=arm, checkpoint_pair=cp, state_id=p,
                                      candidate_id=m) for m in range(M)]
        x0 = np.zeros((M, 8, ai["action_dim"]), np.float32)
        z = np.zeros((M, ai["n_tok"], ai["tok_dim"]), np.float32) if ai["conditioned"] else None
        dseeds = []
        if arm == "amix":
            assert var == "primary"
            for m in range(M):
                x0[m] = torch.randn(8, ai["action_dim"], generator=torch.Generator().manual_seed(cseeds[m])).numpy()
        else:
            for m in range(M):
                z[m] = torch.randn(ai["n_tok"], ai["tok_dim"], generator=torch.Generator().manual_seed(cseeds[m])).numpy()
            if var == "primary":
                ds = self.kr.derive_seed("decoder", experiment=self.exp, arm=arm, checkpoint_pair=cp, state_id=p, candidate_id=-1)
                shared = torch.randn(8, ai["action_dim"], generator=torch.Generator().manual_seed(ds)).numpy()
                x0[:] = shared
                dseeds = [ds] * M
            else:
                for m in range(M):
                    ds = self.kr.derive_seed("decoder", experiment=self.exp, arm=arm, checkpoint_pair=cp, state_id=p, candidate_id=m)
                    x0[m] = torch.randn(8, ai["action_dim"], generator=torch.Generator().manual_seed(ds)).numpy()
                    dseeds.append(ds)
        ep = snap["episode"]
        seg, prop = self.frames.model_inputs(ep["hist"])
        t = time.time()
        a, it = self.dec.decode(arm, {k_: np.repeat(v[None], M, 0) for k_, v in seg.items()},
                                np.repeat(prop[None], M, 0), x0, z)
        lat = time.time() - t
        h = hashlib.sha256()
        for k_ in sorted(seg):
            h.update(seg[k_].tobytes())
        h.update(prop.tobytes())
        out = {"x0": x0, "chunks": a.astype(np.float64), "candidate_seeds": np.array(cseeds, np.int64),
               "decoder_seeds": np.array(dseeds if dseeds else [-1] * M, np.int64),
               "obs_input_sha": np.array(h.hexdigest()[:16]), "decode_latency_s": np.array(lat),
               "ckpt_sha16": np.array(self.ck[arm]), "nfe_per_candidate": np.array(ai["action_nfe"] + (ai["intent_nfe"] if ai["conditioned"] else 0))}
        if z is not None:
            out["z"] = z
            out["intents"] = it.astype(np.float32)
        tmp = f"{path}.tmp{os.getpid()}.npz"
        np.savez(tmp, **out)
        os.replace(tmp, path)
        return out

    # ---- branch group ----
    def group(self, p, arm, var):
        H, SDS, torch = self.H, self.SDS, self.torch
        sp = state_path(self.root, p)
        snap = self.snaps.get(p) or SDS.load(sp + ".pkl")
        self.snaps = {p: snap}
        gdir = group_dir(self.root, arm, var, p)
        os.makedirs(f"{gdir}/_infra", exist_ok=True)
        cands = self.candidates(p, arm, var, snap, gdir)
        k = self.cfg["train_seed"]
        cp = f"train_seed_{k}"
        tseeds = [self.kr.derive_seed("continuation", experiment=self.exp, checkpoint_pair=cp, state_id=p,
                                      continuation_id=r, policy_identity=f"{arm}_tail") for r in range(R)]
        ai = self.ainfo[arm]
        meta = snap["meta"]
        n_done = 0
        for m in range(M):
            todo = []
            for r in range(R):
                out = f"{gdir}/c{m:02d}_r{r}.json"
                if os.path.exists(out):
                    continue
                n_inf = len(glob.glob(f"{gdir}/_infra/c{m:02d}_r{r}.*.json"))
                if n_inf > 2:
                    continue
                todo.append(r)
            if not todo:
                continue
            if self.stop_ev.is_set():
                return {"unit": f"group:{p}:{arm}:{var}", "partial": True, "n_done": n_done}
            t_c = time.time()
            try:
                ep, rs_s = self.restore(snap)
                lut = self.frames.lut(self.env.seg_id_to_class)
                chunk = cands["chunks"][m]
                ended = H.exec_chunk(self.env, ep, chunk, self.frames, lut)
                S8 = SDS.capture(self.env, episode=ep, meta=dict(meta))
                chunk_s = time.time() - t_c
            except Exception:
                for r in todo:
                    n_inf = len(glob.glob(f"{gdir}/_infra/c{m:02d}_r{r}.*.json"))
                    atomic_json({"error": traceback.format_exc(), "stage": "candidate_chunk"}, f"{gdir}/_infra/c{m:02d}_r{r}.{n_inf}.json")
                continue
            for r in todo:
                if self.stop_ev.is_set():
                    return {"unit": f"group:{p}:{arm}:{var}", "partial": True, "n_done": n_done}
                out = f"{gdir}/c{m:02d}_r{r}.json"
                t0 = time.time()
                try:
                    e2, rs2 = self.restore(S8)
                    gen = torch.Generator().manual_seed(tseeds[r])
                    t_tail0 = e2["t"]
                    first_tail = {}
                    infer = 0.0
                    if not e2["done"]:
                        x0t, zt = H.draw_chunk_noise(gen, ai)
                        ch, _it, lat = H.policy_chunk(self.dec, arm, self.frames, e2, x0t, zt)
                        infer += lat
                        first_tail = {"tail_first_action": ch[0].tolist(), "cand_last_action": chunk[7].tolist(),
                                      "arm_cmd_l2": float(np.linalg.norm(ch[0][:6] - chunk[7][:6])),
                                      "gripper_diff": float(ch[0][6] - chunk[7][6])}
                        H.exec_chunk(self.env, e2, ch, self.frames, lut)
                        infer += H.run_policy(self.env, e2, self.dec, arm, ai, self.frames, lut, gen)
                    rec = H.finalize(e2)
                    rec.update({
                        "record_type": "continuation", "pool": self.cfg["pool"], "split": "dev" if "dev" in self.cfg["pool"] else "test",
                        "parent_index": p, "parent_id": meta["parent_id"], "parent_seed": meta["parent_seed"],
                        "reset_family": meta["parent_id"], "training_seed": k, "arm": arm, "variant": var,
                        "proposal_noise": {"amix": "action x0 varies", "primary": "intent z varies, decoder x0 fixed",
                                           "joint": "intent z and decoder x0 vary"}["amix" if arm == "amix" else var],
                        "candidate_id": m, "tail_replicate": r, "tail_role": "rank" if r < 2 else "validate",
                        "candidate_seed": int(cands["candidate_seeds"][m]), "decoder_seed": int(cands["decoder_seeds"][m]),
                        "tail_seed": tseeds[r], "tail_policy": f"{arm} (own, {CKPT}, sha {self.ck[arm]})",
                        "checkpoint_sha256_16": self.ck[arm], "state_t_b": meta["t_b"], "source_arm": meta["source_arm"],
                        "candidate_chunk_sha16": hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest()[:16],
                        "ended_within_candidate_chunk": bool(ended), "handoff": first_tail,
                        "continuation_steps": rec["steps"] - meta["t_b"], "tail_steps": rec["steps"] - t_tail0,
                        "action_nfe": ai["action_nfe"], "intent_nfe": ai["intent_nfe"] if ai["conditioned"] else 0,
                        "controller": "PandaOmron HYBRID_MOBILE_BASE, continuous gripper patch, control 20 Hz",
                        "solver": f"Euler {ai['action_nfe']} (+ intent Euler {ai['intent_nfe']})" if ai["conditioned"] else f"Euler {ai['action_nfe']}",
                        "latency": {"wall_s": time.time() - t0, "restore_s": rs2, "infer_s": infer,
                                    "candidate_chunk_s_shared": chunk_s, "initial_restore_s_shared": rs_s},
                        "sim_steps_executed": rec["steps"] - t_tail0, "run_key": self.cfg["run_key"], "worker": self.wid,
                        "errors": None})
                    atomic_json(rec, out)
                    n_done += 1
                except Exception:
                    n_inf = len(glob.glob(f"{gdir}/_infra/c{m:02d}_r{r}.*.json"))
                    atomic_json({"error": traceback.format_exc(), "stage": "tail", "candidate_id": m, "tail_replicate": r},
                                f"{gdir}/_infra/c{m:02d}_r{r}.{n_inf}.json")
        return {"unit": f"group:{p}:{arm}:{var}", "n_done": n_done, "complete": True}


def worker_main(wid, cfg, req_q, resp_q, task_q, res_q, stop_ev):
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(cfg["render_gpu"])
    os.environ["EGL_DEVICE_ID"] = str(cfg["render_gpu"])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    wt = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"ovsel_{os.getpid()}")
    os.makedirs(wt, exist_ok=True)
    os.environ["TMPDIR"] = wt
    try:
        w = Worker(wid, cfg, req_q, resp_q, stop_ev)
        bank = json.load(open(f"{STUDY}/roster/reset_banks_v1.json"))["tasks"][cfg["task"]]["pools"][cfg["pool"]]["entries"]
        res_q.put({"wid": wid, "event": "ready", "env_build_s": w.env_build_s})
        while True:
            u = task_q.get()
            if u is None:
                break
            t0 = time.time()
            try:
                if u[0] == "collect":
                    e = bank[u[1]]
                    out = w.collect(u[1], {"family": e["reset_id"], "seed": e["seed"], "id": e["reset_id"].replace("/", "__")})
                else:
                    out = w.group(*u[1:])
                res_q.put({"wid": wid, "event": "done", "unit": list(u), "out": out, "wall_s": time.time() - t0})
            except Exception:
                res_q.put({"wid": wid, "event": "error", "unit": list(u), "tb": traceback.format_exc()})
            if stop_ev.is_set():
                break
    except Exception:
        res_q.put({"wid": wid, "event": "crash", "tb": traceback.format_exc()})
    res_q.put({"wid": wid, "event": "exit"})


