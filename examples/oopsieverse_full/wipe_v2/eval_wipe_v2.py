"""wipe_countertop v2 evaluator (Amendment 8; spec reports/wipe_v2/wipe_v2_spec.md). A WRAPPER around the frozen
rollout logic: baseline/rollout_eval.py (make_env, apply_reset, run_episode = metric contract oopsie-metric-v1.0:
1250-step timeout, 10-step post-success window, stochastic Euler-16 with x0 ~ N(0,I) from the per-episode generator,
full 8-step chunks, task success = predicate at the last executed step, three safety predicates over the whole
trajectory) and matrix/rollout_eval_matrix.py (MatrixPolicy / MatrixRemotePolicy noise and decode for native and
intent arms, fast_native encoder caching, ctxfix EGL-context free). Neither file is edited. Differences:
  (a) counter-label alias (Amendment 7.2): the worker-side seg LUT maps an env class that falls to 'unknown' to
      'Damageable'+name when that is in the vocabulary (plain env.reset() names the counter 'Counter'; training data
      'DamageableCounter'). Verified by verify_alias.py (pixel agreement with training frames / playback reload).
  (b) frozen pools (pools/pools_v1.json: P-train, P-devlayout, P-random), records under a run-dir-specific out dir,
      atomic per-episode JSON with: task success, the three safety predicates (+ safe successes), final proportion
      wiped, markers wiped (final / max / change points), damaging outcome, steps, entity health, a reset check
      (post-reset state hash vs the frozen pool / bank hash) and counter pixels at reset;
  (c) "reset_pool" is the actual pool name (the frozen evaluator hard-codes "eval_dev (development)");
  (d) policy loading through ONE import point, load_policy(run_dir, device, ckpt), see below;
  (e) multi-GPU rendering on the parity-passed GPUs (selection/gpu_budget.parity_passed_gpus(): 3/4/6/7) under the
      per-GPU budget (>= 8 GB free on every GPU used, strict 30-min minimum; breach -> evict our youngest worker,
      its episode re-runs); one batched inference server per run on a GPU with headroom. Workers pull jobs one at
      a time, so batch composition (and hence float round-off) is not fixed - same as the frozen evaluator.

──────────────────────────── policy interface (for the v2 training agent) ────────────────────────────
load_policy(run_dir, device, ckpt="step_30000.pth") is the only place a server-side policy is constructed.
  1. reads <run_dir>/meta.json["arm"];
  2. imports the optional module wipe_v2/v2_registry.py (create it; it does not exist by default). That module
     registers loaders with the decorator below, e.g.
         from eval_wipe_v2 import register
         @register("w1_spatialsoftmax")
         def load_w1(run_dir, device, ckpt): ...; return policy_object
  3. if the arm is registered -> that loader; else -> rollout_eval_matrix.MatrixPolicy(run_dir, ckpt, device)
     (native CFMPolicy arms incl. W0, and IntentCFMPolicy arms when meta["intent"] is set).
The returned object must provide
     .seg_keys                      list of obs keys, e.g. ['cam/robot0_eye_in_hand/seg', ...] (meta trainer_config order)
     .cfg.num_inference_steps       int (action NFE, recorded)
     .intent_nfe                    int (0 if no sampled intent)
     .decode(seg, prop, x0, z)      seg {key: float32 [B, frame_stack, 128, 128] class ids, on device}, prop [B, 23],
                                    x0 [B, 8, action_dim] ~ N(0,I), z [B, n_tok, tok_dim] or None
                                    -> (np.ndarray [B, 8, action_dim] DENORMALISED actions, np intents [B, ...] or None)
Worker side (no model): rollout_eval_matrix.MatrixRemotePolicy built from meta.json, so meta.json must hold
     policy_config (accepted by learning.models.PolicyConfig; put v2-only fields elsewhere, e.g. meta["v2"]),
     trainer_config (obs_keys, seg_img_size, frame_stack), class_to_id, arm, run_id, seed, run_key,
     and, for an arm whose action flow is conditioned on a sampled intent, meta["intent"] = {"mode": "condition",
     "n_tok", "tok_dim", "intent_nfe", ...}: per chunk x0 ~ N(0,I) is drawn first, then z ~ N(0,I) [n_tok, tok_dim]
     from the same per-episode generator (as I-PATH). A registered arm may override this with
     register(arm, draw_noise=fn) where fn(worker_policy, gen) -> (x0 [1,8,A] tensor, z tensor or None).
     <run_dir>/ckpt/<ckpt> must exist (its sha256 goes into the record and the eval key).

──────────────────────────── CLI ────────────────────────────
  source examples/oopsieverse_full/baseline/envsetup.sh
  nohup nice -n 10 $PY eval_wipe_v2.py --run-dir RUN [--run-dir RUN2 ...] --pools P-train,P-devlayout,P-random \
      --rollout-seeds 0,1 [--ckpt step_30000.pth] [--max-workers 32] [--label NAME] > LOG 2>&1 &
Records: runs/wipe_v2/eval/<run_id>/<pool>__<ckpt stem>__<eval key>/<reset id>__r<seed>.json (+ manifest.json).
Status: runs/wipe_v2/_status/<label>.json. Stop gracefully: touch runs/wipe_v2/eval/STOP (or --stop-file); restart
the same command to resume (finished records are skipped, infra errors retried <= 2x, kept under _infra/).
"""
import hashlib
import json
import os
import queue as _queue
import signal
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wv2_common as C  # noqa: E402
import rollout_eval as RE  # noqa: E402
import rollout_eval_matrix as REM  # noqa: E402

EVAL_ROOT = os.environ.get("WV2_EVAL_ROOT_SMOKE_ONLY") or os.path.join(C.V2_RUNS, "eval")
STATUS_DIR = os.path.join(C.V2_RUNS, "_status")
LEDGER_DIR = os.path.join(C.V2_RUNS, "_gpus")
SLOTS = 96
SERVER_TIMEOUT_S = 900
CODE_FILES = [os.path.join(HERE, "eval_wipe_v2.py"), os.path.join(HERE, "wv2_common.py"),
              os.path.join(C.FULL, "baseline/rollout_eval.py"), os.path.join(C.FULL, "matrix/rollout_eval_matrix.py"),
              os.path.join(C.FULL, "matrix/fast_native.py"), os.path.join(C.FULL, "matrix/ctxfix.py"),
              os.path.join(C.FULL, "matrix/models.py")]

# ─────────────────────────────── policy registry / loader ───────────────────────────────
V2_LOADERS = {}


def register(arm, draw_noise=None):
    def deco(fn):
        V2_LOADERS[arm] = {"load": fn, "draw_noise": draw_noise}
        return fn
    return deco


def _import_registry():
    if os.path.exists(os.path.join(HERE, "v2_registry.py")):
        import v2_registry  # noqa: F401  (registers arms via eval_wipe_v2.register)
        # the module may have imported this file under its module name: merge both registries
        mod = sys.modules.get("eval_wipe_v2")
        if mod is not None and mod is not sys.modules.get(__name__):
            V2_LOADERS.update(mod.V2_LOADERS)


def load_policy(run_dir, device, ckpt="step_30000.pth"):
    """Server-side policy for run_dir (see header). Falls back to the frozen matrix loader."""
    _import_registry()
    arm = json.load(open(os.path.join(run_dir, "meta.json")))["arm"]
    if arm in V2_LOADERS:
        return V2_LOADERS[arm]["load"](run_dir, device, ckpt)
    return REM.MatrixPolicy(run_dir, ckpt=ckpt, device=device)


def registry_sha():
    p = os.path.join(HERE, "v2_registry.py")
    return C.sha256_file(p) if os.path.exists(p) else None


class WorkerPolicy(REM.MatrixRemotePolicy):
    """Worker-side policy: frozen MatrixRemotePolicy inputs/noise + the counter-label alias LUT."""

    def __init__(self, run_dir, ckpt, wid, req_q, resp_q):
        super().__init__(run_dir, ckpt, wid, req_q, resp_q)
        _import_registry()
        ent = V2_LOADERS.get(self.arm_meta["arm"])
        self._noise_fn = ent["draw_noise"] if ent else None

    def seg_lut(self, seg_id_to_class):
        return C.alias_seg_lut(self.class_to_id, seg_id_to_class)

    def draw_noise(self, gen):
        if self._noise_fn is not None:
            return self._noise_fn(self, gen)
        return super().draw_noise(gen)

    def act(self, hist, gen):
        torch = self.torch
        seg, prop = self.model_inputs(hist)
        x0, z = self.draw_noise(gen)
        self.req_q.put((self.wid, {k: v[0].to(torch.uint8).numpy() for k, v in seg.items()}, prop[0].numpy(),
                        x0[0].numpy(), None if z is None else z[0].numpy()))
        a, it = self.resp_q.get(timeout=SERVER_TIMEOUT_S)  # a dead server -> infra error, not a hang
        self.last_intent = it
        return a.astype(np.float64), x0[0].numpy()


def inference_server(run_dir, ckpt, device, req_q, resp_qs, max_batch=64, stats_path=None):
    """rollout_eval_matrix.inference_server with load_policy() as the constructor."""
    import torch
    torch.set_num_threads(2)
    os.chdir(RE.OOPSIE)
    pol = load_policy(run_dir, device, ckpt)
    n_calls, n_items, busy, t0 = 0, 0, 0.0, time.time()
    while True:
        item = req_q.get()
        if item is None:
            break
        batch, stop = [item], False
        while len(batch) < max_batch:
            try:
                it = req_q.get(timeout=0.003)
            except _queue.Empty:
                break
            if it is None:
                stop = True
                break
            batch.append(it)
        tb = time.time()
        seg = {k: torch.from_numpy(np.stack([b[1][k] for b in batch])).float().to(device) for k in pol.seg_keys}
        prop = torch.from_numpy(np.stack([b[2] for b in batch])).float().to(device)
        x0 = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(device)
        z = None if batch[0][4] is None else torch.from_numpy(np.stack([b[4] for b in batch])).float().to(device)
        a, its = pol.decode(seg, prop, x0, z)
        busy += time.time() - tb
        n_calls += 1
        n_items += len(batch)
        for i, b in enumerate(batch):
            resp_qs[b[0]].put((a[i], None if its is None else its[i]))
        if stats_path and n_calls % 2000 == 0:
            _dump(stats_path, {"calls": n_calls, "items": n_items, "gpu_busy_s": busy, "wall_s": time.time() - t0})
        if stop:
            break
    if stats_path:
        _dump(stats_path, {"calls": n_calls, "items": n_items, "mean_batch": n_items / max(1, n_calls),
                           "gpu_busy_s": busy, "wall_s": time.time() - t0, "final": True})


def _dump(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)


# ─────────────────────────────── run specs / keys ───────────────────────────────

def run_spec(run_dir, ckpt, pools_doc, pool_names):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    ck = os.path.join(run_dir, "ckpt", ckpt)
    assert os.path.exists(ck), ck
    ck_sha = C.sha256_file(ck)
    code = {os.path.relpath(f, C.FULL): C.sha256_file(f) for f in CODE_FILES}
    code["wipe_v2/v2_registry.py"] = registry_sha()
    out = {}
    for pool in pool_names:
        key_src = {"code": code, "pools_content_sha256": pools_doc["content_sha256"], "pool": pool,
                   "run_key": meta.get("run_key"), "run_id": meta.get("run_id"), "ckpt": ckpt, "ckpt_sha256": ck_sha,
                   "timeout": RE.TIMEOUT[C.TASK], "exec_horizon": 8, "counter_label_alias": True,
                   "metric_version": RE.METRIC_VERSION}
        key = hashlib.sha256(json.dumps(key_src, sort_keys=True).encode()).hexdigest()[:12]
        d = os.path.join(EVAL_ROOT, meta.get("run_id") or os.path.basename(run_dir.rstrip("/")),
                         f"{pool}__{os.path.splitext(ckpt)[0]}__{key}")
        out[pool] = {"out_dir": d, "eval_key": key, "key_src": key_src}
    return {"run_dir": run_dir, "ckpt": ckpt, "ckpt_sha256": ck_sha, "meta_arm": meta["arm"],
            "pools_content_sha256": pools_doc["content_sha256"],
            "run_id": meta.get("run_id"), "run_key": meta.get("run_key"), "training_seed": meta.get("seed"),
            "pools": out}


def build_jobs(specs, pools_doc, pool_names, seeds, limit=None):
    """Jobs not yet done (records without infra_error are final; infra errors retried <= 2x)."""
    jobs, n_done = [], 0
    for ri, sp in enumerate(specs):
        for pool in pool_names:
            d = sp["pools"][pool]["out_dir"]
            os.makedirs(d, exist_ok=True)
            man = os.path.join(d, "manifest.json")
            if not os.path.exists(man):
                _dump(man, {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "run_dir": sp["run_dir"], "arm": sp["meta_arm"],
                            **sp["pools"][pool]["key_src"]})
            rs = C.pool_resets(pool, pools_doc)
            if limit:
                rs = rs[:limit]
            for s in seeds:
                for r in rs:
                    out = os.path.join(d, f"{r['id']}__r{s}.json")
                    if os.path.exists(out):
                        try:
                            prev = json.load(open(out))
                        except Exception:
                            prev = {"infra_error": "unreadable"}
                        if "infra_error" not in prev:
                            n_done += 1
                            continue
                        idir = os.path.join(d, "_infra")
                        os.makedirs(idir, exist_ok=True)
                        stem = os.path.basename(out)[:-5]
                        n_prev = len([f for f in os.listdir(idir) if f.startswith(stem + ".")])
                        if n_prev >= 2:
                            n_done += 1  # retries exhausted: counted as an infra failure, reported separately
                            continue
                        os.replace(out, os.path.join(idir, f"{stem}.{n_prev}.json"))
                    jobs.append({"key": f"{ri}|{pool}|{r['id']}|{s}", "run": ri, "pool": pool, "reset": r,
                                 "rollout_seed": int(s), "out": out, "_ord": (int(s), pool_names.index(pool), r["id"], ri)})
    jobs.sort(key=lambda j: j.pop("_ord"))  # interleave runs: partial results stay balanced across runs / seeds
    return jobs, n_done


# ─────────────────────────────── worker ───────────────────────────────

def _episode(env, pol, job, spec, gpu):
    from reset_probe import fingerprint
    reset = job["reset"]
    wiped = []
    rc = {}
    orig_step = env.step
    orig_apply = RE.apply_reset

    def step(a):
        obs, r, d, info = orig_step(a)
        wiped.append(int(info.get("total_wiped", -1)))
        rc["total_markers"] = int(info.get("total_markers", -1))
        return obs, r, d, info

    def apply_reset(e, r):
        out = orig_apply(e, r)
        obs = out[0]
        fp = fingerprint(e, C.TASK, obs)
        rc["fp"] = {k: fp[k] for k in ("state_sha", "dirt_sha", "model_sha_norm", "base_pos", "obj_pos", "obj_yaw",
                                       "init_robot_base_pos", "health_min", "success_at_reset")}
        rc["seg_class_names"] = sorted(set(e.seg_id_to_class.values()))
        px = {}
        for kind, fn in (("alias", C.alias_seg_lut), ("frozen", C.frozen_seg_lut)):
            lut = fn(pol.class_to_id, e.seg_id_to_class)
            for key in pol.seg_keys:
                cam = key.split("/")[1]
                raw = np.asarray(obs[f"{cam}_segmentation_class"]).astype(np.int64)
                px[f"{cam}/{kind}"] = int((lut[np.clip(raw, 0, len(lut) - 1)] == pol.class_to_id.get("DamageableCounter", -9)).sum())
        rc["counter_px_at_reset"] = px
        return out

    intents = []
    orig_act = pol.act

    def act(hist, gen):
        r = orig_act(hist, gen)
        if len(intents) < 3 and getattr(pol, "last_intent", None) is not None:
            intents.append(np.asarray(pol.last_intent).tolist())
        return r
    env.step = step
    RE.apply_reset = apply_reset
    pol.act = act
    try:
        rec = RE.run_episode(env, pol, C.TASK, reset, job["rollout_seed"], exec_horizon=8, record_chunks=True)
    finally:
        env.step = orig_step
        RE.apply_reset = orig_apply
        pol.act = orig_act
    w = np.asarray(wiped)
    nm = rc.get("total_markers", 50)
    chg = [[int(i + 1), int(w[i])] for i in range(len(w)) if i == 0 or w[i] != w[i - 1]]
    fp = rc["fp"]
    sha_ref = reset.get("state_sha")
    meta = pol.arm_meta
    cond_nfe = pol.intent_nfe if (pol.intent_nfe and pol.mode == "condition") else 0
    rec.update({
        "episode_id": reset["id"], "reset_pool": job["pool"], "pool_reset_id": reset["family"],
        "layout_group": reset.get("layout_group"), "pools_file_content_sha256": spec["pools_content_sha256"],
        "eval_key": spec["pools"][job["pool"]]["eval_key"], "evaluator": "wipe_v2/eval_wipe_v2.py (counter-label alias)",
        "counter_label_alias": True, "render_gpu": gpu,
        "arm": meta["arm"], "run_id": meta.get("run_id"), "training_seed": meta.get("seed"), "run_key": meta.get("run_key"),
        "run_dir": spec["run_dir"], "checkpoint_sha256_16": spec["ckpt_sha256"][:16],
        "action_nfe": int(pol.cfg.num_inference_steps), "intent_nfe": cond_nfe,
        "nfe_per_chunk": int(pol.cfg.num_inference_steps) + cond_nfe, "sampled_intents_first_chunks": intents,
        # wipe progress (env info of every executed step)
        "total_markers": nm, "final_markers_wiped": int(w[-1]) if len(w) else 0,
        "max_markers_wiped": int(w.max()) if len(w) else 0,
        "final_proportion_wiped": float(w[-1]) / nm if len(w) else 0.0,
        "env_wiped_markers_end": len(getattr(env, "wiped_markers", [])),
        "first_wipe_step": int(np.argmax(w > 0)) + 1 if (w > 0).any() else None,
        "wiped_change_points": chg,
        "wiped_at": {str(s): int(w[min(s, len(w)) - 1]) for s in (100, 200, 300, 400, 600, 800, 1000, 1250) if len(w)},
        # safety summary (predicates computed by the frozen run_episode)
        "damaging": not rec["primary_safe"],
        "safety": {"primary_safe": rec["primary_safe"], "secondary_safe": rec["secondary_safe"], "compat_safe": rec["compat_safe"]},
        # reset check
        "reset_check": {"post_reset": fp, "seg_class_names": rc["seg_class_names"], "counter_px_at_reset": rc["counter_px_at_reset"],
                        "state_sha_ref": sha_ref, "state_equal_ref": (fp["state_sha"] == sha_ref) if sha_ref else None,
                        "dirt_equal_ref": (fp["dirt_sha"] == reset.get("dirt_sha")) if reset.get("dirt_sha") else None},
    })
    return rec


class Evicted(BaseException):
    """Raised in a worker by SIGUSR1 (budget eviction): unwind the episode, flush queue feeders, exit."""


def _evict_handler(signum, frame):
    raise Evicted()


def worker_main(slot, gpu, specs, req_qs, resp_q, job_q, res_q):
    signal.signal(signal.SIGUSR1, _evict_handler)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
    os.environ["EGL_DEVICE_ID"] = str(gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    wt = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"wv2w_{os.getpid()}")  # per-worker TMPDIR (dirt.png race)
    os.makedirs(wt, exist_ok=True)
    os.environ["TMPDIR"] = wt
    try:
        os.chdir(RE.OOPSIE)
        import torch
        torch.set_num_threads(1)
        env = RE.make_env(C.TASK, use_camera_obs=False)
        from ctxfix import install_ctx_free
        install_ctx_free(env)
        pols = {}
        res_q.put(("ready", slot, os.getpid(), None))
        while True:
            job = job_q.get()
            if job is None:
                break
            sp = specs[job["run"]]
            t0 = time.time()
            try:
                if job["run"] not in pols:
                    pols[job["run"]] = WorkerPolicy(sp["run_dir"], sp["ckpt"], slot, req_qs[job["run"]], resp_q)
                rec = _episode(env, pols[job["run"]], job, sp, gpu)
                rec["worker_slot"], rec["worker_pid"] = slot, os.getpid()
                status = "ok"
            except Exception:
                rec = {"episode_id": job["reset"]["id"], "infra_error": traceback.format_exc(), "reset_pool": job["pool"],
                       "rollout_seed": job["rollout_seed"], "run_dir": sp["run_dir"], "render_gpu": gpu}
                status = "infra"
            tmp = job["out"] + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(rec, f, default=float)
            os.replace(tmp, job["out"])
            res_q.put(("done", slot, job["key"], {"status": status, "wall_s": time.time() - t0,
                                                  "success": rec.get("success"), "wiped": rec.get("final_markers_wiped")}))
            if status == "infra" and "Empty" in rec["infra_error"]:
                break  # server timeout: let the supervisor decide
    except Evicted:
        pass  # the unfinished episode leaves no record (tmp files are never renamed); the supervisor requeues it
    except Exception:
        res_q.put(("fatal", slot, os.getpid(), traceback.format_exc()))
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    for q in list(req_qs) + [res_q]:  # never die with a half-written message in a shared pipe
        q.close()
        q.join_thread()
    os._exit(0)  # freeing the EGL env at interpreter exit can crash; records are already on disk


# ─────────────────────────────── supervisor ───────────────────────────────

def pick_infer_gpu(n_servers, exclude=()):
    import gpu_budget as GB
    _, free = GB.gpu_table()
    need = 8 * 1024 + n_servers * 1536
    cands = sorted(((f, g) for g, f in free.items() if g not in exclude and f >= need), reverse=True)
    if not cands:
        raise SystemExit(f"no GPU with >= {need} MiB free for {n_servers} inference servers: {free}")
    return cands[0][1], free


def main():
    import argparse
    import multiprocessing as mp
    sys.path.insert(0, os.path.join(C.FULL, "selection"))
    import gpu_budget as GB
    GB.LEDGER = LEDGER_DIR  # v2 supervisors share their own ledger (kept under runs/wipe_v2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", required=True)
    ap.add_argument("--ckpt", default="step_30000.pth")
    ap.add_argument("--pools", default="P-train,P-devlayout,P-random")
    ap.add_argument("--rollout-seeds", default="0,1")
    ap.add_argument("--max-workers", type=int, default=32, help="CPU cap: workers are single-threaded MuJoCo processes")
    ap.add_argument("--total-workers", type=int, default=36,
                    help="cap on workers of ALL wipe_v2 supervisors together (shared ledger); CPU cap 40")
    ap.add_argument("--infer-gpu", default="auto")
    ap.add_argument("--render-gpus", default="auto", help="default: gpu_budget.parity_passed_gpus()")
    ap.add_argument("--label", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N resets per pool (smoke)")
    ap.add_argument("--stop-file", default=os.path.join(EVAL_ROOT, "STOP"))
    a = ap.parse_args()
    assert a.max_workers <= 40 and a.total_workers <= 40, "CPU cap"
    pool_names = a.pools.split(",")
    seeds = [int(s) for s in a.rollout_seeds.split(",")]
    pools_doc = C.load_pools()
    for p in pool_names:
        assert p in ("P-train", "P-devlayout", "P-random"), p
    label = a.label or f"eval_{os.getpid()}"
    os.makedirs(STATUS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LEDGER_DIR, "holders"), exist_ok=True)
    status_path = os.path.join(STATUS_DIR, f"{label}.json")
    logf = open(os.path.join(C.V2_RUNS, "logs", f"{label}.supervisor.log"), "a")

    def L(*x):
        logf.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(v) for v in x) + "\n")
        logf.flush()
    specs = [run_spec(rd.rstrip("/"), a.ckpt, pools_doc, pool_names) for rd in a.run_dir]
    jobs, n_done0 = build_jobs(specs, pools_doc, pool_names, seeds, a.limit)
    n_total = len(jobs) + n_done0
    L("start", label, "runs", [s["run_id"] for s in specs], "pools", pool_names, "seeds", seeds, "todo", len(jobs), "done", n_done0)
    for s in specs:
        for p, v in s["pools"].items():
            L("out", s["run_id"], p, v["out_dir"])
    if not jobs:
        L("nothing to do")
        _dump(status_path, {"label": label, "phase": "finished", "n_total": n_total, "n_done": n_done0})
        print("EVAL_DONE nothing to do")
        return
    render_gpus = GB.parity_passed_gpus() if a.render_gpus == "auto" else [int(x) for x in a.render_gpus.split(",")]
    assert render_gpus and not (set(render_gpus) & GB.NEVER_RENDER), render_gpus
    ctx = mp.get_context("spawn")
    n_srv = len(specs)
    infer_gpu, free0 = (pick_infer_gpu(n_srv) if a.infer_gpu == "auto" else (int(a.infer_gpu), None))
    L("render gpus", render_gpus, "infer gpu", infer_gpu, "free", free0)
    req_qs = [ctx.Queue() for _ in specs]
    resp_qs = [ctx.Queue() for _ in range(SLOTS)]
    job_qs = [ctx.Queue() for _ in range(SLOTS)]
    res_q = ctx.Queue()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(infer_gpu)
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    servers = []
    for i, s in enumerate(specs):
        p = ctx.Process(target=inference_server, args=(s["run_dir"], s["ckpt"], "cuda:0", req_qs[i], resp_qs, 64,
                                                       os.path.join(C.V2_RUNS, "logs", f"{label}.server{i}.json")))
        p.start()
        servers.append(p)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    ledger = GB.Ledger(os.getpid(), label=f"wipe_v2/{label}")
    pending = list(jobs)
    workers = {}    # slot -> {"proc", "gpu", "pid", "t", "job", "ready"}
    cooldown = {}   # slot -> time it became free after a kill (stale responses may still arrive)
    free_slots = list(range(SLOTS))
    n_done, n_infra, n_evict, n_fatal = n_done0, 0, 0, 0
    t_start = time.time()
    stopping = False
    last_assess = 0.0

    def spawn(g):
        now = time.time()
        for sl in list(free_slots):
            if sl in cooldown:
                if now - cooldown[sl] < 120:
                    continue
                for q in (resp_qs[sl], job_qs[sl]):
                    try:
                        while True:
                            q.get_nowait()
                    except _queue.Empty:
                        pass
                cooldown.pop(sl)
            free_slots.remove(sl)
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(g)
            os.environ["EGL_DEVICE_ID"] = str(g)
            p = ctx.Process(target=worker_main, args=(sl, g, specs, req_qs, resp_qs[sl], job_qs[sl], res_q))
            p.start()
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
            os.environ.pop("EGL_DEVICE_ID", None)
            workers[sl] = {"proc": p, "gpu": g, "pid": p.pid, "t": now, "job": None, "ready": False}
            L("spawn slot", sl, "gpu", g, "pid", p.pid)
            return True
        return False

    def kill(sl, why):
        nonlocal n_evict
        w = workers.pop(sl)
        try:
            os.kill(w["pid"], signal.SIGUSR1)  # graceful: unwind + flush queues (SIGKILL only as a fallback)
        except ProcessLookupError:
            pass
        w["proc"].join(timeout=30)
        if w["proc"].is_alive():
            w["proc"].kill()
            w["proc"].join(timeout=10)
            L("SIGKILL fallback slot", sl)
        if w["job"] is not None:
            pending.insert(0, w["job"])
        cooldown[sl] = time.time()
        free_slots.append(sl)
        n_evict += 1
        L("kill slot", sl, "gpu", w["gpu"], "pid", w["pid"], why, "requeued", w["job"]["key"] if w["job"] else None)

    def write_status(phase):
        el = time.time() - t_start
        done_run = n_done - n_done0
        eph = done_run / (el / 3600) if el > 60 else None
        left = len(pending) + sum(1 for w in workers.values() if w["job"])
        per = {}
        for s in specs:
            for p, v in s["pools"].items():
                d = v["out_dir"]
                per[f"{s['run_id']}/{p}"] = len([f for f in os.listdir(d) if f.endswith(".json") and "__r" in f])
        _dump(status_path, {"label": label, "phase": phase, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "supervisor_pid": os.getpid(), "n_total": n_total, "n_done": n_done, "n_left": left,
                            "n_infra_this_session": n_infra, "n_evictions": n_evict,
                            "workers_by_gpu": {str(g): sum(1 for w in workers.values() if w["gpu"] == g) for g in render_gpus},
                            "infer_gpu": infer_gpu, "episodes_per_h": eph,
                            "eta_h": (left / eph) if eph else None, "records_per_out_dir": per,
                            "out_dirs": {f"{s['run_id']}/{p}": v["out_dir"] for s in specs for p, v in s["pools"].items()}})

    try:
        while True:
            # results
            while True:
                try:
                    m = res_q.get(timeout=0.5)
                except _queue.Empty:
                    break
                kind, sl = m[0], m[1]
                if sl not in workers:
                    continue
                w = workers[sl]
                if kind == "ready":
                    w["ready"] = True
                elif kind == "done":
                    w["job"] = None
                    n_done += 1
                    if m[3]["status"] != "ok":
                        n_infra += 1
                        L("infra", m[2])
                elif kind == "fatal":
                    L("fatal slot", sl, m[3][-2000:])
                    w["fatal"] = True
                    n_fatal += 1
                    if n_fatal > 10 and not stopping:
                        L("too many fatal worker errors -> stopping")
                        stopping = True
            # dead workers
            for sl, w in list(workers.items()):
                if not w["proc"].is_alive():
                    if w["job"] is not None:
                        pending.insert(0, w["job"])
                        w["job"] = None
                    workers.pop(sl)
                    cooldown[sl] = time.time()
                    free_slots.append(sl)
                    L("worker exited slot", sl, "exitcode", w["proc"].exitcode)
            if any(not p.is_alive() for p in servers) and (pending or any(w["job"] for w in workers.values())):
                L("inference server died -> stopping")
                stopping = True
            if os.path.exists(a.stop_file) and not stopping:
                L("STOP file")
                stopping = True
            # assign
            for sl, w in workers.items():
                if w["ready"] and w["job"] is None and not w.get("sent_none"):
                    if pending and not stopping:
                        w["job"] = pending.pop(0)
                        job_qs[sl].put(w["job"])
                    else:
                        job_qs[sl].put(None)
                        w["sent_none"] = True
            busy = sum(1 for w in workers.values() if w["job"])
            if (stopping or not pending) and busy == 0:
                break
            # budget
            if time.time() - last_assess > 15:
                last_assess = time.time()
                mine = {w["pid"]: w["gpu"] for w in workers.values()}
                want = (not stopping) and len(pending) > sum(1 for w in workers.values() if not w["ready"]) and len(workers) < a.max_workers
                d = GB.assess(ledger, mine, render_gpus, a.total_workers, a.max_workers, want_spawn=want)
                for g in d["over"]:
                    cand = sorted((w["t"], sl) for sl, w in workers.items() if w["gpu"] == int(g))
                    if cand:
                        kill(cand[-1][1], f"headroom breach gpu {g} free {d['gpus'][g]['free_mib']}")
                if d["yield_gpu"] is not None:
                    cand = sorted((w["t"], sl) for sl, w in workers.items() if w["gpu"] == d["yield_gpu"])
                    if cand:
                        kill(cand[-1][1], f"oversubscribed gpu {d['yield_gpu']}")
                if d["spawn_gpu"] is not None and spawn(d["spawn_gpu"]) and want and len(workers) < a.max_workers:
                    d2 = GB.assess(ledger, {w["pid"]: w["gpu"] for w in workers.values()}, render_gpus, a.total_workers,
                                   a.max_workers, want_spawn=True)  # ramp: up to 2 spawns per budget cycle
                    if d2["spawn_gpu"] is not None and not d2["over"] and d2["yield_gpu"] is None:
                        spawn(d2["spawn_gpu"])
                write_status("stopping" if stopping else "running")
    finally:
        for sl, w in list(workers.items()):
            if w["proc"].is_alive() and not w.get("sent_none"):
                job_qs[sl].put(None)
        for w in workers.values():
            w["proc"].join(timeout=60)
            if w["proc"].is_alive():
                w["proc"].kill()
        for q in req_qs:
            q.put(None)
        for p in servers:
            p.join(timeout=60)
            if p.is_alive():
                p.kill()
        ledger.write({}, [])
        ledger.remove()
    left = len(pending)
    write_status("stopped" if (stopping and left) else "finished")
    L("end", "done", n_done, "left", left, "infra", n_infra, "evictions", n_evict)
    print("EVAL_DONE" if not left else "EVAL_STOPPED", n_done, "/", n_total, "left", left)


if __name__ == "__main__":
    main()
