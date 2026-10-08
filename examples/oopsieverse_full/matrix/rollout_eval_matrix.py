"""Rollout evaluator for every matrix arm. Reuses baseline/rollout_eval.py unchanged (env construction, canonical
reset, observation preprocessing, run_episode = frozen contract protocol oopsie-metric-v1.0); only the policy object
differs:
  * native arms (amix, asafe, ssmix): exactly rollout_eval.Policy / generate_action.
  * intent arms: per chunk, x0_action ~ N(0, I) is drawn FIRST from the per-episode generator (same draw as native),
    then, for conditioned arms, z_intent ~ N(0, I) [n_tok x tok_dim] from the same generator; the intent is sampled
    with 16 Euler steps and the action flow (16 Euler steps) is conditioned on it. AUX arms decode actions natively
    (no intent draw). Never zero noise.
Records add: arm, intent_nfe, total NFE per chunk, sampled intent of the first chunks.

  python rollout_eval_matrix.py --run-dir RUN --task pick_egg --resets dev --limit 5 --rollout-seeds 0 \
      --out-dir OUT --workers 5 --egl-devices 5 --torch-device cuda:0
"""

import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(HERE), "baseline")
for p in (HERE, BASE):
    if p not in sys.path:
        sys.path.insert(0, p)
import rollout_eval as RE  # noqa: E402

REPO = "/scr/lizad/personal/learning-to-imitate"
STUDY = os.path.join(REPO, "outputs/oopsieverse_full_study")


def bank_resets(task, pool):
    """Frozen reset_banks_v1.json pool in listed order (read only). in_support = obj_rel_base_inside_demo_box
    (identical to the field used by the baseline screens: 100/100 eval_dev entries agree)."""
    bank = json.load(open(os.path.join(STUDY, "roster/reset_banks_v1.json")))
    out = []
    for e in bank["tasks"][task]["pools"][pool]["entries"]:
        out.append({"id": e["reset_id"].replace("/", "__"), "seed": e["seed"], "family": e["reset_id"],
                    "obj_pos": e["obj_pos"], "base_xy": e["base_xy"], "state_sha": e["state_sha"],
                    "in_support": e.get("obj_rel_base_inside_demo_box")})
    return out


def dev_resets(task):
    return bank_resets(task, "eval_dev")


class MatrixPolicy(RE.Policy):
    def __init__(self, run_dir, ckpt="final.pth", device="cuda", load_model=True):
        import torch
        self.arm_meta = json.load(open(os.path.join(run_dir, "meta.json")))
        self.intent = self.arm_meta.get("intent")
        if self.intent is None:
            super().__init__(run_dir, ckpt=ckpt, device=device, load_model=load_model)
            self.intent_nfe = 0
            return
        super().__init__(run_dir, ckpt=ckpt, device=device, load_model=False)
        from models import IntentCFMPolicy
        self.intent_nfe = int(self.intent["intent_nfe"])
        self.n_tok, self.tok_dim, self.mode = self.intent["n_tok"], self.intent["tok_dim"], self.intent["mode"]
        if load_model:
            self.model = IntentCFMPolicy(self.cfg, self.n_tok, self.tok_dim, self.mode, intent_seed=self.arm_meta["seed"])
            sd = torch.load(os.path.join(run_dir, "ckpt", ckpt), map_location="cpu")["policy_state_dict"]
            self.model.load_state_dict(sd)
            self.model.to(device).eval()

    def draw_noise(self, gen):
        torch = self.torch
        x0 = torch.randn(1, self.cfg.action_chunk_size, self.cfg.action_dim, generator=gen)
        z = None
        if self.intent is not None and self.mode == "condition":
            z = torch.randn(1, self.n_tok, self.tok_dim, generator=gen)
        return x0, z

    def decode(self, seg, prop, x0, z):
        """Batched decode on self.device. Returns (actions np [B,8,A], intents np or None)."""
        torch = self.torch
        with torch.no_grad():
            if self.intent is None:
                # encoders once per chunk; bitwise identical to generate_action (verification/verify_fast_native.json)
                from fast_native import generate_action_cached
                return generate_action_cached(self.model, seg, prop, x0).cpu().numpy(), None
            a, it = self.model.generate(seg, prop, init_noise=x0, intent_noise=z)
            return a.cpu().numpy(), (it.cpu().numpy() if it is not None else None)

    def act(self, hist, gen):
        seg, prop = self.model_inputs(hist)
        x0, z = self.draw_noise(gen)
        a, it = self.decode(seg, prop, x0.to(self.device), None if z is None else z.to(self.device))
        self.last_intent = None if it is None else it[0]
        return a[0].astype(np.float64), x0[0].cpu().numpy()


class MatrixRemotePolicy(MatrixPolicy):
    def __init__(self, run_dir, ckpt, wid, req_q, resp_q):
        super().__init__(run_dir, ckpt=ckpt, device="cpu", load_model=False)
        self.wid, self.req_q, self.resp_q = wid, req_q, resp_q

    def act(self, hist, gen):
        torch = self.torch
        seg, prop = self.model_inputs(hist)
        x0, z = self.draw_noise(gen)
        self.req_q.put((self.wid, {k: v[0].to(torch.uint8).numpy() for k, v in seg.items()}, prop[0].numpy(),
                        x0[0].numpy(), None if z is None else z[0].numpy()))
        a, it = self.resp_q.get()
        self.last_intent = it
        return a.astype(np.float64), x0[0].numpy()


def inference_server(run_dir, ckpt, device, req_q, resp_qs, max_batch=64, stats_path=None):
    import queue
    import torch
    os.chdir(RE.OOPSIE)
    pol = MatrixPolicy(run_dir, ckpt=ckpt, device=device)
    n_calls, n_items, busy = 0, 0, 0.0
    t0 = time.time()
    while True:
        item = req_q.get()
        if item is None:
            break
        batch, stop = [item], False
        while len(batch) < max_batch:
            try:
                it = req_q.get(timeout=0.003)
            except queue.Empty:
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
        if stop:
            break
    if stats_path:
        json.dump({"calls": n_calls, "items": n_items, "mean_batch": n_items / max(1, n_calls), "gpu_busy_s": busy,
                   "wall_s": time.time() - t0}, open(stats_path, "w"))


def _sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def worker(args):
    wid, jobs, run_dir, task, out_dir, egl_dev, device, exec_h, use_cam, ckpt, qs = args[:11]
    go = args[11] if len(args) > 11 else None
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_dev)
    wt = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"ovw_{os.getpid()}")
    os.makedirs(wt, exist_ok=True)
    os.environ["TMPDIR"] = wt
    os.chdir(RE.OOPSIE)
    import torch
    torch.set_num_threads(1)
    t0 = time.time()
    env = RE.make_env(task, use_camera_obs=use_cam)
    from ctxfix import install_ctx_free
    install_ctx_free(env)  # free the previous EGL context on every reset (robosuite leak, see ctxfix.py)
    pol = MatrixRemotePolicy(run_dir, ckpt, wid, qs[0], qs[1])
    meta = pol.arm_meta
    ck_sha = _sha16(os.path.join(run_dir, "ckpt", ckpt))
    t_env = time.time() - t0
    done = 0
    for j in jobs:
        out = os.path.join(out_dir, f"{j['id']}__r{j['rollout_seed']}.json")
        if os.path.exists(out):
            prev = json.load(open(out))
            if "infra_error" not in prev:
                continue
            # infrastructure error: keep it (logged separately) and retry, at most 2 retries
            idir = os.path.join(out_dir, "_infra")
            os.makedirs(idir, exist_ok=True)
            n_prev = len([f for f in os.listdir(idir) if f.startswith(os.path.basename(out)[:-5] + ".")])
            if n_prev >= 2:
                continue
            os.replace(out, os.path.join(idir, f"{os.path.basename(out)[:-5]}.{n_prev}.json"))
        if go is not None:
            go.wait()  # paused while GPU-3 memory limits are exceeded
        try:
            intents = []
            orig_act = pol.act

            def act(hist, gen):
                r = orig_act(hist, gen)
                if len(intents) < 3 and pol.last_intent is not None:
                    intents.append(np.asarray(pol.last_intent).tolist())
                return r
            pol.act = act
            rec = RE.run_episode(env, pol, task, j["reset"], j["rollout_seed"], exec_horizon=exec_h, record_chunks=True)
            pol.act = orig_act
            rec.update({"episode_id": j["id"], "worker": wid, "worker_env_build_s": t_env, "arm": meta["arm"],
                        "run_id": meta["run_id"], "training_seed": meta["seed"], "run_key": meta["run_key"],
                        "checkpoint_sha256_16": ck_sha, "action_nfe": int(pol.cfg.num_inference_steps),
                        # intent NFE is spent at inference only by conditioned arms; AUX decodes actions natively
                        "intent_nfe": pol.intent_nfe if (pol.intent_nfe and pol.mode == "condition") else 0,
                        "nfe_per_chunk": int(pol.cfg.num_inference_steps) + (pol.intent_nfe if (pol.intent_nfe and pol.mode == "condition") else 0),
                        "intent_noise": "z ~ N(0,I) drawn after x0 from the same per-episode generator" if pol.intent_nfe and pol.mode == "condition" else None,
                        "sampled_intents_first_chunks": intents, "reset_pool": "eval_dev (development)"})
        except Exception:
            rec = {"episode_id": j["id"], "infra_error": traceback.format_exc(), "reset": j["reset"],
                   "rollout_seed": j["rollout_seed"], "arm": meta["arm"]}
        tmp = out + ".tmp"
        json.dump(rec, open(tmp, "w"), default=float)
        os.replace(tmp, out)
        done += 1
    return wid, done, time.time() - t0


def _worker_entry(args, rq):
    try:
        rq.put(worker(args))
    except Exception:
        rq.put((args[0], -1, traceback.format_exc()))


def _gpu3_usage(pids):
    """(our MiB on GPU `RENDER_GPU`, free MiB). EGL contexts are graphics ('G') processes: parse the nvidia-smi table."""
    import subprocess
    free = float(subprocess.check_output(["nvidia-smi", "-i", str(RENDER_GPU), "--query-gpu=memory.free",
                                          "--format=csv,noheader,nounits"]).decode().strip())
    ours = 0.0
    for line in subprocess.check_output(["nvidia-smi"]).decode().splitlines():
        parts = line.strip("| ").split()
        if len(parts) >= 7 and parts[0] == str(RENDER_GPU) and parts[3].isdigit() and int(parts[3]) in pids:
            ours += float(parts[-1].rstrip("MiB"))
    return ours, free


RENDER_GPU = 3


def main():
    import argparse
    import multiprocessing as mp
    import threading
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--task", default="pick_egg")
    ap.add_argument("--resets", default="dev", choices=["dev", "test"], help="dev = eval_dev; test = eval_test (Amendment 2)")
    ap.add_argument("--rollout-seeds", default="0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--render-gpu", type=int, default=3, help="Amendment 2: EGL rendering on GPU 3 only")
    ap.add_argument("--infer-gpu", type=int, default=0, help="Amendment 2: policy inference on GPU 0 only")
    ap.add_argument("--max-mem-gb", type=float, default=10.0)
    ap.add_argument("--headroom-gb", type=float, default=8.0)
    ap.add_argument("--exec-horizon", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ckpt", default="final.pth")
    a = ap.parse_args()
    global RENDER_GPU
    assert a.render_gpu == 3 and a.infer_gpu == 0, "Amendment 2: render on GPU 3, infer on GPU 0, nothing else"
    RENDER_GPU = a.render_gpu
    assert a.workers <= 30
    os.makedirs(a.out_dir, exist_ok=True)
    resets = bank_resets(a.task, "eval_dev" if a.resets == "dev" else "eval_test")
    if a.limit:
        resets = resets[: a.limit]
    jobs = [{"id": r["id"], "reset": r, "rollout_seed": int(s)} for s in a.rollout_seeds.split(",") for r in resets]
    W = min(a.workers, len(jobs))
    chunks = [jobs[i::W] for i in range(W)]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    req_q = ctx.Queue()
    resp = [ctx.Queue() for _ in range(W)]
    # inference server: CUDA on GPU 0 only
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.infer_gpu)
    os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
    srv = ctx.Process(target=inference_server, args=(a.run_dir, a.ckpt, "cuda:0", req_q, resp, 64,
                                                     os.path.join(a.out_dir, f"_server_{int(t0)}.json")))
    srv.start()
    # workers: no CUDA at all; EGL rendering on GPU 3
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(a.render_gpu)
    os.environ["EGL_DEVICE_ID"] = str(a.render_gpu)
    go = ctx.Event()
    go.set()
    procs, rq = [], ctx.Queue()
    for i in range(W):
        p = ctx.Process(target=_worker_entry, args=((i, chunks[i], a.run_dir, a.task, a.out_dir, a.render_gpu, "cpu",
                                                     a.exec_horizon, False, a.ckpt, (req_q, resp[i]), go), rq))
        p.start()
        procs.append(p)
    pids = {p.pid for p in procs}
    mon_log = open(os.path.join(a.out_dir, f"_gpu3_monitor_{int(t0)}.jsonl"), "a")
    stop = threading.Event()

    def monitor():
        peak = 0.0
        while not stop.is_set():
            try:
                ours, free = _gpu3_usage(pids)
                peak = max(peak, ours)
                bad = ours > a.max_mem_gb * 1024 or free < a.headroom_gb * 1024
                if bad and go.is_set():
                    go.clear()
                    mon_log.write(json.dumps({"t": time.time(), "event": "pause", "ours_mib": ours, "free_mib": free}) + "\n")
                elif not bad and not go.is_set():
                    go.set()
                    mon_log.write(json.dumps({"t": time.time(), "event": "resume", "ours_mib": ours, "free_mib": free}) + "\n")
                mon_log.write(json.dumps({"t": time.time(), "ours_mib": ours, "free_mib": free, "peak_ours_mib": peak}) + "\n")
                mon_log.flush()
            except Exception as e:  # monitoring failure: pause to be safe
                go.clear()
                mon_log.write(json.dumps({"t": time.time(), "event": "monitor_error", "err": str(e)}) + "\n")
            stop.wait(10)
    th = threading.Thread(target=monitor, daemon=True)
    th.start()
    res = [rq.get() for _ in procs]
    for p in procs:
        p.join()
    stop.set()
    req_q.put(None)
    srv.join()
    json.dump({"workers": W, "n_jobs": len(jobs), "wall_s": time.time() - t0, "per_worker": res},
              open(os.path.join(a.out_dir, f"_pool_{int(t0)}.json"), "w"))
    print("EVAL_DONE", len(jobs), time.time() - t0)


if __name__ == "__main__":
    main()
