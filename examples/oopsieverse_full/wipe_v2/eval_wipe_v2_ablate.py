"""Rollout-time intent ablation for the wipe v2 evaluator (spec section "Secondary analysis: rollout-time intent
ablation", declared 2026-10-26). Entry point = eval_wipe_v2.py's CLI plus

    --intent-ablation {sampled,shuffled,zero}      (default: sampled)

Why a sibling file and not a flag inside eval_wipe_v2.py: the eval key hashes eval_wipe_v2.py itself, and spawned
workers / servers re-import it from disk. Editing it would change the key of every existing run (and swap code under
evaluations that are running), so `sampled` could not stay byte-identical. Here:
  * sampled  -> eval_wipe_v2.main() exactly as is (same code path, same CODE_FILES, same key, same out dirs).
  * shuffled / zero -> the same main(), with only the inference-server and worker targets replaced by the wrappers
    below. The key adds {"intent_ablation": mode, "ablation_code_sha256": sha(this file), "zero_token": ...}; out dirs
    go to runs/wipe_v2/ablation/<run_id>/<pool>__<ckpt>__<mode>__<key>/ (not matched by the main summaries).
Noise draws are unchanged: the worker draws x0 then z ~ N(0,I) from the per-episode generator exactly as in the
sampled run; the server always decodes the sampled intent from z (so it is available to other workers), then
conditions the action flow on the ablated intent via IntentCFMPolicy.generate(..., intent=...).

Modes (condition-mode intent arms only: meta["intent"]["mode"] == "condition"):
  zero      the normalised intent token is set to its training mean: mean over all mixed-TRAIN samples of the
            normalised target (MinMax to [-1, 1], so the mean is NOT the zero vector). Recomputed from the training
            data with the run's own target code (matrix intent_targets.raw_targets, or wipe_v2 dirt_targets for
            "dirt") and checked: refitted normaliser == meta normaliser (max abs diff <= 1e-9), sample count ==
            meta n_train_samples. Cached in runs/wipe_v2/ablation/zero_token__<run_id>.json.
  shuffled  item i of a server batch gets the sampled intent the server decoded for the PREVIOUS chunk of a different
            reset (the next batch item, cyclically, whose worker is running a different reset id and that has a
            previous intent). If none exists in the batch (e.g. batch size 1), the most recent previous-chunk intent
            of any other worker on a different reset (ring of the last 64) is used, provenance "fallback_ring"; if
            the ring is empty too (first chunks of the run), the training-mean token, provenance "fallback_mean".
            Per-episode counts of each provenance are in the record (intent_ablation_provenance).
Records add "intent_ablation" (all modes run through here; the main eval_wipe_v2.py records have no such field and
are the sampled runs).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_wipe_v2 as E  # noqa: E402
import wv2_common as C  # noqa: E402

MODES = ("sampled", "shuffled", "zero")
ABL_DIR = os.path.join(C.V2_RUNS, "ablation")
ENV_MODE = "WV2_INTENT_ABLATION"
RING = 64


# ─────────────────────────────── training-mean token ───────────────────────────────

def zero_token(run_dir):
    """Training mean of the normalised intent target [n_tok, tok_dim] (+ provenance and checks)."""
    import h5py
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    rid = meta.get("run_id") or os.path.basename(run_dir.rstrip("/"))
    cache = os.path.join(ABL_DIR, f"zero_token__{rid}.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        if d["run_key"] == meta.get("run_key"):
            return d
    sys.path.insert(0, os.path.join(C.FULL, "matrix"))
    import intent_targets as IT
    it = meta["intent"]
    name = it["name"]
    norm = IT.MinMax.from_dict(it["normaliser"])
    rows_raw = []
    with h5py.File(meta["train_hdf5"], "r") as f:
        ids = json.loads(f.attrs["source_ids"]) if "source_ids" in f.attrs else None
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            g = f["data"][k]
            p = np.asarray(g["obs/proprio"][:], np.float64)
            a = np.asarray(g["action"][:], np.float32)
            if name == "dirt":
                import dirt_targets as DT
                mpos, ws = DT.load_markers(ids[int(k.split("_")[1])])
                rows_raw.append(DT.raw_dirt_targets(p, a, mpos, ws))
            else:
                rows_raw.append(IT.raw_targets(p, a)[name].astype(np.float64))
    X = np.concatenate(rows_raw)  # [N, n_tok, d] one row per native training sample
    refit = IT.MinMax(X.min(0), X.max(0)) if name != "imean" else None
    chk = {"n_rows": int(X.shape[0]), "meta_n_train_samples": meta.get("n_train_samples"),
           "normaliser_refit_maxabs_vs_meta": (float(max(np.abs(refit.min - norm.min).max(), np.abs(refit.max - norm.max).max()))
                                               if refit is not None else None)}
    assert chk["n_rows"] == chk["meta_n_train_samples"], chk
    assert chk["normaliser_refit_maxabs_vs_meta"] is None or chk["normaliser_refit_maxabs_vs_meta"] <= 1e-9, chk
    Z = norm.normalize(X).astype(np.float32)
    mean = Z.reshape(Z.shape[0], int(it["n_tok"]), int(it["tok_dim"])).mean(0)
    d = {"run_id": rid, "run_key": meta.get("run_key"), "intent_name": name, "token": mean.tolist(),
         "is_zero_vector": bool(np.allclose(mean, 0)), "definition": "mean over mixed-TRAIN samples of the MinMax[-1,1]-normalised target",
         "checks": chk, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    os.makedirs(ABL_DIR, exist_ok=True)
    E._dump(cache, d)
    return d


# ─────────────────────────────── server ───────────────────────────────

def ablation_server(run_dir, ckpt, device, req_q, resp_qs, max_batch=64, stats_path=None):
    """eval_wipe_v2.inference_server with the intent replaced per ENV_MODE. Responses: (actions, info dict)."""
    import queue as _queue
    import torch
    torch.set_num_threads(2)
    mode = os.environ[ENV_MODE]
    assert mode in ("shuffled", "zero"), mode
    os.chdir(E.RE.OOPSIE)
    pol = E.load_policy(run_dir, device, ckpt)
    assert pol.intent is not None and pol.mode == "condition", "intent ablation needs a condition-mode intent arm"
    model = pol.model
    zt = torch.tensor(np.asarray(zero_token(run_dir)["token"], np.float32), device=device)
    last = {}     # wid -> (reset_id, sampled intent np [n_tok, d]) of that worker's previous chunk
    ring = []     # [(wid, reset_id, intent)] most recent last
    n_calls = n_items = 0
    busy, t0 = 0.0, time.time()
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
        z = torch.from_numpy(np.stack([b[4] for b in batch])).float().to(device)
        wids = [b[0] for b in batch]
        rids = [b[5] if len(b) > 5 else None for b in batch]
        B = len(batch)
        with torch.no_grad():
            state, _ = model.split_state(prop)
            sampled = model.sample_intent(model.obs_tokens(seg, state), z)  # same z -> the sampled-mode intent
            use = torch.empty_like(sampled)
            prov = []
            for i in range(B):
                if mode == "zero":
                    use[i] = zt
                    prov.append({"src": "train_mean"})
                    continue
                got = None
                for s in range(1, B):
                    j = (i + s) % B
                    lj = last.get(wids[j])
                    if lj is not None and rids[j] != rids[i] and lj[0] != rids[i]:
                        got = ({"src": "batch_prev_chunk", "from_wid": int(wids[j]), "from_reset": lj[0]}, lj[1])
                        break
                if got is None:
                    for (w, r, v) in reversed(ring):
                        if w != wids[i] and r != rids[i]:
                            got = ({"src": "fallback_ring", "from_wid": int(w), "from_reset": r}, v)
                            break
                if got is None:
                    use[i] = zt
                    prov.append({"src": "fallback_mean"})
                else:
                    use[i] = torch.from_numpy(got[1]).to(device)
                    prov.append(got[0])
            a, _ = model.generate(seg, prop, init_noise=x0, intent=use)
        a = a.cpu().numpy()
        s_np = sampled.cpu().numpy()
        u_np = use.cpu().numpy()
        for i in range(B):  # update AFTER use: a worker's intent serves others from its next chunk on
            last[wids[i]] = (rids[i], s_np[i])
            ring.append((wids[i], rids[i], s_np[i]))
        del ring[:-RING]
        busy += time.time() - tb
        n_calls += 1
        n_items += B
        for i, b in enumerate(batch):
            resp_qs[b[0]].put((a[i], {"intent_used": u_np[i].tolist(), "intent_sampled": s_np[i].tolist(), **prov[i]}))
        if stop:
            break
    if stats_path:
        E._dump(stats_path, {"calls": n_calls, "items": n_items, "mean_batch": n_items / max(1, n_calls),
                             "gpu_busy_s": busy, "wall_s": time.time() - t0, "mode": mode, "final": True})


# ─────────────────────────────── worker ───────────────────────────────

def ablation_worker(slot, gpu, specs, req_qs, resp_q, job_q, res_q):
    """eval_wipe_v2.worker_main with (1) the reset id appended to each request, (2) provenance tallies and the
    intent_ablation field added to every record. Noise draws and episode code unchanged."""
    mode = os.environ[ENV_MODE]
    WP = E.WorkerPolicy

    def act(self, hist, gen):
        torch = self.torch
        seg, prop = self.model_inputs(hist)
        x0, z = self.draw_noise(gen)
        self.req_q.put((self.wid, {k: v[0].to(torch.uint8).numpy() for k, v in seg.items()}, prop[0].numpy(),
                        x0[0].numpy(), None if z is None else z[0].numpy(), getattr(self, "_cur_reset", None)))
        a, info = self.resp_q.get(timeout=E.SERVER_TIMEOUT_S)
        self.last_intent = info["intent_used"]
        tally = self.__dict__.setdefault("_prov", {})
        tally[info["src"]] = tally.get(info["src"], 0) + 1
        return a.astype(np.float64), x0[0].numpy()
    WP.act = act
    orig_episode = E._episode

    def episode(env, pol, job, spec, g):
        pol._cur_reset = job["reset"]["id"]
        pol._prov = {}
        rec = orig_episode(env, pol, job, spec, g)
        rec["intent_ablation"] = mode
        rec["intent_ablation_provenance"] = dict(pol._prov)
        rec["evaluator"] = "wipe_v2/eval_wipe_v2_ablate.py (counter-label alias, intent ablation)"
        return rec
    E._episode = episode
    E.worker_main(slot, gpu, specs, req_qs, resp_q, job_q, res_q)


# ─────────────────────────────── keys / main ───────────────────────────────

def main():
    argv = sys.argv[1:]
    mode = "sampled"
    if "--intent-ablation" in argv:
        i = argv.index("--intent-ablation")
        mode = argv[i + 1]
        del argv[i:i + 2]
    assert mode in MODES, mode
    sys.argv = [sys.argv[0]] + argv
    if mode == "sampled":
        return E.main()  # identical code path, keys and out dirs to eval_wipe_v2.py
    os.environ[ENV_MODE] = mode
    abl_sha = C.sha256_file(os.path.abspath(__file__))
    orig_spec = E.run_spec

    def run_spec(run_dir, ckpt, pools_doc, pool_names):
        sp = orig_spec(run_dir, ckpt, pools_doc, pool_names)
        meta = json.load(open(os.path.join(run_dir, "meta.json")))
        it = meta.get("intent") or {}
        assert it.get("mode") == "condition", f"{run_dir}: intent ablation needs a condition-mode intent arm"
        zt = zero_token(run_dir)
        for pool, v in sp["pools"].items():
            ks = dict(v["key_src"], intent_ablation=mode, ablation_code_sha256=abl_sha,
                      zero_token_sha256=hashlib.sha256(json.dumps(zt["token"]).encode()).hexdigest())
            key = hashlib.sha256(json.dumps(ks, sort_keys=True).encode()).hexdigest()[:12]
            base = os.path.join(os.environ.get("WV2_ABLATION_ROOT_SMOKE_ONLY") or ABL_DIR, sp["run_id"])
            v["out_dir"] = os.path.join(base, f"{pool}__{os.path.splitext(ckpt)[0]}__{mode}__{key}")
            v["eval_key"], v["key_src"] = key, ks
        return sp
    E.run_spec = run_spec
    E.inference_server = ablation_server
    E.worker_main = ablation_worker
    return E.main()


if __name__ == "__main__":
    main()
