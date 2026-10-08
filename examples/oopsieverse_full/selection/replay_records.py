"""H8: re-run completed continuation records in a fresh process and compare (determinism given snapshot + stored
candidate chunk + keyed tail seed). Inference: the same fixed-16 batch decode as the server, on GPU 0, one request at a
time (so different batch companions than in the original run). Rendering: GPU-3 EGL (run under guard.py).

  python replay_records.py --n 6 --out OUT.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import branch_worker as BW  # noqa: E402
import harness as H  # noqa: E402
import snapshot_damage_state as SDS  # noqa: E402
import fast_contacts  # noqa: E402

FIELDS = ["steps", "success", "outcome", "primary_safe_success", "secondary_safe_success", "compat_safe_success",
          "entity_final_health", "link_final_health", "worst_entity_loss", "first_damage_step", "first_success_step",
          "max_contact_force_N", "max_impact_force_target_N", "avg_env_health_over_time", "egg_height_final",
          "termination", "candidate_chunk_sha16", "handoff"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)
    fast_contacts.install()
    root = BW.root_dir("pick_egg", "sel_dev_parents", 0)
    files = sorted(glob.glob(f"{root}/branches/*/p*/c*_r*.json"))
    rng = np.random.default_rng(20261008)
    pick = [files[i] for i in rng.choice(len(files), size=min(a.n, len(files)), replace=False)]
    os.chdir(H.RE.OOPSIE)
    from ctxfix import install_ctx_free
    rd = BW.run_dirs("pick_egg", 0)
    # same inference path as production: a GPU-0 server process (fixed-16 batches); this process has no CUDA
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    req_q, resp_q = ctx.Queue(), ctx.Queue()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    srv = ctx.Process(target=H.inference_server, args=(rd, BW.CKPT, "cuda:0", req_q, [resp_q], None, 16))
    srv.start()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    dec = H.RemoteDecoder(0, req_q, resp_q)
    env = H.RE.make_env("pick_egg", use_camera_obs=False)
    install_ctx_free(env)
    frames = H.Frames(rd["amix"])
    out = []
    for f in pick:
        rec = json.load(open(f))
        gd = os.path.dirname(f)
        snap = SDS.load(BW.state_path(root, rec["parent_index"]) + ".pkl")
        c = np.load(f"{gd}/candidates.npz")
        arm = rec["arm"]
        ai = H.arm_info(rd[arm])
        t0 = time.time()
        ep = SDS.rebuild_and_restore(env, snap)
        lut = frames.lut(env.seg_id_to_class)
        chunk = c["chunks"][rec["candidate_id"]]
        H.exec_chunk(env, ep, chunk, frames, lut)
        gen = torch.Generator().manual_seed(rec["tail_seed"])
        handoff = {}

        class D:
            def decode(self, arm_, seg, prop, x0, z):
                return dec.decode(arm_, seg, prop, x0, z)
        if not ep["done"]:
            x0t, zt = H.draw_chunk_noise(gen, ai)
            ch, _i, _l = H.policy_chunk(D(), arm, frames, ep, x0t, zt)
            handoff = {"tail_first_action": ch[0].tolist(), "cand_last_action": chunk[7].tolist(),
                       "arm_cmd_l2": float(np.linalg.norm(ch[0][:6] - chunk[7][:6])), "gripper_diff": float(ch[0][6] - chunk[7][6])}
            H.exec_chunk(env, ep, ch, frames, lut)
            H.run_policy(env, ep, D(), arm, ai, frames, lut, gen)
        new = H.finalize(ep)
        import hashlib
        new["candidate_chunk_sha16"] = hashlib.sha256(np.ascontiguousarray(chunk).tobytes()).hexdigest()[:16]
        new["handoff"] = handoff
        diff = [k for k in FIELDS if json.dumps(rec.get(k), sort_keys=True, default=float) != json.dumps(new.get(k), sort_keys=True, default=float)]
        r = {"record": os.path.relpath(f, root), "identical": not diff, "differing_fields": diff,
             "steps": rec["steps"], "outcome": rec["outcome"], "wall_s": time.time() - t0}
        out.append(r)
        print(json.dumps(r), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    req_q.put(None)
    srv.join(timeout=60)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    main()
