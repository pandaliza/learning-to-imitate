"""Sampling-mode check: different x0 ~ N(0, I) -> distinct action chunks; same x0 -> same chunk.

At the reset observation of a few resets (and after 40 policy steps), draw M chunks with distinct
noise and 2 with identical noise from the evaluator's Policy (official generate_action, Euler 16).
Writes outputs/oopsieverse_full_study/baseline/noise_check_<tag>.json.
Usage: python noise_check.py RUN_DIR CKPT TAG
"""
import json
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rollout_eval as R  # noqa: E402
from split_data import STUDY  # noqa: E402

M = 16
_KEEP = []


def chunks_at(pol, hist, seeds):
    import torch
    out = []
    for s in seeds:
        a, _ = pol.act(hist, torch.Generator().manual_seed(int(s)))
        out.append(a)
    return np.stack(out)


def stats(C):
    n = len(C)
    d = [np.linalg.norm(C[i, :, :7] - C[j, :, :7]) for i in range(n) for j in range(i + 1, n)]
    return {"mean_pairwise_l2_dims0to6": float(np.mean(d)), "min_pairwise_l2": float(np.min(d)),
            "per_dim_std_first_action": C[:, 0, :].std(axis=0).round(5).tolist(),
            "n_distinct": int(len({c.round(6).tobytes() for c in C}))}


def main(run_dir, ckpt, tag):
    import torch
    os.chdir(R.OOPSIE)
    pol = R.Policy(run_dir, ckpt=ckpt, device="cuda:0")
    env = R.make_env("pick_egg", use_camera_obs=False)
    _KEEP.append(env)
    res = []
    for seed in (900000, 900001, 900002):
        obs, _ = R.apply_reset(env, {"seed": seed})
        lut = pol.seg_lut(env.seg_id_to_class)
        hist = deque([pol.frame(obs, lut)] * 2, maxlen=2)
        for phase in ("reset", "after_40_steps"):
            if phase == "after_40_steps":
                g = torch.Generator().manual_seed(7)
                for _ in range(5):
                    a, _ = pol.act(hist, g)
                    for x in a:
                        obs = env.step(x)[0]
                        hist.append(pol.frame(obs, lut))
            C = chunks_at(pol, hist, range(1000, 1000 + M))
            same = chunks_at(pol, hist, [42, 42])
            x0 = torch.zeros(1, pol.cfg.action_chunk_size, pol.cfg.action_dim, device=pol.device)
            seg, prop = pol.model_inputs(hist)
            with torch.no_grad():
                zero = pol.model.generate_action(seg, prop, init_noise=x0)[0].cpu().numpy()  # diagnostic only
            r = {"reset_seed": seed, "phase": phase, "distinct_noise": stats(C),
                 "same_noise_maxabs_diff": float(np.abs(same[0] - same[1]).max()),
                 "zero_noise_dist_to_sample_mean_l2_DIAGNOSTIC_ONLY": float(np.linalg.norm(zero[:, :7] - C.mean(0)[:, :7])),
                 "sample_mean_first_action": C.mean(0)[0].round(4).tolist()}
            res.append(r)
            print(r, flush=True)
    out = {"run_dir": run_dir, "ckpt": ckpt, "M": M, "sampler": "Euler 16 NFE, x0 ~ N(0,I) via torch.Generator",
           "results": res}
    json.dump(out, open(os.path.join(STUDY, f"baseline/noise_check_{tag}.json"), "w"), indent=1)
    os._exit(0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
