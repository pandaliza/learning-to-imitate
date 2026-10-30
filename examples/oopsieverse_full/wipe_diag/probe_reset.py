"""Wipe diagnosis probe (new script; frozen evaluator untouched).

For a few eval_dev bank resets and a few demo initial states, report:
  * env.ee_force_bias after the reset as the evaluator does it (bank: env.reset() incl. IK + 5 zero steps;
    demo: ep_meta + model_xml + states[0], exactly as playback/replay do),
  * the bias after the first zero-action step, ||ee_force - bias|| at rest,
  * per-view pixel counts of each vocabulary class in the policy's segmentation input.
Rendering: EGL on GPU 3 only (MUJOCO_EGL_DEVICE_ID=3, CUDA_VISIBLE_DEVICES="").
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for p in (os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL):
    sys.path.insert(0, p)
import rollout_eval as RE  # noqa: E402
from wipe_common import demo_reset, VOCAB_RUN  # noqa: E402


def seg_counts(pol, obs, lut):
    segs, _ = pol.frame(obs, lut)
    out = {}
    for k, v in segs.items():
        cam = k.split("/")[1]
        out[cam] = {name: int((v == i).sum()) for name, i in pol.class_to_id.items()}
    return out


def main():
    os.chdir(RE.OOPSIE)
    from ctxfix import install_ctx_free
    env = RE.make_env("wipe_countertop", use_camera_obs=False)
    install_ctx_free(env)
    pol = RE.Policy(VOCAB_RUN, ckpt="step_30000.pth", load_model=False)
    import rollout_eval_matrix as REM
    resets = [("bank", r) for r in REM.dev_resets("wipe_countertop")[:4]]
    resets += [("demo", demo_reset(c, d)) for c, d in [("without", "demo_0"), ("with", "demo_10"), ("with", "demo_0")]]
    rows = []
    for kind, r in resets:
        obs, _ = RE.apply_reset(env, r)
        lut = RE.Policy.seg_lut(pol, env.seg_id_to_class)
        arm = env.robots[0].arms[0]
        b0 = np.asarray(env.ee_force_bias.get(arm, np.zeros(3))).tolist()
        f0 = np.asarray(env.robots[0].ee_force[arm]).tolist()
        sc = seg_counts(pol, obs, lut)
        a = np.zeros(12)
        a[6] = -1.0
        obs, *_ = env.step(a)
        b1 = np.asarray(env.ee_force_bias.get(arm)).tolist()
        f1 = np.asarray(env.robots[0].ee_force[arm])
        rows.append({"kind": kind, "id": r.get("id"), "bias_after_reset": b0, "ee_force_after_reset": f0,
                     "bias_after_step1": b1, "ee_force_step1": f1.tolist(),
                     "force_minus_bias_norm_step1": float(np.linalg.norm(f1 - np.asarray(b1))),
                     "seg_id_to_class_has_dirt": "dirt" in env.seg_id_to_class.values(),
                     "n_markers": len(env.markers), "wiped": len(env.wiped_markers),
                     "seg_counts_reset": sc})
        print(json.dumps(rows[-1]), flush=True)
    out = os.path.join(sys.argv[1] if len(sys.argv) > 1 else ".", "probe_reset.json")
    json.dump(rows, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
