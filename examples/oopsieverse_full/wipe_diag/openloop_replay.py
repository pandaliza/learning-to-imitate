"""H4: open-loop replay of teleop demo actions through the EVALUATOR's env construction
(baseline rollout_eval.make_env: continuous-gripper patch, segmentation rendering on, EGL GPU 3), starting from
the demo's own initial state (ep_meta + model_xml + states[0], via rollout_eval.apply_reset). Steps actions[1:]
with no state forcing; logs markers wiped and success. Compared with the study's earlier open-loop replay
(runs/replay/open_teleop/*.npz, render-free env)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.dirname(HERE)
for p in (os.path.join(FULL, "matrix"), os.path.join(FULL, "baseline"), FULL, HERE):
    sys.path.insert(0, p)
import rollout_eval as RE  # noqa: E402
from wipe_common import demo_actions, demo_reset  # noqa: E402

PRIOR = "/edrive1/lizad/oopsieverse_full_study/runs/replay/open_teleop/wipe_countertop__{cond}__{demo}.npz"


def main():
    demos = sys.argv[2].split(",")
    os.chdir(RE.OOPSIE)
    from ctxfix import install_ctx_free
    env = RE.make_env("wipe_countertop", use_camera_obs=False)
    install_ctx_free(env)
    rows = []
    for d in demos:
        cond, demo = d.split("/")
        r = demo_reset(cond, demo)
        RE.apply_reset(env, r)
        acts = demo_actions(cond, demo)
        wiped, succ = [], []
        for a in acts[1:]:
            obs, rew, done, info = env.step(a)
            wiped.append(int(info["total_wiped"]))
            succ.append(RE.success_now(env))
        prior = np.load(PRIOR.format(cond=cond, demo=demo))["rp_wiped"]
        row = {"demo": d, "steps": len(wiped), "final_wiped": wiped[-1], "final_success": bool(succ[-1]),
               "prior_openloop_final_wiped": int(prior[-1]),
               "wiped_trace_equal_to_prior": bool(len(prior) == len(wiped) and np.array_equal(prior, wiped))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    json.dump(rows, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
